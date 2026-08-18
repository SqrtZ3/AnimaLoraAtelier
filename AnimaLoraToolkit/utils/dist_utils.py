"""单机多卡数据并行（DDP）—— opt-in，未用 torchrun 启动时全程 no-op。

设计取舍：**为什么不用 ``torch.nn.parallel.DistributedDataParallel``**
----------------------------------------------------------------------
本仓库训练的是**冻结底模 + 注入的 LoRA/LoKr**，可训练参数只有几十 MB；而 DDP 的
优势（把 all-reduce 与 backward overlap）要以"把 model 包一层"为代价，那一层在这个
训练器里代价很高：

1. ``anima_train.py`` 有十几处直接调用 ``model.xxx`` / 自定义前向路径
   （NaViT 打包、DPO、GAF、CSFlow、NCP、LoRA-One 预热、telemetry 探针），包一层
   ``DistributedDataParallel`` 后这些调用点全部要改写成 ``model.module.xxx``。
2. 更要命的是 **module_dropout / T-LoRA / LoKr 的逐步条件分支** 会让"这一步哪些参数
   参与了计算"每步都不同 —— 原生 DDP 必须开 ``find_unused_parameters=True``（每步多一遍
   参数图遍历），且与 grad checkpoint 组合时在若干 torch 版本上直接报错。

所以这里走**手动梯度 all-reduce**：只在累积周期边界、只对可训练参数做一次集合通信。
代价是失去了通信/计算 overlap；收益是模型侧**零改动**、所有进阶功能原样可用。
通信量小（LoRA 梯度几十 MB），在 PCIe 互联的国产 8 卡机上这笔交换是划算的 ——
但**具体划不划算要看实测**，``tools/dcu_probe.py --dist`` 会打出本机的 all-reduce 带宽
与"占单步时间的比例"。

数值语义
--------
每个 rank 拿到全局 batch 列表的一个**互斥子集**（见 ``ShardedBatchSampler``），
各自 backward 出本地梯度，边界处对梯度取**跨 rank 平均**，再照常裁剪/step。
这与单卡跑一个 world_size 倍大的 batch **不完全等价**：

* 若各 rank 的 micro-batch 样本数不同（ARB 分桶的桶尾、NaViT 打包的每包图数不等），
  "各 rank 均值再取均值" ≠ "全局样本均值"，小 batch 的样本被加权更高。
  这与 PyTorch 原生 DDP 的语义完全一致（原生 DDP 也是无权平均），不是本实现引入的偏差，
  但**必须知道它存在**。
* 有效 batch 变成 world_size 倍 → 学习率通常要重新考虑。本模块**不自动改 LR**
  （那是未经请求的行为改变），只在启动时把有效 batch 打到日志里。

死锁面（本模块存在的主要理由）
------------------------------
数据并行的坑不是算错，是**挂住**。任何"某个 rank 走了不同分支、少调了一次集合通信"
都会让整个 job 卡在通信上直到超时。本训练器里有两条 per-rank 的提前退出路径
（micro-batch loss 非有限、整周期梯度非有限），它们本来就是数据相关的 —— 在多卡下
必须先把"要不要跳过"变成**所有 rank 一致的决定**，见 ``all_ranks_clean()``。
"""

from __future__ import annotations

import datetime
import logging
import os

logger = logging.getLogger(__name__)


class DistContext:
    """分布式上下文。``enabled=False`` 时每个方法都是 no-op / 恒等映射。

    单卡（没用 torchrun 启动、或 WORLD_SIZE=1）时 ``enabled`` 恒为 False，
    调用方无需写任何 ``if``，行为与本模块加入前逐字节一致。
    """

    def __init__(self, enabled=False, rank=0, world_size=1, local_rank=0, backend="",
                 device=None):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.backend = backend
        # 集合通信缓冲所在设备。nccl/RCCL 只吃显存张量，gloo 吃 CPU 张量 —— 显式记下来
        # 而不是写死 "cuda"，这样 CPU+gloo 也能跑（本地单测就靠它验证数值语义）。
        self._device = device or ("cuda" if backend != "gloo" else "cpu")
        self._flat = None          # 梯度打平缓冲（懒分配，跨 step 复用）
        self._flat_numel = 0

    # ── 基本属性 ──────────────────────────────────────────────────────────────

    @property
    def is_main(self) -> bool:
        """是否是主 rank。单卡时恒为 True。

        所有**有副作用**的操作（写 checkpoint、出采样图、起监控端口、推 wandb）
        都必须只在主 rank 做：8 个进程同时写同一个 safetensors 会写出损坏文件，
        同时 bind 8765 端口会有 7 个失败。
        """
        return (not self.enabled) or self.rank == 0

    @property
    def tag(self) -> str:
        """日志前缀。单卡时为空串（日志逐字节不变）。"""
        return "" if not self.enabled else f"[rank{self.rank}/{self.world_size}] "

    @property
    def rank_suffix(self) -> str:
        """给**每个 rank 都要写**的文件用的文件名后缀。单卡时为空串（文件名不变）。

        用于 stage_timing.csv 这种"各 rank 的值不同、且差异本身有信息量"的诊断产物：
        8 个进程往同一个 csv 追加会写出交错的坏行，而分文件既不冲突、又能看出
        掉队 rank（某张卡持续慢 → 是它在拖整个 job，因为集合通信要等最慢的那个）。
        """
        return "" if not self.enabled else f"_rank{self.rank}"

    # ── 集合通信 ──────────────────────────────────────────────────────────────

    def barrier(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist
        dist.barrier()

    def all_ranks_clean(self, clean: bool) -> bool:
        """跨 rank 的逻辑与：**任一** rank 报 dirty，所有 rank 都得到 False。

        用途：把"这一累积周期要不要作废"变成全体一致的决定。少了它，出 NaN 的那个
        rank 会 ``continue`` 掉、不参与后面的梯度 all-reduce，其余 rank 在集合通信上
        无限等待 —— 表现为训练毫无征兆地卡死，且不报任何错。
        """
        if not self.enabled:
            return bool(clean)
        import torch
        import torch.distributed as dist
        t = torch.tensor([1.0 if clean else 0.0], device=self._device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item() > 0.5)

    def mean_scalar(self, value: float) -> float:
        """标量跨 rank 求平均（用于日志里的 loss —— 否则日志只反映 rank0 的那份数据）。"""
        if not self.enabled:
            return float(value)
        import torch
        import torch.distributed as dist
        t = torch.tensor([float(value)], device=self._device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item()) / self.world_size

    def all_reduce_grads_(self, params) -> int:
        """把 ``params`` 的 ``.grad`` 原地换成跨 rank 平均值。返回参与的元素总数。

        实现要点：

        * **打平成一个 fp32 缓冲再通信**。分成上百个小 all-reduce（LoKr 会注入 100+ 个
          小张量）的延迟开销远大于带宽开销；一次大包是这里唯一合理的做法。
          用 fp32 而不是原 dtype：跨 rank 求和用 bf16 累加会丢有效位，且各 rank grad
          dtype 可能不一致（打平后布局就对不上了）。代价是 bf16 梯度的通信量翻倍 ——
          在几十 MB 的量级上无所谓。
        * **``grad is None`` 的参数也要占位**。module_dropout 会让某个 rank 这一步没碰
          某个模块 → 它的 grad 是 None，而别的 rank 有。若跳过这些参数，各 rank 的打平
          布局就不同了，all-reduce 出来的是**逐元素错位**的垃圾（且不会报错）。
          所以：None 按零参与，reduce 完把平均值写回去（那本来就是它应得的梯度）。
        * 参数顺序必须跨 rank 一致 —— 调用方传的是同一份 ``optimizer.param_groups``
          展开的列表，模型构造与注入顺序在各 rank 相同，故一致。
        """
        if not self.enabled:
            return 0
        import torch
        import torch.distributed as dist

        params = [p for p in params]
        numel = sum(p.numel() for p in params)
        if numel == 0:
            return 0
        if self._flat is None or self._flat_numel != numel:
            self._flat = torch.zeros(numel, dtype=torch.float32, device=self._device)
            self._flat_numel = numel
        flat = self._flat

        off = 0
        for p in params:
            n = p.numel()
            if p.grad is None:
                flat[off:off + n].zero_()
            else:
                flat[off:off + n].copy_(p.grad.detach().reshape(-1))
            off += n

        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)

        off = 0
        for p in params:
            n = p.numel()
            chunk = flat[off:off + n].view_as(p)
            if p.grad is None:
                # 本 rank 这一步没算到它，但别的 rank 算到了 —— 按平均值补上。
                p.grad = chunk.to(dtype=p.dtype).clone()
            else:
                p.grad.copy_(chunk)
            off += n
        return numel

    def shutdown(self) -> None:
        if not self.enabled:
            return
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:      # 退出路径，不让清理失败盖住真正的错误
            logger.warning("[dist] destroy_process_group 失败（忽略）: %s", e)


def env_world_size() -> int:
    """torchrun 注入的 WORLD_SIZE（未用 torchrun 启动时为 1）。不碰设备，可最早调用。"""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
    except ValueError:
        return 1


def bind_device() -> int:
    """把本进程绑到 ``LOCAL_RANK`` 对应的那张卡上，返回 local_rank。单卡时是 no-op。

    **必须在任何会创建设备上下文的调用之前执行**（包括
    ``torch.cuda.get_device_properties()`` 这种看起来只是"查一下"的 API —— 它会在
    device 0 上建上下文）。漏了的后果不是报错，而是 8 个进程全在 0 号卡上各留一份
    上下文，0 号卡凭空少几 GB 显存，且极易被误判成"模型太大"。
    """
    if env_world_size() <= 1:
        return 0
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def init_from_env(backend: str = "", timeout_minutes: int = 60) -> DistContext:
    """按 torchrun 注入的环境变量初始化进程组。没有这些变量就返回 no-op 上下文。

    ``backend`` 留空时按平台选：CUDA / DCU(HIP，走 RCCL) 都用 ``"nccl"``
    —— ROCm 生态里 RCCL 就注册在 ``nccl`` 这个名字下，写 ``"rccl"`` 反而不认。
    可用环境变量 ``ANIMA_DIST_BACKEND`` 覆盖（例如互联有问题时临时退 ``gloo`` 验证正确性）。

    ``timeout_minutes`` 默认放到 60 分钟：主 rank 独占执行采样出图 / eval 全量过一遍
    dataloader 时，其余 rank 就阻塞在下一次集合通信上，NCCL 默认 10~30 分钟的超时
    在大数据集上会误杀。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return DistContext(enabled=False)

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    backend = (backend or os.environ.get("ANIMA_DIST_BACKEND", "") or "nccl").strip().lower()

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"WORLD_SIZE={world_size} 说明是多卡启动，但 torch.cuda.is_available() 为 False。"
            "分布式训练需要每个进程都能看到自己的那张卡。"
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 超出本机可见设备数 {torch.cuda.device_count()}。"
            "检查 torchrun --nproc_per_node 与 HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES 是否一致。"
        )
    # 必须在 init_process_group 之前设好当前设备：NCCL/RCCL 按当前设备建 communicator，
    # 漏了会让多个 rank 抢同一张卡（表现为显存翻倍 + 速度腰斩，而不是报错）。
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=int(timeout_minutes)),
        )

    ctx = DistContext(enabled=True, rank=rank, world_size=world_size,
                      local_rank=local_rank, backend=backend)
    logger.info(
        "[dist] 进程组就绪：backend=%s rank=%d/%d local_rank=%d device=%s",
        backend, rank, world_size, local_rank, torch.cuda.get_device_name(local_rank),
    )
    return ctx


class ShardedBatchSampler:
    """把任意 batch sampler 按 rank 切成互斥子集。

    切法是 **stride 切片**：rank r 取全局第 ``r, r+W, r+2W, ...`` 个 batch。
    这样做而不是"按样本切数据集"，是因为本仓库的三个 sampler
    （``BucketBatchSampler`` / ``FitTokenBatchSampler`` / ``NavitPackBatchSampler``）
    的分批逻辑本身携带约束（同桶同分辨率、token 预算装包），从中间切样本会破坏它们；
    切**已经组好的 batch** 则完全不碰这些逻辑。

    ⚠ **长度必须对齐**：每个 rank 迭代的 batch 数必须严格相等，否则先跑完的 rank 会
    退出循环、不再参与集合通信，剩下的 rank 永远等下去。所以这里把总数截断到
    ``world_size`` 的整数倍，尾部余数（< world_size 个 batch）**每个 epoch 丢弃**。
    每个 epoch 洗牌不同，被丢的不是固定那几张图。
    """

    def __init__(self, base, rank: int, world_size: int):
        if world_size <= 1:
            raise ValueError("ShardedBatchSampler 只应在 world_size>1 时使用")
        self.base = base
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        limit = len(self)
        taken = 0
        for i, batch in enumerate(self.base):
            if i % self.world_size != self.rank:
                continue
            if taken >= limit:
                break
            taken += 1
            yield batch

    def __len__(self):
        return len(self.base) // self.world_size

    def reference_batches_for_batch_index(self, local_idx: int):
        """把本 rank 的**局部** batch 序号换算回全局序号再问底层 sampler。

        训练循环拿到的 ``batch_idx`` 是局部的；底层 sampler 的 reference-step 记账按
        全局序号建立。不换算就会对错行（reference step 计数悄悄偏掉）。
        """
        fn = getattr(self.base, "reference_batches_for_batch_index", None)
        if fn is None:
            return 0
        return fn(int(local_idx) * self.world_size + self.rank)

    def __getattr__(self, name):
        # set_epoch / set_accumulation_offset / batch_size / seed / drop_last …
        # 一律透传给底层 sampler（``dataloader_fingerprint`` 也靠这个读到原始字段；
        # ``world_size`` 是本类自己的实例属性，会先被正常查找命中，用于让指纹区分
        # "8 卡存的 state 拿到 4 卡上 resume"）。
        # 注意：只有实例字典里找不到时才会走到这里，不会遮蔽上面显式定义的成员。
        if name.startswith("__") or name in ("base", "rank", "world_size"):
            # 反 pickle 时 __init__ 不会被调用、__dict__ 为空，此时查 self.base 会再次
            # 落到这里 → 无限递归。显式抛 AttributeError 掐断。
            raise AttributeError(name)
        return getattr(self.base, name)


def guard_incompatible(args, ctx: DistContext) -> None:
    """多卡下已知会**死锁或算错**的配置，构造期 fail-fast。

    只在 ``ctx.enabled`` 时生效；单卡完全不触发。
    """
    if not ctx.enabled:
        return

    problems: list[str] = []

    if int(getattr(args, "effective_batch_size", 0) or 0) > 0:
        problems.append(
            "effective_batch_size（按真实图片数累积的 sample-window）：累积边界由**每个 batch 的"
            "样本数**决定，而各 rank 分到的 batch 图数不同 → 各 rank 在不同的 batch_idx 触发"
            "边界 → 集合通信错位、直接死锁。\n"
            "      多卡请改用 grad_accum（边界只由 batch_idx 决定，跨 rank 恒等）。"
        )

    if bool(getattr(args, "dpo_enabled", False)):
        problems.append(
            "dpo_enabled：输家池按 global_step 周期性重生成，改的是各 rank **本地**的样本池，"
            "没有任何同步机制 → 各 rank 的池会分叉，训的不再是同一个目标。"
            "首版不支持，请设 dpo_enabled: false。"
        )

    _lora_one = int(getattr(args, "lora_one_init_steps", 0) or 0)
    if _lora_one > 0:
        problems.append(
            f"lora_one_init_steps={_lora_one}：LoRA-One 预热阶段自己迭代 dataloader 做全参 backward，"
            "那段没有接梯度同步 → 各 rank 会算出不同的谱对齐初始化因子，训练一开始就不一致。"
            "首版不支持，请设 lora_one_init_steps: 0"
            "（或先单卡跑完预热、存下 LoRA，再多卡 resume_lora 接着训）。"
        )

    if problems:
        raise ValueError(
            f"以下配置项在多卡（world_size={ctx.world_size}）下不受支持：\n  - "
            + "\n  - ".join(problems)
        )

    # ── 不致命但语义会变的：只警告，不拦。用户有权知道自己在训什么 ──────────────
    for name, why in (
        ("gaf_enabled",
         "GAF 的逐图信任是**进程本地**状态，而每个 epoch 重洗牌后同一张图会落到不同 rank → "
         "信任历史被打散、trust_decay 的累积效果被稀释到 1/world_size 量级。不会崩，但"
         "多卡下的 GAF 与单卡不是同一个东西。"),
    ):
        if bool(getattr(args, name, False)):
            logger.warning("[dist] %s 在多卡下语义改变：%s", name, why)

    if bool(getattr(args, "adaptive_timestep", False)):
        logger.warning(
            "[dist] 自适应 timestep 控制器的状态也是进程本地的：各 rank 会独立演化出不同的 "
            "t 采样分布（没有同步）。等价于 world_size 个控制器投票，不等价于单卡的一个控制器。"
        )
