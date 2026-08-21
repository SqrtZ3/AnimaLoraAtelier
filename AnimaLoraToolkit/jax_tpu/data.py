"""TPU 路线的数据侧：读磁盘缓存 -> 打包/分桶 -> 组装 batch。

## 与 PyTorch 侧的分工

TPU 后端**不做任何编码**。两份缓存都由 PyTorch 侧离线产出，两条训练路线（GPU/NPU
与 TPU）共用同一批文件：

  `<stem>.npz`           图像 latent。由仓库现成的 `trainer/data.py:CachedLatentDataset`
                         产出（Kohya 风格），**不需要为 TPU 另造一套**。
                         键：`latent` [C, T, H, W]（C=16）、`bucket_w` / `bucket_h`、
                         `dtype_kind`（"bf16" 时磁盘上是 uint16 view）；
                         开了 flip_augment 时还有 `latent_flipped`。
  `<stem>.textfeat.npz`  文本 cross 条件。由 `tools/cache_text_features.py` 产出。
                         键：`cross` [512, 1024] bf16(uint16 位模式)、`mask`、
                         `caption`、`meta`。

**cross 不是 Qwen hidden**（口径极易抄错）：真实链路是
`llm_adapter(qwen_emb, t5_ids, t5_attn, qwen_attn) * t5_w`（`anima_train.py:3720`）。
底模里有 118 个 `llm_adapter` 键、且默认不注入 LoRA（冻结），所以输出可安全缓存。
只存 Qwen hidden 会让训练一直条件错误且**不报错** —— `cache_text_features.py` 里
那句 `if c.shape[0] != max_length: raise` 就是拦这个的，别删。

## bf16 的读法（抄错不报错）

numpy 没有原生 bfloat16，磁盘上是 uint16 位模式。**直接 `.view(np.float16)` 会把位
模式解释错**（静默出错），必须先转 uint16 再 bitcast：
`jax.lax.bitcast_convert_type(jnp.asarray(a_uint16), jnp.bfloat16)`。

## 缓存目录里**不需要放图片**

`_stems` 优先按图片文件名推 stem，一张图片都没有时回退到按 `<stem>.npz` 推。
TPU 侧从不读像素（只取文件名），caption 也已烘焙进 textfeat —— 而缓存目录通常要
上传到 Kaggle 这类外部平台，那边的内容审查会因为训练图直接删库。所以**上传用的
缓存目录只放 npz**。闸门 `tests/check_cache_scan.py` 钉死「两种模式样本集逐条相同」
与「sidecar 不被当成样本本体」。

## multiscale（`navit_multiscale`）

PyTorch 侧的多尺度阶梯把每张图的**低 token 档缩放副本**编码进独立的 sidecar
`<stem>.ms{target}.npz`（trainer/data.py:2216），原生份的 npz 不受影响。所以 TPU
侧不需要任何图像处理，**只要把 sidecar 也扫进样本列表**即可 —— 副本自带自己的
网格与 token 数，和其它异尺寸图一起被打包器混包，语义与 GPU 侧一致。
没跑过带 `--navit-multiscale` 的缓存就不会有 sidecar，开关打开时会 fail-fast，
不会静默退化成"只有原生份"。

## caption dropout

文本特征是**离线缓存**的，训练时手上没有编码器，没法临时算一个空 caption 的
条件。所以 `caption_dropout_rate > 0` 需要数据目录里有 `_empty.textfeat.npz`
（`tools/cache_text_features.py --empty-caption` 产出）。缺了直接报错 ——
悄悄退回"不 dropout"会让同一份 yaml 在两个后端上训出不同的条件鲁棒性。

## flip

`latent_flipped` 是**像素域翻转后再 encode** 的另一份（VAE conv encoder 非
flip-equivariant，latent 空间翻转 ≠ 像素空间翻转）。这里按 `flip_prob` 每次随机
二选一，与 `trainer/data.py:2515` 同语义。注意翻转会改变 (h, w) 的**内容**但不改
形状，所以不影响分桶/打包。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .packing import BucketPlanner, BucketStep, Packer, Pack
except ImportError:
    from packing import BucketPlanner, BucketStep, Packer, Pack

LATENT_CHANNELS = 16
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
#: `<stem>.ms<档>.npz` 多尺度 sidecar 的后缀（无图目录下推 stem 时要排掉）
_MS_SIDECAR = re.compile(r"\.ms\d+$")


def _read_maybe_bf16(z, key: str) -> np.ndarray:
    """把 npz 里的数组读成 fp32。bf16 存的是 uint16 位模式，要左移 16 位再当 fp32。

    用位运算而不是 jax.lax.bitcast：这一层是纯 host 侧的 numpy，不该把 jax 拖进来
    （数据加载在 CPU 线程里跑，起 jax 会争 TPU 客户端）。
    """
    a = z[key]
    kind = str(z["dtype_kind"]) if "dtype_kind" in z.files else "fp32"
    if kind == "bf16":
        if a.dtype != np.uint16:
            raise ValueError(f"{key}: dtype_kind=bf16 但磁盘 dtype 是 {a.dtype}，"
                             f"缓存不一致，别猜，重新生成")
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def patchify(latent: np.ndarray, patch: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    """latent [C, H, W] -> tokens [N, C*patch^2]，通道序 `(c ph pw)`。

    与 `models/anima_modeling_core.py:1585` 的
    `rearrange(x, "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)")` 同序（T=1）。
    **这个序写错 -> 统计量正常、逐元素全错**，是本仓库记录在案的静默错之一
    （历史上出过一次，rel 1.4）。
    """
    c, h, w = latent.shape
    if h % patch or w % patch:
        raise ValueError(f"latent {h}x{w} 不能被 patch {patch} 整除")
    gh, gw = h // patch, w // patch
    t = latent.reshape(c, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
    return t.reshape(gh * gw, c * patch * patch), (gh, gw)


@dataclass
class Sample:
    """一条样本的磁盘指针 + 形状。**不持有像素/latent**，按需读。"""
    stem: Path
    latent_npz: Path
    text_npz: Path
    grid: Tuple[int, int]          # (gh, gw) = patch 后的 token 网格
    tokens: int                    # gh * gw
    ms_target: int = 0             # >0 = navit_multiscale 的缩放副本（该档 token 上限）
    txt_len: int = 0               # krea2：有效 caption token 数（anima 不用，恒 0）

    @property
    def name(self) -> str:
        return self.stem.name if not self.ms_target else \
            f"{self.stem.name}.ms{self.ms_target}"


class CacheDataset:
    """扫描一个目录，把成对的 `<stem>.npz` / `<stem>.textfeat.npz` 收成样本列表。

    **不做任何编码，也不写盘。** 缺任一份就 fail-fast 并报出缺哪个 —— 静默跳过会
    让"训练集少了一半"这种事完全看不出来。

    `family="krea2"` 时文本缓存换格式：`<stem>.textfeat.npz` 里是
    `txt` [L, 12, 2560]（Qwen3-VL 12 层堆叠，bf16 位模式），**不定长** ——
    navit 只打包有效 token（krea2_modeling.py 的口径），L 被记下来参与装箱。
    """

    def __init__(self, data_dir, patch: int = 2, txt_len: int = 512,
                 crossattn_dim: int = 1024, flip_prob: float = 0.0,
                 rng: Optional[np.random.RandomState] = None,
                 repeats: int = 1, multiscale: bool = False,
                 caption_dropout: float = 0.0, family: str = "anima"):
        self.dir = Path(data_dir)
        self.patch, self.txt_len, self.crossattn_dim = patch, txt_len, crossattn_dim
        self.flip_prob = float(flip_prob)
        self.rng = rng or np.random.RandomState(0)
        self.repeats = max(int(repeats), 1)
        self.multiscale = bool(multiscale)
        self.caption_dropout = float(caption_dropout)
        self.family = str(family or "anima").lower()
        self.samples: List[Sample] = []
        self._empty_ctx: Optional[np.ndarray] = None
        self._scan()
        if self.caption_dropout > 0:
            q = self.dir / "_empty.textfeat.npz"
            if not q.exists():
                raise FileNotFoundError(
                    f"caption_dropout_rate={self.caption_dropout} 需要 {q.name}"
                    f"（空 caption 的文本特征）。文本编码是离线的，训练时算不出来。\n"
                    f"  跑 tools/cache_text_features.py --empty-caption 补上。")
            self._empty_ctx = self._read_ctx(q, "_empty")

    def _stems(self) -> List[Path]:
        """样本 stem 列表。有图片就按图片名推；**一张图片都没有时按 `<stem>.npz` 推**。

        回退分支是为了让缓存 dataset 能**完全不含像素**。TPU 侧本来就不读图片
        （这个函数只取文件名，从不 open），caption 也早烘焙进 `<stem>.textfeat.npz`，
        所以原图放进缓存目录是纯多余的暴露 —— 而缓存目录常常要上传到 Kaggle 这类
        外部平台，那里的内容审查会因为训练图直接删库。别再靠 0 字节占位图绕。

        回退时要把 sidecar 从 stem 里排掉，否则 `a.textfeat.npz` 会被当成一个叫
        `a.textfeat` 的样本，然后去找不存在的 `a.textfeat.npz` 的 textfeat：
          * `<stem>.textfeat.npz` —— 文本特征（`_empty.textfeat.npz` 一并排掉）
          * `<stem>.ms<档>.npz`   —— 多尺度副本，由 `_scan_multiscale` 挂到本体上
        """
        imgs = {p.with_suffix("") for p in self.dir.iterdir()
                if p.suffix.lower() in IMG_EXT}
        if imgs:
            return sorted(imgs)
        out = set()
        for p in self.dir.iterdir():
            if p.suffix.lower() != ".npz":
                continue
            base = p.name[:-len(".npz")]
            if base.endswith(".textfeat") or _MS_SIDECAR.search(base):
                continue
            out.add(p.with_suffix(""))
        return sorted(out)

    def _scan(self) -> None:
        missing_lat, missing_txt, bad, no_flip = [], [], [], []
        n_ms = 0
        stems = self._stems()
        for stem in stems:
            lat, txt = stem.with_suffix(".npz"), Path(str(stem) + ".textfeat.npz")
            if not lat.exists():
                missing_lat.append(stem.name)
                continue
            if not txt.exists():
                missing_txt.append(stem.name)
                continue
            with np.load(lat) as z:
                if "latent" not in z.files:
                    bad.append(f"{stem.name}: 无 latent 键")
                    continue
                shape = z["latent"].shape
                if self.flip_prob > 0 and "latent_flipped" not in z.files:
                    no_flip.append(stem.name)
            if len(shape) != 4 or shape[0] != LATENT_CHANNELS:
                bad.append(f"{stem.name}: latent 形状 {shape}（应为 [16, T, H, W]）")
                continue
            _, _, h, w = shape
            if h % self.patch or w % self.patch:
                bad.append(f"{stem.name}: latent {h}x{w} 不能被 patch {self.patch} 整除")
                continue
            gh, gw = h // self.patch, w // self.patch
            tl = 0
            if self.family == "krea2":
                tl = self._txt_len_of(txt, stem.name, bad)
                if tl < 0:
                    continue
            self.samples.append(Sample(stem, lat, txt, (gh, gw), gh * gw,
                                       txt_len=tl))
            n_ms += self._scan_multiscale(stem, txt, bad, tl)

        if self.multiscale and n_ms == 0:
            raise FileNotFoundError(
                f"multiscale=true 但 {self.dir} 下没有任何 `*.ms<档>.npz` 缓存。\n"
                f"  那些副本由 PyTorch 侧的缓存流程产出（trainer/data.py:2216），"
                f"这里不做图像处理。先带 --navit-multiscale 跑一遍缓存。")
        if no_flip:
            # `load_tokens` 里那句 `"latent_flipped" in z.files` 本身是安全的写法，
            # 但配上 `flip_augment: true` 就成了静默降级：yaml 说要翻转、实际一次
            # 都没翻，日志上完全看不出来。PyTorch 侧不会这样 —— trainer/data.py:2258
            # 在缓存缺 latent_flipped 时直接判缓存失效并重新编码。这里没有编码器可以
            # 重编，所以只能拦下来。
            raise FileNotFoundError(
                f"flip_augment 开着，但 {len(no_flip)} 个 latent 缓存里没有 "
                f"`latent_flipped`（如 {no_flip[:3]}）。\n"
                f"  翻转必须在**像素域**做完再 encode（VAE 卷积不是 flip-等变的），"
                f"训练时没有 VAE 补不了。\n"
                f"  要么带 flip 重跑一遍 PyTorch 侧的 latent 缓存，"
                f"要么把 yaml 的 flip_augment 关掉 —— 别让它静默不生效。")
        if missing_lat or missing_txt or bad:
            parts = []
            if missing_lat:
                parts.append(f"缺 latent 缓存 {len(missing_lat)} 个"
                             f"（如 {missing_lat[:3]}）—— 先用 PyTorch 侧跑一遍缓存")
            if missing_txt:
                parts.append(f"缺 textfeat {len(missing_txt)} 个（如 {missing_txt[:3]}）"
                             f" —— 跑 tools/cache_text_features.py")
            if bad:
                parts.append(f"格式异常 {len(bad)} 个：{bad[:3]}")
            raise FileNotFoundError(
                f"{self.dir} 的缓存不完整，拒绝静默跳过：\n  " + "\n  ".join(parts))
        if not self.samples:
            raise FileNotFoundError(
                f"{self.dir} 里没找到任何样本：既没有图片文件，也没有"
                f"`<stem>.npz`（sidecar `.textfeat.npz` / `.ms<档>.npz` 不算）。")

    def _txt_len_of(self, txt: Path, name: str, bad: List[str]) -> int:
        """krea2：读 textfeat 的 `txt` 键形状，返回有效 caption token 数。出错 -1。"""
        with np.load(txt) as z:
            if "txt" not in z.files:
                bad.append(f"{name}: 无 txt 键（krea2 文本缓存是 [L,12,2560] 的 "
                           f"txt，不是 anima 的 cross —— 检查缓存是不是用错编码器跑的）")
                return -1
            shape = z["txt"].shape
        if len(shape) != 3 or shape[1] == 0 or shape[2] == 0:
            bad.append(f"{name}: txt 形状 {shape}（应为 [L, n_layers, dim]）")
            return -1
        if shape[0] < 1:
            bad.append(f"{name}: caption 0 token —— krea2 每图至少 1 个有效 token")
            return -1
        return int(shape[0])

    def _scan_multiscale(self, stem: Path, txt: Path, bad: List[str],
                         txt_len: int = 0) -> int:
        """扫 `<stem>.ms<档>.npz` sidecar，每个当成一条独立样本（共用同一份 caption）。"""
        if not self.multiscale:
            return 0
        n = 0
        for q in sorted(stem.parent.glob(f"{stem.name}.ms*.npz")):
            m = re.fullmatch(re.escape(stem.name) + r"\.ms(\d+)", q.stem)
            if not m:
                continue
            with np.load(q) as z:
                if "latent" not in z.files:
                    bad.append(f"{q.name}: 无 latent 键")
                    continue
                shape = z["latent"].shape
            if len(shape) != 4 or shape[0] != LATENT_CHANNELS:
                bad.append(f"{q.name}: latent 形状 {shape}（应为 [16, T, H, W]）")
                continue
            _, _, h, w = shape
            if h % self.patch or w % self.patch:
                bad.append(f"{q.name}: latent {h}x{w} 不能被 patch 整除")
                continue
            gh, gw = h // self.patch, w // self.patch
            self.samples.append(Sample(stem, q, txt, (gh, gw), gh * gw,
                                       ms_target=int(m.group(1)), txt_len=txt_len))
            n += 1
        return n

    # ── 读取 ──────────────────────────────────────────────────────────────────
    def load_tokens(self, s: Sample) -> np.ndarray:
        """[N, 64] 的 patchified latent（fp32）。T 维必须是 1（图像模型）。"""
        with np.load(s.latent_npz) as z:
            key = "latent"
            if (self.flip_prob > 0 and "latent_flipped" in z.files
                    and self.rng.rand() < self.flip_prob):
                key = "latent_flipped"          # 像素域翻转后编码的那一份
            lat = _read_maybe_bf16(z, key)
        if lat.shape[1] != 1:
            raise ValueError(f"{s.name}: latent 的 T 维是 {lat.shape[1]}，"
                             f"本路径只支持图像（T=1）")
        tok, grid = patchify(lat[:, 0], self.patch)
        if grid != s.grid:
            raise ValueError(f"{s.name}: 网格 {grid} != 扫描时的 {s.grid}")
        return tok

    def _read_ctx(self, path: Path, name: str) -> np.ndarray:
        with np.load(path) as z:
            if self.family == "krea2":
                if "txt" not in z.files:
                    raise ValueError(f"{name}: 无 txt 键（krea2 文本缓存应为 "
                                     f"[L, n_layers, dim]，不是 anima 的 cross）")
                a = z["txt"]
                # krea2 的 12 层堆叠每条约 30MB fp32 —— **不升 fp32**，uint16 位
                # 模式原样返回，bitcast 留给 assemble 侧（host 内存与带宽都减半）。
                if a.dtype == np.uint16:
                    c = a
                else:
                    c = a.astype(np.float32)
                if c.ndim != 3:
                    raise ValueError(f"{name}: txt 形状 {c.shape}（应为 [L, n_layers, dim]）")
                return c
            a = z["cross"]
            c = ((a.astype(np.uint32) << 16).view(np.float32)
                 if a.dtype == np.uint16 else a.astype(np.float32))
        if c.shape != (self.txt_len, self.crossattn_dim):
            raise ValueError(f"{name}: cross 形状 {c.shape} != "
                             f"({self.txt_len}, {self.crossattn_dim})")
        return c

    def load_ctx(self, s: Sample) -> np.ndarray:
        """anima：[txt_len, crossattn_dim] 的 cross 条件（fp32，定长槽）。
        krea2：[L, n_layers, dim] 的 12 层文本特征（**变长**，量化槽的补齐在
        assemble 侧做）。

        `caption_dropout` 命中时换成**空 caption 的那一份**，不是置零 —— 置零
        得到的不是"无条件"，而是一个模型从没见过的越界条件。
        """
        if self._empty_ctx is not None and self.rng.rand() < self.caption_dropout:
            return self._empty_ctx
        c = self._read_ctx(s.text_npz, s.name)
        if self.family == "krea2" and c.shape[0] != s.txt_len:
            raise ValueError(f"{s.name}: txt 长度 {c.shape[0]} != 扫描时的 "
                             f"{s.txt_len}（缓存中途被改过？删掉重跑缓存）")
        return c

    @property
    def canvas_hw(self) -> Tuple[int, int]:
        """aux_spectral 的 FFT 画布（token 网格单位）= 数据集里各轴的最大值。

        必须逐轴取 max 而不是取"token 数最多那张图的形状"：宽高比不同的两张图
        可能一张最高、另一张最宽。
        """
        return (max(s.grid[0] for s in self.samples),
                max(s.grid[1] for s in self.samples))

    # ── 调度 ──────────────────────────────────────────────────────────────────
    def plan_packed(self, packer, shuffle: bool = True):
        """NaViT 打包：-> (steps, carry)，每个 step 是 `devices` 个同布局的 pack。

        krea2 走 K2Packer（text+image 同序列，caption 长度参与装箱）。"""
        idx = self._order(shuffle)
        samples = [self.samples[i] for i in idx]
        if self.family == "krea2":
            packs = packer.build_packs(samples,
                                       [s.tokens for s in samples],
                                       [s.txt_len for s in samples],
                                       [s.grid for s in samples])
        else:
            packs = packer.build_packs(samples,
                                       [s.tokens for s in samples],
                                       [s.grid for s in samples])
        return packer.plan_steps(packs)

    def plan_buckets(self, planner: BucketPlanner, shuffle: bool = True
                     ) -> Tuple[List[BucketStep], List]:
        """量化分桶（对照/兜底路线）：-> (steps, carry)。"""
        idx = self._order(shuffle)
        return planner.plan_steps([self.samples[i] for i in idx],
                                  [self.samples[i].tokens for i in idx],
                                  [self.samples[i].grid for i in idx])

    def _order(self, shuffle: bool) -> np.ndarray:
        """一个 epoch 的样本下标序列（含 `repeats`）。

        repeats 是"同一张图在一个 epoch 里出现几次"，与 PyTorch 侧同义；对
        "极大 epoch 一直训"的范式而言它只改变 epoch 的长度与洗牌粒度。
        """
        idx = np.tile(np.arange(len(self.samples)), self.repeats)
        return self.rng.permutation(idx) if shuffle else idx

    # ── 取回一步的 latent / ctx ────────────────────────────────────────────────
    def materialize_packed(self, packs):
        """-> (latents, ctxs)，形状与 `train.assemble_batch` 的要求一致：
        每个 pack 一个 list，纯填充段给 None。

        krea2：ctxs 是**变长**的 [L_i, n_layers, dim]（有效 token 原样取出，
        量化槽的补齐由 train.assemble_batch_k2 做）；纯填充段给 None。"""
        lats, ctxs = [], []
        for p in packs:
            ll, cc = [], []
            for it in p.items:
                if it is None:                  # FFD 装箱余量段
                    ll.append(None)
                    cc.append(None if self.family == "krea2" else
                              np.zeros((self.txt_len, self.crossattn_dim), np.float32))
                else:
                    ll.append(self.load_tokens(it))
                    cc.append(self.load_ctx(it))
            lats.append(ll)
            ctxs.append(cc)
        return lats, ctxs

    def materialize_bucket(self, step: BucketStep):
        """-> (latents, ctxs)，与 `train.assemble_batch_ragged` 的要求一致：
        按 `step.items` 顺序的扁平 list。"""
        return ([self.load_tokens(it) for it in step.items],
                [self.load_ctx(it) for it in step.items])

    # ── 诊断 ──────────────────────────────────────────────────────────────────
    def report(self) -> str:
        tk = sorted(s.tokens for s in self.samples)
        ar = sorted(s.grid[1] / s.grid[0] for s in self.samples)
        mid = tk[len(tk) // 2]
        return (f"{self.dir.name}: {len(tk)} 张 | token {tk[0]}-{tk[-1]} 中位 {mid} "
                f"| 离散度 {(tk[-1] - tk[0]) / max(mid, 1):.2f} "
                f"| 宽高比 {ar[0]:.2f}-{ar[-1]:.2f}\n"
                f"（离散度小 -> 分桶占便宜；大 -> 打包的混装能力更值钱。"
                f"用 tests/enum_dataset_routes.py 出完整调度账。）")


def text_meta(path) -> Dict:
    """读 textfeat 的 meta（记了 max_length / 编码器路径 / 是否走 llm_adapter）。
    换过文本编码器或改过口径时，靠它事后核对，别靠记忆。"""
    with np.load(Path(path)) as z:
        return json.loads(str(z["meta"])) if "meta" in z.files else {}
