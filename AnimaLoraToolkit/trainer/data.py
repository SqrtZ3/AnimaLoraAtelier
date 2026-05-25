"""数据集 / 分桶 / VAE latent 缓存。

包含：
- `BucketManager` —— ARB（aspect-ratio bucketing）桶生成与匹配
- `ImageDataset` —— 主数据集类，支持 JSON / TXT caption、tag dropout（含频率均衡）、
  flip 增强、按目录名前缀自动 repeat（kohya 风格 `10_xxx` → ×10）
- `RepeatDataset` —— 简单 N×重复包装
- `MergedDataset` —— 把主数据集与正则数据集（reg）首尾拼接
- `BucketBatchSampler` —— 把同桶样本聚一个 batch，避免 ARB 不同尺寸混 batch；
  `__len__` 用预计算的 per-bucket 加和（drop_last 在每个桶独立生效，旧的 `n // bs`
  实现在多桶时高估批数）
- `CachedLatentDataset` —— Kohya 风格 .npz 缓存，与上游 ImageDataset 共享 samples 列表
- `collate_fn` / `collate_fn_cached` —— 同 batch 内尺寸一致性检查 + 标准 stack

依赖 `utils/caption_utils.py`（动态 import，可选）。本模块对其它 trainer 子模块没有依赖。
"""

from __future__ import annotations

import importlib.util
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def compute_sample_accumulation_steps(dataset_size: int, epochs: int,
                                      effective_batch_size: int) -> int:
    """Return optimizer steps when flushing one final partial sample window.

    This helper is intentionally sample-count based: it does not care how many
    ARB buckets or micro-batches the DataLoader produces.
    """
    total_samples = max(0, int(dataset_size)) * max(0, int(epochs))
    eff_bs = max(1, int(effective_batch_size))
    return (total_samples + eff_bs - 1) // eff_bs


# numpy ↔ torch dtype 映射（CachedLatentDataset 保存 bf16 / fp16 时用）。
# numpy 1.x 无原生 bfloat16，所以 bf16 在磁盘上用 uint16 视图保存，读回来 view 成 bf16。
_NP_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "uint16_bf16": torch.bfloat16,  # 自定义 sentinel
}


class BucketManager:
    """ARB 分桶管理"""
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso)

    def _generate(self, min_r, max_r, step, base):
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w / h, h / w) > 2.0:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw / bh)
            if diff < best_diff:
                best_diff = diff
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """图像数据集，支持 JSON / TXT caption 与频率均衡的 tag dropout。"""
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 freq_balanced_dropout_strength=0.0):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        # ★ v5 ② frequency-balanced tag dropout
        # 0 = 关闭；>0 启用。在数据集 init 时统计 tag 频率，对在数据集中过度共现的 tag 额外提高 dropout
        # 概率，强迫模型把"风格"与"高频共现 tag"解耦。完全数据驱动，自动适配任何画师。
        self.freq_balanced_dropout_strength = float(freq_balanced_dropout_strength or 0.0)
        self.tag_freq = {}  # tag -> frequency in [0, 1]，scan 之后填充

        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                # 直接加载 caption_utils.py。该文件在 AnimaLoraToolkit/utils/caption_utils.py。
                # trainer 包位于 AnimaLoraToolkit/trainer/ —— parent.parent 是仓库根，
                # parent 是 AnimaLoraToolkit/。两级合一指向 utils/caption_utils.py。
                utils_path = Path(__file__).resolve().parent.parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)

                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption 模式已启用（分类 shuffle）")
                else:
                    logger.warning(f"caption_utils.py 未找到: {utils_path}")
            except Exception as e:
                logger.warning(f"caption_utils 加载失败: {e}，回退到 TXT 模式")

        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        logger.info(f"数据集: {len(self.samples)} 样本 (JSON: {json_count}, TXT: {txt_count})")

        # ★ JSON 预 normalize：__getitem__ 时只做 build（shuffle+dropout），不再每次 load+parse 文件。
        # 在 num_workers=0（Windows 默认）时，这一项能把数据 prep CPU 时间显著降下来。
        if json_count > 0 and self.caption_utils is not None:
            self._pre_normalize_json_captions()

        # 用与 __getitem__ 完全一致的 PIL 路径填充 bucket_key
        self._finalize_bucket_keys()
        self.bucket_for_index = [s["bucket_key"] for s in self.samples]
        # 诊断：统计 bucket 分布
        from collections import Counter
        dist = Counter(self.bucket_for_index)
        logger.info(f"  bucket 分布: {len(dist)} 种, 例: {list(dist.most_common(5))}")

        # ★ v5 ② 统计 tag 频率（仅在启用 freq_balanced dropout 时执行；避免无谓 IO）
        if self.freq_balanced_dropout_strength > 0:
            if json_count > 0:
                logger.warning(
                    f"[freq_balanced] 检测到 {json_count} 个 JSON caption；本机制只对 TXT caption 生效。"
                    "JSON 走 caption_utils 内置的分类 shuffle / dropout，与频率均衡无关。"
                )
            self._compute_tag_freq()

    def _finalize_bucket_keys(self):
        """For each sample, compute bucket_key via the same code path that __getitem__ uses,
        so BucketBatchSampler can reliably group same-shape tensors. Caches per unique img path.

        ★ 大数据集（几万张）时 PIL.Image.open 在 OS 层每张都要 open/close fd，
          init 阶段可能卡几十秒。加进度日志让用户知道还活着；并把 unique image set 预先收集，
          避免 repeats 重复样本重复 open。
        """
        from PIL import Image as _PILImage
        # 先收集 unique image paths（_scan 里同一张图可能被 repeats 次添加）
        unique_imgs = []
        seen = set()
        for sample in self.samples:
            ip = sample["image"]
            if ip not in seen:
                seen.add(ip)
                unique_imgs.append(ip)
        n_unique = len(unique_imgs)

        cache = {}
        for i, img_path in enumerate(unique_imgs):
            if self.bucket_mgr is None:
                cache[img_path] = (self.resolution, self.resolution)
            else:
                try:
                    img = _PILImage.open(img_path)
                    w, h = img.width, img.height
                    try:
                        img.close()
                    except Exception:
                        pass
                    bw, bh = self.bucket_mgr.get_bucket(w, h)
                    cache[img_path] = (bh, bw)  # (h, w)
                except Exception as e:
                    logger.warning(f"[bucket_key] 无法读取 {img_path}: {e}，回退到 ({self.resolution},{self.resolution})")
                    cache[img_path] = (self.resolution, self.resolution)
            # 每 1000 张或最后一张时打印进度，让用户知道 init 没卡死
            if n_unique >= 2000 and ((i + 1) % 1000 == 0 or i == n_unique - 1):
                logger.info(f"  bucket_key 解析进度: {i + 1}/{n_unique}")

        for sample in self.samples:
            sample["bucket_key"] = cache[sample["image"]]

    def _pre_normalize_json_captions(self):
        """一次性把所有 JSON caption 加载 + normalize，缓存到 sample["normalized_json"]。

        旧实现 `_process_caption_json` 在每次 __getitem__ 都做 load_json + normalize + build；
        normalize 是结构变换，对同一文件每次产出都一样，没必要重复做。
        load_json 也省下重复 IO（同一 sample dict 会被 repeats 次添加，每次都读盘）。

        预 normalize 后 __getitem__ 走 build_only 路径（只做 shuffle/dropout 这种随机的部分）。
        """
        if self.caption_utils is None:
            return
        cache = {}  # json_path -> normalized dict（同一 JSON 多次出现只算一次）
        ok = 0
        fail = 0
        for sample in self.samples:
            jp = sample.get("json_path")
            if jp is None:
                continue
            cached = cache.get(jp)
            if cached is None:
                try:
                    raw = self.caption_utils["load_json"](jp)
                    if raw is None:
                        fail += 1
                        continue
                    if "tags" in raw and "meta" in raw:
                        normalized = raw
                    else:
                        normalized = self.caption_utils["normalize"](raw)
                    cache[jp] = normalized
                    cached = normalized
                    ok += 1
                except Exception as e:
                    logger.warning(f"JSON pre-normalize 失败 {jp}: {e}")
                    fail += 1
                    continue
            sample["normalized_json"] = cached
        logger.info(f"[dataset] JSON 预 normalize 完成: {ok} unique, {fail} 失败")

    def _compute_tag_freq(self):
        """★ v5 ② 扫描全部 caption 文件，统计每个 tag 在数据集中出现的图片数 / 总图片数。

        每张图只算一次（即使 tag 在 caption 里重复也只计 1），相同图（被 repeat 而进入 samples
        多次）也只算一次，避免 repeat 高的图把它的 tag 频率人为放大。
        """
        from collections import Counter
        cnt = Counter()
        seen_imgs = set()
        total_imgs = 0

        for s in self.samples:
            img_key = str(s.get("image", ""))
            if img_key in seen_imgs:
                continue
            seen_imgs.add(img_key)

            caption_text = None
            # 优先 TXT（与 __getitem__ 的回退顺序一致）；JSON 结构性 tag 不参与此机制
            txt_path = s.get("txt_path")
            if txt_path:
                try:
                    caption_text = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    caption_text = None

            if not caption_text:
                continue

            total_imgs += 1
            if "," in caption_text:
                tags = [t.strip() for t in caption_text.split(",") if t.strip()]
            else:
                tags = [t for t in caption_text.split() if t]
            for t in set(tags):
                cnt[t] += 1

        if total_imgs == 0:
            logger.warning("[freq_balanced] 没有可统计的 TXT caption，禁用频率加权 dropout。")
            self.tag_freq = {}
            return

        self.tag_freq = {t: c / total_imgs for t, c in cnt.items()}
        top = sorted(self.tag_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_str = ", ".join(f"{t}={f:.2f}" for t, f in top)
        logger.info(
            f"[freq_balanced] 已统计 {len(self.tag_freq)} 个 tag 的频率 "
            f"(strength={self.freq_balanced_dropout_strength:.2f}). Top10: {top_str}"
        )

    def _scan(self):
        samples = []
        dir_repeat_stats = {}  # dir_name -> repeat 次数（用于诊断日志）
        unique_images = 0
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() not in self.EXTS:
                continue

            # 解析目录名中的重复次数 (例如 10_tags)
            repeats = 1
            parent_name = img_path.parent.name
            if "_" in parent_name:
                prefix = parent_name.split("_", 1)[0]
                if prefix.isdigit():
                    repeats = max(1, int(prefix))
                    dir_repeat_stats[parent_name] = repeats

            sample = {"image": img_path}

            # 优先查找 JSON
            json_path = img_path.with_suffix(".json")
            if self.prefer_json and json_path.exists():
                sample["json_path"] = json_path
                sample["txt_path"] = None
            else:
                txt_path = img_path.with_suffix(".txt")
                if not txt_path.exists():
                    txt_path = img_path.with_suffix(".caption")
                if not txt_path.exists():
                    continue
                sample["json_path"] = None
                sample["txt_path"] = txt_path

            sample["bucket_key"] = None
            # ★ JSON 预 normalize 槽：__getitem__ 时直接读 dict 而非重新解析文件
            sample["normalized_json"] = None
            unique_images += 1

            for _ in range(repeats):
                samples.append(dict(sample))

        # ★ 旧实现：如果用户同时使用了 `10_xxx` 目录命名 + 顶层 `repeats: 10` YAML 参数，
        # 实际曝光会 ×100，用户无从察觉。这里明确 log 出 dir-level repeats 和 unique image 数。
        if dir_repeat_stats:
            preview = ", ".join(f"{name}×{r}" for name, r in list(dir_repeat_stats.items())[:5])
            logger.info(
                f"[dataset] 检测到 {len(dir_repeat_stats)} 个目录使用 kohya 风格 repeat 前缀: {preview}"
                + (" ..." if len(dir_repeat_stats) > 5 else "")
            )
            logger.info(
                f"[dataset] 唯一图片: {unique_images}, dir-level repeat 后样本: {len(samples)}。"
                f"注意：若 YAML 顶层再设 `repeats: N`，最终曝光 = dir_repeats × N，请确认是否符合预期。"
            )
        return samples

    def _process_caption_txt(self, caption):
        """处理 TXT caption: 传统 tag 打乱 + keep_tokens + tag_dropout

        tag_dropout 与 caption_utils.build_caption_from_json 中的语义一致：
          - 仅丢弃 keep_tokens 之后的可变标签；keep_tokens（角色名/触发词）始终保留
          - 若 dropout 后可变部分全空，强制保留其中随机一个，避免 caption 退化为只有触发词
        """
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",") if t.strip()]
        else:
            tags = [t for t in caption.split() if t]

        if not tags:
            return ""

        keep_n = max(int(self.keep_tokens or 0), 0)
        kept = tags[:keep_n]
        rest = tags[keep_n:]

        if self.shuffle_caption and rest:
            random.shuffle(rest)

        dropout = float(self.tag_dropout or 0.0)
        freq_strength = float(getattr(self, "freq_balanced_dropout_strength", 0.0) or 0.0)
        tag_freq = getattr(self, "tag_freq", {}) or {}

        if rest and (dropout > 0.0 or (freq_strength > 0.0 and tag_freq)):
            survivors = []
            for t in rest:
                # 通用 dropout
                if dropout > 0.0 and random.random() < dropout:
                    continue
                # ★ v5 ② 频率加权 dropout
                # 触发词在 kept 里完全免疫；这里只对 rest 起作用。
                # 公式：extra_drop = strength * max(0, freq - 0.3) / 0.7
                #   freq=1.0 → extra_drop = strength；freq=0.3 → 0；线性插值。
                #   0.3 是经验阈值：低于此值的 tag 视为"真实变量"不需要解耦。
                if freq_strength > 0.0:
                    freq = tag_freq.get(t, 0.0)
                    if freq > 0.3:
                        extra_drop = freq_strength * (freq - 0.3) / 0.7
                        if random.random() < extra_drop:
                            continue
                survivors.append(t)
            if not survivors:
                survivors = [random.choice(rest)]
            rest = survivors

        return ", ".join(kept + rest)

    def _process_caption_json(self, json_path, normalized=None):
        """处理 JSON caption: 分类 shuffle。

        若调用方传入 `normalized`（init 时 _pre_normalize_json_captions 算好的），跳过 load + normalize；
        否则按旧路径 load + normalize（兼容外部直接调用，如 CachedLatentDataset）。
        """
        if self.caption_utils is None:
            return None

        try:
            if normalized is None:
                raw_json = self.caption_utils["load_json"](json_path)
                if raw_json is None:
                    return None

                if "tags" in raw_json and "meta" in raw_json:
                    normalized = raw_json
                else:
                    normalized = self.caption_utils["normalize"](raw_json)

            return self.caption_utils["build"](
                normalized,
                shuffle_appearance=self.shuffle_caption,
                shuffle_tags=self.shuffle_caption,
                shuffle_environment=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"JSON 处理失败 {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")

        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            # 使用预 normalized 缓存（init 阶段算好），跳过每次的 load+normalize 开销
            caption = self._process_caption_json(
                sample["json_path"], normalized=sample.get("normalized_json"),
            )

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)

        if caption is None:
            caption = ""

        # ARB 分桶
        if self.bucket_mgr:
            tw, th = self.bucket_mgr.get_bucket(img.width, img.height)
        else:
            tw = th = self.resolution

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        # 水平翻转增强
        if self.flip_augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption, "image": str(sample["image"])}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """合并主数据集与正则数据集（Kohya 风格 reg）"""
    def __init__(self, main_dataset, reg_dataset):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        bfi = getattr(d, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            return self.main_dataset[idx]
        return self.reg_dataset[idx - self._main_len]


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so tensors in each batch have the same size.

    Per-index resolution: walks the dataset wrapping chain (RepeatDataset / MergedDataset)
    for every outer index to look up the underlying ImageDataset / CachedLatentDataset's
    bucket_for_index. This avoids any indirection bugs in pre-built bucket_for_index lists.
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True, seed=42,
                 effective_batch_size=0, reference_batch_size=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.effective_batch_size = int(effective_batch_size or 0)
        if self.effective_batch_size <= 0:
            self.effective_batch_size = 0
        self.reference_batch_size = int(reference_batch_size or 0)
        if self.reference_batch_size <= 0:
            self.reference_batch_size = 0
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.accumulation_offset = 0
        self._reference_counts_for_epoch = []
        self._bucket_keys = self._build_keys(dataset)

        unique = set(self._bucket_keys)
        none_count = sum(1 for k in self._bucket_keys if k is None)
        logger.info(
            "[BucketBatchSampler] dataset_len=%d unique_buckets=%d none=%d (e.g. %s)",
            len(self._bucket_keys), len(unique), none_count,
            list(unique)[:5],
        )
        if none_count == len(self._bucket_keys):
            logger.warning(
                "[BucketBatchSampler] 没有任何样本能解析到 bucket_key，"
                "将退化为顺序分批（可能在 ARB 模式下因尺寸不一致而崩溃）。"
                "请检查 ImageDataset/CachedLatentDataset 是否正确填充了 bucket_for_index。"
            )
        # 让用户在训练开始时就直观看到 ARB 分桶对样本的影响（丢弃数或小 batch 数）
        from collections import Counter
        _counts = Counter(tuple(k) if k is not None else (0, 0) for k in self._bucket_keys)
        _bs = self.batch_size
        if self.drop_last:
            _dropped = sum(n % _bs for n in _counts.values())
            if _dropped > 0:
                logger.warning(
                    "[BucketBatchSampler] drop_last=True：将丢弃 %d/%d 张图片"
                    "（来自不满 batch_size=%d 的余数）。设 bucket_drop_last=false 可让所有图都参训。",
                    _dropped, len(self._bucket_keys), _bs,
                )
        else:
            _small_batches = sum(1 for n in _counts.values() if n % _bs != 0)
            logger.info(
                "[BucketBatchSampler] drop_last=False：所有 %d 张图都将参训；"
                "其中 %d 个桶会产生 1 个小于 batch_size=%d 的余数 batch。",
                len(self._bucket_keys), _small_batches, _bs,
            )
        if self.effective_batch_size:
            logger.info(
                "[BucketBatchSampler] sample-window accumulation enabled: "
                "effective_batch_size=%d, native batch_size<=%d.",
                self.effective_batch_size, self.batch_size,
            )
        if self.reference_batch_size:
            logger.info(
                "[BucketBatchSampler] reference progress enabled: "
                "reference_batch_size=%d.",
                self.reference_batch_size,
            )

        # 预计算 per-bucket 批数（drop_last 在每个桶内独立生效）。
        # 旧实现 `n // bs` 在多桶 + drop_last 时会高估批数，进而把 cosine 调度器的 T_max 设错。
        self._total_batches = self._compute_total_batches()

    def _compute_total_batches(self):
        total = 0
        pending = self.accumulation_offset
        for batch in self._native_batches():
            split_batches, pending = self._split_for_accumulation_window(batch, pending)
            total += len(split_batches)
        return total

    def _native_batches(self):
        rng = random.Random(self.seed + self.epoch)
        bucket_to_indices = {}
        for idx, key in enumerate(self._bucket_keys):
            if key is None:
                key = (0, 0)
            bucket_to_indices.setdefault(tuple(key), []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def _split_for_accumulation_window(self, batch, pending):
        """Split a same-bucket native batch before forward to hit sample windows.

        The returned sub-batches preserve order and contain each input index
        exactly once. If `effective_batch_size` is disabled, the original batch
        is returned unchanged.
        """
        if not self.effective_batch_size:
            return [batch], pending

        eff = self.effective_batch_size
        pending = int(pending) % eff
        parts = []
        pos = 0
        while pos < len(batch):
            remaining = eff - pending
            take = min(len(batch) - pos, remaining)
            parts.append(batch[pos:pos + take])
            pos += take
            pending += take
            if pending == eff:
                pending = 0
        return parts, pending

    def _reference_boundaries_for_bucket(self, bucket_len):
        if not self.reference_batch_size:
            return []
        ref_bs = self.reference_batch_size
        full = bucket_len // ref_bs
        boundaries = [ref_bs * i for i in range(1, full + 1)]
        if not self.drop_last and bucket_len % ref_bs:
            boundaries.append(bucket_len)
        return boundaries

    def _native_batches_with_reference_boundaries(self):
        rng = random.Random(self.seed + self.epoch)
        bucket_to_indices = {}
        for idx, key in enumerate(self._bucket_keys):
            if key is None:
                key = (0, 0)
            bucket_to_indices.setdefault(tuple(key), []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            reference_boundaries = self._reference_boundaries_for_bucket(len(indices))
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                local_boundaries = [
                    boundary - i
                    for boundary in reference_boundaries
                    if i < boundary <= i + len(batch)
                ]
                yield batch, local_boundaries

    def _build_keys(self, dataset):
        """Build per-outer-index bucket keys。

        ★ 旧实现对每个 outer idx 都递归 walk wrapper（O(N × depth)）。
        优化：一次 walk 找到 leaf 的 bucket_for_index 列表，然后批量索引 → O(N + walk_depth)。
        对 100K samples + 3 层 wrap 提升约 2-3×。

        无法静态决定 leaf 时（如 MergedDataset(reg, main) 两个 leaf 不同），回退到 per-idx walk。
        """
        n = len(dataset)
        # 尝试 fast path：找到唯一 leaf 的 bucket_for_index
        leaf = self._find_unique_leaf(dataset)
        if leaf is not None:
            leaf_keys = getattr(leaf, "bucket_for_index", None)
            leaf_len = len(leaf_keys) if leaf_keys is not None else 0
            if leaf_len > 0:
                # outer idx 通过 % leaf_len 直接映射（RepeatDataset 的语义）
                return [leaf_keys[i % leaf_len] for i in range(n)]

        # 回退：复用旧的 walk 实现（MergedDataset 路径）
        keys = [None] * n
        for i in range(n):
            keys[i] = self._lookup(dataset, i)
        return keys

    def _find_unique_leaf(self, d):
        """Walk down a single chain（RepeatDataset/CachedLatentDataset）的 leaf。

        MergedDataset 有两条分支，不在 fast path 范围；返回 None 由 _lookup 处理。
        """
        # MergedDataset：拒绝
        if getattr(d, "main_dataset", None) is not None and getattr(d, "reg_dataset", None) is not None:
            return None
        cur = d
        # 已有 bucket_for_index 就是 leaf
        bfi = getattr(cur, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return cur
        # 一层层往下找
        for _ in range(10):  # 防御 max depth
            inner = getattr(cur, "dataset", None)
            if inner is None or inner is cur or isinstance(inner, list):
                inner = getattr(cur, "base_dataset", None)
            if inner is None or inner is cur:
                return None
            cur = inner
            # MergedDataset 出现在中间层就不行
            if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
                return None
            bfi = getattr(cur, "bucket_for_index", None)
            if bfi is not None and len(bfi) > 0:
                return cur
        return None

    def _lookup(self, d, idx):
        """Resolve the bucket key for a given outer index by walking dataset wrappers.

        Priority: MergedDataset routing → RepeatDataset (.dataset) → leaf bucket_for_index
        → CachedLatentDataset (.base_dataset).
        """
        main = getattr(d, "main_dataset", None)
        reg = getattr(d, "reg_dataset", None)
        if main is not None and reg is not None:
            ml = getattr(d, "_main_len", len(main))
            if idx < ml:
                return self._lookup(main, idx)
            return self._lookup(reg, idx - ml)
        inner = getattr(d, "dataset", None)
        if inner is not None and inner is not d and not isinstance(inner, list):
            try:
                return self._lookup(inner, idx % len(inner))
            except TypeError:
                pass
        bfi = getattr(d, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return bfi[idx % len(bfi)]
        inner = getattr(d, "base_dataset", None)
        if inner is not None and inner is not d:
            return self._lookup(inner, idx % len(inner))
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_accumulation_offset(self, pending_samples):
        if self.effective_batch_size:
            self.accumulation_offset = int(pending_samples) % self.effective_batch_size
        else:
            self.accumulation_offset = 0

    def reference_batches_for_batch_index(self, batch_idx):
        try:
            return int(self._reference_counts_for_epoch[int(batch_idx)])
        except (IndexError, TypeError, ValueError):
            return 0

    def __len__(self):
        if self.effective_batch_size:
            return self._compute_total_batches()
        return self._total_batches

    def __iter__(self):
        pending = self.accumulation_offset
        self._reference_counts_for_epoch = []
        for batch, reference_boundaries in self._native_batches_with_reference_boundaries():
            split_batches, pending = self._split_for_accumulation_window(batch, pending)
            split_start = 0
            for split_batch in split_batches:
                split_end = split_start + len(split_batch)
                reference_count = sum(
                    1 for boundary in reference_boundaries
                    if split_start < boundary <= split_end
                )
                self._reference_counts_for_epoch.append(reference_count)
                split_start = split_end
                yield split_batch


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集。

    save_dtype: torch.float32 / torch.bfloat16 / torch.float16
        - 默认 bf16：与训练 dtype 对齐，disk 占用 ≈ fp32 的 50%，读取时无精度转换开销。
        - numpy 1.x 没有原生 bf16，bf16 在磁盘上以 uint16 view 保存，并附带 sentinel
          key (`dtype_kind = "bf16"`) 让读端正确还原。
        - 旧 cache（无 dtype_kind 键）默认按 fp32 读，向后兼容。
    """
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None,
                 save_dtype: torch.dtype = torch.bfloat16):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self.save_dtype = save_dtype
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_npz_path(self, img_path):
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
        """检查缓存是否有效（图像未修改，且格式含 latent 键）。
        若为其他模型的不兼容缓存，则删除并返回 False。"""
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        try:
            data = self.np.load(npz_path)
            if "latent" not in data.files:
                npz_path.unlink()
                logger.debug(f"已删除不兼容缓存: {npz_path.name}")
                return False
            latent = data["latent"]
            if latent.ndim != 4:
                logger.warning(f"删除异常 latent 缓存（维度应为 C,T,H,W）: {npz_path}")
                npz_path.unlink()
                return False
            if latent.shape[0] != 16:
                logger.warning(f"删除疑似非 Anima/Qwen VAE 缓存（C={latent.shape[0]}，应为 16）: {npz_path}")
                npz_path.unlink()
                return False
            # bf16 cache 用 uint16 view 保存，跳过 isfinite 检查（uint16 永远 finite）。
            # 旧 fp32 / 新 fp16 cache 仍做 NaN/Inf 校验。
            dtype_kind = str(data["dtype_kind"]) if "dtype_kind" in data.files else "fp32"
            if dtype_kind != "bf16":
                if not self.np.isfinite(latent).all():
                    logger.warning(f"删除非有限 latent 缓存: {npz_path}")
                    npz_path.unlink()
                    return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{len(self.samples)} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {len(self.samples)} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes."""
        self.bucket_for_index = [None] * len(self.samples)
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(self.samples[i]["image"])
            if not npz_path.exists():
                continue
            data = self.np.load(npz_path)
            latent = data["latent"]
            s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))

    def _encode_and_save(self, indices, vae, device, dtype):
        # ★ flip_augment + cache 不能同时生效：base_dataset.__getitem__ 会做一次随机 flip
        # 然后被烘焙进 npz，等价于 50% 数据集预 flip，再也不会每 epoch 随机翻转。
        # 这里临时把 base 的 flip 关掉，编码完恢复。
        base = self.base_dataset
        while hasattr(base, "dataset") and base is not self.base_dataset:
            base = base.dataset
        # 找到最底层的 ImageDataset
        leaf = self.base_dataset
        while hasattr(leaf, "dataset"):
            leaf = leaf.dataset
        _orig_flip = bool(getattr(leaf, "flip_augment", False))
        if _orig_flip:
            logger.warning(
                "[cache] 临时关闭 flip_augment 编码 latent（避免随机 flip 被烘焙进 npz）。"
                "编码完后恢复，但请注意：cache_latents=True 时 flip 不会每 epoch 随机生效。"
            )
            leaf.flip_augment = False

        save_dtype = getattr(self, "save_dtype", torch.bfloat16)
        try:
            for count, i in enumerate(indices):
                item = self.base_dataset[i]
                pixels = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
                _, _, ph, pw = pixels.shape
                bucket_w, bucket_h = pw, ph
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)
                    latent = vae.model.encode(pixels_5d, vae.scale)
                if not torch.isfinite(latent).all():
                    logger.warning(f"VAE 编码产生非有限 latent，跳过缓存: {self.samples[i]['image']}")
                    continue

                latent_gpu = latent.squeeze(0)
                npz_path = self._get_npz_path(self.samples[i]["image"])
                if save_dtype == torch.bfloat16:
                    # bf16 → uint16 view（同等 bit pattern；numpy 1.x 没有原生 bf16）
                    latent_bf16 = latent_gpu.to(dtype=torch.bfloat16).cpu().contiguous()
                    latent_u16 = latent_bf16.view(torch.uint16).numpy()
                    self.np.savez(
                        npz_path,
                        latent=latent_u16,
                        bucket_w=bucket_w, bucket_h=bucket_h,
                        dtype_kind="bf16",
                    )
                elif save_dtype == torch.float16:
                    latent_np = latent_gpu.to(dtype=torch.float16).cpu().numpy()
                    self.np.savez(
                        npz_path,
                        latent=latent_np,
                        bucket_w=bucket_w, bucket_h=bucket_h,
                        dtype_kind="fp16",
                    )
                else:
                    latent_np = latent_gpu.cpu().float().numpy()
                    self.np.savez(
                        npz_path,
                        latent=latent_np,
                        bucket_w=bucket_w, bucket_h=bucket_h,
                        dtype_kind="fp32",
                    )
                if (count + 1) % 10 == 0 or count == len(indices) - 1:
                    logger.info(f"  编码进度: {count + 1}/{len(indices)}")
        finally:
            if _orig_flip:
                leaf.flip_augment = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        latent_np = data["latent"]
        # dtype_kind sentinel：bf16 cache 在磁盘上是 uint16 view，要 view 回 bf16；
        # fp16 / fp32 直接 from_numpy。旧 cache 无 dtype_kind，按 fp32 兼容。
        dtype_kind = str(data["dtype_kind"]) if "dtype_kind" in data.files else "fp32"
        if dtype_kind == "bf16":
            # uint16 → torch.uint16 → view as bf16（bit pattern 相同）
            latent = torch.from_numpy(latent_np).view(torch.bfloat16)
        else:
            latent = torch.from_numpy(latent_np)
        # bf16 不能 isfinite 直接判（pytorch 旧版本可能行为不一致），但我们在保存时已验证过；
        # 这里只对 fp32/fp16 cache 做防御性 isfinite 校验。
        if dtype_kind != "bf16" and not torch.isfinite(latent).all():
            raise RuntimeError(f"读取到非有限 latent 缓存: {npz_path}")

        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset

        # NOTE: 不要在缓存 latent 上做空间 flip！
        # Anima/Qwen VAE 的 conv encoder 不是 flip-equivariant，
        # 即 flip(encode(img)) ≠ encode(flip(img))；
        # 在 latent 空间翻转会喂给训练"非自然"的潜变量，模型学到的是
        # 偏离真实分布的 latent，推理时表现为马赛克 / 边缘溶解。
        # flip 增强在缓存阶段（base ImageDataset 的 __getitem__ 里做图像 flip
        # 后再 encode）已经生效一次；想要每个 epoch 重新 flip，请关闭 cache_latents。

        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            # 使用预 normalized 缓存（避免每次 __getitem__ load+normalize JSON）
            caption = base._process_caption_json(
                sample["json_path"], normalized=sample.get("normalized_json"),
            )

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)

        if caption is None:
            caption = ""

        return {"latent": latent, "caption": caption, "image": str(sample["image"])}


def collate_fn(batch):
    """DataLoader collate"""
    shapes = [tuple(b["pixel_values"].shape) for b in batch]
    if len(set(shapes)) > 1:
        # 诊断信息：BucketBatchSampler 应该已按 bucket 分组，出现混合尺寸说明分桶失效
        details = [
            f"  - {b.get('image', '?')}: shape={tuple(b['pixel_values'].shape)}"
            for b in batch
        ]
        raise RuntimeError(
            "[collate_fn] 同一 batch 出现不同尺寸张量，BucketBatchSampler 分桶失效。\n"
            "Batch 内容:\n" + "\n".join(details)
        )
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"pixel_values": pixels, "captions": captions, "images": images}


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    shapes = [tuple(b["latent"].shape) for b in batch]
    if len(set(shapes)) > 1:
        details = [
            f"  - {b.get('image', '?')}: latent_shape={tuple(b['latent'].shape)}"
            for b in batch
        ]
        raise RuntimeError(
            "[collate_fn_cached] 同一 batch 出现不同 latent 尺寸，BucketBatchSampler 分桶失效。\n"
            "Batch 内容:\n" + "\n".join(details)
        )
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"latents": latents, "captions": captions, "images": images}
