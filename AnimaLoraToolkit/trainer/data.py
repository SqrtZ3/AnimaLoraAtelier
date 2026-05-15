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
        so BucketBatchSampler can reliably group same-shape tensors. Caches per unique img path."""
        from PIL import Image as _PILImage
        cache = {}
        for sample in self.samples:
            img_path = sample["image"]
            key = cache.get(img_path)
            if key is None:
                if self.bucket_mgr is None:
                    key = (self.resolution, self.resolution)
                else:
                    try:
                        img = _PILImage.open(img_path)
                        w, h = img.width, img.height
                        try:
                            img.close()
                        except Exception:
                            pass
                        bw, bh = self.bucket_mgr.get_bucket(w, h)
                        key = (bh, bw)  # (h, w)
                    except Exception as e:
                        logger.warning(f"[bucket_key] 无法读取 {img_path}: {e}，回退到 ({self.resolution},{self.resolution})")
                        key = (self.resolution, self.resolution)
                cache[img_path] = key
            sample["bucket_key"] = key

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

            for _ in range(repeats):
                samples.append(sample.copy())
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

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None

        try:
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
            caption = self._process_caption_json(sample["json_path"])

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
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
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
        # 预计算 per-bucket 批数（drop_last 在每个桶内独立生效）。
        # 旧实现 `n // bs` 在多桶 + drop_last 时会高估批数，进而把 cosine 调度器的 T_max 设错。
        self._total_batches = self._compute_total_batches()

    def _compute_total_batches(self):
        from collections import Counter
        counts = Counter(tuple(k) if k is not None else (0, 0) for k in self._bucket_keys)
        bs = self.batch_size
        total = 0
        if self.drop_last:
            for n in counts.values():
                total += n // bs
        else:
            for n in counts.values():
                total += (n + bs - 1) // bs
        return total

    def _build_keys(self, dataset):
        n = len(dataset)
        keys = [None] * n
        for i in range(n):
            keys[i] = self._lookup(dataset, i)
        return keys

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

    def __len__(self):
        return self._total_batches

    def __iter__(self):
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


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集"""
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
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
            latent_np = latent.squeeze(0).cpu().float().numpy()
            npz_path = self._get_npz_path(self.samples[i]["image"])
            self.np.savez(npz_path, latent=latent_np, bucket_w=bucket_w, bucket_h=bucket_h)
            if (count + 1) % 10 == 0 or count == len(indices) - 1:
                logger.info(f"  编码进度: {count + 1}/{len(indices)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        latent = torch.from_numpy(data["latent"])
        if not torch.isfinite(latent).all():
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
            caption = base._process_caption_json(sample["json_path"])

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
