# -*- coding: utf-8 -*-
"""预览提示词从训练集 caption 随机取（sample_dataset_prompts）的回归测试。

覆盖两块：
1. `trainer.data.collect_dataset_captions` —— caption 池的构建语义：
   - 取原文，**不**做 shuffle_caption / tag_dropout（预览要跨 step 可比）；
   - 按图片路径去重（`10_xxx` 目录前缀会把同一张图重复进 samples）；
   - 空 caption / 读失败的条目跳过而不是让整个 pool 构建炸掉；
   - JSON caption 走 caption_utils，且 shuffle/dropout 参数被强制关闭。
2. YAML → args 的接线与默认值（默认必须是 off，即行为中立）。
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.config import DEFAULTS, YAML_TO_ARGS, apply_yaml_config
from trainer.data import collect_dataset_captions


# ---------------------------------------------------------------- 测试替身

class _FakeDataset:
    """最小数据集替身：collect_dataset_captions 只看 samples / caption_override / caption_utils。"""

    def __init__(self, samples, caption_override=None, caption_utils=None):
        self.samples = samples
        self.caption_override = caption_override
        self.caption_utils = caption_utils
        # 故意把随机化开到最大 —— 正确实现不应该读这两个字段
        self.shuffle_caption = True
        self.tag_dropout = 1.0


def _txt_sample(tmp: Path, name: str, text: str, repeats: int = 1):
    """造一条 TXT caption 样本（repeats>1 模拟 kohya 目录前缀的重复计入）。"""
    txt = tmp / f"{name}.txt"
    txt.write_text(text, encoding="utf-8")
    sample = {"image": tmp / f"{name}.png", "txt_path": txt, "json_path": None,
              "normalized_json": None}
    return [dict(sample) for _ in range(repeats)]


# ---------------------------------------------------------------- 用例

def test_txt_captions_raw_and_deduped():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        samples = []
        samples += _txt_sample(tmp, "a", "trg, 1girl, long hair, smile", repeats=10)
        samples += _txt_sample(tmp, "b", "trg, 2girls, park bench")
        samples += _txt_sample(tmp, "c", "   ")  # 空 caption → 跳过
        ds = _FakeDataset(samples)

        caps = collect_dataset_captions(ds)

        # 去重：a 被重复了 10 次也只算一条
        assert caps == ["trg, 1girl, long hair, smile", "trg, 2girls, park bench"], caps


def test_no_shuffle_no_dropout_even_when_dataset_enables_them():
    """dataset 上 shuffle_caption=True / tag_dropout=1.0，caption 仍须逐字原样返回。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        text = "trg, 1girl, long hair, smile, outdoors, day"
        ds = _FakeDataset(_txt_sample(tmp, "a", text))

        for _ in range(20):  # 多跑几次，随机化若被误用几乎必然暴露
            assert collect_dataset_captions(ds) == [text]


def test_max_count_truncates():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        samples = []
        for i in range(5):
            samples += _txt_sample(tmp, f"img{i}", f"cap {i}")
        ds = _FakeDataset(samples)

        assert collect_dataset_captions(ds, max_count=3) == ["cap 0", "cap 1", "cap 2"]
        assert len(collect_dataset_captions(ds, max_count=0)) == 5


def test_missing_file_is_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        good = _txt_sample(tmp, "good", "trg, 1girl")
        broken = [{"image": tmp / "gone.png", "txt_path": tmp / "gone.txt",
                   "json_path": None, "normalized_json": None}]
        ds = _FakeDataset(good + broken)

        assert collect_dataset_captions(ds) == ["trg, 1girl"]


def test_caption_override_wins():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds = _FakeDataset(_txt_sample(tmp, "a", "原始 caption"), caption_override="reg caption")

        assert collect_dataset_captions(ds) == ["reg caption"]


def test_json_caption_disables_shuffle_and_dropout():
    seen_kwargs = {}

    def _build(normalized, **kwargs):
        seen_kwargs.update(kwargs)
        return normalized["text"]

    cap_utils = {
        "load_json": lambda p: {"tags": {}, "meta": {}, "text": "trg, 1girl, json path"},
        "normalize": lambda raw: raw,
        "build": _build,
    }
    samples = [{"image": Path("x.png"), "txt_path": None, "json_path": Path("x.json"),
                "normalized_json": None}]
    ds = _FakeDataset(samples, caption_utils=cap_utils)

    assert collect_dataset_captions(ds) == ["trg, 1girl, json path"]
    assert seen_kwargs == {
        "shuffle_appearance": False,
        "shuffle_tags": False,
        "shuffle_environment": False,
        "tag_dropout": 0.0,
    }


def test_json_uses_precomputed_normalized_cache():
    """samples 里已有 normalized_json 时不该再去读文件（文件根本不存在）。"""
    cap_utils = {
        "load_json": lambda p: (_ for _ in ()).throw(AssertionError("不该走 load_json")),
        "normalize": lambda raw: raw,
        "build": lambda normalized, **kw: normalized["text"],
    }
    samples = [{"image": Path("x.png"), "txt_path": None,
                "json_path": Path("does-not-exist.json"),
                "normalized_json": {"text": "cached caption"}}]
    ds = _FakeDataset(samples, caption_utils=cap_utils)

    assert collect_dataset_captions(ds) == ["cached caption"]


# ---------------------------------------------------------------- 配置接线

class _Args:
    pass


def test_config_defaults_are_behavior_neutral():
    """默认 off —— 不写这些键时与改动前行为等价。"""
    assert DEFAULTS["sample_dataset_prompts"] == "off"
    assert DEFAULTS["sample_dataset_prompt_pick"] == "each"
    assert DEFAULTS["sample_dataset_prompt_suffix"] == ""


def test_yaml_keys_are_wired():
    keys = [
        "sample_dataset_prompts",
        "sample_dataset_prompt_ratio",
        "sample_dataset_prompt_pick",
        "sample_dataset_prompt_count",
        "sample_dataset_prompt_suffix",
    ]
    for k in keys:
        assert YAML_TO_ARGS.get(k) == k, f"{k} 未映射到同名 args 属性"
        assert k in DEFAULTS, f"{k} 缺 DEFAULTS 条目（apply_yaml_config 的覆盖判断依赖它）"

    args = _Args()
    for k in keys:
        setattr(args, k, DEFAULTS[k])
    args.optimizer_args = {}
    apply_yaml_config(args, {
        "sample_dataset_prompts": "only",
        "sample_dataset_prompt_ratio": 0.25,
        "sample_dataset_prompt_pick": "once",
        "sample_dataset_prompt_count": 6,
        "sample_dataset_prompt_suffix": ",a",
    })
    assert args.sample_dataset_prompts == "only"
    assert args.sample_dataset_prompt_ratio == 0.25
    assert args.sample_dataset_prompt_pick == "once"
    assert args.sample_dataset_prompt_count == 6
    assert args.sample_dataset_prompt_suffix == ",a"


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
