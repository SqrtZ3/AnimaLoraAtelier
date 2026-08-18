"""YAML 未识别键的告警 —— 防"拼错一个键、静默按默认值训一整晚"。

背景：``apply_yaml_config`` 只遍历 ``YAML_TO_ARGS``，不在表里的键被**完全静默**丢掉。
把 ``navit_token_budget`` 拼成 ``navit_token_budge``，训练照常启动、按默认值跑，
不打印任何东西 —— 只有事后对不上账时才可能发现。jax_tpu 侧早就对未知键报错
（``jax_tpu/config.py`` 顶部的设计原则），这里移植同一条原则，但只警告不抛
（抛会让历史 yaml 直接跑不起来）。

本文件测的是**分类是否正确**：
* 拼错的键 → 归入 unknown，且能给出"是不是想写 X"
* 合法键（含 deprecated / 已失效）→ 不能误报成 unknown
* 已失效键只在值"看起来在开启它"时才提示（``foo: false`` 与不写等价，提示纯属噪声）
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.config import (  # noqa: E402
    DEFAULTS,
    DEPRECATED_V5_KEYS,
    INERT_KEYS,
    YAML_TO_ARGS,
    _did_you_mean,
    apply_yaml_config,
    unrecognized_keys,
)


# ── 分类正确性 ────────────────────────────────────────────────────────────────

def test_typo_is_reported_as_unknown():
    unknown, inert = unrecognized_keys({"navit_token_budge": 131072})
    assert unknown == ["navit_token_budge"]
    assert inert == []


def test_typo_gets_a_suggestion():
    """光说"不认识"用处有限，能指出正确拼写才真正省时间。"""
    assert "navit_token_budget" in _did_you_mean("navit_token_budge")
    assert "learning_rate" in _did_you_mean("learnig_rate")


def test_legit_keys_are_never_flagged():
    """全量合法键一个都不能误报 —— 误报会让人学会无视这条警告。"""
    cfg = {k: 1 for k in YAML_TO_ARGS}
    unknown, _ = unrecognized_keys(cfg)
    assert unknown == []


def test_deprecated_keys_are_not_unknown():
    """deprecated 有它自己的告警通道，不能在这里重复报一遍。"""
    cfg = {k: 1 for k in DEPRECATED_V5_KEYS}
    unknown, _ = unrecognized_keys(cfg)
    assert unknown == []


def test_inert_keys_are_separated_from_unknown():
    """已失效键与未知键的**处置方式不同**：前者删掉即可，后者要改对拼写。"""
    unknown, inert = unrecognized_keys({k: True for k in INERT_KEYS})
    assert unknown == []
    assert set(inert) == set(INERT_KEYS)


@pytest.mark.parametrize("falsy", [False, 0, "", "false", "none", None])
def test_inert_key_with_falsy_value_is_quiet(falsy):
    """``use_per_block_checkpoint: false`` 与不写它完全等价，不该刷警告。"""
    unknown, inert = unrecognized_keys({"use_per_block_checkpoint": falsy})
    assert unknown == []
    assert inert == []


def test_inert_key_with_truthy_value_is_reported():
    _, inert = unrecognized_keys({"use_per_block_checkpoint": True})
    assert inert == ["use_per_block_checkpoint"]


def test_empty_config_is_safe():
    assert unrecognized_keys({}) == ([], [])
    assert unrecognized_keys(None) == ([], [])


# ── 接线：apply_yaml_config 真的会发这条警告，且不改变原有行为 ────────────────

def test_apply_yaml_config_warns_on_typo(caplog):
    args = types.SimpleNamespace(**DEFAULTS)
    with caplog.at_level("WARNING"):
        apply_yaml_config(args, {"navit_token_budge": 131072})
    # getMessage() 才是把 %-参数插好的最终文案；直接读 .message 拿到的是模板
    msgs = [r.getMessage() for r in caplog.records]
    assert any("navit_token_budge" in m for m in msgs), msgs
    assert any("navit_token_budget" in m for m in msgs), f"应给出改正建议：{msgs}"
    # 关键：拼错的键**不会**意外写进 args
    assert not hasattr(args, "navit_token_budge")


def test_apply_yaml_config_still_applies_good_keys(caplog):
    """加了警告之后，正常的赋值路径必须一字不差地照旧工作。"""
    args = types.SimpleNamespace(**DEFAULTS)
    with caplog.at_level("WARNING"):
        apply_yaml_config(args, {"navit_token_budget": 65536, "seed": 7})
    assert args.navit_token_budget == 65536
    assert args.seed == 7
    assert not caplog.records, f"合法配置不该有任何警告：{caplog.text}"


def test_repo_configs_have_no_unknown_keys():
    """仓库自带的 config/*.yaml 不应含未知键。

    这条同时是 INERT_KEYS 名单的守卫：将来谁删掉一个 YAML_TO_ARGS 条目却没同步，
    这里会立刻变红，而不是等到某次训练悄悄按默认值跑完。
    """
    import glob

    import yaml

    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    offenders = {}
    for path in sorted(glob.glob(os.path.join(cfg_dir, "*.yaml"))):
        try:
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue          # 解析不了的 yaml 不是本测试的职责
        unknown, _ = unrecognized_keys(cfg)
        if unknown:
            offenders[os.path.basename(path)] = unknown
    assert not offenders, f"仓库配置里出现未知键：{offenders}"
