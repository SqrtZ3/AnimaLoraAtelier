"""`--help` 必须能渲染出来 —— 静态防回归。

背景：argparse 在渲染帮助时会对每条 ``help=`` 文案做一次 **%-格式化**
（``self._get_help_string(action) % params``，见 CPython argparse `_expand_help`）。
所以中文文案里一个裸的 ``%`` 就会让**整个 ``--help`` 直接抛 ValueError**：

    ValueError: unsupported format character '？' (0xff09) at index 138

实测踩到过三条（"±10% 造"、"忙碌≈100%）"、"省 20-40% 算力"）—— 共同点是 ``%`` 后面跟
空格或中文，空格被当成格式化 flag、后面那个汉字被当成转换符。写法上要写 ``%%``。

这类问题的讨厌之处：平时训练完全正常，只有想看帮助的人会撞上，且报错信息完全指不到
是哪条文案。所以用 AST 静态扫一遍（毫秒级），比起真去跑一次 ``--help``（要几十秒的
重量级 import）更适合放进单测。
"""

from __future__ import annotations

import ast
import os

# argparse 在 `_expand_help` 里喂给 % 的键（取自 CPython 实现里 vars(action) 的常用项）。
# 键给全一点没坏处 —— 我们要抓的是"根本不是合法格式化"的写法。
_PARAMS = {
    "default": 0, "prog": "prog", "type": "int", "const": 0,
    "choices": "a, b", "dest": "dest", "metavar": "M", "nargs": 1,
    "required": False, "help": "", "option_strings": "",
}

_TRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "anima_train.py")


def _bad_help_strings(path: str):
    """返回 [(行号, 报错, 文案)]：所有会让 argparse 渲染失败的 help 文案。"""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                # 只处理字面量（含隐式拼接的多行字符串）；f-string / 变量跳过
                value = ast.literal_eval(kw.value)
            except Exception:
                continue
            if not isinstance(value, str) or "%" not in value:
                continue
            try:
                value % _PARAMS
            except Exception as e:
                bad.append((kw.value.lineno, f"{type(e).__name__}: {e}", value))
    return bad


def test_all_argparse_help_strings_are_format_safe():
    bad = _bad_help_strings(_TRAIN_PY)
    if bad:
        detail = "\n".join(
            f"  anima_train.py:{ln}  {err}\n    {txt[:160]}" for ln, err, txt in bad
        )
        raise AssertionError(
            f"以下 help 文案会让 `python anima_train.py --help` 直接崩（共 {len(bad)} 条）。\n"
            f"裸的 % 要写成 %%：\n{detail}"
        )


def test_detector_actually_catches_a_bare_percent(tmp_path):
    """自检：确认上面那个扫描器真的能抓到问题，而不是永远返回空列表。

    没有这条，扫描器写错（比如 add_argument 的匹配条件挂了）会表现为"永远通过"。
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--ok", help="安全写法 ±10%% 造")\n'
        'p.add_argument("--bad", help="危险写法 ±10% 造")\n'
        'p.add_argument("--default-ok", help="默认值 %(default)s 是合法的")\n',
        encoding="utf-8",
    )
    bad = _bad_help_strings(str(sample))
    assert len(bad) == 1, f"应当只抓到 --bad 那一条，实际 {bad}"
    assert "危险写法" in bad[0][2]
