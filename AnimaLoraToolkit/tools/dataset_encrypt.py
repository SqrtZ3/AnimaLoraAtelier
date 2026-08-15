r"""把要上传到公有云（Kaggle）的数据集目录整体"脱敏"：图片按 sd-encrypt-image 的
pixel_shuffle_3 加密，caption 明文剥掉，可选把文件名也换成哈希。

## 这个工具解决什么

TPU 训练要的三样东西都必须挂成 Kaggle Dataset 才能被 job 读到，而 Kaggle 会对
上传内容做自动审查。逐条看暴露面（依据 `jax_tpu/data.py:CacheDataset`）：

  | 文件 | 训练时怎么用 | 暴露面 |
  |---|---|---|
  | `<stem>.<jpg/png/…>` | **只当 stem 锚点**，像素一个字节都没被读 | 图像内容 |
  | `<stem>.npz` | 真正的训练输入（16 通道 VAE latent） | 需要同款 VAE 才能还原 |
  | `<stem>.textfeat.npz` | 只读 `cross` 字段 | 里面**另存了 caption 明文** |
  | `.txt` / `.json` | 训练时完全不读（caption 在缓存阶段就烘进 textfeat 了） | 明文标签 |
  | 文件名 | 只用来配对 | 画师名/来源常写在文件名里 |

所以这里做四件事，每件都可单独关：

  1. 图片按 **pixel_shuffle_3** 逐行/逐列置换（`--no-encrypt-images` 关）；
  2. `textfeat.npz` 去掉 `caption` 字段（`--keep-caption` 关）——训练路径不读它，
     去掉对训练**逐 bit 无影响**（本文件 `--self-test` 里有断言）；
  3. `.txt` / `.json` 默认不带进输出目录（`--copy-captions` 打开）；
  4. `--scramble-names` 把 stem 换成 `sha1(stem+密码)[:16]`，映射表写在**本地**
     `name_map.json`（**别上传它**）。

**它不改动源目录**，一律写到 `--out` 的新目录；那个新目录就是可以直接 zip 上传的。

## 算法与上游的一致性

逐字复刻 spawner1145/sd-encrypt-image 的 `encrypt_image_v3` / `decrypt_image_v3`
（`scripts/core/core.py`、`utils/encrypt_auto.py`），并沿用它命令行工具的密码口径：

    实际用的 psw = sha256(用户输入的密码)          # utils/encrypt_auto.py:main
    x 轴（宽）置换 key = psw
    y 轴（高）置换 key = sha256(psw)
    PNG 文本块：Encrypt=pixel_shuffle_3
                EncryptPwdSha=sha256(f"{psw}Encrypt")   # scripts/encrypt_image.py

因此产物能被上游的 `decrypt_auto.py`、encrypt_gallery 客户端、以及 webui 插件
用**同一个用户密码**正常解开。`--self-test` 会把本文件的向量化实现与上游那份逐行
python 实现对拍（要求逐 bit 相同），跑真数据前先跑它。

## 加密后的图还能不能用来重跑缓存

**不能**——像素被置换了，直接拿去 VAE encode 会得到完全不同的 latent。加密件只用于
"上传到云端当 stem 锚点"。要在云端重跑缓存就得先解密（那等于把密码也带上云，
失去意义）。正确做法是 latent/textfeat 缓存在本地（或私有云）算好再上传。

## 用法

    # 0. 先自检（零风险，几秒）
    <python> dataset_encrypt.py --self-test

    # 1. 加密 + 打包出可上传目录
    <python> dataset_encrypt.py encrypt -d ./Dataset/your-dataset \
        -o ./Dataset/your-dataset-enc -p "你的密码"

    # 2. 事后要看原图：解密（也能直接喂给上游的 decrypt_auto.py）
    <python> dataset_encrypt.py decrypt -d ./Dataset/your-dataset-enc \
        -o ./restore -p "你的密码"

密码也可以走环境变量 `ANIMA_ENC_PW`，免得进 shell 历史。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
ENCRYPT_TAG = "pixel_shuffle_3"


# ── 上游算法（逐字复刻，别"优化"它）────────────────────────────────────────────
def get_range(s: str, offset: int, range_len: int = 4) -> str:
    offset = offset % len(s)
    return (s * 2)[offset:offset + range_len]


def get_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def shuffle_arr_v2(arr: List[int], key: str) -> List[int]:
    """上游 `shuffle_arr_v2`。**从尾往头**换位，`to_index` 由 sha 的 8 位十六进制取模。

    注意它和 `shuffle_arr`（v1/v2 加密用的那个）不是同一个置换 —— 抄错不会报错，
    只会得到一份谁也解不开的图。
    """
    sha_key = get_sha256(key)
    n = len(arr)
    for i in range(n):
        s_idx = n - i - 1
        to_index = int(get_range(sha_key, i, range_len=8), 16) % (n - i)
        arr[s_idx], arr[to_index] = arr[to_index], arr[s_idx]
    return arr


def _perms(width: int, height: int, psw: str) -> Tuple[np.ndarray, np.ndarray]:
    """(x_arr, y_arr)。x 用 psw、y 用 sha256(psw) —— 与上游一致。"""
    x = np.asarray(shuffle_arr_v2(list(range(width)), psw), np.int64)
    y = np.asarray(shuffle_arr_v2(list(range(height)), get_sha256(psw)), np.int64)
    return x, y


def encrypt_pixels(arr: np.ndarray, psw: str) -> np.ndarray:
    """≡ 上游 `encrypt_image_v3`。arr [H, W, C] -> 同形。

    上游那两个循环就是"按 y_arr 取行"再"按 x_arr 取列"，这里用花式索引一次做完
    （`--self-test` 对拍逐 bit）。
    """
    x, y = _perms(arr.shape[1], arr.shape[0], psw)
    return arr[y][:, x]


def decrypt_pixels(arr: np.ndarray, psw: str) -> np.ndarray:
    """≡ 上游 `decrypt_image_v3`（散射赋值 = 加密置换的逆）。"""
    x, y = _perms(arr.shape[1], arr.shape[0], psw)
    out = np.empty_like(arr)
    out[y] = arr                      # 逆行置换
    tmp = np.empty_like(out)
    tmp[:, x] = out                   # 逆列置换
    return tmp


# ── 上游的逐行实现（只给 --self-test 当参考，别在批处理里用，慢）──────────────
def _ref_encrypt_v3(arr: np.ndarray, psw: str) -> np.ndarray:
    width, height = arr.shape[1], arr.shape[0]
    x_arr = np.arange(width)
    shuffle_arr_v2(x_arr, psw)
    y_arr = np.arange(height)
    shuffle_arr_v2(y_arr, get_sha256(psw))
    a = arr.copy()
    _a = a.copy()
    for i in range(height):
        a[i] = _a[y_arr[i]]
    a = np.transpose(a, axes=(1, 0, 2))
    _a = a.copy()
    for i in range(width):
        a[i] = _a[x_arr[i]]
    return np.transpose(a, axes=(1, 0, 2))


def _ref_decrypt_v3(arr: np.ndarray, psw: str) -> np.ndarray:
    width, height = arr.shape[1], arr.shape[0]
    x_arr = np.arange(width)
    shuffle_arr_v2(x_arr, psw)
    y_arr = np.arange(height)
    shuffle_arr_v2(y_arr, get_sha256(psw))
    a = arr.copy()
    _a = a.copy()
    for i in range(height):
        a[y_arr[i]] = _a[i]
    a = np.transpose(a, axes=(1, 0, 2))
    _a = a.copy()
    for i in range(width):
        a[x_arr[i]] = _a[i]
    return np.transpose(a, axes=(1, 0, 2))


# ── 单张图 ────────────────────────────────────────────────────────────────────
def _open_rgb(path: Path):
    """读成 3 维数组能用的模式。灰度/调色板图会被转成 RGB —— v3 的
    `transpose(1,0,2)` 要求三维，2 维数组直接报错。转换只影响上传用的加密件。"""
    from PIL import Image
    im = Image.open(path)
    im.load()
    info = dict(im.info or {})
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode in ("LA", "PA")
                        else "RGB")
    return im, info


def process_image(src: Path, dst: Path, psw: str, decrypt: bool) -> str:
    """加密/解密一张图并写盘。返回一行日志。**输出恒为 PNG 格式**。

    加密件写成 PNG 而不是原格式：jpg/webp 是有损的，置换后的图像素间毫无相关性，
    有损压缩会把它压烂 —— 解密出来是一堆噪点（上游 readme 也强调了这条）。
    """
    from PIL import Image, PngImagePlugin
    im, info = _open_rgb(src)
    tag = str(info.get("Encrypt", ""))
    if not decrypt and tag == ENCRYPT_TAG:
        return f"  [跳过] {src.name} 已经是加密件"
    if decrypt and tag != ENCRYPT_TAG:
        return f"  [跳过] {src.name} 不带 Encrypt 标记，不是本工具的加密件"

    arr = np.array(im)
    out = decrypt_pixels(arr, psw) if decrypt else encrypt_pixels(arr, psw)
    res = Image.fromarray(out)

    png = PngImagePlugin.PngInfo()
    if not decrypt:
        png.add_text("Encrypt", ENCRYPT_TAG)
        png.add_text("EncryptPwdSha", get_sha256(f"{psw}Encrypt"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    res.save(dst, format="PNG", pnginfo=png)
    return f"  {'解密' if decrypt else '加密'} {src.name} -> {dst.name} {arr.shape}"


# ── textfeat 去 caption ───────────────────────────────────────────────────────
def strip_caption(src: Path, dst: Path) -> bool:
    """把 `<stem>.textfeat.npz` 里的 `caption` 字段去掉后另存。

    训练路径只读 `cross`（`jax_tpu/data.py:_read_ctx`），`caption` 与 `mask` 都只是
    事后核对用的。这里保留 `mask` / `meta`（meta 里是编码器路径与口径，没有画面
    信息，出了问题要靠它对账），只丢 `caption`。
    返回是否真的丢掉了什么。
    """
    with np.load(src, allow_pickle=False) as z:
        keep = {k: z[k] for k in z.files if k != "caption"}
        had = "caption" in z.files
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **keep)
    return had


# ── 目录级 ────────────────────────────────────────────────────────────────────
def _sidecars(stem: Path) -> Dict[str, List[Path]]:
    """一个 stem 名下的全部伴生文件，按用途分类。"""
    out: Dict[str, List[Path]] = {"latent": [], "text": [], "caption": []}
    lat = stem.with_suffix(".npz")
    if lat.exists():
        out["latent"].append(lat)
    for q in sorted(stem.parent.glob(f"{stem.name}.ms*.npz")):
        out["latent"].append(q)
    t = Path(str(stem) + ".textfeat.npz")
    if t.exists():
        out["text"].append(t)
    for ext in (".txt", ".caption", ".json"):
        p = stem.with_suffix(ext)
        if p.exists():
            out["caption"].append(p)
    return out


def run_dir(a) -> int:
    src_dir, out_dir = Path(a.data_dir), Path(a.out)
    if out_dir.resolve() == src_dir.resolve():
        raise SystemExit("--out 不能等于 --data-dir：本工具不原地改源目录")
    psw = get_sha256(_password(a))
    decrypt = a.cmd == "decrypt"

    imgs = sorted(p for p in src_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"{src_dir} 下没有图片（本工具按图片 stem 组织整个目录）")
    print(f"{src_dir} -> {out_dir}\n{len(imgs)} 张图，模式 {'解密' if decrypt else '加密'}")

    name_map: Dict[str, str] = {}
    if a.scramble_names and not decrypt:
        for p in imgs:
            name_map[p.stem] = hashlib.sha1(
                (p.stem + psw).encode("utf-8")).hexdigest()[:16]
    elif a.scramble_names and decrypt:
        mp = Path(a.name_map or (src_dir / "name_map.json"))
        if not mp.exists():
            raise SystemExit(f"解密时 --scramble-names 需要映射表 {mp}（加密时生成的）")
        name_map = {v: k for k, v in json.loads(mp.read_text("utf-8")).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_lat = n_txt = n_cap = n_strip = 0
    for p in imgs:
        stem = p.with_suffix("")
        new = name_map.get(p.stem, p.stem)
        if a.encrypt_images or decrypt:
            print(process_image(p, out_dir / f"{new}.png", psw, decrypt))
            n_img += 1
        else:
            shutil.copy2(p, out_dir / f"{new}{p.suffix}")
            n_img += 1
        side = _sidecars(stem)
        for q in side["latent"]:
            # `<stem>.npz` 与 `<stem>.ms4096.npz` 的后缀不同，用相对 stem 的剩余部分拼
            tail = q.name[len(stem.name):]
            shutil.copy2(q, out_dir / f"{new}{tail}")
            n_lat += 1
        for q in side["text"]:
            dst = out_dir / f"{new}.textfeat.npz"
            if a.keep_caption or decrypt:
                shutil.copy2(q, dst)
            else:
                n_strip += int(strip_caption(q, dst))
            n_txt += 1
        if a.copy_captions:
            for q in side["caption"]:
                shutil.copy2(q, out_dir / f"{new}{q.suffix}")
                n_cap += 1

    empty = src_dir / "_empty.textfeat.npz"
    if empty.exists():                       # caption_dropout 要它，且本来就是空 caption
        shutil.copy2(empty, out_dir / empty.name)
        print("  已带上 _empty.textfeat.npz（caption_dropout 用）")

    if name_map and not decrypt:
        mp = Path(a.name_map or "name_map.json")
        mp.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), "utf-8")
        print(f"[!] 文件名映射表写在 {mp} —— **留在本地，别上传**，"
              f"丢了就再也对不回原名")

    print(f"\n完成：图 {n_img} | latent {n_lat} | textfeat {n_txt}"
          f"（去 caption {n_strip}）| caption 文件 {n_cap}")
    if not decrypt:
        _postcheck(out_dir, a)
    return 0


def _postcheck(out_dir: Path, a) -> None:
    """出门前的自查清单：把还可能泄漏的东西点出来，别等审查来点。"""
    warn = []
    leftovers = [p.name for p in out_dir.iterdir()
                 if p.suffix.lower() in (".txt", ".json", ".caption")
                 and p.name != "name_map.json"]
    if leftovers:
        warn.append(f"目录里还有 {len(leftovers)} 个明文标签文件"
                    f"（如 {leftovers[:3]}）—— 训练不需要它们")
    if a.keep_caption:
        warn.append("textfeat.npz 里保留了 caption 明文（--keep-caption）")
    if not a.encrypt_images:
        warn.append("图片没加密（--no-encrypt-images），像素是明文")
    if not a.scramble_names:
        warn.append("文件名原样保留；来源/画师信息常写在文件名里，"
                    "需要就加 --scramble-names")
    warn.append("`.npz` 里的是 16 通道 VAE latent：不是可直接看的图像，"
                "但拿到同一份 Qwen VAE 就能解回接近原图 —— 它是这条路上"
                "**没有被加密**的那一环，请据此判断可接受性")
    print("\n[自查]")
    for w in warn:
        print(f"  - {w}")


def _password(a) -> str:
    pw = a.password or os.environ.get("ANIMA_ENC_PW", "")
    if not pw:
        import getpass
        pw = getpass.getpass("密码：")
    if not pw:
        raise SystemExit("密码不能为空")
    return pw


# ── 自检 ──────────────────────────────────────────────────────────────────────
def self_test() -> int:
    """三条断言，跑真数据前先过：

      1. 本文件的向量化实现 ≡ 上游逐行实现（逐 bit）；
      2. 解密 ∘ 加密 = 恒等（逐 bit）；
      3. `strip_caption` 之后 `cross` 逐 bit 不变（= 训练输入不受影响）。

    第 1 条是关键：置换写反了不会报错，只会得到一份上游工具解不开的图，
    而那时原图可能已经不在手边了。
    """
    ok = fail = 0

    def chk(name, cond, note=""):
        nonlocal ok, fail
        print(f"  [{'OK ' if cond else 'FAIL'}] {name:<34} {note}")
        ok, fail = ok + int(bool(cond)), fail + int(not cond)

    rng = np.random.RandomState(0)
    psw = get_sha256("测试密码 test-123")
    print("T1 向量化实现 ≡ 上游逐行实现")
    for h, w, c in ((37, 53, 3), (64, 64, 4), (1, 9, 3), (11, 1, 3)):
        a = rng.randint(0, 256, (h, w, c)).astype(np.uint8)
        e_mine, e_ref = encrypt_pixels(a, psw), _ref_encrypt_v3(a, psw)
        d_mine, d_ref = decrypt_pixels(e_ref, psw), _ref_decrypt_v3(e_ref, psw)
        chk(f"{h}x{w}x{c} 加密逐 bit", np.array_equal(e_mine, e_ref))
        chk(f"{h}x{w}x{c} 解密逐 bit", np.array_equal(d_mine, d_ref))

    print("T2 解密 ∘ 加密 = 恒等")
    for h, w in ((37, 53), (128, 96)):
        a = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        chk(f"{h}x{w} 往返", np.array_equal(decrypt_pixels(encrypt_pixels(a, psw), psw), a))
    a = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    chk("换密码解不开", not np.array_equal(
        decrypt_pixels(encrypt_pixels(a, psw), get_sha256("别的密码")), a))

    print("T3 去 caption 不动训练输入")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "a.textfeat.npz", Path(td) / "b.textfeat.npz"
        cross = rng.randint(0, 65536, (512, 1024)).astype(np.uint16)
        np.savez(src, cross=cross, mask=np.ones(512, np.uint8),
                 caption=np.array("1girl, 某画师名"), meta=np.array("{}"))
        had = strip_caption(src, dst)
        with np.load(dst) as z:
            chk("caption 已去掉", had and "caption" not in z.files)
            chk("cross 逐 bit 不变", np.array_equal(z["cross"], cross))
            chk("mask/meta 保留", {"mask", "meta"} <= set(z.files))

    print(f"\n*** {'通过' if not fail else '失败'} ***  ({ok} OK / {fail} FAIL)")
    return 1 if fail else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("encrypt", "decrypt"), default=None)
    ap.add_argument("-d", "--data-dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("-p", "--password", default="",
                    help="不传就读环境变量 ANIMA_ENC_PW，再没有就交互输入")
    ap.add_argument("--self-test", action="store_true",
                    help="与上游实现对拍 + 往返自检，不碰任何数据")
    ap.add_argument("--no-encrypt-images", dest="encrypt_images",
                    action="store_false",
                    help="图片原样拷贝（只做去 caption / 改名那几件事）")
    ap.add_argument("--keep-caption", action="store_true",
                    help="保留 textfeat.npz 里的 caption 明文（默认剥掉）")
    ap.add_argument("--copy-captions", action="store_true",
                    help="把 .txt/.json 标签也带进输出目录（默认不带 —— 训练不读它们）")
    ap.add_argument("--scramble-names", action="store_true",
                    help="stem 换成 sha1(stem+密码)[:16]，映射表写本地 name_map.json")
    ap.add_argument("--name-map", default="",
                    help="映射表路径（默认加密时写 ./name_map.json、解密时读输入目录下的）")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if not (a.cmd and a.data_dir and a.out):
        ap.error("需要 `encrypt|decrypt -d <目录> -o <输出目录>`，或 --self-test")
    return run_dir(a)


if __name__ == "__main__":
    sys.exit(main())
