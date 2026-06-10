"""Batch-render LoRA variants through a running ComfyUI instance.

取参考工作流（默认 = ComfyUI /history 里最新一条；建议先手动跑一张目标 LoRA 的
正常评图），然后对每个变体文件：
  1. 把工作流里引用原 LoRA 的字符串替换成变体的相对路径（按磁盘相对位置推算）
  2. 固定 seed（沿用参考图的 seed，或 --seed 覆盖）
  3. PreviewImage -> SaveImage，filename_prefix=<前缀>/<变体名>，落盘可对比
  4. POST /prompt 入队

用法:
  python comfy_lora_scan.py --orig-file D:/models/LoRA/anima/goutong10-1/xxx_step1200.safetensors ^
      --variants D:/models/LoRA/anima/goutong10-1/blockscan --dry-run
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
import urllib.request


def api(port, path, payload=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def newest_history_prompt(port):
    h = api(port, "/history?max_items=1")
    if not h:
        sys.exit("ComfyUI history is empty - run one reference generation first")
    item = h[next(iter(h))]
    return item["prompt"][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--orig-file", required=True,
                    help="disk path of the LoRA referenced by the reference workflow")
    ap.add_argument("--variants", required=True,
                    help="directory or glob of variant .safetensors files")
    ap.add_argument("--workflow", help="API-format workflow json file (default: newest /history entry)")
    ap.add_argument("--seed", type=int, help="override seed (default: keep reference seed)")
    ap.add_argument("--strength", type=float, help="override strength on the matched slot")
    ap.add_argument("--save-prefix", default="lora_scan")
    ap.add_argument("--include-orig", action="store_true",
                    help="also queue the unmodified original LoRA as baseline")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.workflow:
        with open(args.workflow, encoding="utf-8") as f:
            wf = json.load(f)
            if "prompt" in wf and isinstance(wf["prompt"], list):  # full history item
                wf = wf["prompt"][2]
    else:
        wf = newest_history_prompt(args.port)

    orig_base = os.path.basename(args.orig_file)
    orig_dir = os.path.dirname(os.path.abspath(args.orig_file))

    # find every string input that references the original lora file
    hits = []  # (node_id, input_name, original_string)
    for nid, node in wf.items():
        for iname, val in node.get("inputs", {}).items():
            if isinstance(val, str) and os.path.basename(val.replace("\\", "/")) == orig_base:
                hits.append((nid, iname, val))
    if not hits:
        sys.exit(f"workflow does not reference {orig_base}; run a reference gen with that LoRA loaded")
    print(f"matched lora slots: {[(n, i) for n, i, _ in hits]}")

    if os.path.isdir(args.variants):
        files = sorted(glob.glob(os.path.join(args.variants, "*.safetensors")))
    else:
        files = sorted(glob.glob(args.variants))
    if args.include_orig:
        files = [args.orig_file] + files
    if not files:
        sys.exit("no variant files found")

    queued = []
    for vf in files:
        rel = os.path.relpath(os.path.abspath(vf), orig_dir)  # e.g. blockscan\xxx.safetensors
        stem = os.path.splitext(os.path.basename(vf))[0]
        short = stem.split("__")[-1] if "__" in stem else stem
        p = copy.deepcopy(wf)
        for nid, iname, oval in hits:
            sep = "\\" if "\\" in oval else "/"
            prefix = oval.rsplit(sep, 1)[0]
            p[nid]["inputs"][iname] = prefix + sep + rel.replace(os.sep, sep)
            if args.strength is not None:
                sname = iname.replace("lora", "strength")
                if sname in p[nid]["inputs"]:
                    p[nid]["inputs"][sname] = args.strength
        for node in p.values():
            if args.seed is not None and "seed" in node.get("inputs", {}):
                node["inputs"]["seed"] = args.seed
            if node.get("class_type") == "PreviewImage":
                node["class_type"] = "SaveImage"
                node["inputs"] = {"images": node["inputs"]["images"],
                                  "filename_prefix": f"{args.save_prefix}/{short}"}
            elif node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"{args.save_prefix}/{short}"
        if args.dry_run:
            nid, iname, _ = hits[0]
            print(f"  [dry] {short:<18} -> {p[nid]['inputs'][iname]}")
            continue
        r = api(args.port, "/prompt", {"prompt": p})
        queued.append((short, r.get("prompt_id", "?")))
        print(f"  queued {short:<18} prompt_id={r.get('prompt_id', '?')}")
        time.sleep(0.2)

    if queued:
        print(f"\n{len(queued)} renders queued; images -> ComfyUI output/{args.save_prefix}/")


if __name__ == "__main__":
    main()
