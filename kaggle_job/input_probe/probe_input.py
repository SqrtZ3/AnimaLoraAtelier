import os
for base in ("/kaggle/input", "/kaggle/input/datasets"):
    print(f"[PROBE] {base} exists:", os.path.exists(base))
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count("/")
    if depth <= 5:
        print(f"[PROBE] {root}  dirs={sorted(dirs)[:4]} files={len(files)}")
    if files and depth <= 6:
        print(f"[PROBE]   first files: {sorted(files)[:4]}")
print("[PROBE] done")
