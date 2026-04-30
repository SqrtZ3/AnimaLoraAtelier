#!/usr/bin/env python3
"""
Keep a CUDA GPU looking busy by reserving otherwise-free VRAM.

This is meant for personally owned/shared machines where the scheduler decides
availability mostly from memory usage. It dynamically grows and shrinks its
allocation so existing workloads still have a configurable safety margin.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass


MB = 1024 * 1024


@dataclass
class Block:
    tensor: object
    size_mb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reserve idle CUDA VRAM while keeping a safety margin free."
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU index to guard.")
    parser.add_argument(
        "--keep-free",
        type=int,
        default=2048,
        help="Target free VRAM to leave available, in MiB.",
    )
    parser.add_argument(
        "--hard-min-free",
        type=int,
        default=None,
        help="Immediately release memory below this free-VRAM threshold, in MiB. Defaults to keep-free / 2.",
    )
    parser.add_argument(
        "--max-reserve",
        type=int,
        default=0,
        help="Maximum VRAM this process may reserve, in MiB. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=256,
        help="Allocation/release block size, in MiB.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print important events.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hard_min_free = args.hard_min_free
    if hard_min_free is None:
        hard_min_free = max(256, args.keep_free // 2)

    if args.block <= 0 or args.keep_free <= 0 or hard_min_free <= 0:
        print("block, keep-free, and hard-min-free must be positive.", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print("PyTorch is required: pip install torch with CUDA support.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    blocks: list[Block] = []
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def reserved_mb() -> int:
        return sum(block.size_mb for block in blocks)

    def free_total_mb() -> tuple[int, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return free_bytes // MB, total_bytes // MB

    def release_one() -> bool:
        if not blocks:
            return False
        blocks.pop()
        torch.cuda.empty_cache()
        return True

    free_mb, total_mb = free_total_mb()
    print(
        f"Guarding cuda:{args.gpu}: total={total_mb} MiB, initial_free={free_mb} MiB, "
        f"keep_free={args.keep_free} MiB, hard_min_free={hard_min_free} MiB"
    )

    try:
        while not stopping:
            free_mb, _total_mb = free_total_mb()
            current_reserved = reserved_mb()

            if free_mb < hard_min_free:
                released = release_one()
                if released and not args.quiet:
                    print(
                        f"free={free_mb} MiB below hard minimum; released {args.block} MiB, "
                        f"reserved={reserved_mb()} MiB",
                        flush=True,
                    )
                time.sleep(max(0.1, args.interval / 2))
                continue

            can_reserve_more = args.max_reserve <= 0 or current_reserved + args.block <= args.max_reserve
            should_allocate = free_mb > args.keep_free + args.block and can_reserve_more
            should_release = free_mb < args.keep_free - args.block

            if should_allocate:
                alloc_mb = min(args.block, free_mb - args.keep_free)
                try:
                    # uint8 makes requested bytes easy to reason about.
                    tensor = torch.empty(alloc_mb * MB, dtype=torch.uint8, device=device)
                    tensor.fill_(1)
                    blocks.append(Block(tensor=tensor, size_mb=alloc_mb))
                    if not args.quiet:
                        print(
                            f"allocated={alloc_mb} MiB, free_before={free_mb} MiB, "
                            f"reserved={reserved_mb()} MiB",
                            flush=True,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if not args.quiet:
                        print("allocation hit OOM; backing off", flush=True)
                    time.sleep(args.interval * 2)
            elif should_release:
                if release_one() and not args.quiet:
                    print(
                        f"free={free_mb} MiB below target; reserved={reserved_mb()} MiB",
                        flush=True,
                    )
            elif not args.quiet:
                print(f"free={free_mb} MiB, reserved={current_reserved} MiB", flush=True)

            time.sleep(args.interval)
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
        print("Released reserved VRAM.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
