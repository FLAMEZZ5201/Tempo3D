#!/usr/bin/env python3
"""Convert *_watertight.obj files from a preprocessed dataset into GLB format.

Usage:
    python tools/convert_to_glb.py \
        --src /path/to/preprocessed \
        --dst /path/to/TempoDetail_glb \
        --count 49
"""
import argparse
import trimesh
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Source directory containing preprocessed subfolders")
    parser.add_argument("--dst", required=True,
                        help="Output directory for GLB files")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of new files to convert (default: all)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True)

    existing_count = len([d for d in dst.iterdir() if d.is_dir()])
    print(f"Already have {existing_count} folders in destination.")

    start_index = existing_count + 1
    converted = 0
    failed = 0

    folders = sorted(f for f in src.iterdir() if f.is_dir())

    for folder in folders:
        if args.count is not None and converted >= args.count:
            break
        obj_files = list((folder / "geo_data").glob("*_watertight.obj"))
        if not obj_files:
            continue
        idx = start_index + converted
        out_dir = dst / f"{idx:05d}"
        if out_dir.exists():
            continue
        try:
            mesh = trimesh.load(str(obj_files[0]), process=False)
            if mesh.is_empty:
                print(f"  Skip {folder.name}: empty mesh")
                failed += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{idx:05d}.glb"
            mesh.export(str(out_file), file_type="glb")
            converted += 1
            print(f"[{converted}] {folder.name} -> {idx:05d}/{idx:05d}.glb"
                  f"  ({len(mesh.vertices)}v {len(mesh.faces)}f)")
        except Exception as e:
            print(f"  Skip {folder.name}: {e}")
            failed += 1

    print(f"\nDone. Converted: {converted}, Failed: {failed}")


if __name__ == "__main__":
    main()