#!/usr/bin/env python3
import os
from PIL import Image, UnidentifiedImageError

def check_png(root_dir):
    bad_files = []
    total = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.png'):
                total += 1
                path = os.path.join(dirpath, f)
                try:
                    with Image.open(path) as img:
                        img.verify()  # verify header + chunks
                except (SyntaxError, UnidentifiedImageError, OSError) as e:
                    bad_files.append((path, str(e)))

    print(f"\nChecked {total} PNG files under {root_dir}")
    if bad_files:
        print(f"\n⚠️  Found {len(bad_files)} corrupted PNGs:\n")
        for path, err in bad_files:
            print(f" - {path}\n   → {err}")
    else:
        print("\n✅ No corrupted PNGs found!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_bad_pngs.py /path/to/dataset")
        sys.exit(1)
    check_png(sys.argv[1])
