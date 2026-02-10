#!/usr/bin/env python3
"""
Extract numeric Quantum Error Correction (QEC) benchmark data
from markdown or patch files and output a clean CSV
ready for sonification or visualization.
"""

import re
import csv
from pathlib import Path

# Where to look for data (you can add more extensions if needed)
SEARCH_EXT = [".md", ".patch", ".txt"]
FILES = [f for ext in SEARCH_EXT for f in Path(".").rglob(f"*{ext}")]

# Matches numeric rows (scientific or decimal) separated by pipes
ROW_PATTERN = re.compile(
    r"\|\s*([\d.eE\+\-]+)\s*\|\s*([\d.eE\+\-]+)\s*\|\s*([\d.eE\+\-]+)\s*\|\s*([\d.eE\+\-]+)\s*\|\s*([\d.eE\+\-]+)\s*\|"
)

rows = []

for file in FILES:
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file.read_text(encoding="latin-1")
            print(f"⚠️  File {file} read with latin-1 encoding due to UTF-8 decode error.")
        except Exception as e:
            print(f"⚠️  Could not read file {file}: {e}")
            continue
    matches = ROW_PATTERN.findall(text)
    for match in matches:
        try:
            rows.append([float(x.replace("−", "-")) for x in match])
        except Exception:
            pass

if not rows:
    print("⚠️  No numeric data found. Check formatting or spacing in source files.")
else:
    # Deduplicate and sort by error_rate
    rows = sorted(set(tuple(r) for r in rows), key=lambda x: x[0])
    out_path = Path("qec_data_prepared.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_rate", "steane", "surface", "reed_muller", "fusion_qec_photonic"])
        writer.writerows(rows)

    print(f"✅ Extracted {len(rows)} rows from {len(FILES)} files → {out_path}")

    # Quick check: print sample
    print("\nSample:")
    for r in rows[:3]:
        print("  ", r)
