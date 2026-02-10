#!/usr/bin/env python3
import re, csv
from pathlib import Path

target = Path("IMPLEMENTATION_SUMMARY.md")
text = target.read_text()

# This matches any line with 5 numbers separated by |, flexible spacing
pattern = re.compile(r"\|\s*([\deE\.\-\+]+)\s*\|\s*([\deE\.\-\+]+)\s*\|\s*([\deE\.\-\+]+)\s*\|\s*([\deE\.\-\+]+)\s*\|\s*([\deE\.\-\+]+)\s*\|")

matches = pattern.findall(text)

if not matches:
    print("⚠️  No numeric rows found — try showing me one example line from the markdown.")
else:
    print(f"Found {len(matches)} numeric rows:")
    for m in matches:
        print("  ", m)

    rows = [tuple(float(x.replace("−", "-")) for x in m) for m in matches]
    rows = sorted(set(rows), key=lambda x: x[0])
    with open("pr_data_sample.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_rate", "steane", "surface", "reed_muller", "fusion_qec_photonic"])
        writer.writerows(rows)
    print(f"✅ Extracted {len(rows)} rows to pr_data_sample.csv")
