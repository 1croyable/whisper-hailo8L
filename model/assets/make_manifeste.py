import csv
from pathlib import Path

base_dir = Path("C:\\Users\\jerem\\Downloads\\cv-corpus-22.0-2025-06-20-fr\\cv-corpus-22.0-2025-06-20\\fr")
clips_dir = base_dir / "clips"
tsv_path = base_dir / "train.tsv"
manifest_out = "model/assets/manifest.tsv"

with open(tsv_path, "r", encoding="utf-8") as fin, open(manifest_out, "w", encoding="utf-8") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    for row in reader:
        path = row["path"].strip()
        sentence = row["sentence"].strip()
        audio_path = clips_dir / path
        if audio_path.exists():
            fout.write(f"{audio_path}\t{sentence}\n")

print(f"✅ Manifest saved to {manifest_out}")
