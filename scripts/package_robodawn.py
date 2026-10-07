#!/usr/bin/env python3
"""Package Colab scripts and collected references, excluding credentials/assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/robodawn_reproduction.zip")
    args = parser.parse_args()
    paths = [*sorted((ROOT/"colab/robotwin").glob("*.sh")),ROOT/"colab/robotwin/README.md",
             ROOT/"colab/robotwin/reproduce_robodawn.ipynb",
             *[ROOT/"scripts"/name for name in ["audit_robodawn.py","run_robodawn_baseline.py","compare_robodawn.py"]],
             *[ROOT/"robodawn_site"/name for name in ["tasks.csv","episodes.csv","collection_report.json",
                 "harness_audit.json","reproduction_manifest.json","openrouter_model.json"]],
             *sorted((ROOT/"robodawn_site/inspect").glob("*.md"))]
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,"branchlab/"+path.relative_to(ROOT).as_posix())
        archive.writestr("branchlab/bundle_checksums.json",json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip():
            raise RuntimeError("Bundle CRC validation failed")
    print(f"{args.output}: {len(paths)} files, {args.output.stat().st_size} bytes")


if __name__=="__main__":
    main()
