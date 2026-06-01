import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
OUTPUT = os.path.join(ROOT, "act_submission.zip")

FILES = [
    "solution.py",
    "policy.pt",
    "Dockerfile",
    "act/__init__.py",
    "act/detr/__init__.py",
    "act/detr/backbone.py",
    "act/detr/detr_vae.py",
    "act/detr/position_encoding.py",
    "act/detr/transformer.py",
    "act/detr/utils.py",
]

missing = []
with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        path = os.path.join(DEMO, f)
        if not os.path.isfile(path):
            missing.append(f)
            continue
        arcname = f
        zf.write(path, arcname=arcname)
        print(f"  + {f}")

if missing:
    print(f"\n  WARNING - missing {len(missing)} file(s):")
    for f in missing:
        print(f"    ! {f}")
else:
    print(f"\n  All {len(FILES)} files packaged successfully.")

print(f"\n  -> {OUTPUT}")
