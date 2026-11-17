"""Build the submission archive dist/contract_risk_dss.zip (source, README, requirements, test data, tests)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = ["app.py", "contract_checker.py", "telegram_bot.py", "make_zip.py", "risk_patterns.json",
           "requirements.txt", "README.md", "cognitive/*.py", "data/*", "tests/*.py"]


def main():
    out = ROOT / "dist" / "contract_risk_dss.zip"
    out.parent.mkdir(exist_ok=True)
    files = sorted({p for pattern in INCLUDE for p in ROOT.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("contract_risk_dss") / f.relative_to(ROOT))
    print(f"{out} — {len(files)} files, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
