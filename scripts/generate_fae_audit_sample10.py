"""Generate the ten-asset global FAE calibration sample (30 charts).

This is a local test artifact only.  It never edits the mini-program frontend
or backend.  The sample uses the same evaluator as the historical 100-asset
batch but writes to a separate directory so the two review sets remain
reproducible.
"""

from __future__ import annotations

from pathlib import Path

from generate_fae_audit_batch import GLOBAL_REPRESENTATIVE_ASSETS, generate_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "fae" / "preview" / "audit_batch_10_global"


def main() -> None:
    generate_batch(
        assets=GLOBAL_REPRESENTATIVE_ASSETS,
        years=12,
        fetch_workers=3,
        output_dir=OUTPUT_DIR,
        batch_id="audit_batch_10_global",
    )


if __name__ == "__main__":
    main()
