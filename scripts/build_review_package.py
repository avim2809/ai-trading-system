"""Build the third-party review package for the 2026-09 research programme.

Two outputs:

1. ``review/`` (committed). It holds:
   - derived artifacts per study: reports, daily return series, placebo draws, ledgers and logs <= 10 MB;
   - every ad-hoc script that ran from the session scratchpad;
   - a SHA-256 manifest of every input data file;
   - the software environment;
   - a UTC timeline of commits and run logs.
2. ``<bundle-dir>/review_bundle_<stamp>.tar.zst`` (NOT committed). It holds:
   - every input data file byte for byte, including the licensed vendor data (EODHD, Tiingo, cached vendor prices) and the public data (SEC Form 4, FRED, CBOE);
   - the artifacts too large for git.
   Its SHA-256 goes into ``review/DATA_BUNDLE.md``. The owner decides who receives it, because the vendor licences restrict redistribution. A reviewer with their own subscription can re-download the vendor data and check it against the manifest instead.

    python scripts/build_review_package.py --scratch <session scratchpad> [--bundle-dir DIR] [--no-bundle]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review"
GIT_LIMIT = 10 * 1024 * 1024       # larger artifacts go to the bundle only
HOST_TZ_OFFSET = timedelta(hours=3)  # Asia/Jerusalem (IDT) on 2026-09-30; logs are local time

# study -> scratch run directories holding its artifacts
STUDIES = {
    "edge_search_step1_standalone": ["runs/step1"],
    "edge_search_step2_alt_premia": ["runs/step2"],
    "insider_clusters": ["runs/insider", "runs/insider_recompute"],
    "shortlist_S1_industry_momentum": ["runs/S1"],
    "shortlist_S2_breadth_overlay": ["runs/S2", "runs/S2_recompute"],
    "shortlist_S3_bond_commodity_trend": ["runs/S3"],
    "shortlist_S4_52wk_high": ["runs/S4"],
    "shortlist_S5_crypto_momentum": ["runs/S5"],
}
# (dataset, root, licence class). Scratch-relative roots start with "scratch:".
DATA_ROOTS = [
    ("eodhd", "data/research/eodhd", "licensed: EODHD Historian (no redistribution without the owner's decision)"),
    ("fred", "data/research/fred", "public: FRED"),
    ("sec_form4_derived", "data/research/insider", "public: derived from SEC EDGAR Form 4"),
    ("sec_form4_source", "scratch:insider_data", "public: SEC EDGAR Form 4 fetch + parse"),
    ("alt_premia_inputs", "scratch:data", "mixed: Tiingo (licensed), CBOE, FRED, FOMC calendar (public)"),
    ("system_cache", "data/cache", "licensed: vendor prices cached by the system (edge-search step 1 input)"),
]
DATE_COLS = ("date", "Date", "timestamp", "known_date", "entry_date")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: str, scratch: Path) -> Path:
    return scratch / root.split(":", 1)[1] if root.startswith("scratch:") else ROOT / root


def parquet_meta(path: Path) -> tuple[int | None, str | None, str | None]:
    try:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        col = next((c for c in DATE_COLS if c in pf.schema_arrow.names), None)
        if col is None or rows == 0:
            return rows, None, None
        s = pd.to_datetime(pf.read(columns=[col]).column(0).to_pandas(), errors="coerce")
        return rows, str(s.min())[:10], str(s.max())[:10]
    except Exception as exc:  # a malformed file is recorded, not fatal
        log.warning("parquet metadata failed for %s (%s)", path, exc)
        return None, None, None


def build_manifest(scratch: Path) -> pd.DataFrame:
    rows = []
    for dataset, root, licence in DATA_ROOTS:
        base = resolve(root, scratch)
        if not base.exists():
            log.warning("data root missing: %s", base)
            continue
        files = sorted(p for p in base.rglob("*") if p.is_file() and not p.name.endswith((".tmp", ".lock")))
        log.info("manifest %s: %d files", dataset, len(files))
        for p in files:
            n, first, last = parquet_meta(p) if p.suffix == ".parquet" else (None, None, None)
            rows.append({"dataset": dataset, "licence": licence, "path": f"{root}/{p.relative_to(base)}",
                         "bytes": p.stat().st_size, "sha256": sha256(p), "rows": n,
                         "first_date": first, "last_date": last})
    return pd.DataFrame(rows)


def _git_safe(rel: Path) -> Path:
    """``runs/...`` is gitignored repo-wide; store scratch run dirs as ``run_outputs/...``."""
    return Path("run_outputs", *rel.parts[1:]) if rel.parts and rel.parts[0] == "runs" else rel


def copy_artifacts(scratch: Path) -> tuple[list[dict], list[Path]]:
    """Copy <= GIT_LIMIT artifacts into review/studies; return the index and the large files."""
    index, large = [], []
    for study, dirs in STUDIES.items():
        for d in dirs:
            src = scratch / d
            if not src.exists():
                log.warning("%s: run dir missing %s", study, src)
                continue
            for p in sorted(src.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts:
                    continue
                rel = p.relative_to(scratch)
                entry = {"study": study, "scratch_path": str(rel), "bytes": p.stat().st_size, "sha256": sha256(p)}
                if p.stat().st_size <= GIT_LIMIT:
                    dst = REVIEW / "studies" / study / "artifacts" / _git_safe(rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                    entry["location"] = f"review/studies/{study}/artifacts/{_git_safe(rel)}"
                else:
                    large.append(p)
                    entry["location"] = f"bundle:{rel}"
                index.append(entry)
    return index, large


def copy_scratch_scripts(scratch: Path) -> list[str]:
    out = []
    for p in sorted(list(scratch.glob("*.py")) + list(scratch.glob("*.sh")) + list(scratch.glob("runs/**/*.py"))
                    + list(scratch.glob("insider_data/*.py"))):
        rel = p.relative_to(scratch)
        dst = REVIEW / "session_scratch_scripts" / _git_safe(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        out.append(str(_git_safe(rel)))
    return out


def environment() -> dict:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (REVIEW / "requirements-lock.txt").write_text(freeze)
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                                   capture_output=True, text=True).stdout.strip(),
            "host_timezone": "Asia/Jerusalem (UTC+3 on 2026-09-30); log timestamps are local time",
            "requirements_lock": "review/requirements-lock.txt"}


def timeline(scratch: Path) -> None:
    fmt = "%H|%ad|%D|%s"
    out = subprocess.run(["git", "log", "--all", "--since=2026-09-28", f"--format={fmt}", "--date=iso-strict-local"],
                         cwd=ROOT, capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout
    lines = ["# UTC timeline", "",
             "Every commit on every branch since 2026-09-28, in UTC. Freeze commits precede the run logs of the same study.",
             "", "## Commits", "", "| UTC | commit | refs | subject |", "|---|---|---|---|"]
    for ln in out.strip().splitlines():
        h, ad, refs, subj = ln.split("|", 3)
        lines.append(f"| {ad} | `{h[:8]}` | {refs.replace('|', '/')} | {subj.replace('|', '/')} |")
    lines += ["", "## Run logs (first/last timestamped line, converted from host local time to UTC)", "",
              "| log | first (UTC) | last (UTC) |", "|---|---|---|"]
    for p in sorted(scratch.rglob("*.log")) + sorted(scratch.rglob("*.out")):
        stamps = []
        with open(p, errors="replace") as f:
            for raw in f:
                try:
                    stamps.append(datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
        if stamps:
            to_utc = lambda t: (t - HOST_TZ_OFFSET).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
            lines.append(f"| {p.relative_to(scratch)} | {to_utc(stamps[0])} | {to_utc(stamps[-1])} |")
    (REVIEW / "TIMELINE.md").write_text("\n".join(lines) + "\n")


def make_bundle(scratch: Path, large: list[Path], bundle_dir: Path, stamp: str) -> tuple[Path, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tar_path = bundle_dir / f"review_bundle_{stamp}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for dataset, root, _ in DATA_ROOTS:
            base = resolve(root, scratch)
            if base.exists():
                tar.add(base, arcname=f"inputs/{root.replace('scratch:', 'scratchpad/')}")
        for p in large:
            tar.add(p, arcname=f"large_artifacts/{p.relative_to(scratch)}")
        tar.add(REVIEW / "data" / "MANIFEST.csv.gz", arcname="MANIFEST.csv.gz")
    zst = tar_path.with_suffix(".tar.zst")
    subprocess.run(["zstd", "-q", "-T4", "-10", "--rm", "-f", str(tar_path), "-o", str(zst)], check=True)
    return zst, sha256(zst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--bundle-dir", type=Path, default=Path("/local/store/review_bundles"))
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (REVIEW / "data").mkdir(parents=True, exist_ok=True)
    man = build_manifest(args.scratch)
    with gzip.open(REVIEW / "data" / "MANIFEST.csv.gz", "wt") as f:
        man.to_csv(f, index=False)
    summary = man.groupby(["dataset", "licence"]).agg(files=("path", "size"), bytes=("bytes", "sum")).reset_index()
    summary.to_csv(REVIEW / "data" / "MANIFEST_SUMMARY.csv", index=False)
    log.info("manifest: %d files, %.2f GB", len(man), man["bytes"].sum() / 1e9)
    index, large = copy_artifacts(args.scratch)
    (REVIEW / "ARTIFACT_INDEX.json").write_text(json.dumps(index, indent=1))
    scripts = copy_scratch_scripts(args.scratch)
    env = environment()
    (REVIEW / "ENVIRONMENT.json").write_text(json.dumps(env, indent=1))
    timeline(args.scratch)
    log.info("artifacts: %d (%d large -> bundle); scratch scripts: %d", len(index), len(large), len(scripts))
    if not args.no_bundle:
        zst, digest = make_bundle(args.scratch, large, args.bundle_dir, stamp)
        (REVIEW / "DATA_BUNDLE.md").write_text(
            "# Data bundle (not in git)\n\n"
            f"- File: `{zst}` ({zst.stat().st_size / 1e9:.2f} GB, zstd)\n- SHA-256: `{digest}`\n"
            f"- Built: {stamp} from git HEAD `{env['git_head'][:10]}`\n"
            "- Contents: `inputs/` (every input data file, byte for byte), `large_artifacts/` (run outputs > 10 MB, "
            "e.g. edge-search step-1 logs, S4 panel, S5 per-coin cache), `MANIFEST.csv.gz`.\n"
            "- Licensing: EODHD, Tiingo and cached vendor prices are licensed. Sharing this bundle is the "
            "owner's decision. A reviewer with their own subscription can re-download the vendor data with the "
            "committed scripts and compare it against `review/data/MANIFEST.csv.gz`. Vendors revise history, so "
            "some hashes may differ.\n"
            "- Extract: `zstd -d review_bundle_*.tar.zst -c | tar -x`, then place `inputs/data/...` under the repo root "
            "and `inputs/scratchpad/...` in a scratch directory passed to the scripts.\n")
        log.info("bundle %s sha256 %s", zst, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
