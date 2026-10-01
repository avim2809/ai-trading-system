"""Freeze the five EODHD shortlist pre-registrations (one timestamp, cumulative trials)."""
import re
import subprocess
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad")
PY = "/local/store/git/ai-trading-system/.venv/bin/python3"
OLD_FP = "f62cb2e4a139ea1d3cf240ce938f1d6f573d6208a9e2c004cedee66a47d07ccd"
NEW_FP = "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054"
PLACEHOLDER = '"190 + other shortlist variants (fixed at freeze)"'
N_VARIANTS = {1: 4, 2: 4, 3: 5, 4: 4, 5: 3}
PRIOR = 190
TS = sys.argv[1]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"FAILED in {cwd}: {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


for i, n in N_VARIANTS.items():
    wt = S / f"wt_S{i}"
    prior = PRIOR + sum(v for k, v in N_VARIANTS.items() if k != i)
    run("git merge -q --no-edit research/eodhd-shortlist", wt)
    pre = next((wt / "scripts").glob(f"eodhd_s{i}_*preregistered_bars.py"))
    tests = [p for p in (wt / "tests").glob(f"test_eodhd_s{i}*.py")]
    t = pre.read_text()
    assert t.count("\nDRAFT = True") == 1 and t.count("\nPREREGISTERED_AT = None") == 1 and t.count(PLACEHOLDER) == 1, pre
    t = t.replace("\nDRAFT = True", "\nDRAFT = False").replace("\nPREREGISTERED_AT = None", f'\nPREREGISTERED_AT = "{TS}"')
    t = t.replace(PLACEHOLDER, str(prior))  # 190 prior ledgers + the other candidates' variants (S1-S5 = 20)
    t = t.replace(OLD_FP, NEW_FP).replace("f62cb2e4", "fc0690f0")
    if i == 4:
        t = t.replace('    "cleaning_fingerprint_v2": ec.cleaning_fingerprint(),',
                      f'    "cleaning_fingerprint_v2": "{NEW_FP}",')
        t = t.rstrip("\n") + (
            "\n\n# Frozen cleaning rule: fail loudly if scripts/eodhd_clean.py changed after the freeze.\n"
            'assert ec.cleaning_fingerprint() == OBSERVED["cleaning_fingerprint_v2"], "cleaning rule changed since freeze"\n')
    pre.write_text(t)
    for tp in tests:
        u = tp.read_text()
        u = re.sub(r"assert (\w+)\.DRAFT is True", r"assert \1.DRAFT is False", u)
        u = re.sub(r"assert (\w+)\.PREREGISTERED_AT is None", rf'assert \1.PREREGISTERED_AT == "{TS}"', u)
        u = u.replace(f'DSR["prior_trials"] == {PLACEHOLDER}', f'DSR["prior_trials"] == {prior}')
        u = u.replace(OLD_FP, NEW_FP).replace("f62cb2e4", "fc0690f0")
        tp.write_text(u)
    out = run(f"PYTHONPATH={wt}/src {PY} -m pytest -q -p no:cacheprovider " + " ".join(str(p) for p in tests)
              + f" {wt}/tests/test_eodhd_clean.py", wt)
    fp = run(f"PYTHONPATH={wt}/scripts:{wt}/src {PY} -c \"import importlib.util as u; s=u.spec_from_file_location('m','{pre}'); "
             f"m=u.module_from_spec(s); s.loader.exec_module(m); print(m.bars_fingerprint(), m.DRAFT, m.DSR['prior_trials'])\"", wt)
    run(f"git add -A scripts tests && git commit -q -m 'Freeze S{i} pre-registration (before any return on the window)\n\n"
        f"DRAFT=False, PREREGISTERED_AT={TS}, DSR prior_trials={prior} (190 prior ledgers + the other\n"
        f"shortlist candidates variants; S1-S5 = 20). Cleaning v2 fingerprint {NEW_FP[:8]}.\n\n"
        f"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'", wt)
    head = run("git rev-parse --short HEAD", wt).strip()
    print(f"S{i} {head} tests: {out.strip().splitlines()[-1]} | fp {fp.strip()}")
