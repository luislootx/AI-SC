"""Master automation script — runs the full pipeline unattended.

Steps:
  1) agentic-smoke (sanity check with new fitness weights, ~2 min)
  2) agentic-demo  (workshop-target run, ~30-60 min)
  3) validate_baselines (apples-to-apples baseline comparison, ~5 min)
  4) write a SUMMARY.md combining all results

Each step writes its own log file under results/logs/ so you can read them
later. Exits with code 1 only if step 1 (smoke) fails — demo failure is
logged but does not abort the pipeline.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PYTHON = sys.executable
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
os.makedirs(LOGS, exist_ok=True)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(name: str, cmd: list[str], timeout_s: int) -> dict:
    log_path = os.path.join(LOGS, f"{name}.log")
    print(f"\n[{ts()}] ▶ STEP: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(
                cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout_s, text=True)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
            f.write(f"\n[{ts()}] ✗ Timed out after {timeout_s}s")
    dt = time.time() - t0
    print(f"[{ts()}] ◀ {name} rc={rc} in {dt/60:.1f} min")
    return {"name": name, "rc": rc, "elapsed_min": round(dt / 60, 2),
            "log": log_path}


def write_summary(steps: list[dict]):
    out = os.path.join(RESULTS, "SUMMARY.md")
    lines = [
        f"# Pipeline summary  ({ts()})", "",
        "| Step | rc | minutes | log |",
        "| --- | --- | --- | --- |",
    ]
    for s in steps:
        lines.append(f"| {s['name']} | {s['rc']} | {s['elapsed_min']} | "
                     f"`{os.path.basename(s['log'])}` |")
    lines += ["", "## Results files"]
    for fname in ("results_agentic-smoke.json",
                  "results_agentic-demo.json",
                  "validation_baselines.json"):
        fp = os.path.join(RESULTS, fname)
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    blob = json.load(fh)
                gb = blob.get("global_best") or {}
                if gb:
                    lines.append(f"\n### `{fname}` — global best")
                    lines.append(f"- fitness: **{gb.get('fitness'):.4f}**")
                    lines.append(f"- lab id: {gb.get('lab_id')}")
                    g = gb.get("genome") or {}
                    if g.get("block_sequence"):
                        lines.append(f"- arch: `{' → '.join(g['block_sequence'])}`")
                        lines.append(f"- ch={g.get('hidden_channels')}, "
                                     f"modes={g.get('fourier_modes')}, "
                                     f"act={g.get('activation')}")
                elif blob.get("results"):
                    lines.append(f"\n### `{fname}` — baseline rel L2")
                    rows = sorted(blob["results"], key=lambda r: r["rel_l2_clean"])
                    for r in rows:
                        lines.append(f"- {r['name']}: rel L2 = "
                                     f"**{r['rel_l2_clean']:.4f}** "
                                     f"({r['params']:,} params)")
            except Exception as e:
                lines.append(f"\n### `{fname}` — could not parse: {e}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[{ts()}] Summary → {out}")


def main():
    print(f"[{ts()}] Pipeline starting. Logs: {LOGS}")
    steps: list[dict] = []

    s1 = run_step("01_smoke_sanity", [PYTHON, "-u", "main.py", "agentic-smoke"],
                  timeout_s=600)
    steps.append(s1)
    if s1["rc"] != 0:
        print(f"[{ts()}] ✗ Sanity smoke failed; aborting demo.")
        write_summary(steps)
        sys.exit(1)

    s2 = run_step("02_agentic_demo", [PYTHON, "-u", "main.py", "agentic-demo"],
                  timeout_s=7200)  # 2h cap
    steps.append(s2)

    s3 = run_step("03_validate_baselines",
                  [PYTHON, "-u", "validate_baselines.py",
                   "--epochs", "15", "--samples", "256", "--test-samples", "64"],
                  timeout_s=1200)
    steps.append(s3)

    write_summary(steps)
    print(f"[{ts()}] Pipeline complete.")


if __name__ == "__main__":
    main()
