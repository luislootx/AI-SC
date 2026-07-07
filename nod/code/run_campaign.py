"""Resume-safe driver for the robust v2 paper campaign.

Matrix: PDEs x conditions (LLM=gemma3:12b / PSO=rules) x seeds.
Each sub-run is itself resume-safe (checkpoint per iteration), and this driver
SKIPS runs whose FINAL.json already exists, so re-running after a reboot simply
continues the campaign where it left off.

Hardening for unattended multi-day runs:
  * lockfile (PID) prevents two driver instances racing on the same run dirs
  * before any LLM job it WAITS for ollama to be ready, so a post-reboot run is
    never silently degraded to the PSO fallback (which would taint the
    "agents are LLMs" guarantee). PSO jobs do not need ollama.

Usage:
  PY=python
  OLLAMA_MODEL=gemma3:12b $PY code/run_campaign.py            # full campaign
  OLLAMA_MODEL=gemma3:12b $PY code/run_campaign.py llm        # only LLM runs
  OLLAMA_MODEL=gemma3:12b $PY code/run_campaign.py llm ns     # only LLM Navier-Stokes
  $PY code/run_campaign.py --status                           # progress table, no runs
"""
from __future__ import annotations
import os, sys, json, subprocess, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))   # nod/code
ROOT = os.path.dirname(HERE)                          # nod
RUNS = os.path.join(ROOT, "results", "swarm_runs")
LOCK = os.path.join(RUNS, "campaign.lock")
PY = sys.executable
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX", "8192")

SEEDS = [42, 137, 2024]
PDES_1D = ["pwreg", "advec", "burgers"]
PDES_2D = ["ns", "darcy"]
CONDS = [("llm", "on"), ("pso", "off")]   # LLM (gemma3:12b) first, ablation second


def job_list():
    jobs = []
    for cond, agents in CONDS:           # llm first, then pso
        for seed in SEEDS:               # 42 first -> full seed-42 slice early
            for pde in PDES_1D:          # 1D (fast) before 2D (slow)
                jobs.append(dict(dim="1d", preset="paper-grade-1d", pde=pde,
                                 cond=cond, agents=agents, seed=seed))
            for pde in PDES_2D:
                jobs.append(dict(dim="2d", preset="paper-grade", pde=pde,
                                 cond=cond, agents=agents, seed=seed))
    return jobs


def tag_for(j):
    return f"v2_{j['cond']}_{j['pde']}_seed{j['seed']}"


def is_done(tag):
    return os.path.exists(os.path.join(RUNS, tag, "FINAL.json"))


# ----------------------------------------------------------------- ollama gate

def _ollama_ready(model, timeout=25):
    import urllib.request
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": "ping"}],
                       "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_ollama(model, max_wait_min=30):
    waited, deadline = 0, max_wait_min * 60
    while waited < deadline:
        if _ollama_ready(model):
            return True
        print(f"  [wait] ollama/{model} not ready; retry in 30s "
              f"({waited}s/{deadline}s)", flush=True)
        time.sleep(30)
        waited += 30
    return False


# ----------------------------------------------------------------- lockfile

def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return True   # assume alive (safer: do not double-run)


def acquire_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
        except Exception:
            pid = None
        if pid and _pid_alive(pid):
            return False
    os.makedirs(RUNS, exist_ok=True)
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    try:
        os.remove(LOCK)
    except Exception:
        pass


# ----------------------------------------------------------------- run one job

def run_job(j):
    tag = tag_for(j)
    runner = "run_swarm_1d.py" if j["dim"] == "1d" else "run_swarm_resumable.py"
    cmd = [PY, "-u", runner, j["preset"], "--pde", j["pde"],
           "--tag", tag, "--seed", str(j["seed"]), "--agents", j["agents"]]
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = OLLAMA_MODEL
    env["OLLAMA_NUM_CTX"] = OLLAMA_NUM_CTX
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(RUNS, exist_ok=True)
    log = os.path.join(RUNS, tag + ".log")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== launch {datetime.now():%Y-%m-%d %H:%M:%S} "
                f"(model={OLLAMA_MODEL}) =====\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    return rc, (time.time() - t0) / 60


def matches(j, toks):
    return all(t == j["cond"] or t == j["pde"] or t == j["dim"]
               or t == f"seed{j['seed']}" for t in toks)


def status_table(jobs):
    done = sum(is_done(tag_for(j)) for j in jobs)
    print(f"  Campaign progress: {done}/{len(jobs)} runs complete\n")
    for j in jobs:
        tag = tag_for(j)
        print(f"  {tag:40} {'DONE' if is_done(tag) else 'pending'}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    jobs = job_list()
    if args:
        jobs = [j for j in jobs if matches(j, args)]

    if "--status" in flags:
        status_table(jobs)
        return

    if not acquire_lock():
        print("[campaign] another instance holds the lock; exiting.")
        return
    try:
        pending = [j for j in jobs if not is_done(tag_for(j))]
        print(f"[campaign] {len(jobs)} jobs selected, {len(pending)} pending, "
              f"model={OLLAMA_MODEL}", flush=True)
        for i, j in enumerate(jobs, 1):
            tag = tag_for(j)
            if is_done(tag):
                print(f"[{i}/{len(jobs)}] SKIP {tag} (FINAL.json present)",
                      flush=True)
                continue
            if j["agents"] == "on" and not wait_for_ollama(OLLAMA_MODEL):
                print(f"[campaign] ollama/{OLLAMA_MODEL} unavailable after wait; "
                      f"deferring LLM job {tag}. Exiting to retry later.",
                      flush=True)
                return
            print(f"[{i}/{len(jobs)}] RUN  {tag} ...", flush=True)
            rc, mins = run_job(j)
            print(f"[{i}/{len(jobs)}] {'OK ' if rc == 0 else 'rc=' + str(rc)} "
                  f"{tag}  ({mins:.1f} min)", flush=True)
        # Retry sweep: re-attempt any job that still has no FINAL.json (e.g. a
        # transient torch/DLL load failure). Resume-safe, so a partial run
        # continues; a failed-at-import run starts fresh. Bounded to a few sweeps.
        for sweep in range(3):
            missing = [j for j in jobs if not is_done(tag_for(j))]
            if not missing:
                break
            print(f"[campaign] retry sweep {sweep+1}: {len(missing)} job(s) "
                  f"still missing", flush=True)
            for j in missing:
                tag = tag_for(j)
                if j["agents"] == "on" and not wait_for_ollama(OLLAMA_MODEL):
                    return
                print(f"[retry] RUN  {tag} ...", flush=True)
                rc, mins = run_job(j)
                print(f"[retry] {'OK ' if rc == 0 else 'rc=' + str(rc)} "
                      f"{tag}  ({mins:.1f} min)", flush=True)
        print("[campaign] pass complete", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
