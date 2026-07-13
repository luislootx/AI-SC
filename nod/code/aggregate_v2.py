"""Aggregate the v2 campaign into paper-ready tables + an evidence audit.

Reads results/swarm_runs/v2_{cond}_{pde}_seed{seed}/ :
  FINAL.json          -> discovered architecture + best fitness
  iter_*.json         -> best lab's measured rel L2 (clean/noisy) + params
  llm_transcript.jsonl-> proof the agents were LLMs (call counts, parse_ok)
  <tag>.log           -> fallback audit (LLM->PSO degradations, must be 0)
Also re-tabulates the honest cross-method baselines in results/exp4v3/.

Outputs (robust to partially-complete campaigns):
  results/aggregate_v2.json
  results/results_summary_v2.md
  paper/tables/tab_discovery.tex, tab_evidence.tex, tab_baselines.tex,
  tab_timing.tex, tab_trainop.tex, tab_actions.tex, tab_perseed.tex
Safe to run anytime; missing runs are simply reported as pending.
"""
from __future__ import annotations
import os, re, json, glob
from collections import defaultdict
from statistics import mean, pstdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # nod
RUNS = os.path.join(ROOT, "results", "swarm_runs")
EXP4 = os.path.join(ROOT, "results", "exp4v3")
PAPER_TAB = os.path.join(ROOT, "paper", "tables")
SEEDS = [42, 137, 2024]
PDES = ["pwreg", "advec", "burgers", "ns", "darcy"]
DISPLAY = {"DeepONet faithful": "DeepONet"}
PDE_LABEL = {"pwreg": "Piecewise Reg. (1D)", "advec": "Advection (1D)",
             "burgers": "Burgers (1D)", "ns": "Navier-Stokes (2D)",
             "darcy": "Darcy (2D)"}
CONDS = ["llm", "pso"]


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_summary(tag):
    """Return dict for one run dir, or None if not started/complete."""
    d = os.path.join(RUNS, tag)
    final = _load(os.path.join(d, "FINAL.json"))
    if not final:
        return None
    gid = final.get("global_best_lab_id")
    blocks = (final.get("global_best_genome") or {}).get("block_sequence")
    # best lab's measured rel L2: scan iter snapshots for that lab's best entry
    best = {"rel_l2_clean": None, "rel_l2_noisy": None, "params": None}
    best_acc = 1e9
    for it in sorted(glob.glob(os.path.join(d, "iter_*.json"))):
        snap = _load(it) or {}
        for e in snap.get("leaderboard", []):
            if e.get("lab_id") == gid and e.get("rel_l2_clean") is not None:
                if e["rel_l2_clean"] < best_acc:
                    best_acc = e["rel_l2_clean"]
                    best = {"rel_l2_clean": e.get("rel_l2_clean"),
                            "rel_l2_noisy": e.get("rel_l2_noisy"),
                            "params": e.get("params")}
    # evidence: transcript + fallback audit
    tr = os.path.join(d, "llm_transcript.jsonl")
    n_plan = n_rev = n_ok = 0
    if os.path.exists(tr):
        for line in open(tr, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            role = (r.get("ctx") or {}).get("role")
            n_plan += role == "planner"
            n_rev += role == "reviewer"
            n_ok += bool(r.get("parse_ok"))
    log = os.path.join(RUNS, tag + ".log")
    fb = 0
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="ignore").read()
        fb = len(re.findall(r"falling back to (PSO|accuracy)", txt))
    return {"tag": tag, "blocks": blocks, "fitness": final.get("global_best_fitness"),
            "rel_l2_clean": best["rel_l2_clean"], "rel_l2_noisy": best["rel_l2_noisy"],
            "params": best["params"], "llm_calls": n_plan + n_rev,
            "planner_calls": n_plan, "reviewer_calls": n_rev,
            "parse_ok": n_ok, "fallbacks": fb}


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": mean(xs), "std": pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def baselines():
    """Mean rel L2 + params per (pde, model) over seeds from exp4v3.

    Display policy: the v3-campaign winners ("Hybrid v3 (NS/Darcy)") are THE
    paper's discovered artifacts and are renamed accordingly; the precursor
    poster-era "Discovered Hybrid" is superseded by them and excluded from the
    paper (its raw jsons remain in results/exp4v3/ as history)."""
    RENAME = {"Hybrid v3 (NS)": "Discovered hybrid (NS)",
              "Hybrid v3 (Darcy)": "Discovered hybrid (Darcy)"}
    EXCLUDE = {"Discovered Hybrid"}
    agg = defaultdict(lambda: defaultdict(list))
    par = defaultdict(list)
    for f in glob.glob(os.path.join(EXP4, "*.json")):
        obj = _load(f)
        for d in (obj if isinstance(obj, list) else [obj] if obj else []):
            if isinstance(d, dict) and "rel_l2_clean" in d:
                m = d.get("model", "?")
                if m in EXCLUDE:
                    continue
                m = RENAME.get(m, m)
                agg[d.get("pde", "?")][m].append(d["rel_l2_clean"])
                if d.get("params"):
                    par[m].append(d["params"])
    out = {}
    for pde, models in agg.items():
        out[pde] = {m: stat(v) for m, v in models.items()}
    out["_params"] = {m: (sorted(v)[len(v)//2] if v else None) for m, v in par.items()}
    return out


def human_params(n):
    if n is None:
        return "--"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def planner_actions():
    """Mine planner decisions from the transcripts: per-PDE distribution of the
    planner's chosen action, plus representative rationales (evidence that the
    LLM plans from swarm context rather than emitting boilerplate)."""
    acts = defaultdict(lambda: defaultdict(int))
    rats = defaultdict(list)
    for pde in PDES:
        for seed in SEEDS:
            tr = os.path.join(RUNS, f"v2_llm_{pde}_seed{seed}", "llm_transcript.jsonl")
            if not os.path.exists(tr):
                continue
            for line in open(tr, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if (r.get("ctx") or {}).get("role") != "planner":
                    continue
                resp = r.get("response") or ""
                i, j = resp.find("{"), resp.rfind("}")
                if i < 0 or j <= i:
                    continue
                try:
                    obj = json.loads(resp[i:j + 1])
                except Exception:
                    continue
                a = str(obj.get("action", "?")).strip().lower() or "?"
                acts[pde][a] += 1
                rat = str(obj.get("rationale", "")).strip()
                if rat and len(rats[pde]) < 5:
                    rats[pde].append(rat)
    return {"counts": {p: dict(v) for p, v in acts.items()},
            "rationales": {p: v for p, v in rats.items()}}


def reviewer_behavior():
    """What the LLM reviewer rewards: per call, Spearman rank correlation
    between its score distribution and the candidates' measured accuracy
    (-rel_l2), plus rationale theme frequencies. Backs the 'What the reviewer
    rewards' paragraph in the paper."""
    def spearman(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        def ranks(v):
            order = sorted(range(n), key=lambda i: v[i])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[order[k]] = (i + j) / 2.0 + 1
                i = j + 1
            return r
        rx, ry = ranks(xs), ranks(ys)
        mx = sum(rx) / n; my = sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = sum((a - mx) ** 2 for a in rx) ** 0.5
        dy = sum((b - my) ** 2 for b in ry) ** 0.5
        return None if dx == 0 or dy == 0 else num / (dx * dy)

    themes = {"novelty": r"novel|divers|unusual|groupthink|unique|explor",
              "accuracy": r"accura|fitness|error|l2|perform",
              "efficiency": r"efficien|param|parsimon|compact|small",
              "generalization": r"generaliz|noise|robust",
              "theory": r"sound|theor|principl|periodic|spectral"}
    rhos, calls = [], 0
    tcount = {k: 0 for k in themes}
    for tr in glob.glob(os.path.join(RUNS, "v2_llm_*", "llm_transcript.jsonl")):
        for line in open(tr, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get("ctx") or {}).get("role") != "reviewer":
                continue
            calls += 1
            cand = re.findall(
                r'"lab_id":\s*(\d+).*?"measured_rel_l2":\s*([0-9.eE+-]+)',
                d.get("user", ""), flags=re.S)
            resp = d.get("response", "")
            i, j = resp.find("{"), resp.rfind("}")
            try:
                obj = json.loads(resp[i:j + 1])
            except Exception:
                continue
            scores = obj.get("scores") or {}
            pairs = [(float(scores[l]), -float(x)) for l, x in cand if l in scores]
            if len(pairs) >= 3:
                r = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                if r is not None:
                    rhos.append(r)
            rat = str(obj.get("rationale", "")).lower()
            for k, pat in themes.items():
                if re.search(pat, rat):
                    tcount[k] += 1
    return {"calls": calls, "scoreable": len(rhos),
            "mean_spearman": (mean(rhos) if rhos else None),
            "pct_positive": (100 * sum(r > 0 for r in rhos) / len(rhos)) if rhos else None,
            "pct_perfect": (100 * sum(r > 0.999 for r in rhos) / len(rhos)) if rhos else None,
            "theme_pct": {k: 100 * v / max(calls, 1) for k, v in tcount.items()}}


def fmt(s, p=4):
    if not s:
        return "--"
    return f"{s['mean']:.{p}f}" + (f"$\\pm${s['std']:.{p}f}" if s["n"] > 1 else "")


def timing():
    """Real wall-clock timing per (pde, cond): total minutes (from campaign.log),
    per-iteration seconds and per-lab-evaluation seconds (from iter saved_at,
    excluding reboot gaps>1h). Also single-operator training time (exp4v3)."""
    import re
    from datetime import datetime
    mins = {}
    clog = os.path.join(ROOT, "results", "campaign.log")
    if os.path.exists(clog):
        # The log mixes UTF-8 and UTF-16LE segments (PowerShell restarts after
        # reboots re-open it with a different encoding). Reading it as UTF-8
        # silently drops the UTF-16 lines, which made the timing table fall
        # back to STALE durations from the archived e4b campaign for every
        # post-reboot run. Strip NULs so UTF-16LE ASCII decodes as plain text.
        raw = open(clog, "rb").read().replace(b"\x00", b"")
        for l in raw.decode("utf-8", errors="ignore").splitlines():
            m = re.search(r"OK\s+(v2_\w+)\s+\(([\d.]+) min\)", l)
            if m:
                mins[m.group(1)] = float(m.group(2))   # last occurrence wins (= v3)

    def per_iter(tag):
        its = sorted(glob.glob(os.path.join(RUNS, tag, "iter_*.json")))
        ts = []
        for it in its:
            s = _load(it)
            if s and s.get("saved_at"):
                try:
                    ts.append(datetime.strptime(s["saved_at"], "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    pass
        d = [(ts[i] - ts[i-1]).total_seconds() for i in range(1, len(ts))]
        d = [x for x in d if x > 0]
        # Robust reboot-gap guard: a fixed 1h cap misses shorter outages (e.g.
        # a 39-min BSOD gap) while a tight cap would clip legit NS-2D
        # iterations. Drop deltas > 5x the run's median instead.
        if len(d) >= 4:
            med = sorted(d)[len(d) // 2]
            d = [x for x in d if x <= 5 * med]
        return d

    out = {}
    for pde in PDES:
        out[pde] = {}
        for cond in CONDS:
            tot = [mins[f"v2_{cond}_{pde}_seed{s}"] for s in SEEDS
                   if f"v2_{cond}_{pde}_seed{s}" in mins]
            pit = []
            for s in SEEDS:
                pit += per_iter(f"v2_{cond}_{pde}_seed{s}")
            out[pde][cond] = {"total_min": stat(tot),
                              "iter_s": stat(pit),
                              "n": len(tot)}
    # single-operator training times (NS-2D, exp4v3); same display policy as
    # baselines(): v3 winners renamed, superseded poster hybrid excluded
    RENAME = {"Hybrid v3 (NS)": "Discovered hybrid (NS)",
              "Hybrid v3 (Darcy)": "Discovered hybrid (Darcy)"}
    tt = defaultdict(list)
    for f in glob.glob(os.path.join(EXP4, "ns2d__*.json")):
        o = _load(f)
        for d in (o if isinstance(o, list) else [o] if o else []):
            if isinstance(d, dict) and "train_time_s" in d:
                m = d.get("model")
                if m == "Discovered Hybrid":
                    continue
                tt[RENAME.get(m, m)].append(d["train_time_s"])
    out["_train_op"] = {m: stat(v) for m, v in tt.items()}
    return out


def main():
    os.makedirs(PAPER_TAB, exist_ok=True)
    runs = {}
    for cond in CONDS:
        for pde in PDES:
            for seed in SEEDS:
                tag = f"v2_{cond}_{pde}_seed{seed}"
                s = run_summary(tag)
                if s:
                    runs[tag] = s

    # ---- discovery + ablation table (per pde: LLM vs PSO best rel L2) ----
    disc = {}
    for pde in PDES:
        disc[pde] = {}
        for cond in CONDS:
            ss = [runs[f"v2_{cond}_{pde}_seed{s}"] for s in SEEDS
                  if f"v2_{cond}_{pde}_seed{s}" in runs]
            disc[pde][cond] = {
                "rel_l2_clean": stat([r["rel_l2_clean"] for r in ss]),
                "params": stat([r["params"] for r in ss]),
                "n_runs": len(ss),
                "example_arch": (ss[0]["blocks"] if ss else None),
            }

    # ---- evidence audit (LLM runs only) ----
    evid = {t: r for t, r in runs.items() if t.startswith("v2_llm_")}
    total_calls = sum(r["llm_calls"] for r in evid.values())
    total_fb = sum(r["fallbacks"] for r in evid.values())

    tim = timing()
    pacts = planner_actions()
    agg = {"runs": runs, "discovery": disc, "baselines_exp4v3": baselines(),
           "timing": tim, "planner_actions": pacts,
           "reviewer_behavior": reviewer_behavior(),
           "evidence": {"llm_runs": len(evid), "total_llm_calls": total_calls,
                        "total_fallbacks": total_fb}}
    with open(os.path.join(ROOT, "results", "aggregate_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)

    # ---- LaTeX: discovery table ----
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Problem & \multicolumn{2}{c}{Discovered rel.\ $L_2$ (clean)} & "
         r"\multicolumn{2}{c}{Params} \\",
         r" & LLM (Gemma~3) & PSO & LLM & PSO \\", r"\midrule"]
    def fmtp(s):
        if not s:
            return "--"
        out = human_params(s["mean"])
        if s["n"] > 1:
            out += f"$\\pm${human_params(s['std'])}"
        return out
    for pde in PDES:
        a = disc[pde]
        L.append(f"{PDE_LABEL[pde]} & {fmt(a['llm']['rel_l2_clean'])} & "
                 f"{fmt(a['pso']['rel_l2_clean'])} & "
                 f"{fmtp(a['llm']['params'])} & {fmtp(a['pso']['params'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_discovery.tex"), "w").write("\n".join(L))

    # ---- LaTeX: evidence table ----
    E = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"LLM run & planner & reviewer & parse-ok & fallbacks \\", r"\midrule"]
    for t in sorted(evid):
        r = evid[t]
        E.append(f"\\texttt{{{t.replace('v2_llm_','').replace('_',' ')}}} & "
                 f"{r['planner_calls']} & {r['reviewer_calls']} & "
                 f"{r['parse_ok']}/{r['llm_calls']} & {r['fallbacks']} \\\\")
    E += [r"\midrule",
          f"\\textbf{{total}} & \\multicolumn{{3}}{{c}}{{{total_calls} LLM calls}} "
          f"& \\textbf{{{total_fb}}} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_evidence.tex"), "w").write("\n".join(E))

    # ---- LaTeX: baselines table (NS + Darcy from exp4v3, with params) ----
    bl = agg["baselines_exp4v3"]
    bpar = bl.get("_params", {})
    B = [r"\begin{tabular}{lrcc}", r"\toprule",
         r"Model & Params & NS-2D rel.\ $L_2$ & Darcy-2D rel.\ $L_2$ \\",
         r"\midrule"]
    models = sorted({m for pde in bl if pde != "_params" for m in bl[pde]})
    for m in models:
        ns = fmt(bl.get("ns2d", {}).get(m))
        dc = fmt(bl.get("darcy2d", {}).get(m))
        B.append(f"{DISPLAY.get(m, m)} & {human_params(bpar.get(m))} & {ns} & {dc} \\\\")
    B += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_baselines.tex"), "w").write("\n".join(B))

    # ---- LaTeX: planner action distribution (how the LLM plans) ----
    ACTS = ["exploit", "explore", "hybridize", "mutate"]
    seen_acts = sorted({a for p in pacts["counts"].values() for a in p})
    cols = [a for a in ACTS if a in seen_acts] + \
           [a for a in seen_acts if a not in ACTS]
    A = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Problem & " + " & ".join(rf"\textsc{{{c}}}" for c in cols) + r" \\",
         r"\midrule"]
    for pde in PDES:
        c = pacts["counts"].get(pde, {})
        tot = sum(c.values()) or 1
        A.append(PDE_LABEL[pde] + " & " +
                 " & ".join(f"{100*c.get(k,0)/tot:.0f}\\%" for k in cols) + r" \\")
    allc = defaultdict(int)
    for p in pacts["counts"].values():
        for k, v in p.items():
            allc[k] += v
    tot = sum(allc.values()) or 1
    A += [r"\midrule", r"\textbf{all} & " +
          " & ".join(f"{100*allc.get(k,0)/tot:.0f}\\%" for k in cols) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_actions.tex"), "w").write("\n".join(A))

    # ---- LaTeX: per-seed appendix table (full disclosure, 30 runs) ----
    ABBR = {"fourier": "F", "attention": "A", "wavelet": "W",
            "residual_conv": "R", "branch_trunk": "T"}
    P = [r"\begin{tabular}{llllrr}", r"\toprule",
         r"Problem & Cond. & Seed & Discovered blocks & rel.\ $L_2$ & Params \\",
         r"\midrule"]
    for pde in PDES:
        for cond in CONDS:
            for seed in SEEDS:
                r = runs.get(f"v2_{cond}_{pde}_seed{seed}")
                if not r:
                    continue
                bl_seq = r"\texttt{" + "-".join(
                    ABBR.get(b, "?") for b in (r["blocks"] or [])) + "}"
                l2 = "--" if r["rel_l2_clean"] is None else f"{r['rel_l2_clean']:.4f}"
                P.append(f"{PDE_LABEL[pde]} & {cond.upper()} & {seed} & "
                         f"{bl_seq} & {l2} & {human_params(r['params'])} \\\\")
        P.append(r"\midrule")
    P[-1] = r"\bottomrule"
    P.append(r"\end{tabular}")
    open(os.path.join(PAPER_TAB, "tab_perseed.tex"), "w").write("\n".join(P))

    # ---- LaTeX: timing table (cost of LLM agency) ----
    def fm(s, p=1):
        return "--" if not s else (f"{s['mean']:.{p}f}" +
               (f"$\\pm${s['std']:.{p}f}" if s["n"] > 1 else ""))
    T = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Problem & \multicolumn{2}{c}{Total run (min)} & "
         r"\multicolumn{2}{c}{Per lab-eval (s)} \\",
         r" & LLM & PSO & LLM & PSO \\", r"\midrule"]
    for pde in PDES:
        t = tim[pde]
        lt, pt = t["llm"]["total_min"], t["pso"]["total_min"]
        # per lab-eval = per-iteration / num_labs(16)
        li = t["llm"]["iter_s"]; pi = t["pso"]["iter_s"]
        lle = {"mean": li["mean"]/16, "std": li["std"]/16, "n": li["n"]} if li else None
        ppe = {"mean": pi["mean"]/16, "std": pi["std"]/16, "n": pi["n"]} if pi else None
        T.append(f"{PDE_LABEL[pde]} & {fm(lt)} & {fm(pt)} & "
                 f"{fm(lle)} & {fm(ppe)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_timing.tex"), "w").write("\n".join(T))

    # ---- LaTeX: single-operator training time (screening fidelity) ----
    ot = tim["_train_op"]
    O = [r"\begin{tabular}{lc}", r"\toprule",
         r"Operator (NS-2D, screening fidelity) & Train time (s) \\", r"\midrule"]
    for m, s in sorted(ot.items(), key=lambda kv: (kv[1]["mean"] if kv[1] else 0)):
        O.append(f"{DISPLAY.get(m, m)} & {fm(s)} \\\\")
    O += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_TAB, "tab_trainop.tex"), "w").write("\n".join(O))

    # ---- human-readable summary ----
    M = [f"# v2 aggregate ({len(runs)}/30 runs present)", "",
         f"Evidence: {len(evid)} LLM runs, {total_calls} LLM calls, "
         f"**{total_fb} fallbacks** (0 = agents were LLMs throughout).", "",
         "## Discovery (best rel L2 clean, mean over seeds)", "",
         "| PDE | LLM | PSO | n(LLM/PSO) |", "|---|---|---|---|"]
    for pde in PDES:
        a = disc[pde]
        M.append(f"| {PDE_LABEL[pde]} | {fmt(a['llm']['rel_l2_clean'])} | "
                 f"{fmt(a['pso']['rel_l2_clean'])} | "
                 f"{a['llm']['n_runs']}/{a['pso']['n_runs']} |")
    open(os.path.join(ROOT, "results", "results_summary_v2.md"), "w",
         encoding="utf-8").write("\n".join(M))

    print(f"[aggregate] {len(runs)}/30 runs; LLM calls={total_calls}, "
          f"fallbacks={total_fb}; wrote tables + aggregate_v2.json")


if __name__ == "__main__":
    main()
