"""Assemble the paper end-to-end: aggregate v2 results -> generate tables +
figures -> compile paper/main.tex to PDF. Resilient: each step is guarded, and
the LaTeX is written to compile even with partial data, so this never hard-fails
the supervisor. Run with the jax-env python (needs matplotlib)."""
from __future__ import annotations
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))     # nod/code
ROOT = os.path.dirname(HERE)                            # nod
PAPER = os.path.join(ROOT, "paper")
sys.path.insert(0, HERE)

MIKTEX = r"C:\Users\luisl\AppData\Local\Programs\MiKTeX\miktex\bin\x64"


def step(name, fn):
    print(f"[build] {name} ...", flush=True)
    try:
        fn()
        return True
    except Exception as e:
        print(f"[build] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def compile_latex():
    # Best-effort: let MiKTeX install missing packages without prompting.
    initexmf = os.path.join(MIKTEX, "initexmf.exe")
    if os.path.exists(initexmf):
        try:
            subprocess.run([initexmf, "--set-config-value", "[MPM]AutoInstall=1"],
                           capture_output=True, text=True, timeout=60)
        except Exception:
            pass

    def tool(name):
        p = os.path.join(MIKTEX, name + ".exe")
        return p if os.path.exists(p) else name

    def run(cmd):
        return subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True,
                              timeout=900)

    # Explicit pdflatex -> bibtex -> pdflatex -> pdflatex (latexmk did not
    # reliably trigger bibtex in this environment, leaving citations undefined).
    pdflatex = [tool("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    run(pdflatex)
    run([tool("bibtex"), "main"])
    run(pdflatex)
    r = run(pdflatex)

    log = os.path.join(PAPER, "main.log")
    undef = 0
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="ignore").read()
        undef = txt.count("Citation") + 0  # informational
        und_cit = sum(1 for _ in __import__("re").finditer(r"Citation .* undefined", txt))
        und_ref = sum(1 for _ in __import__("re").finditer(r"Reference .* undefined", txt))
        if und_cit or und_ref:
            print(f"[build] WARNING: {und_cit} undefined citations, "
                  f"{und_ref} undefined refs")
    pdf = os.path.join(PAPER, "main.pdf")
    if os.path.exists(pdf):
        print(f"[build] PDF OK: {pdf} ({os.path.getsize(pdf)} bytes)")
        return True
    print(f"[build] no PDF (pdflatex rc={r.returncode}); last lines:")
    print("\n".join((r.stdout or "").splitlines()[-25:]))
    return False


def main():
    import aggregate_v2
    import make_v2_figures
    step("aggregate results", aggregate_v2.main)
    step("make figures", make_v2_figures.main)
    ok = step("compile LaTeX", compile_latex)
    print("[build] done" + ("" if ok else " (LaTeX step incomplete)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
