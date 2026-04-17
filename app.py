"""Streamlit dashboard: AI Science Community for Automated CINN Discovery.

Run with: streamlit run app.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

from core import build_feature_matrix, generate_tanh_weights, solve_ridge, l2_relative
from pde_problems import PROBLEMS
from config import PRESETS, PARAM_LABELS, get_search_space
from lab_agent import LabAgent
from ec_coordinator import PSOCoordinator, GACoordinator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
COORDINATORS = {
    "PSO (Particle Swarm Optimization)": PSOCoordinator,
    "GA (Genetic Algorithm)": GACoordinator,
}

st.set_page_config(
    page_title="AI Science Community - CINN Discovery",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
FITNESS_CMAP = plt.cm.RdYlGn_r  # red = bad, green = good (reversed)


def fitness_color(fitness, vmin=0.0, vmax=0.5):
    t = np.clip((fitness - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgba = FITNESS_CMAP(t)
    return mcolors.to_hex(rgba)


def status_icon(status, is_elite=False):
    if is_elite:
        return "⭐"
    icons = {"idle": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}
    return icons.get(status, "?")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def render_swarm_grid(labs, problem, n_cols=4, citations=None):
    """Render a grid of mini-plots, one per lab."""
    n = len(labs)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.0 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Validation grid for exact solution
    x_plot = np.linspace(problem.x_min, problem.x_max, 800)
    if problem.mode == "pde":
        y_exact = problem.exact_solution(x_plot, problem.t_final)
    else:
        y_exact = problem.exact_solution(x_plot)

    # Fitness range for coloring
    fitnesses = [lab.result["fitness"] for lab in labs if lab.result]
    vmin = min(fitnesses) if fitnesses else 0.0
    vmax = max(max(fitnesses), vmin + 1e-6) if fitnesses else 0.5

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        if i >= n:
            ax.axis("off")
            continue

        lab = labs[i]
        ax.plot(x_plot, y_exact, color="#999999", linewidth=0.8, label="Exact")

        if lab.result:
            r = lab.result
            color = fitness_color(r["fitness"], vmin, vmax)

            # Re-predict on plot grid for smooth rendering
            cfg = lab.config
            W, b = generate_tanh_weights(cfg["n_tanh"], cfg["seed_tanh"], cfg["scale"])
            if problem.mode == "pde" and hasattr(problem, "characteristic_shift"):
                xi = problem.characteristic_shift(x_plot, problem.t_final)
            else:
                xi = x_plot
            H = build_feature_matrix(xi, W, b, r["positions"], cfg["kappa"])
            y_pred = H @ r["beta"]
            ax.plot(x_plot, y_pred, color=color, linewidth=1.4)

            # Mark step positions
            for pos in r["positions"]:
                if problem.mode == "pde" and hasattr(problem, "characteristic_shift"):
                    # Show where the step appears at t_final
                    display_pos = pos + (problem.x_max - problem.x_min) * 0.0
                    # For advection: step at pos in IC => pos + v*t in solution
                    if hasattr(problem, "v"):
                        display_pos = pos + problem.v * problem.t_final
                    elif hasattr(problem, "shock_speed"):
                        display_pos = pos + problem.shock_speed * problem.t_final
                    else:
                        display_pos = pos
                else:
                    display_pos = pos
                ax.axvline(display_pos, color="red", linestyle="--",
                           alpha=0.4, linewidth=0.7)

            steps_label = f"{r['n_steps']}s" if r["n_steps"] > 0 else "tanh"
            cit_str = ""
            if citations is not None and i < len(citations):
                cit_str = f"  [{int(citations[i])} cit]"
            ax.set_title(
                f"Lab {lab.lab_id}  |  L2={r['l2_error']:.1e}  |  "
                f"{steps_label}{cit_str}",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
        else:
            ax.set_title(f"Lab {lab.lab_id} | waiting...", fontsize=8, color="#aaa")

        ax.tick_params(labelsize=6)
        ax.set_xlim(problem.x_min, problem.x_max)

    plt.tight_layout(pad=1.0)
    return fig


def render_fitness_chart(coordinator_history, all_lab_results):
    """Line chart: best & mean fitness per meta-generation."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    gens = list(range(len(coordinator_history)))
    best = [h["best_fitness"] for h in coordinator_history]
    mean = [h["mean_fitness"] for h in coordinator_history]

    ax.plot(gens, best, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Best")
    ax.plot(gens, mean, "s--", color="#e67e22", linewidth=1.5, markersize=5, label="Mean")

    # Scatter all labs per generation
    for g, gen_labs in enumerate(all_lab_results):
        for lab in gen_labs:
            if lab.result:
                c = "#2ecc71" if lab.result["n_steps"] > 0 else "#3498db"
                ax.scatter(g, lab.result["fitness"], color=c, alpha=0.3, s=20, zorder=1)

    ax.set_xlabel("Meta-Generation", fontsize=10)
    ax.set_ylabel("Fitness (lower = better)", fontsize=10)
    ax.set_xticks(gens)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def render_best_solution(best_lab, problem):
    """Large plot of the best solution vs exact."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x_plot = np.linspace(problem.x_min, problem.x_max, 1000)
    if problem.mode == "pde":
        y_exact = problem.exact_solution(x_plot, problem.t_final)
    else:
        y_exact = problem.exact_solution(x_plot)

    cfg = best_lab.config
    r = best_lab.result
    W, b = generate_tanh_weights(cfg["n_tanh"], cfg["seed_tanh"], cfg["scale"])

    if problem.mode == "pde" and hasattr(problem, "characteristic_shift"):
        xi = problem.characteristic_shift(x_plot, problem.t_final)
    else:
        xi = x_plot
    H = build_feature_matrix(xi, W, b, r["positions"], cfg["kappa"])
    y_pred = H @ r["beta"]

    # Left: solution comparison
    ax1.plot(x_plot, y_exact, "k-", linewidth=2, label="Exact", alpha=0.7)
    ax1.plot(x_plot, y_pred, "-", color="#2ecc71", linewidth=2, label="Prediction")
    for pos in r["positions"]:
        ax1.axvline(pos, color="red", linestyle="--", alpha=0.5, linewidth=1,
                     label="Step neuron" if pos == r["positions"][0] else "")
    ax1.set_title(f"Best Solution  (L2 = {r['l2_error']:.2e})", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: convergence of inner GA
    gens = [h["gen"] for h in best_lab.history]
    best_f = [h["best_fitness"] for h in best_lab.history]
    mean_f = [h["mean_fitness"] for h in best_lab.history]
    ax2.semilogy(gens, best_f, "o-", color="#2ecc71", label="Best fitness")
    ax2.semilogy(gens, mean_f, "s--", color="#e67e22", alpha=0.7, label="Mean fitness")
    ax2.set_title("Inner GA Convergence", fontsize=12)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness (log scale)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # --- Theme toggle ---
    header_cols = st.columns([20, 1])
    with header_cols[0]:
        st.title("AI Science Community for Automated CINN Discovery")
    with header_cols[1]:
        light_mode = st.toggle("☀️", value=False, help="Light mode")

    if light_mode:
        plt.rcParams.update(plt.rcParamsDefault)
        st.markdown(
            """<style>
            .stApp, [data-testid="stAppViewContainer"] {
                background-color: #ffffff !important;
            }
            [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
                background-color: #f0f2f6 !important;
            }
            .stApp *, [data-testid="stSidebar"] *,
            .stMarkdown, .stCaption, p, span, label,
            h1, h2, h3, h4, h5, h6,
            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"],
            .stSelectbox label, .stSlider label,
            [data-testid="stWidgetLabel"] {
                color: #1a1a1a !important;
            }
            [data-testid="stHeader"] {
                background-color: #ffffff !important;
            }
            /* Force ALL text black in light mode */
            [data-baseweb] *,
            [data-baseweb="select"] *,
            [data-baseweb="popover"] *,
            [data-baseweb="menu"] *,
            [data-baseweb="input"] *,
            [role="listbox"] *,
            [role="option"] *,
            .stSelectbox *,
            .stMultiSelect *,
            .stTextInput *,
            div[data-baseweb="select"] {
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }
            /* Selectbox background */
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border-color: #cccccc !important;
            }
            /* Dropdown list background */
            [data-baseweb="popover"],
            [data-baseweb="popover"] > div,
            [data-baseweb="menu"],
            [data-baseweb="menu"] > div,
            ul[role="listbox"],
            ul[role="listbox"] > li,
            li[role="option"],
            li[role="option"] > div {
                background-color: #ffffff !important;
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }
            li[role="option"]:hover,
            li[role="option"][aria-selected="true"] {
                background-color: #e8f5e9 !important;
            }
            /* Slider values */
            [data-testid="stThumbValue"],
            [data-testid="stTickBarMin"],
            [data-testid="stTickBarMax"] {
                color: #1a1a1a !important;
            }
            /* Expander text */
            [data-testid="stExpander"] * {
                color: #1a1a1a !important;
            }
            /* Button text stays white on green */
            .stButton button[kind="primary"] {
                color: #ffffff !important;
            }
            .stButton button[kind="primary"] * {
                color: #ffffff !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
    else:
        plt.style.use("dark_background")

    st.caption(
        "A virtual lab community (Braga-Neto, arXiv:2603.21344) where "
        "population-based metaheuristics coordinate labs to discover optimal "
        "CINN architectures for PDEs with discontinuities. Labs cite each "
        "other, successful labs grow, unsuccessful labs shrink."
    )
    st.caption(
        "Luis Loo & Ulisses Braga-Neto  |  Scientific Machine Learning Lab  "
        "|  Texas A&M University"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        problem_name = st.selectbox(
            "PDE Problem", list(PROBLEMS.keys()),
            key="pde_problem",
        )
        problem = PROBLEMS[problem_name]()

        coordinator_name = st.selectbox(
            "Coordination Strategy",
            list(COORDINATORS.keys()),
            key="coord_strategy",
            help="Population-based metaheuristic that coordinates the "
                 "virtual labs. All strategies include citation-based "
                 "influence and lab dynamics.",
        )
        CoordinatorClass = COORDINATORS[coordinator_name]
        use_pso = coordinator_name.startswith("PSO")

        preset_name = st.selectbox(
            "Demo Preset", list(PRESETS.keys()), index=1,
            key="demo_preset",
        )
        preset = PRESETS[preset_name]

        with st.expander("Strategy Parameters", expanded=False):
            n_labs = st.slider("Number of labs", 4, 16, preset["n_labs"])
            n_meta_gen = st.slider("Iterations", 2, 8, preset["n_meta_gen"])
            meta_elite = st.slider("Elite count", 1, 4, preset["meta_elite"])

            # PSO-specific params (always defined with defaults)
            pso_w = 0.6
            pso_c1 = 1.5
            pso_c2 = 2.0
            meta_mut = 0.4

            if use_pso:
                pso_w = st.slider("Inertia (w)", 0.2, 0.9, 0.6, step=0.1)
                pso_c1 = st.slider("Cognitive (c1)", 0.5, 3.0, 1.5, step=0.1)
                pso_c2 = st.slider("Social (c2)", 0.5, 3.0, 2.0, step=0.1)
            else:
                meta_mut = st.slider("Mutation rate", 0.1, 0.8, 0.4, step=0.05)

        run_btn = st.button("Run Demo", type="primary", use_container_width=True)

        st.divider()
        st.markdown("**How it works:**")
        st.markdown(
            "1. Initialize N virtual labs with diverse configs\n"
            "2. Each lab runs a Step-CINN experiment (inner GA)\n"
            "3. Labs cite each other's results (peer review)\n"
            "4. Metaheuristic evolves configs toward cited leaders\n"
            "5. Cited labs grow, uncited labs get culled\n"
            "6. Repeat until community consensus emerges"
        )

    # --- Main area ---
    if not run_btn:
        # Show problem description
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Problem: {problem.name}")
            st.write(problem.description)

            fig_preview, ax = plt.subplots(figsize=(8, 3))
            x_preview = np.linspace(problem.x_min, problem.x_max, 800)
            if problem.mode == "pde":
                ax.plot(x_preview, problem.initial_condition(x_preview),
                        "b-", linewidth=2, label="Initial Condition")
                ax.plot(x_preview,
                        problem.exact_solution(x_preview, problem.t_final),
                        "r--", linewidth=2, label=f"Solution at t={problem.t_final}")
            else:
                ax.plot(x_preview, problem.exact_solution(x_preview),
                        "b-", linewidth=2, label="Target function")
            ax.legend()
            ax.grid(True, alpha=0.2)
            ax.set_title(problem.name)
            st.pyplot(fig_preview)
            plt.close(fig_preview)

        with col2:
            st.metric("Domain", f"[{problem.x_min}, {problem.x_max}]")
            if problem.mode == "pde":
                st.metric("t_final", f"{problem.t_final}")
            st.metric("Labs per gen", n_labs)
            st.metric("Meta-generations", n_meta_gen)
            total_labs = n_labs * n_meta_gen
            st.metric("Total experiments", total_labs)

        st.info("Press **Run Demo** in the sidebar to start the evolutionary swarm.")
        return

    # --- Run the demo ---
    coord_kwargs = dict(n_labs=n_labs, elite_count=meta_elite, preset=preset_name)
    if use_pso:
        coord_kwargs.update(w=pso_w, c1=pso_c1, c2=pso_c2)
    else:
        coord_kwargs.update(mutation_rate=meta_mut)
    coordinator = CoordinatorClass(**coord_kwargs)
    configs = coordinator.initialize_population()

    # Placeholders for real-time updates
    progress_ph = st.empty()
    swarm_ph = st.empty()
    col_best, col_fitness = st.columns([1, 1])
    best_ph = col_best.empty()
    fitness_ph = col_fitness.empty()
    log_ph = st.empty()
    discovery_ph = st.empty()
    table_ph = st.empty()

    all_gen_labs = []
    global_best_lab = None
    log_lines = []
    t_start = time.time()

    for meta_gen in range(n_meta_gen):
        gen_labs = []

        for lab_idx, config in enumerate(configs):
            # Progress
            total = n_meta_gen * n_labs
            current = meta_gen * n_labs + lab_idx + 1
            with progress_ph.container():
                st.progress(current / total)
                elapsed = time.time() - t_start
                st.caption(
                    f"Meta-Gen {meta_gen + 1}/{n_meta_gen}  |  "
                    f"Lab {lab_idx + 1}/{n_labs}  |  "
                    f"Elapsed: {elapsed:.0f}s"
                )

            # Run lab
            lab_id = f"{meta_gen + 1}.{lab_idx + 1}"
            lab = LabAgent(lab_id=lab_id, config=config, problem=problem)
            lab.run()
            gen_labs.append(lab)

            # Track global best
            if lab.result and (
                global_best_lab is None
                or lab.result["fitness"] < global_best_lab.result["fitness"]
            ):
                global_best_lab = lab
                log_lines.append(
                    f"**New best!** Lab {lab_id}: "
                    f"L2={lab.result['l2_error']:.2e}, "
                    f"{lab.result['n_steps']} steps"
                )

            # Log
            log_lines.append(lab.summary())

            # Update swarm (in-progress, no citations yet)
            with swarm_ph.container():
                st.subheader(
                    f"Swarm View  --  Iteration {meta_gen + 1}/{n_meta_gen}"
                )
                fig = render_swarm_grid(gen_labs, problem, n_cols=min(4, n_labs))
                st.pyplot(fig)
                plt.close(fig)

        # PSO swarm update
        configs_with_fitness = [
            (lab.config, lab.result["fitness"]) for lab in gen_labs
        ]
        configs, gen_info = coordinator.select_and_evolve(configs_with_fitness)
        all_gen_labs.append(gen_labs)

        # Update swarm grid with citations
        with swarm_ph.container():
            st.subheader(
                f"Swarm View  --  Iteration {meta_gen + 1}/{n_meta_gen}"
            )
            fig = render_swarm_grid(
                gen_labs, problem, n_cols=min(4, n_labs),
                citations=gen_info.get("cumulative_citations"),
            )
            st.pyplot(fig)
            plt.close(fig)

        # Mark elite
        elite_ids = {
            gen_labs[i].lab_id for i in gen_info.get("elite_indices", [])
        }
        cit_summary = ", ".join(
            f"L{i+1}:{int(c)}"
            for i, c in enumerate(gen_info.get("citations", []))
            if c > 0
        )
        log_lines.append(
            f"--- Iter {meta_gen + 1} | "
            f"Best: {gen_info['best_fitness']:.2e} | "
            f"Mean: {gen_info['mean_fitness']:.2e} | "
            f"Elite: {', '.join(elite_ids)} | "
            f"Citations: [{cit_summary}] ---"
        )

        # Update fitness chart
        with fitness_ph.container():
            st.subheader("Fitness Evolution")
            fig = render_fitness_chart(coordinator.history, all_gen_labs)
            st.pyplot(fig)
            plt.close(fig)

        # Update best solution
        if global_best_lab:
            with best_ph.container():
                st.subheader("Best Solution Found")
                fig = render_best_solution(global_best_lab, problem)
                st.pyplot(fig)
                plt.close(fig)

        # Update log
        with log_ph.container():
            with st.expander("Agent Log", expanded=False):
                st.text("\n".join(log_lines[-20:]))

    # --- Final results ---
    elapsed_total = time.time() - t_start
    with progress_ph.container():
        st.progress(1.0)
        st.caption(f"Demo complete in {elapsed_total:.1f}s")

    # Discoveries
    findings = coordinator.discoveries()
    if findings:
        with discovery_ph.container():
            st.subheader("Discoveries")
            for f in findings:
                st.success(f)

    # Config comparison table
    if global_best_lab:
        with table_ph.container():
            st.subheader("Configuration Comparison")
            import pandas as pd

            rows = []
            for gen_labs in all_gen_labs:
                for lab in gen_labs:
                    if lab.result:
                        row = {
                            "Lab": lab.lab_id,
                            "L2 Error": f"{lab.result['l2_error']:.2e}",
                            "Steps": lab.result["n_steps"],
                            "n_tanh": lab.config["n_tanh"],
                            "kappa": lab.config["kappa"],
                            "max_steps": lab.config["max_steps"],
                            "scale": lab.config["scale"],
                            "lam": lab.config["lam"],
                        }
                        rows.append(row)

            df = pd.DataFrame(rows).sort_values("L2 Error")
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
