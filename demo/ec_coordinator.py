"""Swarm coordinators for the Evolutionary Virtual Lab demo.

Two swarm strategies for the AI Science Community (Braga-Neto, arXiv:2603.21344):
- PSOCoordinator: PSO dynamics + citation mechanism + lab growth/shrinkage
- GACoordinator: GA operators (tournament/crossover/mutation) + citations + lab dynamics
Both implement citation-based influence and natural selection of research directions.
"""

import numpy as np
from config import SEARCH_SPACE, get_search_space


# ======================================================================
# GA Coordinator (baseline)
# ======================================================================
class GACoordinator:
    """GA-based AI-SC coordinator.

    Uses tournament selection + crossover + mutation for search,
    with the same citation-based influence and lab dynamics from the
    AI-SC framework. Citations bias tournament selection toward
    well-cited labs and adjust compute budgets.
    """

    def __init__(self, n_labs=8, elite_count=2, mutation_rate=0.4,
                 seed=42, preset="Standard (~90s)", **_kwargs):
        self.n_labs = n_labs
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.search_space, _ = get_search_space(preset)
        self.history = []
        self.all_configs_tested = []
        self.citations = np.zeros(n_labs)
        self.cumulative_citations = np.zeros(n_labs)

    def random_config(self):
        config = {}
        for key, values in self.search_space.items():
            config[key] = values[self.rng.integers(len(values))]
        config["seed_ga"] = int(self.rng.integers(0, 10000))
        return config

    def initialize_population(self):
        configs = []
        baseline = self.random_config()
        baseline["max_steps"] = 0
        configs.append(baseline)
        step_heavy = self.random_config()
        step_heavy["max_steps"] = max(self.search_space["max_steps"])
        step_heavy["kappa"] = max(self.search_space["kappa"])
        configs.append(step_heavy)
        while len(configs) < self.n_labs:
            configs.append(self.random_config())
        return configs

    def _compute_citations(self, fitnesses):
        """Each lab votes for its top-2 peers (same as PSO version)."""
        n = len(fitnesses)
        self.citations = np.zeros(n)
        for i in range(n):
            candidates = [(j, fitnesses[j]) for j in range(n) if j != i]
            candidates.sort(key=lambda x: x[1])
            for rank, (j, _) in enumerate(candidates[:2]):
                self.citations[j] += 2 - rank
        self.cumulative_citations += self.citations

    def _citation_weighted_tournament(self, configs, fitnesses, k=3):
        """Tournament selection biased by citations.

        Labs with more citations get a fitness bonus, making them more
        likely to be selected as parents.
        """
        idx = self.rng.choice(len(configs),
                              size=min(k, len(configs)), replace=False)
        # Effective fitness: citation bonus reduces apparent fitness
        citation_bonus = self.cumulative_citations + 1
        max_bonus = citation_bonus.max()
        effective = [fitnesses[i] / (0.5 + 0.5 * citation_bonus[i] / max_bonus)
                     for i in idx]
        best = idx[int(np.argmin(effective))]
        return dict(configs[best])

    def _adjust_compute_budget(self, configs, fitnesses):
        """Same lab dynamics as PSO: cited labs grow, uncited shrink."""
        median_cit = np.median(self.citations)
        for i, config in enumerate(configs):
            if self.citations[i] > median_cit:
                pop_vals = self.search_space["ga_pop_size"]
                gen_vals = self.search_space["ga_n_gen"]
                curr_pop_idx = (pop_vals.index(config["ga_pop_size"])
                                if config["ga_pop_size"] in pop_vals else 0)
                curr_gen_idx = (gen_vals.index(config["ga_n_gen"])
                                if config["ga_n_gen"] in gen_vals else 0)
                config["ga_pop_size"] = pop_vals[min(curr_pop_idx + 1,
                                                     len(pop_vals) - 1)]
                config["ga_n_gen"] = gen_vals[min(curr_gen_idx + 1,
                                                  len(gen_vals) - 1)]
            elif self.citations[i] == 0:
                pop_vals = self.search_space["ga_pop_size"]
                gen_vals = self.search_space["ga_n_gen"]
                config["ga_pop_size"] = pop_vals[0]
                config["ga_n_gen"] = gen_vals[0]
        return configs

    def select_and_evolve(self, configs_with_fitness):
        fitnesses = np.array([f for _, f in configs_with_fitness])
        configs = [c for c, _ in configs_with_fitness]

        # Citation mechanism
        self._compute_citations(fitnesses)

        ranked_indices = np.argsort(fitnesses)
        elite_set = set(ranked_indices[:self.elite_count].tolist())

        # Elite survive
        elite = [dict(configs[i]) for i in ranked_indices[:self.elite_count]]
        new_configs = list(elite)

        # Fill with citation-weighted tournament + crossover + mutation
        while len(new_configs) < self.n_labs:
            p1 = self._citation_weighted_tournament(configs, fitnesses)
            p2 = self._citation_weighted_tournament(configs, fitnesses)
            child = {}
            for key in self.search_space:
                child[key] = p1[key] if self.rng.random() < 0.5 else p2[key]
            for key, values in self.search_space.items():
                if self.rng.random() < self.mutation_rate:
                    child[key] = values[self.rng.integers(len(values))]
            child["seed_ga"] = int(self.rng.integers(0, 10000))
            new_configs.append(child)

        # Cull uncited labs (replace with random)
        for i in range(self.elite_count, len(new_configs)):
            orig_idx = i if i < len(fitnesses) else -1
            if (orig_idx >= 0
                    and self.cumulative_citations[orig_idx] == 0
                    and len(self.history) >= 2):
                new_configs[i] = self.random_config()

        # Adjust compute budgets
        new_configs = self._adjust_compute_budget(new_configs, fitnesses)

        raw_best_idx = int(np.argmin(fitnesses))
        gen_info = {
            "best_fitness": float(fitnesses[raw_best_idx]),
            "best_config": dict(configs[raw_best_idx]),
            "worst_fitness": float(np.max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "citations": self.citations.copy(),
            "cumulative_citations": self.cumulative_citations.copy(),
            "elite_indices": sorted(elite_set),
            "configs_ranked": [
                (dict(configs[i]), float(fitnesses[i]))
                for i in ranked_indices
            ],
        }
        self.history.append(gen_info)
        self.all_configs_tested.extend([dict(c) for c in configs])

        return new_configs, gen_info

    def best_config(self):
        if not self.history:
            return None
        best_gen = min(self.history, key=lambda g: g["best_fitness"])
        return best_gen["best_config"], best_gen["best_fitness"]

    def discoveries(self):
        if not self.history:
            return []
        findings = []
        step_errors, no_step_errors = [], []
        for gen_info in self.history:
            for config, fitness in gen_info["configs_ranked"]:
                (no_step_errors if config["max_steps"] == 0
                 else step_errors).append(fitness)
        if step_errors and no_step_errors:
            ms, mn = np.mean(step_errors), np.mean(no_step_errors)
            if ms < mn * 0.5:
                findings.append(
                    f"Step neurons improve accuracy by {mn/ms:.1f}x over pure "
                    f"tanh (avg L2: {ms:.2e} vs {mn:.2e})")
        best_config, _ = self.best_config()
        if best_config["max_steps"] > 0:
            findings.append(
                f"Optimal: kappa={best_config['kappa']}, "
                f"{best_config['max_steps']} max steps")
        findings.append(
            f"Best architecture: {best_config['n_tanh']} tanh + "
            f"{best_config['max_steps']} step slots, "
            f"scale={best_config['scale']}, lam={best_config['lam']}")
        # Citation dynamics
        if len(self.history) >= 2:
            final_cit = self.cumulative_citations
            top_cited = int(np.argmax(final_cit))
            if final_cit[top_cited] > 0:
                findings.append(
                    f"Most-cited lab received {int(final_cit[top_cited])} "
                    f"cumulative citations (community consensus leader)")
            culled = int(np.sum(final_cit == 0))
            if culled > 0:
                findings.append(
                    f"{culled} lab(s) received 0 citations and were replaced "
                    f"(natural selection)")
        return findings


# ======================================================================
# PSO Coordinator (AI Science Community)
# ======================================================================


class PSOCoordinator:
    """PSO-based coordinator implementing AI Science Community dynamics.

    Each lab is a particle in a discrete search space. Positions are encoded
    as continuous index vectors and snapped to the nearest valid value.
    """

    def __init__(self, n_labs=8, w=0.6, c1=1.5, c2=2.0,
                 elite_count=2, seed=42, preset="Standard (~90s)"):
        self.n_labs = n_labs
        self.w = w            # inertia weight
        self.c1 = c1          # cognitive coefficient (personal best)
        self.c2 = c2          # social coefficient (global best)
        self.elite_count = elite_count
        self.rng = np.random.default_rng(seed)
        self.search_space, _ = get_search_space(preset)

        # Ordered parameter keys (defines the vector dimensions)
        self.param_keys = sorted(self.search_space.keys())
        self.n_dims = len(self.param_keys)
        self.dim_sizes = np.array([len(self.search_space[k])
                                   for k in self.param_keys], dtype=float)

        # Particle state
        self.positions = []           # continuous index vectors
        self.velocities = []          # velocity vectors
        self.personal_bests = []      # best position per particle
        self.personal_best_fit = []   # best fitness per particle
        self.global_best = None       # best position across all particles
        self.global_best_fit = np.inf

        # Citation / influence tracking
        self.citations = np.zeros(n_labs)
        self.cumulative_citations = np.zeros(n_labs)
        self.influence = np.ones(n_labs) / n_labs

        # History
        self.history = []
        self.all_configs_tested = []

    # ------------------------------------------------------------------
    # Encoding: config <-> continuous vector
    # ------------------------------------------------------------------
    def _config_to_vector(self, config):
        vec = np.zeros(self.n_dims)
        for i, key in enumerate(self.param_keys):
            values = self.search_space[key]
            if config[key] in values:
                vec[i] = float(values.index(config[key]))
            else:
                # nearest match
                diffs = [abs(v - config[key]) if isinstance(v, (int, float))
                         else 0 for v in values]
                vec[i] = float(np.argmin(diffs))
        return vec

    def _vector_to_config(self, vec):
        config = {}
        for i, key in enumerate(self.param_keys):
            values = self.search_space[key]
            idx = int(np.clip(round(vec[i]), 0, len(values) - 1))
            config[key] = values[idx]
        config["seed_ga"] = int(self.rng.integers(0, 10000))
        return config

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def random_config(self):
        config = {}
        for key, values in self.search_space.items():
            config[key] = values[self.rng.integers(len(values))]
        config["seed_ga"] = int(self.rng.integers(0, 10000))
        return config

    def initialize_population(self):
        """Create initial swarm with diversity guarantees."""
        configs = []

        # Force one pure-tanh baseline
        baseline = self.random_config()
        baseline["max_steps"] = 0
        configs.append(baseline)

        # Force one step-heavy config
        step_heavy = self.random_config()
        step_heavy["max_steps"] = max(self.search_space["max_steps"])
        step_heavy["kappa"] = max(self.search_space["kappa"])
        configs.append(step_heavy)

        while len(configs) < self.n_labs:
            configs.append(self.random_config())

        # Initialize PSO state
        self.positions = [self._config_to_vector(c) for c in configs]
        self.velocities = [
            self.rng.uniform(-1, 1, size=self.n_dims)
            for _ in range(self.n_labs)
        ]
        self.personal_bests = [pos.copy() for pos in self.positions]
        self.personal_best_fit = [np.inf] * self.n_labs
        self.global_best = self.positions[0].copy()
        self.global_best_fit = np.inf
        self.citations = np.zeros(self.n_labs)
        self.cumulative_citations = np.zeros(self.n_labs)

        return configs

    # ------------------------------------------------------------------
    # Citation mechanism (Section 2 & 4 of Braga-Neto 2026)
    # ------------------------------------------------------------------
    def _compute_citations(self, fitnesses):
        """Each lab votes for its top-2 peers (excluding itself).

        Mirrors the citation-based influence system: labs that produce
        good results receive more citations from peers, gaining more
        weight in the swarm movement.
        """
        n = len(fitnesses)
        self.citations = np.zeros(n)

        for i in range(n):
            # Each lab votes for the 2 best other labs
            candidates = [(j, fitnesses[j]) for j in range(n) if j != i]
            candidates.sort(key=lambda x: x[1])
            for rank, (j, _) in enumerate(candidates[:2]):
                self.citations[j] += 2 - rank  # 2 pts for 1st, 1 pt for 2nd

        self.cumulative_citations += self.citations

        # Influence = normalized cumulative citations (with smoothing)
        total = self.cumulative_citations.sum() + 1e-10
        self.influence = (self.cumulative_citations + 1) / (total + n)

    def _citation_weighted_best(self, fitnesses):
        """Compute global best as citation-weighted best, not just raw best.

        Labs with high citations have their fitness "boosted", making
        well-cited solutions more attractive to the swarm.
        """
        # Effective fitness: lower is better, citations reduce it
        citation_bonus = self.influence / self.influence.max()
        effective_fitness = fitnesses * (1.0 / (0.5 + 0.5 * citation_bonus))

        best_idx = int(np.argmin(effective_fitness))
        return best_idx

    # ------------------------------------------------------------------
    # Lab dynamics (Section 3 & 5 of Braga-Neto 2026)
    # ------------------------------------------------------------------
    def _adjust_compute_budget(self, configs, fitnesses):
        """Successful labs grow, unsuccessful labs shrink.

        Labs with above-median citations get more inner GA budget.
        Labs with 0 citations get reduced budget. This mirrors natural
        selection in scientific communities.
        """
        median_cit = np.median(self.citations)

        for i, config in enumerate(configs):
            if self.citations[i] > median_cit:
                # Successful lab: increase budget (cap at search space max)
                pop_vals = self.search_space["ga_pop_size"]
                gen_vals = self.search_space["ga_n_gen"]
                curr_pop_idx = pop_vals.index(config["ga_pop_size"]) \
                    if config["ga_pop_size"] in pop_vals else 0
                curr_gen_idx = gen_vals.index(config["ga_n_gen"]) \
                    if config["ga_n_gen"] in gen_vals else 0
                config["ga_pop_size"] = pop_vals[min(curr_pop_idx + 1,
                                                     len(pop_vals) - 1)]
                config["ga_n_gen"] = gen_vals[min(curr_gen_idx + 1,
                                                  len(gen_vals) - 1)]
            elif self.citations[i] == 0:
                # Unsuccessful lab: reduce budget
                pop_vals = self.search_space["ga_pop_size"]
                gen_vals = self.search_space["ga_n_gen"]
                config["ga_pop_size"] = pop_vals[0]
                config["ga_n_gen"] = gen_vals[0]

        return configs

    # ------------------------------------------------------------------
    # PSO update (Section 6 of Braga-Neto 2026)
    # ------------------------------------------------------------------
    def select_and_evolve(self, configs_with_fitness):
        """PSO velocity + position update with citation influence.

        Replaces GA selection/crossover/mutation with PSO dynamics:
        v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest_cited - x)
        x(t+1) = x(t) + v(t+1)

        Returns:
            new_configs: list of config dicts for next iteration
            generation_info: dict with stats
        """
        fitnesses = np.array([f for _, f in configs_with_fitness])
        configs = [c for c, _ in configs_with_fitness]

        # --- Citation mechanism ---
        self._compute_citations(fitnesses)

        # --- Update personal bests ---
        for i in range(self.n_labs):
            if fitnesses[i] < self.personal_best_fit[i]:
                self.personal_best_fit[i] = fitnesses[i]
                self.personal_bests[i] = self.positions[i].copy()

        # --- Citation-weighted global best ---
        best_idx = self._citation_weighted_best(fitnesses)
        if fitnesses[best_idx] < self.global_best_fit:
            self.global_best_fit = fitnesses[best_idx]
            self.global_best = self.positions[best_idx].copy()

        # --- Rank labs for elite preservation ---
        ranked_indices = np.argsort(fitnesses)
        elite_set = set(ranked_indices[:self.elite_count].tolist())

        # --- PSO velocity + position update ---
        max_v = (self.dim_sizes - 1) * 0.4  # clamp velocity

        for i in range(self.n_labs):
            r1 = self.rng.random(self.n_dims)
            r2 = self.rng.random(self.n_dims)

            cognitive = self.c1 * r1 * (self.personal_bests[i] - self.positions[i])
            social = self.c2 * r2 * (self.global_best - self.positions[i])

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            self.velocities[i] = np.clip(self.velocities[i], -max_v, max_v)

            if i in elite_set:
                # Elite labs keep their position (successful labs persist)
                continue

            # Check for culling: 0 cumulative citations -> replace with random
            if self.cumulative_citations[i] == 0 and len(self.history) >= 2:
                self.positions[i] = self._config_to_vector(self.random_config())
                self.velocities[i] = self.rng.uniform(-1, 1, size=self.n_dims)
                self.personal_bests[i] = self.positions[i].copy()
                self.personal_best_fit[i] = np.inf
                continue

            # Standard PSO position update
            self.positions[i] = self.positions[i] + self.velocities[i]

            # Clamp to valid index ranges
            for j in range(self.n_dims):
                self.positions[i][j] = np.clip(self.positions[i][j],
                                               0, self.dim_sizes[j] - 1)

        # --- Convert positions back to configs ---
        new_configs = [self._vector_to_config(pos) for pos in self.positions]

        # --- Adjust compute budgets based on citations ---
        new_configs = self._adjust_compute_budget(new_configs, fitnesses)

        # --- Record generation info ---
        raw_best_idx = int(np.argmin(fitnesses))
        gen_info = {
            "best_fitness": float(fitnesses[raw_best_idx]),
            "best_config": dict(configs[raw_best_idx]),
            "worst_fitness": float(np.max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "citations": self.citations.copy(),
            "cumulative_citations": self.cumulative_citations.copy(),
            "influence": self.influence.copy(),
            "elite_indices": sorted(elite_set),
            "configs_ranked": [
                (dict(configs[i]), float(fitnesses[i]))
                for i in ranked_indices
            ],
        }
        self.history.append(gen_info)
        self.all_configs_tested.extend([dict(c) for c in configs])

        return new_configs, gen_info

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def best_config(self):
        if not self.history:
            return None
        best_gen = min(self.history, key=lambda g: g["best_fitness"])
        return best_gen["best_config"], best_gen["best_fitness"]

    def discoveries(self):
        """Analyze what the swarm discovered."""
        if not self.history:
            return []

        findings = []

        # Compare step vs no-step
        step_errors = []
        no_step_errors = []
        for gen_info in self.history:
            for config, fitness in gen_info["configs_ranked"]:
                if config["max_steps"] == 0:
                    no_step_errors.append(fitness)
                else:
                    step_errors.append(fitness)

        if step_errors and no_step_errors:
            mean_step = np.mean(step_errors)
            mean_no_step = np.mean(no_step_errors)
            if mean_step < mean_no_step * 0.5:
                ratio = mean_no_step / mean_step
                findings.append(
                    f"Step neurons improve accuracy by {ratio:.1f}x over pure "
                    f"tanh (avg L2: {mean_step:.2e} vs {mean_no_step:.2e})"
                )
            elif mean_no_step < mean_step * 0.8:
                findings.append(
                    "Pure tanh performs competitively; step neurons not "
                    "essential for this problem"
                )

        best_config, best_fitness = self.best_config()

        if best_config["max_steps"] > 0:
            findings.append(
                f"Optimal step steepness: kappa={best_config['kappa']} "
                f"with {best_config['max_steps']} max steps allowed"
            )

        findings.append(
            f"Best architecture: {best_config['n_tanh']} tanh + "
            f"{best_config['max_steps']} step slots, "
            f"scale={best_config['scale']}, lam={best_config['lam']}"
        )

        # Citation dynamics
        if len(self.history) >= 2:
            final_cit = self.cumulative_citations
            top_cited = int(np.argmax(final_cit))
            if final_cit[top_cited] > 0:
                findings.append(
                    f"Most-cited lab received {int(final_cit[top_cited])} "
                    f"cumulative citations (community consensus leader)"
                )

            culled = int(np.sum(final_cit == 0))
            if culled > 0:
                findings.append(
                    f"{culled} lab(s) received 0 citations and were replaced "
                    f"(natural selection of research directions)"
                )

        return findings
