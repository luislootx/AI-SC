"""Virtual Lab Agent: wraps Step-CINN as an independent experiment runner.

Each LabAgent receives a configuration dict and a PDE problem, then runs a
complete inner GA to discover optimal step neuron positions. The result
includes fitness, error metrics, learned positions, and per-generation history
(for real-time visualization).
"""

import numpy as np
from core import (
    generate_tanh_weights,
    build_feature_matrix,
    tanh_features,
    solve_ridge,
    l2_relative,
)


class LabAgent:
    """A virtual laboratory that runs one Step-CINN experiment."""

    def __init__(self, lab_id, config, problem):
        self.lab_id = lab_id
        self.config = config
        self.problem = problem
        self.status = "idle"
        self.result = None
        self.history = []

    def _evaluate(self, positions, x_train, y_train, x_val, y_val, W, b):
        """Evaluate one individual: build features, solve ridge, compute error."""
        cfg = self.config
        H = build_feature_matrix(x_train, W, b, positions, cfg["kappa"])
        beta = solve_ridge(H, y_train, cfg["lam"])

        # Validation prediction
        H_val = build_feature_matrix(x_val, W, b, positions, cfg["kappa"])
        y_pred = H_val @ beta
        error = l2_relative(y_val, y_pred)
        fitness = error + cfg["parsimony"] * len(positions)
        return fitness, error, beta, y_pred

    def _init_individual(self, rng):
        """Create a random individual (list of step positions)."""
        n = rng.integers(1, self.config["max_steps"] + 1)
        margin = 0.05 * (self.problem.x_max - self.problem.x_min)
        positions = sorted(
            rng.uniform(
                self.problem.x_min + margin,
                self.problem.x_max - margin,
                size=n,
            ).tolist()
        )
        return positions

    def _tournament_select(self, population, rng, k=3):
        """Tournament selection: pick k random, return best."""
        idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
        best = min(idx, key=lambda i: population[i]["fitness"])
        return population[best]

    def _crossover(self, p1, p2, rng):
        """Combine step positions from two parents."""
        all_pos = sorted(set(p1["positions"] + p2["positions"]))
        if not all_pos:
            return self._init_individual(rng)
        n = rng.integers(1, min(len(all_pos), self.config["max_steps"]) + 1)
        return sorted(rng.choice(all_pos, size=n, replace=False).tolist())

    def _mutate(self, positions, rng):
        """Mutate step positions: shift, add, or remove."""
        cfg = self.config
        domain_width = self.problem.x_max - self.problem.x_min
        margin = 0.02 * domain_width
        pos = list(positions)

        for i in range(len(pos)):
            if rng.random() < cfg["mutation_rate"]:
                pos[i] += rng.normal(0, 0.1 * domain_width)
                pos[i] = np.clip(pos[i], self.problem.x_min + margin,
                                 self.problem.x_max - margin)

        if rng.random() < 0.12 and len(pos) < cfg["max_steps"]:
            pos.append(
                float(rng.uniform(self.problem.x_min + margin,
                                  self.problem.x_max - margin))
            )

        if rng.random() < 0.10 and len(pos) > 1:
            pos.pop(rng.integers(len(pos)))

        return sorted(pos)

    def run(self, callback=None):
        """Run the full experiment. Returns result dict.

        Args:
            callback: optional callable(lab_agent, gen_idx) invoked after each
                      GA generation, for real-time UI updates.
        """
        self.status = "running"
        cfg = self.config
        prob = self.problem

        # Data
        x_train, y_train = prob.training_data(n_points=800)
        x_val_raw, y_val_raw = prob.validation_data(n_points=1200)

        # For PDE mode, apply characteristic shift on validation
        if prob.mode == "pde" and hasattr(prob, "characteristic_shift"):
            xi_val = prob.characteristic_shift(x_val_raw, prob.t_final)
            x_val = xi_val
            y_val = y_val_raw
        else:
            x_val = x_val_raw
            y_val = y_val_raw

        # Fixed tanh weights
        W, b = generate_tanh_weights(cfg["n_tanh"], seed=cfg["seed_tanh"],
                                     scale=cfg["scale"])

        # Pure tanh baseline (no step neurons)
        if cfg["max_steps"] == 0:
            H = tanh_features(x_train, W, b)
            beta = solve_ridge(H, y_train, cfg["lam"])
            H_val = tanh_features(x_val, W, b)
            y_pred = H_val @ beta
            error = l2_relative(y_val, y_pred)

            self.result = {
                "fitness": error,
                "l2_error": error,
                "positions": [],
                "beta": beta,
                "y_pred_val": y_pred,
                "n_steps": 0,
                "config": cfg,
            }
            self.history = [{"gen": 0, "best_fitness": error,
                             "mean_fitness": error, "best_n_steps": 0}]
            self.status = "done"
            if callback:
                callback(self, 0)
            return self.result

        # --- Inner GA ---
        rng = np.random.default_rng(cfg.get("seed_ga", 42))
        pop_size = cfg["ga_pop_size"]
        n_gen = cfg["ga_n_gen"]
        elite_count = cfg["elite_count"]

        population = [
            {"positions": self._init_individual(rng), "fitness": np.inf}
            for _ in range(pop_size)
        ]

        best_ever = None

        for gen in range(n_gen):
            # Evaluate all individuals
            for ind in population:
                fit, err, beta, y_pred = self._evaluate(
                    ind["positions"], x_train, y_train, x_val, y_val, W, b
                )
                ind["fitness"] = fit
                ind["l2_error"] = err
                ind["beta"] = beta
                ind["y_pred"] = y_pred

            population.sort(key=lambda x: x["fitness"])

            # Track best
            top = population[0]
            if best_ever is None or top["fitness"] < best_ever["fitness"]:
                best_ever = {
                    "positions": list(top["positions"]),
                    "fitness": top["fitness"],
                    "l2_error": top["l2_error"],
                    "beta": top["beta"].copy(),
                    "y_pred": top["y_pred"].copy(),
                }

            self.history.append({
                "gen": gen,
                "best_fitness": top["fitness"],
                "mean_fitness": float(np.mean([ind["fitness"] for ind in population])),
                "best_n_steps": len(top["positions"]),
            })

            if callback:
                # Provide intermediate result for visualization
                self.result = {
                    "fitness": best_ever["fitness"],
                    "l2_error": best_ever["l2_error"],
                    "positions": best_ever["positions"],
                    "y_pred_val": best_ever["y_pred"],
                    "n_steps": len(best_ever["positions"]),
                    "config": cfg,
                }
                callback(self, gen)

            # --- Selection + reproduction ---
            elite = [
                {"positions": list(ind["positions"]), "fitness": np.inf}
                for ind in population[:elite_count]
            ]
            new_pop = list(elite)

            while len(new_pop) < pop_size:
                p1 = self._tournament_select(population, rng)
                p2 = self._tournament_select(population, rng)
                child_pos = self._crossover(p1, p2, rng)
                child_pos = self._mutate(child_pos, rng)
                new_pop.append({"positions": child_pos, "fitness": np.inf})

            population = new_pop

        # Final result
        self.result = {
            "fitness": best_ever["fitness"],
            "l2_error": best_ever["l2_error"],
            "positions": best_ever["positions"],
            "beta": best_ever["beta"],
            "y_pred_val": best_ever["y_pred"],
            "n_steps": len(best_ever["positions"]),
            "config": cfg,
        }
        self.status = "done"
        return self.result

    def summary(self):
        """One-line summary for logging."""
        if self.result is None:
            return f"Lab-{self.lab_id}: not run"
        r = self.result
        steps_str = f"{r['n_steps']} steps" if r["n_steps"] > 0 else "pure tanh"
        return (
            f"Lab-{self.lab_id} | L2={r['l2_error']:.2e} | "
            f"{steps_str} | n_tanh={self.config['n_tanh']} "
            f"kappa={self.config['kappa']}"
        )
