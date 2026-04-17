"""Search space and presets for the Evolutionary Swarm demo."""

SEARCH_SPACE = {
    "n_tanh": [40, 60, 80, 100, 120, 160, 200],
    "kappa": [50, 100, 200, 500, 1000],
    "scale": [1.0, 2.0, 2.5, 3.0, 5.0],
    "lam": [1e-8, 1e-6, 1e-4],
    "max_steps": [0, 2, 4, 6, 8],
    "parsimony": [0.0, 0.001, 0.002, 0.005],
    "ga_pop_size": [15, 25, 40],
    "ga_n_gen": [15, 25, 40],
    "mutation_rate": [0.1, 0.3, 0.5],
    "elite_count": [2, 3, 5],
    "seed_tanh": [7, 42, 123, 456, 789],
}

# Descriptive labels for display
PARAM_LABELS = {
    "n_tanh": "Tanh neurons",
    "kappa": "Step steepness",
    "scale": "Weight scale",
    "lam": "Regularization",
    "max_steps": "Max step neurons",
    "parsimony": "Parsimony penalty",
    "ga_pop_size": "Inner GA pop",
    "ga_n_gen": "Inner GA gens",
    "mutation_rate": "Mutation rate",
    "elite_count": "Elite count",
    "seed_tanh": "ELM seed",
}

# Presets for different demo durations
PRESETS = {
    "Cloud (~15s)": {
        "n_labs": 4,
        "n_meta_gen": 2,
        "meta_elite": 1,
        "meta_mutation_rate": 0.4,
        "override": {"ga_pop_size": [8, 10], "ga_n_gen": [8, 10],
                      "n_tanh": [40, 60, 80]},
    },
    "Quick (~30s)": {
        "n_labs": 6,
        "n_meta_gen": 3,
        "meta_elite": 2,
        "meta_mutation_rate": 0.4,
        "override": {"ga_pop_size": [10, 15], "ga_n_gen": [10, 15]},
    },
    "Standard (~90s)": {
        "n_labs": 8,
        "n_meta_gen": 4,
        "meta_elite": 2,
        "meta_mutation_rate": 0.4,
        "override": {},
    },
    "Full (~3min)": {
        "n_labs": 12,
        "n_meta_gen": 5,
        "meta_elite": 3,
        "meta_mutation_rate": 0.3,
        "override": {"ga_pop_size": [25, 40], "ga_n_gen": [25, 40]},
    },
}


def get_search_space(preset_name="Standard (~90s)"):
    """Return search space with optional preset overrides applied."""
    preset = PRESETS[preset_name]
    space = dict(SEARCH_SPACE)
    for key, values in preset.get("override", {}).items():
        space[key] = values
    return space, preset
