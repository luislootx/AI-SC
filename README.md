# AI Science Community for Automated CINN Discovery

## What is this?

This is a working implementation of the AI Science Community framework (Braga-Neto, arXiv:2603.21344) applied to automated discovery of optimal neural network architectures for PDEs with discontinuities.

The system coordinates a swarm of virtual laboratories, where each lab independently runs a Step-CINN experiment and the community collectively discovers which architecture works best through citation-based influence and natural selection.

## The Problem

When solving PDEs with discontinuities (shocks, contact surfaces), standard neural networks with tanh activations fail because smooth basis functions cannot represent sharp jumps. Step-CINN solves this by adding step neurons (sigmoid functions at learnable positions) that capture discontinuities exactly. But the question remains: **what is the best configuration?** How many tanh neurons? How many step neurons? What steepness? What regularization?

This demo answers that question automatically.

## Algorithm: Two Levels of Optimization

The system has two nested loops:

```
OUTER LOOP: Swarm of Virtual Labs (coordinated by PSO or GA)
  Evolves LAB CONFIGURATIONS (n_tanh, kappa, max_steps, ...)
  
  INNER LOOP: Each Lab (runs a complete Step-CINN experiment)
    Evolves STEP NEURON POSITIONS ([x1, x2, ..., xK])
```

### Outer Loop: The AI Science Community

The swarm manages N labs (default 8), each with a different configuration. After all labs run, the community evaluates results, exchanges citations, and evolves toward better configurations.

```
Initialize N labs with diverse configs
  (at least one pure-tanh baseline, one step-heavy)

For each iteration:
  1. Each lab runs its experiment (inner loop) -> fitness
  2. Citation: each lab votes for the 2 best peers
     - 1st place gets 2 points, 2nd gets 1 point
     - Citations accumulate across iterations
  3. Lab dynamics:
     - Cited labs get more compute budget (larger inner GA)
     - Labs with 0 cumulative citations get replaced (culled)
  4. Coordination strategy evolves configs:
     PSO: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
     GA:  tournament selection -> crossover -> mutation
     (both use citation-weighted fitness for selection)
  5. Elite labs keep their config (successful labs persist)
```

### Inner Loop: What Each Lab Does

Each lab receives a configuration and runs a complete Step-CINN experiment:

```
Input: config = {n_tanh=100, kappa=500, max_steps=6, ga_pop=25, ga_gen=25, ...}

1. Generate training data (PDE initial condition, 800 points)
2. Create fixed tanh weights (ELM-style, one random draw)
3. If max_steps = 0: solve ridge regression with tanh only -> done
4. If max_steps > 0: run inner GA to find step positions:

   Initialize population of 25 individuals:
     Individual 1: steps at [0.3, 0.7, 1.2]      (3 steps)
     Individual 2: steps at [0.5]                  (1 step)
     Individual 3: steps at [0.2, 0.9, 1.1, 1.5]  (4 steps)
     ...variable length, 1 to max_steps

   For each generation (25 total):
     For each individual:
       a. Build feature matrix: H = [tanh(Wx+b) | sigmoid(kappa*(x-pos))]
       b. Solve output weights analytically: beta = (H'H + lam*I)^{-1} H'y
       c. Compute validation error: L2 = ||y_exact - H_val @ beta|| / ||y_exact||
       d. Fitness = L2 + parsimony * num_steps
     
     Sort by fitness, keep elite
     Tournament selection -> crossover positions -> mutate (shift/add/remove)

5. Return best individual: fitness, positions, predictions
```

**Key insight**: There is no gradient descent anywhere. The tanh weights are fixed random (ELM), the step positions are optimized by GA, and the output weights are solved analytically via ridge regression. This makes each evaluation ~1ms, enabling the entire swarm to run in seconds.

### For PDE Problems: Characteristic Transform

Step-CINN is **not a PINN**. Instead of minimizing a PDE residual loss, it uses the method of characteristics:

- **Linear advection** (u_t + v*u_x = 0): The exact solution is u(x,t) = u_0(x - vt). The network learns the initial condition u_0, and the solution at any time t is obtained by evaluating the network at the shifted coordinate x - vt. The PDE is satisfied by construction.

- **Burgers equation** (u_t + u*u_x = 0): The shock speed is given by the Rankine-Hugoniot condition. Step neurons at the shock position capture the discontinuity exactly.

The step neurons are interpretable: their positions directly correspond to the physical locations of discontinuities in the solution.

## Concrete Example

```
ITERATION 1 (8 labs):
  Lab 1.1: n_tanh=60,  max_steps=0             -> L2=6.54  (pure tanh, fails)
  Lab 1.2: n_tanh=120, max_steps=8, kappa=1000 -> L2=0.012 (step neurons, excellent)
  Lab 1.3: n_tanh=80,  max_steps=4, kappa=100  -> L2=0.18  (too few steps, low kappa)
  Lab 1.4: n_tanh=200, max_steps=2, kappa=50   -> L2=0.45  (weak steps)
  ...
  
  Citations: Lab 1.2 -> 14 citations (everyone votes for it)
             Lab 1.1 -> 0 citations (pure tanh failed)
  
  Lab 1.2 gets more compute budget (inner GA grows)
  Lab 1.1 gets minimum budget (will be culled if uncited again)

ITERATION 2:
  Configs evolved toward Lab 1.2's architecture (high kappa, many steps)
  Lab 1.1 replaced with random new config (natural selection)
  Best fitness improves: 0.012 -> 0.009

ITERATION 3-4:
  Community consensus: step neurons with kappa >= 500 dominate
  Discovery: "Step neurons improve accuracy by 323x over pure tanh"
```

## Demo Output

After all iterations complete, the dashboard displays:

- **Swarm View**: Grid of mini-plots showing each lab's solution vs. the exact solution, color-coded by fitness, with citation counts displayed per lab.

- **Best Solution**: Large plot of the best solution found across all iterations, alongside the inner GA convergence curve of the winning lab.

- **Fitness Evolution**: Chart tracking best and mean fitness across iterations, with individual lab results as scatter points.

- **Discoveries**: Automatically generated findings from the swarm's collective behavior, such as:
  - "Step neurons improve accuracy by 323x over pure tanh"
  - "Most-cited lab received 26 cumulative citations (community consensus leader)"
  - "1 lab(s) received 0 citations and were replaced (natural selection of research directions)"

- **Agent Log**: Chronological record of every lab's result and every iteration's citation dynamics, including which labs were selected as elite and which were culled.

- **Configuration Table**: Sortable table comparing all lab configurations tested across iterations, ranked by L2 error.

## Run Locally

```bash
git clone https://github.com/luislootx/AI-SC.git
cd AI-SC
pip install -r requirements.txt
python -m streamlit run app.py
```

## Try Online

https://ai-sciencecommunity.streamlit.app/

Use the "Cloud" preset for fast execution. For deeper experiments, run locally with "Standard" or "Full" presets.

## References

- Braga-Neto, U. "The AI Scientific Community: Agentic Virtual Lab Swarms." arXiv:2603.21344, 2026.
- Lu, C. et al. "Towards end-to-end automation of AI research." Nature 651, 914-919, 2026.
- Toscano, J.D., Chen, D.T., Karniadakis, G.E. "Athena: Agentic team for hierarchical evolutionary numerical algorithms." arXiv:2512.03476, 2025.

## Authors

Luis Loo & Ulisses Braga-Neto
Scientific Machine Learning Lab, Texas A&M University
