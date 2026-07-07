# v2 aggregate (30/30 runs present)

Evidence: 15 LLM runs, 9623 LLM calls, **0 fallbacks** (0 = agents were LLMs throughout).

## Discovery (best rel L2 clean, mean over seeds)

| PDE | LLM | PSO | n(LLM/PSO) |
|---|---|---|---|
| Piecewise Reg. (1D) | 0.0417$\pm$0.0011 | 0.0405$\pm$0.0007 | 3/3 |
| Advection (1D) | 0.0159$\pm$0.0038 | 0.0183$\pm$0.0007 | 3/3 |
| Burgers (1D) | 0.0799$\pm$0.0067 | 0.0709$\pm$0.0047 | 3/3 |
| Navier-Stokes (2D) | 0.0009$\pm$0.0005 | 0.0007$\pm$0.0000 | 3/3 |
| Darcy (2D) | 0.0095$\pm$0.0003 | 0.0096$\pm$0.0015 | 3/3 |