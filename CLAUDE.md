# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Companion code for the manuscript "From genes to collective modes: biological constraints shape metabolic evolution." Simulates how metabolic networks evolve under selective pressure by mutating gene activity bounds using constraint-based metabolic modeling (FBA).

## Environment Setup

```bash
conda env create -f evcm_env.yml
conda activate evcm_env
```

Key dependencies: cvxpy, cobra (pip), numpy, pandas, polars, scipy, qpsolvers, scikit-learn, sympy.

## Workflow

The project is notebook-driven with no CLI entry points or test suite:

1. **Run simulation:** `toynet_run_simulation.ipynb` — runs evolutionary simulations on toy metabolic networks (original + 5-gene augmented version)
2. **Analyze results:** `toynet_simulation_analysis.ipynb` — generates Figure 2 from simulation output

Both notebooks must be run from the repo root with the `evcm_env` conda environment.

## Architecture

### `evcm/` — Main Python package (4 submodules)

- **`sim/sim.py`** — Core simulation engine. `run_sim()` is the main entry point: takes constraint matrices (Au, Al, S, Gu, Gl, beta), runs population-level evolution with mutation/fixation dynamics, returns 16 DataFrames tracking fluxes, bounds, biomass, mutations, and selective pressures over time.

- **`utils/utils.py`** — Utility functions for matrix I/O (`mat2file`/`np.savez`), random network generation, initialization strategies (fixed/random/noisy/biological start), flux optimization (FBA via cvxpy, nearest feasible solutions via qpsolvers), mutation mechanics, fixation probability, and selective pressure calculations.

- **`analysis/analysis.py`** — Post-simulation analysis (uses polars DataFrames). Includes FBA, collective mode finding (`find_cm()`), chain sampling for predictions, and theory-related calculations (Di matrices, dual variables).

- **`biggmatrices/biggmatrices.py`** — Converts COBRApy metabolic models into the matrix format used by the simulator. `cmsim_biggmatrices_double()` extracts Au, Al, S, Gu, Gl, beta matrices from COBRA models, handling gene-reaction rules (CNF conversion) and immutable bounds.

### Key mathematical objects

- **S** — Stoichiometric matrix (metabolites × reactions)
- **Au, Al** — Upper/lower constraint matrices mapping reactions to gene-controlled bounds
- **Gu, Gl** — Gene-to-bound mapping matrices
- **beta** — Biomass/fitness objective vector
- **Sigmau, Sigmal** — Mutation covariance matrices

### Data

- `networks/` — Pre-computed network files (.npz) for toy networks
- `toynet_simulation_data/` — Stored simulation results (5 independent runs), each containing CSVs and JSONs for bounds, fluxes, selective pressures, and predictions
