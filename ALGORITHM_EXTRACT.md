# EM Algorithm Extraction

## Overview
The core EM (Expectation-Maximization) algorithm has been extracted from `learn_bn.cpp` into `em_algorithm.cpp` for easier analysis and optimization.

## File Structure

### `em_algorithm.cpp` - **THE CORE ALGORITHM**
**This is what you want to focus on for optimization.**

Contains:
- `run_em_algorithm()` - The main EM learning algorithm
- `print_algorithm_summary()` - Result formatting

**Algorithm Flow:**
1. **Preprocessing** (one-time setup)
   - Build topological order for efficient variable traversal
   - Compute Markov blankets (minimal context needed for each variable)
   - Create data buckets (organize rows by missing patterns)

2. **Alpha Sweep** (hyperparameter search)
   - Try smoothing parameters: 0.1, 0.2, ..., 1.0
   - Run full EM for each alpha
   - Track best by log-likelihood

3. **EM Iteration** (per alpha)
   - **Initialize**: Use complete data rows
   - **E-Step**: Infer missing values, compute expected counts
   - **M-Step**: Update parameters using counts + smoothing
   - **Check**: Convergence by log-likelihood improvement

### `learn_bn.cpp` - **INFRASTRUCTURE CODE**
Contains everything else (you can mostly ignore when optimizing):
- BIF file parsing (Bayesian network structure)
- Data loading and CSV parsing
- Tokenizer and parser utilities
- CPT index computation helpers
- E-step and M-step implementation (called by algorithm)
- File I/O and output formatting

## Key Algorithm Components

### 1. E-Step Variants
- `soft_e_step_bucketed()` - Uses probabilistic inference for missing values
- `hard_e_step_bucketed()` - Uses MAP (maximum) assignment for missing values
- **Currently using**: Soft E-step (more accurate)

### 2. M-Step
- `m_step()` - Updates parameters using expected counts
- Applies Dirichlet smoothing controlled by alpha
- Formula: θ[i] = (count[i] + α) / (total_count + α * num_states)

### 3. Optimization Strategies Already Implemented
- **Bucketing**: Rows grouped by first missing variable
- **Richness sorting**: Process most informative rows first
- **Markov blanket**: Only use relevant variables for inference
- **OpenMP**: Parallel processing of E-step
- **Early stopping**: Convergence based on log-likelihood

## When Optimizing

**Read `em_algorithm.cpp` for**:
- Overall algorithm structure
- Parameter sweep strategy
- Convergence criteria
- What gets optimized (alpha selection)

**Read `learn_bn.cpp` for**:
- E-step implementation details (how missing values are handled)
- M-step implementation (how parameters are updated)
- Likelihood computation
- Data structures (Variable, BN, Data)

## Compilation Note
Both files are compiled together - `em_algorithm.cpp` is not standalone.
It calls functions defined in `learn_bn.cpp`.

## Usage
No change to usage - just run `learn_bn` as before:
```bash
./learn_bn input.bif records.dat output.bif
```

The separation is purely for code readability during optimization work.
