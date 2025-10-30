## QUICK REFERENCE: What's Where

### 🎯 em_algorithm.cpp (FOCUS HERE FOR OPTIMIZATION)
```
LINES: ~150 (vs 1400+ in original)
CONTENT: Pure algorithm logic

Key Function: run_em_algorithm()
  ├─ Preprocessing
  │   ├─ topological_order()        [in learn_bn.cpp]
  │   ├─ build_markov_blankets()    [in learn_bn.cpp]
  │   └─ build_buckets()            [in learn_bn.cpp]
  │
  ├─ For each alpha (0.1 to 1.0)
  │   ├─ initialize_from_complete_rows()  [in learn_bn.cpp]
  │   │
  │   └─ EM Loop (max 10 iterations)
  │       ├─ E-Step: soft_e_step_bucketed()  [in learn_bn.cpp]
  │       ├─ M-Step: m_step()                [in learn_bn.cpp]
  │       ├─ Compute: compute_loglik()       [in learn_bn.cpp]
  │       └─ Check convergence
  │
  └─ Return best alpha & parameters
```

**What to optimize:**
- Alpha sweep strategy (currently 0.1-1.0 in steps of 0.1)
- Convergence threshold (currently 1e-5)
- Max iterations (currently 10)
- Whether to use hard vs soft E-step
- Initialization strategy

---

### 🔧 learn_bn.cpp (SUPPORT FUNCTIONS)
```
LINES: ~1400
CONTENT: Everything else

Data Structures:
  - Variable: BN node info (name, states, parents, children)
  - BN: Complete network structure
  - Data: Training data matrix
  - Buckets: Organized data for efficient E-step

Core Functions Used by Algorithm:
  - soft_e_step_bucketed()     → Infer missing values (soft)
  - hard_e_step_bucketed()     → Infer missing values (MAP)
  - m_step()                   → Parameter update
  - compute_loglik()           → Likelihood calculation
  - initialize_from_complete_rows() → Initialization

Infrastructure (ignore for optimization):
  - parse_bif()                → Read .bif files
  - load_data()                → Read .dat files  
  - write_bif()                → Write output
  - Tokenizer, ArgParser       → Parsing utilities
  - family_index()             → CPT indexing
```

---

### 📊 main() Function (learn_bn.cpp)
```cpp
Now simplified to:
  1. Parse arguments
  2. Load BN and data
  3. Call run_em_algorithm()     ← THE ALGORITHM (see em_algorithm.cpp)
  4. Check CPTs (optional)
  5. Write output
  6. Print summary
```

---

## Optimization Checklist

When analyzing the algorithm, focus on:

- [ ] **em_algorithm.cpp**: Overall structure, alpha sweep, convergence
- [ ] **soft_e_step_bucketed()**: How missing values are inferred
- [ ] **m_step()**: How parameters are updated
- [ ] **compute_loglik()**: Likelihood calculation (expensive!)
- [ ] Data structures: BN, Variable, Buckets (how data is organized)

Ignore (unless debugging I/O issues):
- [ ] BIF parsing
- [ ] Data loading
- [ ] Tokenization
- [ ] File writing
