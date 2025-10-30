/*
 * ALGORITHM STRUCTURE:
 * ====================
 * 
 * 1. PREPROCESSING (one-time setup)
 *    - Build topological order of variables (for efficient inference)
 *    - Compute Markov blankets (minimal set of variables affecting each variable)
 *    - Create data buckets (organize rows by missing pattern for optimized processing)
 * 
 * 2. ALPHA SWEEP (hyperparameter search)
 *    - Try Dirichlet smoothing parameters: 0.1, 0.2, ..., 1.0
 *    - For each alpha, run full EM to convergence
 *    - Select alpha with best log-likelihood
 * 
 * 3. EM ITERATION (for each alpha)
 *    a. INITIALIZATION
 *       - Use complete data rows to estimate initial parameters
 *       - Fallback to uniform for parameters without complete data
 *    
 *    b. E-STEP (Expectation)
 *       - For rows with missing values, infer soft assignments
 *       - Compute expected sufficient statistics (weighted counts)
 *       - Use Markov blanket for efficient local inference
 *    
 *    c. M-STEP (Maximization)
 *       - Update CPT parameters using expected counts
 *       - Apply Dirichlet smoothing (controlled by alpha)
 *    
 *    d. CONVERGENCE CHECK
 *       - Compute log-likelihood
 *       - Stop if relative improvement < threshold
 * 
 * 4. OUTPUT
 *    - Return parameters from best alpha
 */

// No includes needed - this is #included into learn_bn.cpp

/**
 * Core EM Algorithm for Bayesian Network Parameter Learning
 * 
 * @param bn The Bayesian Network structure (variables, parents, CPT layout)
 * @param data Training data with possible missing values (-1 indicates missing)
 * @param max_iterations Maximum EM iterations per alpha value
 * @param convergence_threshold Relative log-likelihood improvement threshold for early stopping
 * @param best_theta [out] Learned CPT parameters for best alpha
 * @param best_alpha [out] Best Dirichlet smoothing parameter found
 * @param best_iters_run [out] Number of iterations run for best alpha
 * @return Best log-likelihood achieved across all alphas
 */

/**
 * Core EM Algorithm for Bayesian Network Parameter Learning
 * 
 * Algorithm Overview:
 * 1. Precompute topological order and Markov blankets for efficient inference
 * 2. Build bucketed data structure for optimized E-step processing
 * 3. Alpha sweep: Try Dirichlet smoothing parameters from 0.1 to 1.0
 * 4. For each alpha:
 *    - Initialize CPTs from complete data rows (with fallback to uniform)
 *    - Iterate EM steps (E-step: soft inference, M-step: parameter update)
 *    - Track log-likelihood and convergence
 * 5. Select best alpha by maximum log-likelihood
 * 
 * @param bn The Bayesian Network structure
 * @param data The training data with possible missing values
 * @param max_iterations Maximum EM iterations per alpha value
 * @param convergence_threshold Relative log-likelihood improvement threshold
 * @param best_theta [out] Learned CPT parameters for best alpha
 * @param best_alpha [out] Best Dirichlet smoothing parameter found
 * @param best_iters_run [out] Number of iterations run for best alpha
 * @return Best log-likelihood achieved
 */
double run_em_algorithm(
    const BN& bn,
    const Data& data,
    int max_iterations,
    double convergence_threshold,
    std::vector<double>& best_theta,
    double& best_alpha,
    int& best_iters_run
) {
    // ========================================================================
    // PREPROCESSING: Build data structures for efficient EM
    // ========================================================================
    
    // Topological ordering helps process variables in dependency order
    auto topo = topological_order(bn);
    
    // Markov blanket: for each variable, the minimal set affecting its inference
    // (parents + children + co-parents of children)
    auto mb = build_markov_blankets(bn);
    
    // Bucket rows by first missing variable, sorted by Markov blanket richness
    // This optimization improves E-step efficiency
    auto buckets = build_buckets(bn, data, mb);

    // ========================================================================
    // ALPHA SWEEP: Try different Dirichlet smoothing parameters
    // ========================================================================
    
    // Alpha controls the strength of the Dirichlet prior for smoothing
    // Smaller alpha = less smoothing, larger alpha = more uniform smoothing
    std::vector<double> alpha_sweep;
    for (int i = 1; i <= 10; ++i) {
        alpha_sweep.push_back(static_cast<double>(i) / 10.0);
    }

    double best_loglik = -std::numeric_limits<double>::infinity();
    best_alpha = -1.0;
    best_iters_run = 0;

    // Try each alpha value
    for (double this_alpha : alpha_sweep) {
        
        // ====================================================================
        // INITIALIZATION: Start with parameters from complete data
        // ====================================================================
        
        std::vector<double> theta(bn.total_cpt_entries, 0.0);
        
        // Initialize CPTs using only complete rows (no missing values)
        // Falls back to uniform distribution where no complete data exists
        initialize_from_complete_rows(bn, data, this_alpha, 
                                     /*use_bdeu=*/false, /*ess=*/0.0, theta);

        // ====================================================================
        // EM ITERATION LOOP
        // ====================================================================
        
        std::vector<double> counts(bn.total_cpt_entries, 0.0);
        double prev_loglik = -std::numeric_limits<double>::infinity();
        double loglik = prev_loglik;
        int iterations_run = 0;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            
            // E-STEP: Compute expected sufficient statistics
            // For rows with missing values, compute soft assignments using
            // current parameters and accumulate fractional counts
            soft_e_step_bucketed(bn, data, theta, buckets, topo, counts, 
                               /*threads=*/0);
            
            // M-STEP: Update parameters to maximize expected complete log-likelihood
            // Uses counts from E-step with Dirichlet smoothing (alpha)
            m_step(bn, counts, this_alpha, 
                  /*use_bdeu=*/false, /*ess=*/0.0, theta);
            
            // Compute log-likelihood of data under current parameters
            loglik = compute_loglik(bn, data, theta, /*threads=*/0);
            
            ++iterations_run;
            
            // ================================================================
            // CONVERGENCE CHECK
            // ================================================================
            
            if (iter > 0) {
                // Check relative improvement in log-likelihood
                double denom = std::max(1.0, std::fabs(prev_loglik));
                double relative_improvement = (loglik - prev_loglik) / denom;
                
                if (relative_improvement < convergence_threshold) {
                    // Converged - stop early
                    break;
                }
            }
            
            prev_loglik = loglik;
        }
        
        // ====================================================================
        // TRACK BEST ALPHA
        // ====================================================================
        
        // Keep the alpha that gives best log-likelihood
        if (loglik > best_loglik) {
            best_loglik = loglik;
            best_theta = std::move(theta);
            best_alpha = this_alpha;
            best_iters_run = iterations_run;
        }
    }
    
    return best_loglik;
}

/**
 * Print algorithm summary statistics
 */
void print_algorithm_summary(
    size_t total_rows,
    size_t single_missing_rows, 
    int iterations_run,
    double best_alpha,
    double best_loglik
) {
    double single_missing_pct = total_rows > 0 
        ? 100.0 * static_cast<double>(single_missing_rows) / total_rows 
        : 0.0;
    
    std::ostringstream summary;
    summary << "rows=" << total_rows 
            << " single_missing=" << std::fixed << std::setprecision(2)
            << single_missing_pct << "% iters=" << iterations_run
            << " best_alpha=" << best_alpha
            << " loglik=" << std::setprecision(6) << best_loglik << "\n";
    
    std::cerr << summary.str();
}
