// Stan implementation of softmax regression analysis from Seymour & Daw (2012)
// 
// Seymour, B., Daw, N. D., Roiser, J. P., Dayan, P., & Dolan, R. (2012). Serotonin selectively 
// modulates reward value in human decision-making. Journal of Neuroscience, 32(17), 5833-5842.

functions {

    // Exponentiate scalar to first-h integers (0-indxed)
    vector fexp(real x, int h) {
        vector[h] arr;
        for (i in 1:h){ arr[i] = x^(i-1); }
        return arr;
    }

}
data {

    // Metadata
    int  N;                           // Number of subjects
    int  K;                           // Number of bandits
    int  T;                           // Number of trials
    int  H;                           // Length of history window
    
    // Data
    int          Y[N,T];              // Choices             [1,2,3,4]
    matrix[T,H]  R[N,K];              // Reward outcomes     [ 1,0]
    matrix[T,H]  P[N,K];              // Punishment outcomes [-1,0]
    matrix[T,H]  C[N,K];              // Previous choices    [ 1,0]

}
parameters {

    // Group-level correlation matrix of scaling constants
    cholesky_factor_corr[3]  R_cholesky; 

    // Group-level parameter means
    vector[3]  mu_omega_pr;
    vector[3]  mu_alpha_pr;
    
    // Group-level parameter SDs
    vector<lower=0>[3]  sigma_omega;
    vector<lower=0>[3]  sigma_alpha;
    
    // Individual-level parameters
    matrix[N,3]  omega_pr;
    matrix[N,3]  alpha_pr;

}
transformed parameters {

    vector[3]                    omega[N];      // Scaling constants
    vector<lower=0, upper=1>[H]  alpha[N,3];    // Decay weights
    
    { // Computation block
    
        // Precompute individual scaling offsets
        matrix[3,N] omega_tilde = diag_pre_multiply(sigma_omega, R_cholesky) * omega_pr';         

        for (n in 1:N) {
        
            // Precompute scaling constants
            omega[n] = mu_omega_pr + omega_tilde[:,n];               
        
            // Precompute decay weights
            alpha[n,1] = fexp( Phi_approx( mu_alpha_pr[1] + sigma_alpha[1] * alpha_pr[n,1] ), H );
            alpha[n,2] = fexp( Phi_approx( mu_alpha_pr[2] + sigma_alpha[2] * alpha_pr[n,2] ), H );
            alpha[n,3] = fexp( Phi_approx( mu_alpha_pr[3] + sigma_alpha[3] * alpha_pr[n,3] ), H );
            
        }
        
    }
    
}
model {

    // Correlation matrix prior
    R_cholesky ~ lkj_corr_cholesky(1);

    // Group-level mean priors
    mu_omega_pr ~ std_normal( );
    mu_alpha_pr ~ std_normal( );
    
    // Group-level SD priors
    sigma_omega ~ gamma(1, 0.5);
    sigma_alpha ~ gamma(1, 0.5);
    
    // Individual-level priors
    to_vector(omega_pr) ~ std_normal( );
    to_vector(alpha_pr) ~ std_normal( );
    
    // Likelihood
    for (n in 1:N) {
    
        // Generated quantities
        matrix[K,T] W;
        
        // Precompute weights        
        for (k in 1:K) {
            W[k,:] = (omega[n,1] * C[n,k] * alpha[n,1] + 
                      omega[n,2] * R[n,k] * alpha[n,2] + 
                      omega[n,3] * P[n,k] * alpha[n,3])';
        }
    
        // Compute likelihood of choice
        for (t in 1:T) {
            Y[n,t] ~ categorical_logit( W[:,t] );
        }
    
    }

}
generated quantities { 

    // Reconstruct correlation matrix
    corr_matrix[3]  R = R_cholesky * R_cholesky';
    
    // Compute log-likelihood
    vector[N]  log_lik = rep_array(0, N);
    
    for (n in 1:N) {
    
        // Generated quantities
        matrix[K,T] W;
        
        // Precompute weights        
        for (k in 1:K) {
            W[k,:] = (omega[n,1] * C[n,k] * alpha[n,1] + 
                      omega[n,2] * R[n,k] * alpha[n,2] + 
                      omega[n,3] * P[n,k] * alpha[n,3])';
        }
    
        // Compute likelihood of choice
        for (t in 1:T) {
            log_lik[N] += categorical_logit_lpmf( Y[n,t] | W[:,t] );
        }
    
    }
  
} 