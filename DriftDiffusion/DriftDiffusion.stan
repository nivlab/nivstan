// Hierarchical drift-diffusion model

data {

    // Metadata
    int   N;            // Number of subjects
    int   T[N];         // Number of trials

    // Observed data
    vector[max(T)]    Z[N];  // Reaction times (seconds)
    vector[max(T)]    Y[N];  // Choice (accept = 1, reject = 0)
    vector[N]         minZ;  // Fastest reaction time (per subj)

}
parameters {

    // Group-level parameters
    vector[4] mu_pr;
    vector<lower=0>[4] sigma;

    // Subject-level parameters
    vector[N] alpha_pr;      // Decision bounds
    vector[N]  beta_pr;      // Initial bias
    vector[N]   tau_pr;      // Non-decision time
    vector[N] delta_pr;      // Drift rate

}
transformed parameters{

    vector[N] alpha;
    vector[N]  beta;
    vector[N]   tau;
    vector[N] delta;

    for (i in 1:N) {
        alpha[i] = Phi_approx( mu_pr[1] + sigma[1] * alpha_pr[i] ) * 4 + 1;
        beta[i]  = Phi_approx( mu_pr[2] + sigma[2] * beta_pr[i] );
        tau[i]   = Phi_approx( mu_pr[3] + sigma[3] * tau_pr[i] ) * minZ[i];
        delta[i] = tanh( mu_pr[4] + sigma[4] * delta_pr[i] );
    }

}
model {

    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ normal(0, 0.5);

    // Subject-level priors
    for (i in 1:N) {
        alpha_pr[i] ~ normal(0, 1);
        beta_pr[i] ~ normal(0, 1);
        tau_pr[i]   ~ normal(0, 1);
        delta_pr[i] ~ normal(0, 1);
    }

    // Likelihood
    for (i in 1:N) {

        // Generated quantities
        vector[T[i]] beta_vec;
        vector[T[i]] delta_vec;

        for (j in 1:T[i]) {
            if ( Y[i,j] > 0 ) {
                beta_vec[j] = beta[i];
                delta_vec[j] = delta[i];
            } else {
                beta_vec[j] = 1 - beta[i];
                delta_vec[j] = -delta[i];
            }
        }

        // Wiener first passage times
        Z[i, :T[i]] ~ wiener(rep_vector(alpha[i], T[i]),
                             rep_vector(tau[i], T[i]),
                             beta_vec, delta_vec);

    }

}
