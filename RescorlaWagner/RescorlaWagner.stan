// Hierarchical Rescorla-Wagner model
// Assumes each participant must learn the appropriate SR contingencies
// for N stimuli across T trials (where T can vary per participant)

data {

    // Metadata
    int N;              // Number of subjects
    int T[N];           // Number of trials (per subject)

    // Data
    int X[N,max(T)];    // Stimulus per subject/trial
    int Y[N,max(T)];    // Response per subject/trial
    int R[N,max(T)];    // Outcome  per subject/trial

}
transformed data {

    int S = 4;          // Number of stimuli
    int A = 3;          // Number of actions

}
parameters {

    // Group-level parameters
    vector[2] mu_pr;
    vector<lower=0>[2] sigma;

    // Individual-level parameters (unscaled)
    vector[N] beta_pr;    // Inverse temperature
    vector[N] eta_pr;     // Learning rate

}
transformed parameters {

    // Individual-level parameters (scaled)
    vector<lower=0,upper=20>[N] beta;
    vector<lower=0,upper=1>[N] eta;

    for (i in 1:N) {
        beta[i] = Phi_approx(mu_pr[1] + sigma[1] * beta_pr[i]) * 20;
        eta[i] = Phi_approx(mu_pr[2] + sigma[2] * eta_u_pr[i]);
    }

}
model {

    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ normal(0, 1);

    // Individual-level priors
    beta_pr ~ normal(0, 1);
    eta_pr ~ normal(0, 1);

    // Main loop
    for (i in 1:N) {

        // Initialize Q-table per participant
        vector[A] Q[S];
        for (s in 1:S){ Q[s] = rep_vector(0, A); }

        for (j in 1:T[i]) {

            // Likelihood of choice
            Y[i,j] ~ categorical_logit( beta[i] * Q[X[i,j]] );

            // Compute RPE and update
            Q[X[i,j], Y[i,j]] += eta[i] * (R[i,j] - Q[X[i,j], Y[i,j]]);

        }

    }

}
generated quantities {

    vector[2] mu;              // Scaled group-level parameters
    real log_lik[N,max(T)];    // Log-likelihood of choice per subject/trial

    mu[1] = Phi_approx(mu_pr[1]) * 20;
    mu[2] = Phi_approx(mu_pr[2]);

    // Main loop
    for (i in 1:N) {

        // Initialize Q-table per participant
        vector[A] Q[S];
        for (s in 1:S){ Q[s] = rep_vector(0, A); }

        for (j in 1:T[i]) {

            // Likelihood of choice
            log_lik[i,j] = categorical_logit_lpmf( Y[i,j] | beta[i] * Q[X[i,j]] );

            // Compute RPE and update
            Q[X[i,j], Y[i,j]] += eta[i] * (R[i,j] - Q[X[i,j], Y[i,j]]);

        }

    }

}
