// Cumulative Prospect Theory - 4 parameter model
// 
// Hierarchical model fitting:
// -- alpha: risk sensitivity
// -- gamma: probability curvature
// -- delta: probability elevation
// -- theta: choice sensitivity
// 
// Notes: The probably weighting function is modeled according to the linear-in-log-odds (GE-87).
// We also transform the prospects as suggseted by Stewart et al. (2018) [A bug fix for stochastic 
// prospect theory] to remove the correlation between risk and choice sensitivity.

functions {

    // Vectorized power function
    vector fpow(vector x, real y) {
        vector[num_elements(x)] z;
        for (i in 1:num_elements(x)) { z[i] = pow(x[i], y); }
        return z;
    }

    // CPT diminishing marginal utility function 
    vector risk_sensitivity(vector x, real alpha) {         
        return fpow(x, alpha);
    }
    
    // CPT probability weights function
    vector probability_weights(vector p, real gamma, real delta) {
        vector[num_elements(p)] a = delta * fpow(p, gamma);
        vector[num_elements(p)] b = fpow(1-p, gamma);
        return a ./ (a + b + machine_precision());
    }

}
data {

    // Metadata
    int  N;              // Number of subjects
    int  T;              // Number of (total) trials
    int  ix [N];          // Number of completed trials per subject
    
    // Data
    int       Y[N,T];    // Choices
    vector[T] X[N,4];    // Gamble values
    vector[T] P[N,2];    // Gamble probabilities

}
transformed data {

    // Unpacking data (user convenience)
    vector[T] X1[N] = X[:,1,:];
    vector[T] X2[N] = X[:,2,:];
    vector[T] X3[N] = X[:,3,:];
    vector[T] X4[N] = X[:,4,:];
    vector[T] P1[N] = P[:,1,:];
    vector[T] P2[N] = P[:,2,:];  

}
parameters {

    // Group-level
    vector[4] mu_pr;
    vector<lower=0>[4] sigma;
    
    // Subject-level (pre-transform)
    vector[N] alpha_pr;
    vector[N] gamma_pr;
    vector[N] delta_pr;
    vector[N] theta_pr;

}
transformed parameters {

    // Subject-level (transformed)
    vector<lower=0,upper=2>[N] alpha;
    vector<lower=0,upper=2>[N] gamma;
    vector<lower=0,upper=2>[N] delta;
    vector<lower=-5,upper=5>[N] theta;

    alpha = Phi_approx( mu_pr[1] + sigma[1] * alpha_pr ) * 2;
    gamma = Phi_approx( mu_pr[2] + sigma[2] * gamma_pr ) * 2;
    delta = Phi_approx( mu_pr[3] + sigma[3] * delta_pr ) * 2;
    theta = tanh( mu_pr[4] + sigma[4] * theta_pr ) * 5;

}
model { 

    // Group-level priors
    mu_pr ~ normal(0, 0.5);
    sigma ~ gamma(1, 0.5);

    // Subject-level priors
    alpha_pr ~ normal(0, 0.5);
    gamma_pr ~ normal(0, 0.5);
    delta_pr ~ normal(0, 0.5);
    theta_pr ~ normal(0, 0.5);
    
    for (i in 1:N) {
   
        // Gamble 1: value / probability
        vector[ix[i]] u1 = risk_sensitivity( X1[i,:ix[i]], alpha[i] );
        vector[ix[i]] u2 = risk_sensitivity( X2[i,:ix[i]], alpha[i] );
        vector[ix[i]] w1 = probability_weights( P1[i,:ix[i]], gamma[i], delta[i] );
        
        // Gamble 2: value / probability
        vector[ix[i]] u3 = risk_sensitivity( X3[i,:ix[i]], alpha[i] );
        vector[ix[i]] u4 = risk_sensitivity( X4[i,:ix[i]], alpha[i] );
        vector[ix[i]] w2 = probability_weights( P2[i,:ix[i]], gamma[i], delta[i] );
        
        // Compute difference in expected value
        vector[T] dEV;
        dEV = fpow(u1 .* w1 + u2 .* (1-w1), 1/alpha[i]) - fpow(u3 .* w2 + u4 .* (1-w2), 1/alpha[i]);
        
        // Likelihood
        Y[i,:ix[i]] ~ bernoulli_logit( theta[i] * dEV );
        
    }

}
generated quantities {

    vector[4] mu;

    mu[1] = Phi_approx( mu_pr[1] ) * 2;
    mu[2] = Phi_approx( mu_pr[2] ) * 2;
    mu[3] = Phi_approx( mu_pr[3] ) * 2;
    mu[4] = tanh( mu_pr[4] ) * 5;

}