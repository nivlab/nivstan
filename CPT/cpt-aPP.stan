// Cumulative Prospect Theory - Partial Pooling - Version A
// 
// Parameters
// -- alpha:    risk aversion
// -- gamma:    probability curvature
// -- delta_p:  probability elevation (gains)
// -- delta_n:  probability elevation (losses)
// -- theta:    choice sensitivity
// 
// Notes: 
// -- Decision weighting function is linear-in-log-odds (GE-87).
// -- Implemented fix from Stewart et al. (2018) [A bug fix for stochastic prospect theory].
// -- Assumes prospects normalized in range [0, 100].

functions {

    // Vectorized power function
    vector fpow(vector x, real y) {
        vector[num_elements(x)] z;
        for (i in 1:num_elements(x)) { z[i] = pow(x[i], y); }
        return z;
    }

    // CPT value function (gains)
    vector risk_sensitivity_pos(vector x, real alpha) {         
        return fpow(x, alpha);
    }
    
    // CPT value function (losses)
    vector risk_sensitivity_neg(vector x, real alpha, real lambda) {         
        return -lambda * fpow(fabs(x), alpha);
    }
    
    // CPT decision weights function
    vector probability_weights(vector p, real gamma, real delta) {
        vector[num_elements(p)] a = delta * fpow(p, gamma);
        vector[num_elements(p)] b = fpow(1-p, gamma);
        return a ./ (a + b + machine_precision());
    }

}
data {

    // Metadata
    int  N;                         // Number of subiects (gains)
    int  M;                         // Number of subiects (losses)
    int  pos_ix[N];                 // Number of completed trials (gains)
    int  neg_ix[M];                 // Number of completed trials (losses)
    int  sub_ix[M];                 // Subiect mapping between gains/losses
     
    // Data
    int       Yp[N,max(pos_ix)];    // Choices (gains)
    vector[max(pos_ix)] Xp[N,4];    // Gamble values (gains)
    vector[max(pos_ix)] Pp[N,2];    // Gamble probabilities (gains)
    
    int       Yn[M,max(neg_ix)];    // Choices (losses)
    vector[max(neg_ix)] Xn[M,4];    // Gamble values (losses)
    vector[max(neg_ix)] Pn[M,2];    // Gamble probabilities (losses)

}
parameters {

    // Group-level
    vector[5] mu_pr;
    vector<lower=0>[5] sigma;
    
    // Subiect-level (pre-transform)
    vector[N] alpha_pr;
    vector[N] gamma_pr;
    vector[N] delta_p_pr;
    vector[M] delta_n_pr;
    vector[N] theta_pr;

}
transformed parameters {

    // Subiect-level (transformed)
    vector<lower=0,upper=2>[N] alpha;
    vector<lower=0,upper=2>[N] gamma;
    vector<lower=0,upper=2>[N] delta_p;
    vector<lower=0,upper=2>[M] delta_n;
    vector<lower=0,upper=1>[N] theta;

    alpha   = Phi_approx( mu_pr[1] + sigma[1] * alpha_pr ) * 2;
    gamma   = Phi_approx( mu_pr[2] + sigma[2] * gamma_pr ) * 2;
    delta_p = Phi_approx( mu_pr[3] + sigma[3] * delta_p_pr ) * 2;
    delta_n = Phi_approx( mu_pr[4] + sigma[4] * delta_n_pr ) * 2;
    theta   = Phi_approx( mu_pr[5] + sigma[5] * theta_pr );

}
model { 

    // Group-level priors
    mu_pr ~ normal(0, 0.5);
    sigma ~ gamma(1, 0.5);

    // Subiect-level priors
    alpha_pr ~ normal(0, 0.5);
    gamma_pr ~ normal(0, 0.5);
    delta_p_pr ~ normal(0, 0.5);
    delta_n_pr ~ normal(0, 0.5);
    theta_pr ~ normal(0, 0.5);
    
    // Likelihood (gains)
    for (i in 1:N) {
   
        // Generated quantities
        int k = pos_ix[i];
   
        // Gamble 1: value / probability
        vector[k] u1 = risk_sensitivity_pos( Xp[i,1,:k], alpha[i] );
        vector[k] u2 = risk_sensitivity_pos( Xp[i,2,:k], alpha[i] );
        vector[k] w1 = probability_weights( Pp[i,1,:k], gamma[i], delta_p[i] );
        
        // Gamble 2: value / probability
        vector[k] u3 = risk_sensitivity_pos( Xp[i,3,:k], alpha[i] );
        vector[k] u4 = risk_sensitivity_pos( Xp[i,4,:k], alpha[i] );
        vector[k] w2 = probability_weights( Pp[i,2,:k], gamma[i], delta_p[i] );
        
        // Compute difference in expected value
        vector[k] dEV;
        dEV = fpow(u1 .* w1 + u2 .* (1-w1), 1/alpha[i]) - 
              fpow(u3 .* w2 + u4 .* (1-w2), 1/alpha[i]);
        
        // Likelihood
        Yp[i,:k] ~ bernoulli_logit( theta[i] * dEV );
        
    }
    
    // Likelihood (losses)
    for (i in 1:M) {
   
        // Generated quantities
        int k = neg_ix[i];
   
        // Gamble 1: value / probability
        vector[k] u1 = risk_sensitivity_neg( Xn[i,1,:k], alpha[sub_ix[i]], 1 );
        vector[k] u2 = risk_sensitivity_neg( Xn[i,2,:k], alpha[sub_ix[i]], 1 );
        vector[k] w1 = probability_weights( Pn[i,1,:k], gamma[i], delta_n[i] );
        
        // Gamble 2: value / probability
        vector[k] u3 = risk_sensitivity_neg( Xn[i,3,:k], alpha[sub_ix[i]], 1 );
        vector[k] u4 = risk_sensitivity_neg( Xn[i,4,:k], alpha[sub_ix[i]], 1 );
        vector[k] w2 = probability_weights( Pn[i,2,:k], gamma[i], delta_n[i] );
        
        // Compute difference in expected value
        vector[k] dEV;
        dEV = -fpow(fabs(u1 .* w1 + u2 .* (1-w1)), 1/alpha[sub_ix[i]]) + 
               fpow(fabs(u3 .* w2 + u4 .* (1-w2)), 1/alpha[sub_ix[i]]);
        
        // Likelihood
        Yn[i,:k] ~ bernoulli_logit( theta[sub_ix[i]] * dEV );
        
    }

}
generated quantities {

    vector[5] mu;

    mu[1] = Phi_approx( mu_pr[1] ) * 2;
    mu[2] = Phi_approx( mu_pr[2] ) * 2;
    mu[3] = Phi_approx( mu_pr[3] ) * 2;
    mu[4] = Phi_approx( mu_pr[4] ) * 2;
    mu[5] = Phi_approx( mu_pr[5] );

}