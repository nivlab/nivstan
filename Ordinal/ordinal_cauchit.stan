data {

    // Metadata
    int  N;                         // Number of subjects
    int  K;                         // Number of predictors
    int  U;                         // Number of response options
    
    // Data
    int<lower=1, upper=U>  Y[N];    // Responses
    row_vector[K]          X[N];    // Predictors
    
}
parameters {

    // Regression coefficients
    vector[K]  beta;
    
    // Thresholds
    vector[U-1] tau_pr;

}
transformed parameters {

    // Thresholds
    vector[U-1] tau;
    
    tau[1] = tau_pr[1];
    for (i in 2:U-1) {
        tau[i] = exp(tau_pr[i]);
    }    
    tau = cumulative_sum(tau);

}
model {

    // Coefficient priors
    beta ~ normal(0, 1);
    
    // Threshold priors
    tau_pr[1] ~ normal(-log(U-1), 1);
    for (i in 2:U-1) {
        tau_pr[i] ~ normal(0, 1);
    }    
    
    // Likelihood
    for (i in 1:N) {
    
        // Generated quantities
        real mu = X[i] * beta;      // Latent mean
        real p;                     // Response probability
        
        // Compute likelihood
        if ( Y[i] == 1 ) {
            p = cauchy_cdf( tau[1] - mu, 0, 1 );
        } else if ( Y[i] == U ) {
            p = 1 - cauchy_cdf( tau[U-1] - mu, 0, 1 );
        } else {
            p = cauchy_cdf( tau[Y[i]] - mu, 0, 1 ) - cauchy_cdf( tau[Y[i]-1] - mu, 0, 1 );
        }
    
        // Increment target
        target += log(p);
    
    }

}