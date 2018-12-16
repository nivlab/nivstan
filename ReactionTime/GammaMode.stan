// Gamma mode distribution

functions {

    // (Mode, SD)-paramterized Gamma distribution
    real gamma_mode_lpdf(vector z, vector mu, real sigma){
        vector[rows(mu)] beta = (mu + sqrt(square(mu) + 4*sigma^2)) / (2 * sigma^2);
        vector[rows(mu)] alpha = 1 + mu .* beta;
        return gamma_lpdf(z | alpha, beta);
    }

}
data {

    // Metadata
    int T;          // Number of trials

    // Data
    vector[T] Z;    // Reaction times

}
parameters {

    real<lower=0> mu;       // Center of distribution
    real<lower=0> shape;    // Deviation around mean

}
model {

    // Priors
    mu ~ uniform(0, 2);
    shape ~ normal(0, 1);

    // Likelihood
    Z ~ gamma_mode(mu, shape);

}
