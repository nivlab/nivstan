// Shifted Wald distribution

functions {

    real shifted_wald_lpdf(real x, real gamma, real alpha, real theta) {
        real tmp1;
        real tmp2;

        tmp1 = alpha / (sqrt(2 * pi() * (pow((x - theta), 3))));
        tmp2 = exp(-1 * (pow((alpha - gamma * (x-theta)),2)/(2*(x-theta))));
        return log(tmp1*tmp2);
    }

}
data {

    // Metadata
    int  T;         // Number of trials

    // Data
    vector[T] Z;    // Reaction times

}
parameters {

    real<lower=0> gamma;    // Extent of right-skew
    real<lower=0> alpha;    // Deviation around mean
    real<lower=0> theta;    // Shift in onset

}
model {

    // Priors
    gamma ~ gamma(1, 0.5);
    alpha ~ gamma(1, 0.5);
    theta ~ normal(0.5, 0.5);

    // Likelihood
    for (i in 1:T) {
        target += shifted_wald_lpdf( Z[i] | gamma, alpha, theta );
    }

}
