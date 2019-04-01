import pystan
import stan_utility
import numpy as np
import matplotlib.pyplot as plt

# Generate samples from a mixture of distributions
numSamples = 100
samples = [None] * numSamples
mu = [-2, 2]
sigma = [1, 3]
weights = [0.7, 0.3]
for i in range(numSamples):
     comp = np.random.choice(np.arange(0, len(mu)), p=weights);
     samples[i] = np.random.normal(loc=mu[comp], scale=sigma[comp], size=1)[0];

#plt.hist(samples, normed=True, bins='auto');
#plt.show();

paramEstModel = """
data {
    int<lower=1> K; // number of mixture components
    int<lower=1> N; // number of data points
    real y[N]; // observations
}
parameters {
    simplex[K] theta; // mixing proportions
    ordered[K] mu; // locations of mixture components
    vector<lower=0.001>[K] sigma; // scales of mixture components
}
model {
    vector[K] log_theta = log(theta); // cache log calculation
    sigma ~ lognormal(0,2);
    mu ~ normal(0,10);
    for (n in 1:N) {
        vector[K] lps = log_theta;
        for (k in 1:K)
            lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
        target += log_sum_exp(lps);
    }
}
"""

modelData = {'K': 2,
             'N': numSamples,
             'y': samples}

sm = pystan.StanModel(model_code=paramEstModel)
fit = sm.sampling(data=modelData, iter=4000, chains=4, algorithm="HMC")

print(fit)

#print(fit.get_sampler_params(inc_warmup=False))

#stan_utility.check_treedepth(fit)
#stan_utility.check_energy(fit)
#stan_utility.check_div(fit)
