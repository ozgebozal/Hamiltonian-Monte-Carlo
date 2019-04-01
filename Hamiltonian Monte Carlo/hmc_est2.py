import pystan
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import savemat
import pickle

# Generate samples from a mixture of distributions
numSamples = 2000
samples = [None] * numSamples
mu = [-2, 5, 1, 8, 15, -3, 0.4, 9]
sigma = [1, 3, 0.1, 4, 2, 0.4, 0.5, 0.7]
weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
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
    vector[K] mu; // locations of mixture components
    vector<lower=0.001>[K] sigma; // scales of mixture components
}
model {
    vector[K] log_theta = log(theta); // cache log calculation
    sigma ~ lognormal(0, 2);
    mu ~ normal(0, 10);
    for (n in 1:N) {
        vector[K] lps = log_theta;
        for (k in 1:K)
            lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
        target += log_sum_exp(lps);
    }
}
"""

modelData = {'K': 8,
             'N': numSamples,
             'y': samples}

sm = pystan.StanModel(model_code=paramEstModel)
fit = sm.sampling(data=modelData, algorithm='HMC',iter=1000, chains=1, thin=2, n_jobs=-1, )

print(fit)

fit.plot()
plt.show()

file_handler = open('fit.obj', 'w') 
pickle.dump(fit, file_handler) 

extracted_data = fit.extract(permuted=True) 
#np.save( '/home/korhan/Desktop/mc/extracted.txt' , extracted_data)
savemat('/home/korhan/Desktop/mc/extracted.mat', extracted_data)
