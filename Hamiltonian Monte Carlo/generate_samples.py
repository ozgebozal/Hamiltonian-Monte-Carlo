import numpy as np
import matplotlib.pyplot as plt


# Generate samples from a mixture of distributions
numSamples = 1000
samples = [None] * numSamples
mu = [-2, 2]
sigma = [1, 3]
weights = [0.7, 0.3]
for i in range(numSamples):
     comp = np.random.choice(np.arange(0, 2), p=weights);
     samples[i] = np.random.normal(loc=mu[comp], scale=sigma[comp], size=1)[0];

plt.hist(samples, normed=True, bins='auto');
plt.show();




