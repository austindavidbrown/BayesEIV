import sys
sys.path.append("../")
from bayeseiv import EIVGibbsSampler

import numpy as np
import pandas as pd

# Load dataset
df = dict(pd.read_csv('https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p1/M_sigma.csv'))
X = np.expand_dims(np.array(df['obsx']), 1)
errX = np.expand_dims(np.array(df['errx']), 1)
Y = np.array(df['obsy'])
errY = np.array(df['erry'])
n_samples = X.shape[0]


# Initialize the sampler with data and hyper-parameters
Z = np.column_stack((np.ones(n_samples), np.diag(np.ones(n_samples))))
V_mu = np.diag(np.concatenate((np.array([10**3]), errY**2)))
V_A = [errX[i]**(2) for i in range(0, n_samples)]
V_beta = 10**(3) * np.eye(1)

gibbs_sampler = EIVGibbsSampler(
                  Z, X, Y,
                  a_0 = 10**(-3), s_0 = 10**(-3),
                  V_mu = V_mu, V_beta = V_beta,
                  V_A = V_A,
                )

# Run the sampler from initial values
burn_in = 10**3
n_iterations = 10**4
sigma2s, mus, betas, As = gibbs_sampler.sample(n_iterations = n_iterations)

###
# Results
###
output = pd.DataFrame({'mean' : [mus[burn_in:, 0].mean(), betas[burn_in:, 0].mean(), np.sqrt(sigma2s[burn_in:]).mean()],
                   'std' : [mus[burn_in:, 0].std(), betas[burn_in:, 0].std(), np.sqrt(sigma2s[burn_in:]).std()],
                   '2.5%' : [np.quantile(mus[burn_in:, 0], .025), np.quantile(betas[burn_in:, 0], .025), np.quantile(np.sqrt(sigma2s[burn_in:]), .025)],
                   '97.5%' : [np.quantile(mus[burn_in:, 0], .975), np.quantile(betas[burn_in:, 0], .975), np.quantile(np.sqrt(sigma2s[burn_in:]), .975)]
                   },
                   index = ["mu", "beta", "sigma"])
print(output.to_markdown())






###
# This following is modified but the full citation is listed
###
import pystan 

# prepare data for Stan
data = {}
data['obsx'] = np.array(df['obsx'])
data['errx'] = np.array(df['errx'])
data['obsy'] = np.array(df['obsy'])
data['erry'] = np.array(df['erry'])
data['N'] = len(data['obsx'])

# From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication
#
# Code 10.2 Normal linear model, in Python using Stan, for assessing the relationship
#           between central black hole mass and bulge velocity dispersion
#
# Statistical Model: Gaussian regression considering errors in variables
#                    in Python using Stan
#
# Astronomy case: Relation between mass of galaxy central supermassive black hole
#                 and its stelar bulge velocity dispersion
#                 taken from Harris, Poole and Harris, 2013, MNRAS, 438 (3), p.2117-2130
#
# 1 response (obsy - mass) and 1 explanatory variable (obsx - velocity dispersion)
#
# Data from: http://www.physics.mcmaster.ca/~harris/GCS_table.txt

# Stan Gaussian model with errors
stan_code="""
data{
    int<lower=0> N;                   // number of data points
    vector[N] obsx;                   // obs velocity dispersion
    vector<lower=0>[N] errx;          // errors in velocity dispersion measurements
    vector[N] obsy;                   // obs black hole mass
    vector<lower=0>[N] erry;          // errors in black hole mass measurements
}
parameters{
    real alpha;                       // intercept
    real beta;                        // angular coefficient
    real<lower=0> epsilon;            // scatter around true black hole mass
    vector[N] x;                      // true velocity dispersion
    vector[N] y;                      // true black hole mass
}
model{
  // likelihood and priors
  alpha ~ normal(0, 1000);
  beta ~ normal(0, 1000); 
  epsilon ~ gamma(0.001, 0.001);

  obsx ~ normal(x, errx);
  y ~ normal(alpha + beta * x, epsilon); 
  obsy ~ normal(y, erry);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, 
                  data=data, 
                  iter=n_iterations, 
                  warmup=burn_in, 
                  n_jobs = 3,
                  chains = 1)

# Output
output = str(fit).split('\n')
for item in output[:8]:
    print(item) 

















###
# Plot
###

light_blue_color = (3./255, 37./255, 76./255)
dark_blue_color = (24./255, 123./255, 205./255)

red_color = (0.86, 0.3712, 0.33999999999999997)
blue_color = (0.33999999999999997, 0.43879999999999986, 0.86)
green_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
purple_color = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)

import matplotlib.pyplot as plt
import seaborn as sns

plt.clf()
plt.style.use("ggplot")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.histplot(mus[burn_in:, 0], ax=ax, color=red_color, kde=False, label="Gibbs")
sns.histplot(fit["alpha"], ax=ax, color=blue_color, kde=False, label="STAN")
plt.xlabel(r"$\mu$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
ax.legend(fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("mu.png", pad_inches=0, bbox_inches='tight',)


plt.clf()
plt.style.use("ggplot")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.histplot(betas[burn_in:, 0], ax=ax, color=red_color, kde=False, label="Gibbs")
sns.histplot(fit["beta"], ax=ax, color=blue_color, kde=False, label="STAN")
plt.xlabel(r"$\beta$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
ax.legend(fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("beta.png", pad_inches=0, bbox_inches='tight',)


plt.clf()
plt.style.use("ggplot")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.histplot(np.sqrt(sigma2s[burn_in:]), ax=ax, color=red_color, kde=False, label="Gibbs")
sns.histplot(fit["epsilon"], ax=ax, color=blue_color, kde=False, label="STAN")
plt.xlabel(r"$\sigma$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
ax.legend(fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("sigma.png", pad_inches=0, bbox_inches='tight',)




'''
###
# Plot
###
import matplotlib.pyplot as plt
import seaborn as sns

plt.clf()
sns.set(rc={"figure.figsize": (100, 100)})
plt.style.use("ggplot")

#sns.lineplot(np.arange(10**4 - burn_in), betas[burn_in:, 0])
sns.histplot(mus[burn_in:, 0], kde = False)
plt.xlabel(r"$\mu$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("mu.png", pad_inches=0, bbox_inches='tight',)

plt.clf()
sns.set(rc={"figure.figsize": (100, 100)})
plt.style.use("ggplot")

#sns.lineplot(np.arange(10**4 - burn_in), betas[burn_in:, 0])
sns.histplot(betas[burn_in:, 0], kde = False)
plt.xlabel(r"$\beta$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("beta.png", pad_inches=0, bbox_inches='tight',)

plt.clf()
sns.set(rc={"figure.figsize": (100, 100)})
plt.style.use("ggplot")

#sns.lineplot(np.arange(10**4 - burn_in), betas[burn_in:, 0])
sns.histplot(np.sqrt(sigma2s[burn_in:]), kde = False)
plt.xlabel(r"$\sigma$", fontsize = 30, color="black")
plt.ylabel(r"Density", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=15, borderpad=.05, framealpha=0)
plt.savefig("sigma.png", pad_inches=0, bbox_inches='tight',)
'''



