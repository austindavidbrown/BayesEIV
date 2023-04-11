import numpy as np
import pandas as pd

###
# Univariate Error-in-variables regression Gibbs sampler
###
class EIVGibbsSamplerUnivariate:
  def __init__(self,
               r, M,
               a_0, B_0,
               C_theta, m_theta,
               C_beta, m_beta,
               C_A, m_A):
    self.n_features = C_beta.shape[1]
    self.n_samples = M.shape[0]
    self.q = M.shape[1]

    # Store data
    self.r = r
    self.M = M

    # Hyper-parameters for sigma2
    self.a_0 = a_0
    self.B_0 = B_0

    # Hyper-parameters for mu, beta
    self.m_theta_beta = np.concatenate((m_theta, m_beta))
  
    # Create covariance matrix for (mu, beta) and invert it
    C_theta_L_inv = np.linalg.inv(np.linalg.cholesky(C_theta))
    C_theta_inv = C_theta_L_inv.T @ C_theta_L_inv

    C_beta_L_inv = np.linalg.inv(np.linalg.cholesky(C_beta))
    C_beta_inv = C_beta_L_inv.T @ C_beta_L_inv

    self.C_theta_beta_inv = np.block(
                           [[C_theta_inv, np.zeros((self.q, self.n_features))], 
                           [np.zeros((self.n_features, self.q)), C_beta_inv]]
                         )

    # Hyper-parameters for A

    # Invert all of the covariance matrices for A
    self.D_0_inv = []
    for i in range(0, self.n_samples):
      self.D_0_inv.append(1./C_A[i])

    self.d_0 = m_A

  def sample(self, n_iterations):
    # The Markov chain
    sigma2s = np.ones(n_iterations)
    mus_betas = np.zeros((n_iterations, self.q + self.n_features))
    As = np.zeros((n_iterations, self.n_samples, self.n_features))

    for t in range(1, n_iterations):
      if t % 10**4 == 0:
        print("Iteration", t)
      ###
      # Sample sigma2
      ###
      Z_A_t_1 = np.column_stack((self.M, As[t-1]))
      a_n = self.a_0 + self.n_samples/2
      B_n = self.B_0 + 1/2 * ((self.r - Z_A_t_1 @ mus_betas[t-1])**2).sum()
      
      sigma2s[t] = 1./np.random.gamma(shape = a_n, scale = 1/B_n)

      ###
      # Sample mu, beta
      ###
      C_n_inv = 1/sigma2s[t] * Z_A_t_1.T @ Z_A_t_1 + self.C_theta_beta_inv
      C_n_inv_L = np.linalg.cholesky(C_n_inv)
      C_n_inv_L_inv = np.linalg.inv(C_n_inv_L)
      C_n = C_n_inv_L_inv.T @ C_n_inv_L_inv

      xi = np.random.normal(size=self.q + self.n_features, loc=0, scale=1)
      mus_betas[t] = C_n @ (1/sigma2s[t] * Z_A_t_1.T @ self.r) + C_n @ self.C_theta_beta_inv @ self.m_theta_beta \
                     + C_n_inv_L_inv.T @ xi


      ###
      # Sample A
      ###
      theta_t = mus_betas[t][:self.q]
      beta_t = mus_betas[t][self.q:]
      for i in range(0, self.n_samples):
        D_i_inv = 1/sigma2s[t] * np.outer(beta_t, beta_t) + self.D_0_inv[i]
        D_i = 1./D_i_inv

        d_i = D_i * ( self.D_0_inv[i] @ self.d_0[i] + beta_t* 1/sigma2s[t] * (self.r[i] -  self.M[i] @ theta_t ) )
        
        xi_i = np.random.normal(size=self.n_features, loc=0, scale=1)
        As[t, i, :] = d_i + D_i**(1/2) * xi_i

        #As[t, i, :] = self.m_A[i] \
        #              + (self.r[i] - np.concatenate((self.M[i], self.m_A[i]), axis=0) @ mus_betas[t]) * (1/sigma2s[t] * C_i @ beta_t) \
        #              + D_i**(1/2) @ xi_i

    return {
       "sigma2s": sigma2s, 
       "thetas": mus_betas[:, 0:self.q], 
       "betas": mus_betas[:, self.q:], 
       "As": As
    }
    

def acf(x, length=20):
  correlations = np.ones(length)
  for i in range(1, length):
    correlations[i] = np.corrcoef(x[:-i], x[i:])[0,1]
  return correlations

def bm_var(X):
  N = X.shape[0]
  M = int(N**(1/3)) # how many batches
  B = int(N/M) # batch size

  m = np.mean(X, 0)
  S = np.zeros(M)
  for k in range(0, M):
    X_m = X[(k * B):(k + 1)*B].mean(0)
    S[k]  = X_m - m
  return B/(M - 1) * S.T @ S

def ess(X):
  N = X.shape[0]

  m = X.mean(0)
  var = 1/(N - 1) * (X - m).T @ (X - m)
  bmvar = bm_var(X)

  return N * (var / bmvar)

def compute_diagnostics(X):
  n_computes = 1 + int((N_ITERATIONS - SHIFT) / SHIFT)

  se_values = np.zeros(n_computes)
  ess_values = np.zeros(n_computes)
  for k in range(0, n_computes):
    X_k = X[0:(SHIFT * (k + 1))]
    se_values[k] = bm_var(X_k)**(1/2)
    ess_values[k] = ess(X_k) 
  return se_values, ess_values


####
# Simulation
####

# Load dataset
df = dict(pd.read_csv('https://raw.githubusercontent.com/astrobayes/BMAD/master/data/Section_10p1/M_sigma.csv'))
X = np.expand_dims(np.array(df['obsx']), 1)
errX = np.expand_dims(np.array(df['errx']), 1)
Y = np.array(df['obsy'])
errY = np.array(df['erry'])
n_samples = X.shape[0]


# Initialize the sampler with data and hyper-parameters
M = np.column_stack( (np.ones(n_samples), -np.eye(n_samples)  ) )

C_theta = np.diag( np.concatenate( (np.array([10**3]), errY**2 ) ) )
C_beta = 10**(3) * np.eye(1)

C_A = [(errX[i]**(-2) + 10**(-3))**(-1) for i in range(0, n_samples)]
m_A = [C_A[i] * errX[i]**(-2) * X[i] for i in range(0, n_samples)]

gibbs_sampler = EIVGibbsSamplerUnivariate(
                  r = np.zeros(n_samples), M = M,
                  a_0 = 10**(-3), B_0 = 10**(-3),
                  C_theta = C_theta, m_theta = np.concatenate((np.zeros(1), Y)),
                  C_beta = C_beta, m_beta = np.zeros(1),
                  C_A = C_A, m_A = m_A,
                )

print("Running Gibbs sampler")

# Run the Gibbs sampler
np.random.seed(2)

BURN_IN = 10**(4)
N_ITERATIONS = 10**5 + BURN_IN
SHIFT = int(N_ITERATIONS / 4)

sim = gibbs_sampler.sample(n_iterations = N_ITERATIONS)
sigma2s = sim["sigma2s"][BURN_IN:]
alphas = sim["thetas"][BURN_IN:]
betas = sim["betas"][BURN_IN:]
As = sim["As"][BURN_IN:]

###
# Results
###
output = pd.DataFrame({'mean' : [alphas[:, 0].mean(), betas[:, 0].mean(), np.sqrt(sigma2s[:]).mean()],
                       '2.5%' : [np.quantile(alphas[:, 0], .025), np.quantile(betas[:, 0], .025), np.quantile(np.sqrt(sigma2s[:]), .025)],
                       '97.5%' : [np.quantile(alphas[:, 0], .975), np.quantile(betas[:, 0], .975), np.quantile(np.sqrt(sigma2s[:]), .975)]
                      },
                      index = ["alpha", "beta", "sigma"])
print(output.to_markdown())

# Save to csv
pd.DataFrame({'sigma2s' : sigma2s,
              'alphas' : alphas[:, 0],
              'betas' : betas[:, 0]
             }).to_csv("astro_example.csv", index = False)


sigma2s_se, sigma2s_ess = compute_diagnostics(sigma2s)
alphas_se, alphas_ess = compute_diagnostics(alphas[:, 0])
betas_se, betas_ess = compute_diagnostics(betas[:, 0])

# Save to csv
pd.DataFrame({"sigma2s_se": sigma2s_se,
              "sigma2s_ess": sigma2s_ess,
              "alphas_se": alphas_se,
              "alphas_ess": alphas_ess,
              "betas_se": betas_se,
              "betas_ess": betas_ess,
              }).to_csv("astro_diagnostics.csv", index = False)

####
# Compute autocorrelations
####
sigma2s_acf = acf(sigma2s, 20)
alphas_acf = acf(alphas[:, 0], 20)
betas_acf = acf(betas[:, 0], 20)

# Save to csv
pd.DataFrame({"sigma2s_acf": sigma2s_acf,
              "alphas_acf": alphas_acf,
              "betas_acf": betas_acf,
              }).to_csv("astro_acf.csv", index = False)





















'''
###
# Notes:
# use Python3.8 and pystan < 3.
###

###
# This following is modified but the full citation is listed
###
import pystan

# prepare data for Stan
data = {}
data['obsx'] = np.array(df['obsx']).tolist()
data['errx'] = np.array(df['errx']).tolist()
data['obsy'] = np.array(df['obsy']).tolist()
data['erry'] = np.array(df['erry']).tolist()
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
data {
    int<lower=0> N;                   // number of data points
    vector[N] obsx;                   // obs velocity dispersion
    vector<lower=0>[N] errx;          // errors in velocity dispersion measurements
    vector[N] obsy;                   // obs black hole mass
    vector<lower=0>[N] erry;          // errors in black hole mass measurements
}
parameters {
    real alpha;                       // intercept
    real beta;                        // angular coefficient
    real<lower=0> epsilon;            // scatter around true black hole mass
    vector[N] x;                      // true velocity dispersion
    vector[N] y;                      // true black hole mass
}
model {
  alpha ~ normal(0, 1000);
  beta ~ normal(0, 1000); 
  epsilon ~ gamma(0.001, 0.001);

  for (i in 1:N){
    x[i] ~ normal(0, 1000); 
  }

  obsx ~ normal(x, errx);
  y ~ normal(alpha + beta * x, epsilon); 
  obsy ~ normal(y, erry);
}
"""

# Run mcmc
fit = pystan.stan(model_code=stan_code, 
                  data=data, 
                  iter=N_ITERATIONS + 10, 
                  warmup = 10,
                  chains = 1,
                  n_jobs = -1,
                  algorithm = "HMC",
                  control = {
                    "stepsize": .01,
                    "metric": "diag_e",
                    "inv_metric": 1/100,
                    "adapt_engaged": False
                  })

# Output
output = str(fit).split('\n')
for item in output[:8]:
    print(item) 


'''
