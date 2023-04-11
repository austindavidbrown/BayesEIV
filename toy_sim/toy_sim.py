import numpy as np
import pandas as pd

import numpy as np
from scipy.stats import wishart

def make_spd_matrix(m, rho = .6):
  A = np.eye(m)
  for i in range(0, m):
    for j in range(0, m):
      A[i, j] = rho**(abs(i - j))
  return A

def cholesky_inv(M):
  M_L = np.linalg.cholesky(M)
  M_inv_L = np.linalg.inv(M_L).T
  M_inv = M_inv_L @ M_inv_L.T
  return M_inv_L, M_inv

###
# Error-in-variables regression Gibbs sampler
###
class EIVGibbsSamplerMultivariate:
  def __init__(self,
               R, M,
               a_0, B_0,
               C_theta, m_theta,
               C_beta, m_beta,
               C_A, m_A):
    self.n_samples = R.shape[0]
    self.m = R.shape[1]

    self.q = int(m_theta.shape[0] / self.m)
    self.p = int(m_beta.shape[0] / self.m)

    # Store data
    self.R = R
    self.r = self.R.flatten("F")
    self.M = M

    # Hyper-parameters for sigma2
    self.a_0 = a_0
    self.B_0 = B_0

    # Hyper-parameters for mu, beta
    self.c_0 = np.concatenate((m_theta, m_beta))
  
    # Create covariance matrix for (mu, beta) and invert it
    C_theta_inv_L, C_theta_inv = cholesky_inv(C_theta)
    C_beta_inv_L, C_beta_inv = cholesky_inv(C_beta)
    self.C_0_inv = np.block([[C_theta_inv, np.zeros((self.q * self.m, self.p * self.m))], [np.zeros((self.p * self.m, self.q * self.m)), C_beta_inv]])

    # Hyper-parameters for A

    # Invert all of the covariance matrices for A
    self.D_0_inv = []
    for i in range(0, self.n_samples):
      D_0_i_L_inv = np.linalg.inv(np.linalg.cholesky(C_A[i]))
      self.D_0_inv.append(D_0_i_L_inv.T @ D_0_i_L_inv)

    self.d_0 = m_A

  def sample(self, n_iterations):
    # The Markov chain
    Sigma_invs = np.ones((n_iterations, self.m, self.m))

    Thetas = np.zeros((n_iterations, self.q, self.m))
    thetas = np.zeros((n_iterations, self.q * self.m))

    Bs = np.zeros((n_iterations, self.p, self.m))
    betas = np.zeros((n_iterations, self.p *self.m))

    As = np.zeros((n_iterations, self.n_samples, self.p))

    for t in range(1, n_iterations):
      if t % 10**4 == 0:
        print("Iteration", t)

      ###
      # Sample sigma2
      ###
      a_n = self.a_0 + self.n_samples
      S_n = self.R - self.M @ Thetas[t-1]  - As[t-1] @ Bs[t-1]
      B_n = self.B_0 + S_n.T @ S_n

      B_n_inv_L, B_n_inv = cholesky_inv(B_n)
      Sigma_invs[t] = wishart.rvs(size = 1, df = a_n, scale = B_n_inv)

      ###
      # Sample mu, beta
      ###
      Z_A_t_1 = np.column_stack((self.M, As[t-1]))
      C_n_L, C_n = cholesky_inv( np.kron(Sigma_invs[t], Z_A_t_1.T @ Z_A_t_1) + self.C_0_inv)
      m_n = C_n @ (np.kron(Sigma_invs[t], Z_A_t_1.T) @ self.r + self.C_0_inv @ self.c_0 )

      thetas_betas_t = m_n + C_n_L @ np.random.normal(size=self.m * ( self.q + self.p ), loc=0, scale=1)
      
      thetas[t] = thetas_betas_t[0:self.q * self.m]
      betas[t] = thetas_betas_t[(self.q * self.m):]

      Thetas[t] = thetas[t].reshape(self.m, self.q).T
      Bs[t] = betas[t].reshape(self.m, self.p).T
      
      ###
      # Sample A
      ###
      for i in range(0, self.n_samples):
        D_i_L, D_i = cholesky_inv( Bs[t] @ Sigma_invs[t] @ Bs[t].T + self.D_0_inv[i] )
        m_i = D_i @ ( self.D_0_inv[i] @ self.d_0[i] + Bs[t] @ Sigma_invs[t] @ (self.R[i] -  Thetas[t].T @ self.M[i]) )

        As[t, i, :] = m_i + D_i_L @ np.random.normal(size=self.p, loc=0, scale=1)

    return {
      "Sigma_invs": Sigma_invs, 
      "thetas": thetas, 
      "betas": betas, 
      "As": As
    }


def bm_cov(X):
  N = X.shape[0]
  M = int(N**(1/3)) # how many batches
  B = int(N/M) # batch size

  m = np.mean(X, 0)
  S = np.zeros((M, X.shape[1]))
  for k in range(0, M):
    X_m = X[(k * B):(k + 1)*B, :].mean(0)
    S[k]  = X_m - m
  return B/(M - 1) * S.T @ S


def multi_ess(X):
  N = X.shape[0]
  d = X.shape[1]

  m = X.mean(0)
  cov = 1/(N - 1) * (X - m).T @ (X - m)
  L = np.linalg.cholesky(cov)

  bmcov = bm_cov(X)
  L_bm = np.linalg.cholesky(bmcov)

  det_cov = np.linalg.det(L)**2
  det_bmcov = np.linalg.det(L_bm)**2

  return N * (det_cov / det_bmcov)**(1/d)


def compute_diagnostics(X):
  se = np.zeros(N_COMPUTES)
  ess = np.zeros(N_COMPUTES)
  for k in range(0, N_COMPUTES):
    X_k = X[0:(SHIFT * (k + 1))]
    lams_k, _ = np.linalg.eig(np.linalg.cholesky(bm_cov(X_k)))
    se[k] = np.max(lams_k)
    ess[k] = multi_ess(X_k) 
  return se, ess


def run_sim(q, p, m, n):
  ###
  # Generate data
  ###
  Sigma_true = make_spd_matrix(m)
  Sigma_half_true = np.linalg.cholesky(Sigma_true)

  Theta_true = np.random.uniform(size=(q, m), low=-2, high=2)
  B_true = np.random.uniform(size=(p, m), low=-2, high=2)
  
  C_A_true = [.2 * np.eye(p) for i in range(0, n)]

  Z = np.zeros((n, q))
  Z[:, 0] = 1
  for i in range(1, n):
    Z[i] = np.random.uniform(size=q, low=-1, high=1)

  X = np.zeros((n, p))
  A_true = np.zeros((n, p))
  for i in range(0, n):
    X[i] = np.random.uniform(size=p, low=-1, high=1)
    A_true[i] = X[i] + np.random.multivariate_normal(mean=np.zeros(p),
                                                     cov = C_A_true[i])
  Y = np.zeros((n, m))
  for i in range(0, n):
    Y[i] = Theta_true.T @ Z[i] + B_true.T @ A_true[i] + Sigma_half_true @ np.random.normal(size = m, loc=0, scale=1)

  # Run the sampler from initial values
  gibbs_sampler = EIVGibbsSamplerMultivariate(
                    R = Y, M = Z,
                    a_0 = m, B_0 = 10**(-3) * np.eye(m),
                    C_theta = 100 * np.eye(q * m), m_theta = np.zeros(q * m),
                    C_beta = 100 * np.eye(p * m), m_beta = np.zeros(p * m),
                    C_A = C_A_true, m_A = X,
                  )

  ###
  # Generate diagnostics
  ###
  se_vector = np.zeros((N_REPS, N_COMPUTES))
  ess_vector = np.zeros((N_REPS, N_COMPUTES))
  for r in range(0, N_REPS):
    print("Rep:", r + 1)
    sim = gibbs_sampler.sample(n_iterations = N_ITERATIONS)
    betas = sim["betas"][BURN_IN:]
    se, ess = compute_diagnostics(betas)
    se_vector[r] = se
    ess_vector[r] = ess

    print("SE", se)
    print("ESS", ess)

  se_mean = se_vector.mean(0)
  ess_mean = ess_vector.mean(0)
  return se_mean, ess_mean


######
# Simulations
######

np.random.seed(1)

BURN_IN = 10**4
N_ITERATIONS = 10**5 + BURN_IN
N_REPS = 5

SHIFT = int(N_ITERATIONS / 4)
N_COMPUTES = 1 + int((N_ITERATIONS - SHIFT) / SHIFT)

p = 6
m = 3
print("Running sim")
se, ess = run_sim(q = 1, p = p, m = m, n = 50)

###
# Save to csv
###
pd.DataFrame({"se": se,
              "ess": ess
              }).to_csv("sim_%s_%s.csv" % (p, m), index = False)


