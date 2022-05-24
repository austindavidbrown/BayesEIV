import numpy as np


###
# Error-in-variables regression Gibbs sampler
###
class EIVGibbsSampler:
  def __init__(self,
               Z, X, Y,
               a_0, s_0,
               V_mu, V_beta,
               V_A):
    self.n_samples = X.shape[0]
    self.n_features = X.shape[1]
    self.q = Z.shape[1]

    # Store data
    self.Z = Z
    self.X = X
    self.Y = Y

    # Hyper-parameters
    self.a_0 = a_0
    self.s_0 = s_0

    # Create covariance matrix for (mu, beta) and invert it
    V_mu_L_inv = np.linalg.inv(np.linalg.cholesky(V_mu))
    V_mu_inv = V_mu_L_inv.T @ V_mu_L_inv

    V_beta_L_inv = np.linalg.inv(np.linalg.cholesky(V_beta))
    V_beta_inv = V_beta_L_inv.T @ V_beta_L_inv

    self.V_mu_beta_inv = np.block(
                           [[V_mu_inv, np.zeros((self.q, self.n_features))], 
                           [np.zeros((self.n_features, self.q)), V_beta_inv]]
                         )

    # Invert all of the covariance matrices for A
    self.V_A_inv = []
    for i in range(0, self.n_samples):
      if V_A[i].shape[0] == 1:
        self.V_A_inv.append(1./V_A[i])
      else:
        V_A_i_L_inv = np.linalg.inv(np.linalg.cholesky(V_A[i]))
        self.V_A_inv.append(V_A_i_L_inv.T @ V_A_i_L_inv)

  def sample(self, n_iterations):
    # The Markov chain
    sigma2s = np.ones(n_iterations)
    mus_betas = np.zeros((n_iterations, self.q + self.n_features))
    As = np.zeros((n_iterations, self.n_samples, self.n_features))

    for t in range(1, n_iterations):
      ###
      # Sample sigma2
      ###
      Z_X_A_t_1 = np.column_stack((self.Z, self.X + As[t-1]))

      a_n = self.a_0 + self.n_samples/2
      s_n = self.s_0 + 1/2 * ((self.Y - Z_X_A_t_1 @ mus_betas[t-1])**2).sum()
      
      sigma2s_t_inv = np.random.gamma(shape = a_n, scale = 1/s_n)
      sigma2s[t] = 1.0/sigma2s_t_inv
      
      ###
      # Sample A
      ###
      beta_t_1 = mus_betas[t-1][self.q:]
      for i in range(0, self.n_samples):
        V_i_inv = 1/sigma2s[t] * np.outer(beta_t_1, beta_t_1) + self.V_A_inv[i]

        if V_i_inv.shape[0] == 1:
          V_i_inv_L_inv = 1./V_i_inv**(1./2.)
          V_i = 1./V_i_inv
        else:
          V_i_inv_L = np.linalg.cholesky(V_i_inv)
          V_i_inv_L_inv = np.linalg.inv(V_i_inv_L)
          V_i = V_i_inv_L_inv.T @ V_i_inv_L_inv

        X_Z_i = np.concatenate((self.Z[i], self.X[i]), axis=0)
        r_i = (self.Y[i] - X_Z_i @ mus_betas[t-1]) * (1/sigma2s[t] * V_i @ beta_t_1)
        xi_i = np.random.normal(size=self.n_features, loc=0, scale=1)
        As[t, i, :] = r_i + V_i_inv_L_inv.T @ xi_i

      ###
      # Sample mu, beta
      ###
      Z_X_A_t = np.column_stack((self.Z, self.X + As[t]))
      C_n_inv = 1/sigma2s[t] * Z_X_A_t.T @ Z_X_A_t +  self.V_mu_beta_inv
      C_n_inv_L = np.linalg.cholesky(C_n_inv)
      C_n_inv_L_inv = np.linalg.inv(C_n_inv_L)
      C_n = C_n_inv_L_inv.T @ C_n_inv_L_inv

      m_n = 1/sigma2s[t] * C_n @ Z_X_A_t.T @ self.Y
      xi = np.random.normal(size=self.q + self.n_features, loc=0, scale=1)
      mus_betas[t] = m_n + C_n_inv_L_inv.T @ xi


    return sigma2s, mus_betas[:, 0:self.q], mus_betas[:, self.q:], As

