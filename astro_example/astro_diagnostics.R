library("mcmcse")

df = read.csv("astro_example.csv")

sigma2s = as.vector(unlist(df["sigma2s"]))
alphas = as.vector(unlist(df["alphas"]))
betas = as.vector(unlist(df["betas"]))

N_ITERATIONS = length(betas) 
SHIFT = N_ITERATIONS / 4
N_COMPUTES = N_ITERATIONS / SHIFT

sigma2s_se = numeric(N_COMPUTES)
sigma2s_ess = numeric(N_COMPUTES)

alphas_se = numeric(N_COMPUTES)
alphas_ess = numeric(N_COMPUTES)

betas_se = numeric(N_COMPUTES)
betas_ess = numeric(N_COMPUTES)

for (i in 1:N_COMPUTES) {
  shift = SHIFT * i
  print(shift)
  sigma2s_se[i] = mcse(sigma2s[1:shift])$se
  sigma2s_ess[i] = ess(sigma2s[1:shift])

  alphas_se[i] = mcse(alphas[1:shift])$se
  alphas_ess[i] = ess(alphas[1:shift])

  betas_se[i] = mcse(betas[1:shift])$se
  betas_ess[i] = ess(betas[1:shift])
}

csv = data.frame(sigma2s_se = sigma2s_se, 
                 sigma2s_ess = sigma2s_ess,
                 alphas_se = alphas_se, 
                 alphas_ess = alphas_ess,
                 betas_se = betas_se, 
                 betas_ess = betas_ess)
write.csv(csv, "astro_diagnostics.csv", row.names = FALSE)