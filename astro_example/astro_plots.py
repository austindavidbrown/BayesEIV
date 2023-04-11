import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

N_ITERATIONS = 10**(5)
SHIFT = int(N_ITERATIONS / 4)

COLORS = sns.color_palette("bright")

LINEWIDTH = 3
MARKERSIZE = 8
OPACITY = 1

################################################################################################
# Plot ACF 
################################################################################################

df = dict(pd.read_csv('./astro_acf.csv'))
sigma2s_acf = np.array(df["sigma2s_acf"])
alphas_acf = np.array(df["alphas_acf"])
betas_acf = np.array(df["betas_acf"])

plt.clf()
plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 8))
iterations = np.arange(1, sigma2s_acf.shape[0] + 1)

plt.plot(iterations, sigma2s_acf, 
         '-', label=r"$\sigma^2$",  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, alphas_acf, 
         '-', label=r"$\alpha$", alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)
plt.plot(iterations, betas_acf, 
        '-', label=r"$\beta$", alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)

plt.tick_params(axis='x', labelsize=20)
plt.xticks(iterations)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Lag", fontsize = 25, color="black")
plt.ylabel(r"Autocorrelation", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("astro_acf.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


################################################################################################
# Plot ERRORS 
################################################################################################

df = dict(pd.read_csv('./astro_diagnostics.csv'))

sigma2s_se = np.array(df["sigma2s_se"])
sigma2s_ess = np.array(df["sigma2s_ess"])

alphas_se = np.array(df["alphas_se"])
alphas_ess = np.array(df["alphas_ess"])

betas_se = np.array(df["betas_se"])
betas_ess = np.array(df["betas_ess"])

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")

iterations = np.arange(SHIFT, N_ITERATIONS + SHIFT, SHIFT)

plt.plot(iterations, sigma2s_se, '-', label=r"$\sigma^2$", 
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, alphas_se, '-', label=r"$\alpha$",
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)
plt.plot(iterations, betas_se, '-', label=r"$\beta$", 
  alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel("MCMC standard error", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("astro_se.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")

plt.plot(iterations, sigma2s_ess, '-', label=r"$\sigma^2$",  
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, alphas_ess, '-', label=r"$\alpha$", 
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)
plt.plot(iterations, betas_ess, '-', label=r"$\beta$", 
  alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel("MCMC ESS", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("astro_ess.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

