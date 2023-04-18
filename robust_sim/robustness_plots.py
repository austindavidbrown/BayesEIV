import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

####
# Plot
####
COLORS = sns.color_palette("bright")

LINEWIDTH = 3
MARKERSIZE = 8
OPACITY = 1

SIM1_LABEL = "df = 2"
SIM2_LABEL = "df = 10"


################################################################################################
# Plot ERRORS 
################################################################################################


N_SAMPLES = 10**(5)
SHIFT = int(N_SAMPLES / 4)

iterations = np.arange(SHIFT, N_SAMPLES + SHIFT, SHIFT)

df = dict(pd.read_csv('./sim_2.csv'))
min_se1 = np.array(df["min_se"])
max_se1 = np.array(df["max_se"])
det_se1 = np.array(df["det_se"])
ess1 = np.array(df["ess"])

df = dict(pd.read_csv('./sim_10.csv'))
min_se2 = np.array(df["min_se"])
max_se2 = np.array(df["max_se"])
det_se2 = np.array(df["det_se"])
ess2 = np.array(df["ess"])

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")


plt.plot(iterations, min_se1, 
  '-', label=SIM1_LABEL,  
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, min_se2, 
  '-', label=SIM2_LABEL, 
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"$\lambda_{min}$ of standard error matrix", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("robust_se_min.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")


plt.plot(iterations, max_se1, 
  '-', label=SIM1_LABEL,  
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, max_se2, 
  '-', label=SIM2_LABEL, 
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"$\lambda_{max}$ of standard error matrix", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("robust_se_max.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")

plt.plot(iterations, ess1, 
  '-', label=SIM1_LABEL,  
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, ess2, 
  '-', label=SIM2_LABEL, 
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"Multivariate ESS", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("robust_ess.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

