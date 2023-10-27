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

SIM1_LABEL = "p = 1, m = 1"
SIM3_LABEL = "p = 7, m = 3"


def plot_estimate(iterations, med, lower, upper, marker, color, label):
  plt.plot(iterations, med, 
    '-', label=label, 
    alpha = OPACITY, 
    marker=marker, 
    markersize=MARKERSIZE, 
    color=color, 
    linewidth = LINEWIDTH)
  plt.fill_between(iterations, 
                   lower, upper, 
                   alpha=0.1,
                   color=color)

################################################################################################
# Plot ERRORS 
################################################################################################


N_SAMPLES = 10**(5)
SHIFT = int(N_SAMPLES / 4)

iterations = np.arange(SHIFT, N_SAMPLES + SHIFT, SHIFT)

df = dict(pd.read_csv('./sim_1_1.csv'))
min_se_lower_1 = np.array(df["min_se_lower"])
min_se_med_1 = np.array(df["min_se_med"])
min_se_upper_1 = np.array(df["min_se_upper"])

max_se_lower_1 = np.array(df["max_se_lower"])
max_se_med_1 = np.array(df["max_se_med"])
max_se_upper_1 = np.array(df["max_se_upper"])

ess_lower_1 = np.array(df["ess_lower"])
ess_med_1 = np.array(df["ess_med"])
ess_upper_1 = np.array(df["ess_upper"])

df = dict(pd.read_csv('./sim_7_3.csv'))
min_se_lower_3 = np.array(df["min_se_lower"])
min_se_med_3 = np.array(df["min_se_med"])
min_se_upper_3 = np.array(df["min_se_upper"])

max_se_lower_3 = np.array(df["max_se_lower"])
max_se_med_3 = np.array(df["max_se_med"])
max_se_upper_3 = np.array(df["max_se_upper"])

ess_lower_3 = np.array(df["ess_lower"])
ess_med_3 = np.array(df["ess_med"])
ess_upper_3 = np.array(df["ess_upper"])


plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

plot_estimate(iterations, 
              min_se_med_1, min_se_lower_1, min_se_upper_1, 
              "s", COLORS[0], SIM1_LABEL)
plot_estimate(iterations, 
              min_se_med_3, min_se_lower_3, min_se_upper_3, 
              "p", COLORS[3], SIM3_LABEL)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"$\lambda_{min}$ of standard error matrix", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("toy_se_min.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")


plot_estimate(iterations, 
              max_se_med_1, max_se_lower_1, max_se_upper_1, 
              "s", COLORS[0], SIM1_LABEL)
plot_estimate(iterations, 
              max_se_med_3, max_se_lower_3, max_se_upper_3, 
              "p", COLORS[3], SIM3_LABEL)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"$\lambda_{max}$ of standard error matrix", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("toy_se_max.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

plot_estimate(iterations, 
              ess_med_1, ess_lower_1, ess_upper_1, 
              "s", COLORS[0], SIM1_LABEL)
plot_estimate(iterations, 
              ess_med_3, ess_lower_3, ess_upper_3, 
              "p", COLORS[3], SIM3_LABEL)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"Multivariate ESS", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("toy_ess.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

