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

SIM1_LABEL = "p = 2, m = 2"
SIM2_LABEL = "p = 4, m = 2"
SIM3_LABEL = "p = 8, m = 2"


################################################################################################
# Plot ERRORS 
################################################################################################


N_SAMPLES = 10**(5)
SHIFT = int(N_SAMPLES / 4)

iterations = np.arange(SHIFT, N_SAMPLES + SHIFT, SHIFT)

df = dict(pd.read_csv('./sim1.csv'))
se1 = np.array(df["se"])
ess1 = np.array(df["ess"])

df = dict(pd.read_csv('./sim2.csv'))
se2 = np.array(df["se"])
ess2 = np.array(df["ess"])

df = dict(pd.read_csv('./sim3.csv'))
se3 = np.array(df["se"])
ess3 = np.array(df["ess"])

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-whitegrid")


plt.plot(iterations, se1, 
  '-', label=SIM1_LABEL,  
  alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, se2, 
  '-', label=SIM2_LABEL, 
  alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)
plt.plot(iterations, se3, 
  '-', label=SIM3_LABEL, 
  alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Iteration", fontsize=30)
plt.ylabel(r"$\lambda_{max}$ of standard error matrix", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("toy_se.pdf", 
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
plt.plot(iterations, ess3, 
  '-', label=SIM3_LABEL, 
  alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)


plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
#plt.xticks(iterations, ['%.2E' % i for i in iterations])
plt.xlabel("Iteration", fontsize=30)
plt.ylabel("Multivariate ESS", fontsize=30)
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
plt.savefig("toy_ess.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


'''
################################################################################################
# Plot ACF 
################################################################################################

BURN_IN = 0 # int(N_SAMPLES / 2)
betas1 = pd.read_csv('./betas1.csv').values
betas2 = pd.read_csv('./betas2.csv').values
betas3 = pd.read_csv('./betas3.csv').values

####
# Compute autocorrelations
####

def acf(x, length=20):
  correlations = np.ones(length)
  for i in range(1, length):
    correlations[i] = np.corrcoef(x[:-i], x[i:])[0,1]
  return correlations

length = 10
acf1 = acf(betas1[BURN_IN:, 0], length)
acf2 = acf(betas2[BURN_IN:, 0], length)
acf3 = acf(betas3[BURN_IN:, 0], length)

###
# Plot acf
###
plt.clf()
plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 8))
iterations = np.arange(1, length + 1)

plt.plot(iterations, acf1, 
         '-', label="p = 1, m = 2",  
         alpha = OPACITY, marker="s", markersize=MARKERSIZE, color=COLORS[0], linewidth = LINEWIDTH)
plt.plot(iterations, acf2, 
         '-', label="p = 2, m = 2", 
         alpha = OPACITY, marker="v", markersize=MARKERSIZE, color=COLORS[2], linewidth = LINEWIDTH)
plt.plot(iterations, acf3, 
         '-', label="p = 3, m = 2", 
         alpha = OPACITY, marker="p", markersize=MARKERSIZE, color=COLORS[3], linewidth = LINEWIDTH)


plt.tick_params(axis='x', labelsize=20)
#plt.xticks(iterations)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Lag", fontsize = 25, color="black")
plt.ylabel(r"Autocorrelation", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("acf.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')
'''
