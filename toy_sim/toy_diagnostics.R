library("mcmcse")

betas1 = as.matrix(read.csv("betas1.csv"))
betas2 = as.matrix(read.csv("betas2.csv"))
betas3 = as.matrix(read.csv("betas3.csv"))

N_SAMPLES = dim(betas1)[1]
SHIFT = round(N_SAMPLES / 4)
BATCH_SIZE = round(N_SAMPLES**(1/3))
METHOD = "bm"

N_COMPUTES = 1 + (N_SAMPLES - SHIFT) / SHIFT

se1 = numeric(N_COMPUTES)
ess1 = numeric(N_COMPUTES)

se2 = numeric(N_COMPUTES)
ess2 = numeric(N_COMPUTES)

se3 = numeric(N_COMPUTES)
ess3 = numeric(N_COMPUTES)

for (i in 1:N_COMPUTES) {
  shift = SHIFT * i
  se1[i] = max(mcse.multi(betas1[1:shift, ], size = BATCH_SIZE, method = METHOD, r = 1)$eigen_values)
  ess1[i] = multiESS(betas1[1:shift, ])

  se2[i] = max(mcse.multi(betas2[1:shift, ], size = BATCH_SIZE, method = METHOD, r = 1)$eigen_values)
  ess2[i] = multiESS(betas2[1:shift, ])

  se3[i] = max(mcse.multi(betas3[1:shift, ], size = BATCH_SIZE, method = METHOD, r = 1)$eigen_values)
  ess3[i] = multiESS(betas3[1:shift, ])
}

csv = data.frame(se1 = betas1_se, 
                 ess1 = betas1_ess,
                 se2 = betas2_se, 
                 ess3 = betas2_ess,
                 se3 = betas3_se, 
                 ess3 = betas3_ess)
write.csv(csv, "toy_se.csv", row.names = FALSE)



