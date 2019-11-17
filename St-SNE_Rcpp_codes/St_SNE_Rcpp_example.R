## Created by Seongoh Park, Nov 17,2019
## Email: inmybrain6@gmail.com
## modified from the R version of the St-SNE code.

rm(list = ls())
library("Rcpp")
library("RcppArmadillo")


source("tsne_internal_v2.R")
source("simu_data_lib.R")

## cpp functions
### Replace by cpp functions iterative functions within a function for each stage
sourceCpp("StSNE_benchmark/group_tSNE_FUN_v1.0.cpp")
## Replace functions for stage 1,2,3 and the main function
source("stsne_wrappers.R")

## generate data, first 100 are the training set; the last 50 are the test set
n = 100
p = 20
train.id = 1:95
pre.id = setdiff(1:n,train.id)
data = simu_twosided(n=n,p=p)
X = data$X
Y = data$Y

W = matrix(rep(1,n*n),n)


res0 = stsne1_conv_cpp(X = X,Y = Y,W = W, k = 2, train.id = train.id, niter = 100, rho = .3,
                       epoch = epoch, perp.x = 50, max_iter = 500, show_figure = FALSE) 

