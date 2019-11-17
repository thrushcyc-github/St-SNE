# a working example of St-SNE

setwd("E:/Dropbox/research/Shared_folder/SmuGsu/dimension_reduction/Rcode_organized_for_github/")
setwd("C:/Users/ycheng11/Dropbox/research/Shared_folder/SmuGsu/dimension_reduction/Rcode_organized_for_github/")
source("simu_data.R")
source("tsne_internel.R")
source("st_sne_lib.R")

n = 100
p = 20

data = simu_twosided(n=n,p=p)
X = data$X
Y = data$Y

train.id = 1:95
pre.id = setdiff(1:n,train.id)

## for training (dimension reduction)
res_tsne = stsne1_conv(X,Y,train.id,perp.x=50,rho = .3,niter=100,max_iter = 500) ## try different values of rho between 0 and 1
res_tsne$zdata ## data with reduced dimension

## for prediction

## unweighted version
res_tsne = stsne_tsne(X,Y,train.id,pre.id,perp.x=50,rho = .3,niter1=100,niter2=100,max_iter = 500) ## try different values of rho between 0 and 1
mean(abs(Y[pre.id] - res_tsne$ypred[pre.id]))

## weighted versions
res_tsne_max = stsne_tsne_max(X,Y,train.id,pre.id,perp.x=50,rho = .3,niter1=100,niter2=100,max_iter = 500) ## try different values of rho between 0 and 1
res_tsne_dist = stsne_tsne_dist(X,Y,train.id,pre.id,perp.x=50,rho = .3,niter1=100,niter2=100,max_iter = 500) ## try different values of rho between 0 and 1
