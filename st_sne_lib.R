### this code is modified based on the source code of t-SNE
### please visit https://lvdmaaten.github.io/tsne/ for reference
###
### Author: Yichen Cheng, Nov, 17, 2019
###         Email: ycheng11@gsu.edu
###
### Stage 1 functions are related with st-sne training
###        function: stsne1_conv
### Stages 2 and 3 functions are for prediction using testing data
###        functions: stage_2_tsne, stage_2_tsne_v2, stsne_pred

## stage 1
stsne1_conv <- function(X,Y,train.id,k=2,perp.x=30,niter=500,max_iter = 500,whiten=TRUE,epoch=100,rho=.1){
  #######################################################################
  ## standardize X and Y
  #######################################################################
  initial_dims = ncol(X)
  #if (whiten) X<-.whiten(as.matrix(X),n.comp=initial_dims)
  n.tot = nrow(X)
  pre.id = setdiff(1:n.tot,train.id)
  Y = as.matrix(Y)
  
  #######################################################################
  ## set parameters
  #######################################################################
  momentum = .5
  final_momentum = .8
  
  epsilon = 500
  min_gain = .01
  #max_iter = 5e3
  eps = 2^(-52) # typical machine precision
  
  #######################################################################
  ## generate zdata
  #######################################################################
  zdata = matrix(rnorm(k * n.tot),n.tot)
  
  #######################################################################
  ## intialize n and matrices
  #######################################################################
  n.train = length(train.id)
  grads.t =  matrix(0,n.train,ncol(zdata)) ## gradient for training wrt z
  incs.t =  matrix(0,n.train,ncol(zdata))
  gains.t = matrix(1,n.train,ncol(zdata))
  
  
  ###########################################################################
  ##
  ##  Part I: Fitting the training data set to obtain zdata.t for S1
  ##
  ###########################################################################
  
  zdata.t = zdata[train.id,]
  
  P.t = .x2p(X[train.id,],perp.x, 1e-5)$P
  P.t = .5 * (P.t + t(P.t));    P.t[P.t < eps]<-eps;    P.t = P.t/sum(P.t)
  P.t = P.t*4
  
  O.t = .Y2O(Y[train.id,])$O;
  O.t = O.t*4
  for (iter in 1:niter){
    ###########################################################################
    ## calculate grad for z train.
    ###########################################################################
    rz.t = .Z2Q(zdata.t)
    Q.t = rz.t$Q
    Ep.t = rz.t$Ep
    
    stiffnesses = 4 * (P.t*rho + O.t*(1-rho) -Q.t) * Ep.t
    for (i in 1:n.train) grads.t[i,] = apply(sweep(-zdata.t, 2, -zdata.t[i,]) * stiffnesses[,i],2,sum)
    
    ###########################################################################
    ## update zdata.t
    ###########################################################################
    gains.t = ((gains.t + .2) * abs(sign(grads.t) != sign(incs.t)) + gains.t * .8 * abs(sign(grads.t) == sign(incs.t)))
    gains.t[gains.t < min_gain] = min_gain
    incs.t = momentum * incs.t - epsilon * (gains.t * grads.t)
    zdata.t = zdata.t + incs.t
    zdata.t = sweep(zdata.t,2,apply(zdata.t,2,mean)) ## we should center zdata at stage 1, but no need to center zdata at stage 2, since at stage 2, only part of zdata is updated.
    if (iter == round(niter/2)) momentum = final_momentum
    if (iter == 100) {P.t = P.t/4; O.t = O.t/4}
  }
  
  while(max(abs(incs.t))>1e-2&&iter<max_iter){
    momentum = 0.5
    iter = iter+1
    rz.t = .Z2Q(zdata.t)
    Q.t = rz.t$Q
    Ep.t = rz.t$Ep
    
    stiffnesses = 4 * (P.t*rho + O.t*(1-rho) -Q.t) * Ep.t
    for (i in 1:n.train) grads.t[i,] = apply(sweep(-zdata.t, 2, -zdata.t[i,]) * stiffnesses[,i],2,sum)
    
    gains.t = ((gains.t + .2) * abs(sign(grads.t) != sign(incs.t)) + gains.t * .8 * abs(sign(grads.t) == sign(incs.t)))
    gains.t[gains.t < min_gain] = min_gain
    incs.t = momentum * incs.t - epsilon * (gains.t * grads.t)
    zdata.t = zdata.t + incs.t
    zdata.t = sweep(zdata.t,2,apply(zdata.t,2,mean)) ## we should center zdata at stage 1, but no need to center zdata at stage 2, since at stage 2, only part of zdata is updated.
  }
  
  cost =  rho*sum(apply(P.t * log((P.t+eps)/(Q.t+eps)),1,sum)) + (1-rho)*sum(apply(O.t * log((O.t+eps)/(Q.t+eps)),1,sum)) # P for high dimension, Q for low dim, O for outcome
  cost
  max(abs(incs.t))
  
  zdata[train.id,] = zdata.t
  #plot(zdata[train.id,],col=Y[train.id]+1)
  res = list()
  res$zdata = zdata
  res$cost = cost
  res$incs = incs.t
  res$niter = iter
  return(res)
}

## stage 2
stage_2_tsne <- function(X,Y,zdata,train.id,pred.id,D0,Ep0,Zp0,k=2,perp.x=30,niter = niter2,max_iter = 500,epoch_callback=NULL,epoch=epoch2,rho=.1){
  eps = 2^(-52) # typical machine precision
  n.pred = 1 ## the prediction is done one by one
  
  grads.p =  matrix(0,n.pred,ncol(zdata)) ## gradient for prediction wrt z
  incs.p =  matrix(0,n.pred,ncol(zdata))
  gains.p = matrix(1,n.pred,ncol(zdata))
  
  # first one is the prediction one
  tot.id = c(pred.id,train.id)
  Xnow = X[tot.id,]
  Ynow = Y[tot.id]
  zdatanow = zdata[tot.id,]
  
  # first use KNN to initialize zdata
  ##################################
  ##
  ##  change place one: can we not recalculate the P every time?
  ##
  ####################################
  
  
  P = .x2p(Xnow,perp.x, 1e-5)$P
  P = .5 * (P + t(P));  P[P < eps]<-eps;        P = P/sum(P)
  
  Dx = as.matrix(dist(Xnow))
  th = sort(Dx[1,-1])[4]
  id = which(Dx[1,-1]<th+1e-6)
  zdatanow[1,] = apply(zdatanow[id+1,],2,mean)
  
  ##################################
  ##
  ##  change place one: the function .z2q include a known part and unkonwn part?
  ##
  ####################################
  
  rz = .Z2Q_one(zdatanow,D0,Ep0,Zp0)
  Dp = rz$D;
  Q = rz$Q
  Ep = rz$Ep;
  Zp = rz$Zp
  
  epsilon = 500
  momentum = .1
  final_momentum = .2
  mom_switch_iter = 250
  
  min_gain = .01
  #max_iter = 1e4
  P = P*4
  
  for (iter in 1:niter){
    if (iter %% epoch == 0) { # epoch
      # P for high dimension (X), Q for low dim (zdata), O for outcome (Y)
      cost1 =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
      #message("Epoch: prediction part I Iteration #",iter," error is: ",cost1)
      if (!is.null(epoch_callback)) epoch_callback(zdata)
    }
    
    ###########################################################################
    ## calculate gradient for z pred
    ###########################################################################
    rz = .Z2Q_one(zdatanow,D0,Ep0,Zp0)
    Q = rz$Q
    Ep = rz$Ep
    stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    
    ###########################################################################
    ## update zdata.t
    ###########################################################################
    gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    gains.p[gains.p < min_gain] = min_gain
    incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    zdata.p = zdatanow[1,]
    zdata.p = zdata.p + incs.p
    zdatanow[1,] = zdata.p
    
    if (iter == niter/2) momentum = final_momentum
    if (iter == 100) {P = P/4}
  }
  
  while( (max(abs(incs.p))>1e-2)&&(iter < max_iter)){
    iter = iter+1
    ###########################################################################
    ## calculate gradient for z pred
    ###########################################################################
    rz = .Z2Q_one(zdatanow,D0,Ep0,Zp0)
    Q = rz$Q
    Ep = rz$Ep
    stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    
    ###########################################################################
    ## update zdata.t
    ###########################################################################
    gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    gains.p[gains.p < min_gain] = min_gain
    incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    zdata.p = zdatanow[1,]
    zdata.p = zdata.p + incs.p
    zdatanow[1,] = zdata.p
  }
  
  cost1 =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
  res = list()
  res$cost = cost1
  res$zdata = zdatanow
  res$niter = iter
  res$incs.p = incs.p
  
  return(res)
}

stage_2_tsne_v2 <- function(X,Y,zdata,train.id,pred.id,k=2,perp.x=30,niter = niter2,max_iter = 500,epoch_callback=NULL,epoch=epoch2,rho=.1){
  eps = 2^(-52) # typical machine precision
  n.pred = 1 ## the prediction is done one by one
  
  grads.p =  matrix(0,n.pred,ncol(zdata)) ## gradient for prediction wrt z
  incs.p =  matrix(0,n.pred,ncol(zdata))
  gains.p = matrix(1,n.pred,ncol(zdata))
  
  # first one is the prediction one
  tot.id = c(pred.id,train.id)
  Xnow = X[tot.id,]
  Ynow = Y[tot.id]
  zdatanow = zdata[tot.id,]
  
  # first use KNN to initialize zdata
  ##################################
  ##
  ##  change place one: can we not recalculate the P every time?
  ##
  ####################################
  
  
  P = .x2p(Xnow,perp.x, 1e-5)$P
  P = .5 * (P + t(P));  P[P < eps]<-eps;        P = P/sum(P)
  
  Dx = as.matrix(dist(Xnow))
  th = sort(Dx[1,-1])[4]
  id = which(Dx[1,-1]<th)
  zdatanow[1,] = apply(zdatanow[id+1,],2,mean)
  
  ##################################
  ##
  ##  change place one: the function .z2q include a known part and unkonwn part?
  ##
  ####################################
  
  rz = .Z2Q(zdatanow)
  Dp = rz$D;
  Q = rz$Q
  Ep = rz$Ep;
  Zp = rz$Zp
  epsilon = 500
  momentum = .1
  final_momentum = .2
  mom_switch_iter = 250
  
  min_gain = .01
  #max_iter = 1e4
  P = P*4
  
  for (iter in 1:niter){
    if (iter %% epoch == 0) { # epoch
      # P for high dimension (X), Q for low dim (zdata), O for outcome (Y)
      cost1 =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
      #message("Epoch: prediction part I Iteration #",iter," error is: ",cost1)
      if (!is.null(epoch_callback)) epoch_callback(zdata)
    }
    
    ###########################################################################
    ## calculate gradient for z pred
    ###########################################################################
    rz = .Z2Q(zdatanow)
    Q = rz$Q
    Ep = rz$Ep
    stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    
    ###########################################################################
    ## update zdata.t
    ###########################################################################
    gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    gains.p[gains.p < min_gain] = min_gain
    incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    zdata.p = zdatanow[1,]
    zdata.p = zdata.p + incs.p
    zdatanow[1,] = zdata.p
    
    if (iter == niter/2) momentum = final_momentum
    if (iter == 100) {P = P/4}
  }
  
  while( (max(abs(incs.p))>1e-2)&&(iter < max_iter)){
    iter = iter+1
    ###########################################################################
    ## calculate gradient for z pred
    ###########################################################################
    rz = .Z2Q(zdatanow)
    Q = rz$Q
    Ep = rz$Ep
    stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    
    ###########################################################################
    ## update zdata.t
    ###########################################################################
    gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    gains.p[gains.p < min_gain] = min_gain
    incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    zdata.p = zdatanow[1,]
    zdata.p = zdata.p + incs.p
    zdatanow[1,] = zdata.p
  }
  
  cost1 =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
  res = list()
  res$cost = cost1
  res$zdata = zdatanow
  res$niter = iter
  res$incs.p = incs.p
  
  return(res)
}

## stage 3
stsne_pred <- function(Y,zdata,train.id,pred.id,k=2){
  eps = 2^(-52) # typical machine precision
  Ynow = c(Y[pred.id],Y[train.id])
  zdatanow = rbind(zdata[pred.id,],zdata[train.id,])
  
  ry = .Y2O(Ynow)
  O = ry$O;
  E = ry$E; E[E < eps]<-eps
  D = ry$D;
  Z = ry$Z
  
  rz = .Z2Q(zdatanow)
  Dp = rz$D;
  Q = rz$Q
  Ep = rz$Ep;
  Zp = rz$Zp
  
  y0 = Ynow; y0[1] = 0; c0 = .cost2(y0,zdatanow)
  y1 = Ynow; y1[1] = 1; c1 = .cost2(y1,zdatanow)
  if(c1>c0){Ynow = y0;Y[pred.id]=0}else{Ynow = y1;Y[pred.id]=1}
  
  #cost2 = .cost2(Ynow,zdatanow)
  #cost2 =  sum(apply(Q * log(1/(O+eps)),1,sum))
  #message("Epoch: prediction error is: ",cost2)
  #print(Y[pred.id])
  
  return(Y[pred.id])
}




## dist version of the stsne implementation (one of the weighted version)
stsne_tsne_dist = function(X,Y,train.id,pre.id,perp.x,rho,niter1,niter2,max_iter = 500, epoch = 1000, epoch2 = 1000){
  n = length(Y)
  p = ncol(X)
  ypred = c()
  
  #cor1 = cor(X,Y)
  #X = t(t(X)*c(cor1))
  Ytr = Y[train.id]
  Xtr = X[train.id,]
  n1 = length(train.id)
  cor = c()
  Bt = Bt = matrix(abs(rep(Ytr,n1)-rep(Ytr,each=n1)),n1)
  m1 = apply(Bt,1,mean)
  m2 = apply(Bt,2,mean)
  B = Bt - m1 - matrix(rep(m2,each=n1),n1)+mean(Bt)
  
  for(i in 1:p){
    At = matrix(abs(rep(Xtr[,i],n1)-rep(Xtr[,i],each=n1)),n1)
    m1 = apply(At,1,mean)
    m2 = apply(At,2,mean)
    A = At - m1 - matrix(rep(m2,each=n1),n1)+mean(At)
    cor[i] = mean(A*B)
  }
  X = t(t(X)*c(cor)/max(cor))
  
  print("stage 1 start")
  #stage 1
  res0 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  res1 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  if(res0$cost<res1$cost){res = res0}else{res = res1}
  zdata = res$zdata
  print("stage 1 finish")
  ## at the end of stage 1, calculate D0, Ep0 and Zp0
  zdatatemp = rbind(zdata[train.id[1],],zdata[train.id,])
  D0 = as.matrix(dist(zdatatemp));
  Ep0 =  1/(1 + D0^2) ## num = 1/(1+|z_i - z_j|^2) it is a matrix
  diag(Ep0)=0
  Zp0 = sum(Ep0)
  print("stage 2 :")
  #stage 2
  for(pred.id in pre.id){
    # print(pred.id)
    res2 = stage_2_tsne(X,Y,zdata,train.id,pred.id,D0,Ep0,Zp0,perp.x=perp.x,niter = niter2,max_iter = max_iter,epoch=epoch2,rho=rho)
    zdata[pred.id,] = res2$zdata[1,]
  } #5s for n=100
  #plot(zdata,col=Y+1)
  
  #stage 3, use a loop to update Y[pre.id]
  print("stage 3 :")
  for(pred.id in pre.id){
    #  print(pred.id)
    ypred[pred.id] = stsne_pred(Y,zdata,train.id,pred.id,k=2)
  }
  ## store the results
  res_tsne = list()
  res_tsne$pre.id = pre.id
  res_tsne$zdata = zdata
  res_tsne$ypred = ypred
  return(res_tsne)
  
}

## max version of the stsne implementation (weighted version)
stsne_tsne_max = function(X,Y,train.id,pre.id,perp.x,rho,niter1,niter2,max_iter = 500,epoch = 1000, epoch2 = 1000){
  n = length(Y)
  p = ncol(X)
  ypred = c()
  
  Ytr = Y[train.id]
  Xtr = X[train.id,]
  cor1 = cor(Xtr,Ytr)
  cor2 = cor(Xtr^2,Ytr)
  X = t(t(X)*c(pmax(abs(cor1),abs(cor2))))
  
  print("stage 1 start")
  #stage 1
  res0 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  res1 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  if(res0$cost<res1$cost){res = res0}else{res = res1}
  zdata = res$zdata
  print("stage 1 finish")
  ## at the end of stage 1, calculate D0, Ep0 and Zp0
  zdatatemp = rbind(zdata[train.id[1],],zdata[train.id,])
  D0 = as.matrix(dist(zdatatemp));
  Ep0 =  1/(1 + D0^2) ## num = 1/(1+|z_i - z_j|^2) it is a matrix
  diag(Ep0)=0
  Zp0 = sum(Ep0)
  print("stage 2 :")
  #stage 2
  for(pred.id in pre.id){
    # print(pred.id)
    res2 = stage_2_tsne(X,Y,zdata,train.id,pred.id,D0,Ep0,Zp0,perp.x=perp.x,niter = niter2,max_iter = max_iter,epoch=epoch2,rho=rho)
    zdata[pred.id,] = res2$zdata[1,]
  } #5s for n=100
  #plot(zdata,col=Y+1)
  
  #stage 3, use a loop to update Y[pre.id]
  print("stage 3 :")
  for(pred.id in pre.id){
    #  print(pred.id)
    ypred[pred.id] = stsne_pred(Y,zdata,train.id,pred.id,k=2)
  }
  ## store the results
  res_tsne = list()
  res_tsne$pre.id = pre.id
  res_tsne$zdata = zdata
  res_tsne$ypred = ypred
  return(res_tsne)
  
}

## unweighted version of St-SNE

stsne_tsne = function(X,Y,train.id,pre.id,perp.x,rho,niter1,niter2,max_iter = 500,epoch = 1000, epoch2 = 1000){
  n = length(Y)
  p = ncol(X)
  ypred = c()
  
  print("stage 1 start")
  #stage 1
  res0 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  res1 = stsne1_conv(X,Y,train.id,niter=niter1,max_iter = max_iter,rho=rho,epoch=epoch,perp.x=perp.x) # 10s for n=100
  if(res0$cost<res1$cost){res = res0}else{res = res1}
  zdata = res$zdata
  print("stage 1 finish")
  ## at the end of stage 1, calculate D0, Ep0 and Zp0
  zdatatemp = rbind(zdata[train.id[1],],zdata[train.id,])
  D0 = as.matrix(dist(zdatatemp));
  Ep0 =  1/(1 + D0^2) ## num = 1/(1+|z_i - z_j|^2) it is a matrix
  diag(Ep0)=0
  Zp0 = sum(Ep0)
  print("stage 2 :")
  #stage 2
  for(pred.id in pre.id){
    # print(pred.id)
    res2 = stage_2_tsne(X,Y,zdata,train.id,pred.id,D0,Ep0,Zp0,perp.x=perp.x,niter = niter2,max_iter = max_iter,epoch=epoch2,rho=rho)
    zdata[pred.id,] = res2$zdata[1,]
  } #5s for n=100
  #plot(zdata,col=Y+1)
  
  #stage 3, use a loop to update Y[pre.id]
  print("stage 3 :")
  for(pred.id in pre.id){
    #  print(pred.id)
    ypred[pred.id] = stsne_pred(Y,zdata,train.id,pred.id,k=2)
  }
  ## store the results
  res_tsne = list()
  res_tsne$pre.id = pre.id
  res_tsne$zdata = zdata
  res_tsne$ypred = ypred
  return(res_tsne)
}