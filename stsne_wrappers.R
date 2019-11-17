## Created by Seongoh Park, Nov 17,2019
## Email: inmybrain6@gmail.com
## modified from the R version of the St-SNE code.

stsne1_conv_cpp <- function(X,Y,W,train.id,k=2,perp.x=30,niter=500,epoch=100,rho=.1, max_iter, show_figure = FALSE){
  # ## generate data, first 100 are the training set; the last 50 are the test set
  # train.id = 1:100
  # pre.id = 101:150
  # data = simu_twosided(n=150,p=30)
  # X = data$X
  # Y = data$Y
  # ## then generate the study ID and calculate the weight matrix
  # S = sample(1:3,150,replace = TRUE)
  # s = 0.3
  # W = as.matrix(stats::dist(S))
  # W = (W==0)+s*(W!=0)
  # k=2
  # perp.x=30
  # niter=500
  # epoch=100
  # rho=.1
  
  #######################################################################
  ## standardize X and Y
  #######################################################################
  initial_dims = ncol(X)
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
  # max_iter = 5e3
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
  
  P.t = .x2p(X[train.id,,drop=FALSE], W[train.id,train.id], perp.x, 1e-5)$P
  P.t = .5 * (P.t + t(P.t));	P.t[P.t < eps]<-eps;	P.t = P.t/sum(P.t)
  P.t = P.t*4
  
  O.t = .Y2O(Y[train.id,,drop=FALSE],W[train.id,train.id])$O;	
  O.t = O.t * 4
  
  res_cpp <- list()
  res_cpp$grads_t <- grads.t
  res_cpp$zdata_t <- zdata.t
  res_cpp$gains_t <- gains.t
  res_cpp$incs_t <- incs.t
  for(iter in 1:niter){
    # list_res_cpp[[i]] <- res_cpp
    res_cpp <- stage1_iteration(
      P_t = P.t,
      rho = rho,
      O_t = O.t,
      # Q_t = rz.t$Q,
      # Ep_t = rz.t$Ep,
      
      grads_t = res_cpp$grads_t,
      zdata_t = res_cpp$zdata_t,
      gains_t = res_cpp$gains_t,
      incs_t  = res_cpp$incs_t,
      
      min_gain = min_gain,
      momentum = momentum,
      epsilon = epsilon
    )
    if (iter == round(niter/2)) momentum = final_momentum
    if (iter == 100) {P.t = P.t/4; O.t = O.t/4}
  }
  momentum = 0.5
  while(max(abs(res_cpp$incs_t))>1e-2&&iter<max_iter){
    iter = iter+1
    res_cpp <- stage1_iteration(
      P_t = P.t,
      rho = rho,
      O_t = O.t,
      # Q_t = rz.t$Q,
      # Ep_t = rz.t$Ep,
      
      grads_t = res_cpp$grads_t,
      zdata_t = res_cpp$zdata_t,
      gains_t = res_cpp$gains_t,
      incs_t  = res_cpp$incs_t,
      
      min_gain = min_gain,
      momentum = momentum,
      epsilon = epsilon
    )
  }
  
  cost =  rho*sum(apply(P.t * log((P.t+eps)/(res_cpp$Q_t+eps)),1,sum)) + 
    (1-rho)*sum(apply(O.t * log((O.t+eps)/(res_cpp$Q_t+eps)),1,sum)) # P for high dimension, Q for low dim, O for outcome
  # cost
  # max(abs(res_cpp$incs.t))
  
  zdata[train.id,] = res_cpp$zdata_t
  if(show_figure){
    plot(zdata[train.id,],col=Y[train.id]+1)
  }
  
  res = list()
  res$zdata = zdata
  res$cost = cost
  res$incs = res_cpp$incs_t
  res$niter = iter
  return(res)
}

stage_2_tsne_cpp <- function(X,Y,W,zdata,train.id, k=2, pred.id,D0,Ep0,Zp0,perp.x=30,niter,epoch_callback=NULL,epoch=epoch2,rho=.1, max_iter){
  eps = 2^(-52) # typical machine precision
  n.pred = 1 ## the prediction is done one by one
  
  grads.p =  matrix(0,n.pred,ncol(zdata)) ## gradient for prediction wrt z
  incs.p =  matrix(0,n.pred,ncol(zdata))
  gains.p = matrix(1,n.pred,ncol(zdata))
  
  # first one is the prediction one
  tot.id = c(pred.id,train.id)
  Xnow = X[tot.id,]
  Ynow = Y[tot.id]
  Wnow = W[tot.id,tot.id]
  zdatanow = zdata[tot.id,]
  
  # first use KNN to initialize zdata
  ##################################
  ## 
  ##  change place one: can we not recalculate the P every time?
  ##
  ####################################
  
  
  P = .x2p(Xnow,Wnow,perp.x, 1e-5)$P
  P = .5 * (P + t(P));	P[P < eps]<-eps;	P = P/sum(P)
  
  Dx = as.matrix(stats::dist(Xnow))
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
  # max_iter = 1e4
  P = P*4
  
  for (iter in 1:niter){
    if (iter %% epoch == 0) { # epoch
      # P for high dimension (X), Q for low dim (zdata), O for outcome (Y)
      cost1 =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
      #message("Epoch: prediction part I Iteration #",iter," error is: ",cost1)
      if (!is.null(epoch_callback)) epoch_callback(zdata)
    }
    
    # ###########################################################################
    # ## calculate gradient for z pred
    # ###########################################################################
    # rz = .Z2Q_one(zdatanow,D0,Ep0,Zp0)
    # Q = rz$Q
    # Ep = rz$Ep
    # stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    # grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    # 
    # ###########################################################################
    # ## update zdata.t
    # ###########################################################################
    # gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    # gains.p[gains.p < min_gain] = min_gain
    # incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    # zdata.p = zdatanow[1,]
    # zdata.p = zdata.p + incs.p
    # zdatanow[1,] = zdata.p
    res_cpp <- list()
    res_cpp$grads_p <- grads.p
    res_cpp$zdatanow <- zdatanow
    res_cpp$gains_p <- gains.p
    res_cpp$incs_p <- incs.p
    res_cpp <- stage2_iteration(
      D0 = D0,
      Ep0 = Ep0,
      Zp0 = Zp0,
      P = P,
      
      zdatanow = res_cpp$zdatanow,
      grads_p = res_cpp$grads_p,
      gains_p = res_cpp$gains_p,
      incs_p = res_cpp$incs_p,
      
      min_gain = min_gain,
      momentum = momentum,
      epsilon = epsilon
    )
    if (iter == niter/2) momentum = final_momentum
    if (iter == 100) {P = P/4}
  }
  
  while( (max(abs(res_cpp$incs_p))>1e-2)&&(iter < max_iter)){
    iter = iter+1
    # ###########################################################################
    # ## calculate gradient for z pred
    # ###########################################################################
    # rz = .Z2Q_one(zdatanow,D0,Ep0,Zp0)
    # Q = rz$Q
    # Ep = rz$Ep
    # stiffnesses = 4 * (P - Q) * Ep ## stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
    # grads.p[1,] = apply(sweep(-zdatanow, 2, -zdatanow[1,]) * stiffnesses[,1],2,sum)
    # 
    # ###########################################################################
    # ## update zdata.t
    # ###########################################################################
    # gains.p = ((gains.p + .2) * abs(sign(grads.p) != sign(incs.p)) + gains.p * .8 * abs(sign(grads.p) == sign(incs.p)))
    # gains.p[gains.p < min_gain] = min_gain
    # incs.p = momentum * incs.p - epsilon * (gains.p * grads.p)
    # zdata.p = zdatanow[1,]
    # zdata.p = zdata.p + incs.p
    # zdatanow[1,] = zdata.p
    res_cpp <- stage2_iteration(
      D0 = D0,
      Ep0 = Ep0,
      Zp0 = Zp0,
      P = P,
      
      zdatanow = res_cpp$zdatanow,
      grads_p = res_cpp$grads_p,
      gains_p = res_cpp$gains_p,
      incs_p = res_cpp$incs_p,
      
      min_gain = min_gain,
      momentum = momentum,
      epsilon = epsilon
    )
  }
  
  cost1 =  sum(apply(P * log((P+eps)/(res_cpp$Q+eps)),1,sum))
  res = list()
  res$cost = cost1
  res$zdata = res_cpp$zdatanow
  res$niter = iter
  res$incs.p = res_cpp$incs_p
  
  return(res)
}


stsne_tsne_cpp = function(X,Y,W, k, train.id,pre.id,perp.x,rho,niter1,niter2,epoch = 1000, epoch2 = 1000, 
                          repredict = FALSE, whiten = TRUE, weight = "Identidy",
                          max_iter1 = 5000, max_iter2 = 10000, show_figure = FALSE){
  # X = coord_high[,!colnames(coord_high) %in% c("study", "y"), drop = F]
  # Y = coord_high$y
  # W = W
  # train.id = 1:nrow(coord_high)
  # weight = "Identidy"
  # pre.id = NULL
  # perp.x = 50
  # rho = .5
  # niter1 = 1000
  # niter2 = 500
  # 
  # perp.x=50
  # rho = .5
  # niter1=1000
  # niter2=500
  # repredict=FALSE
  # whiten = FALSE
  # weight = "Identidy"
  # max_iter1 = 5000
  # max_iter2 = 10000
  # show_figure = FALSE
  # epoch = 1000
  # epoch2 = 1000
  
  
  n = length(Y)
  p = ncol(X)
  ypred = c()
  
  if (whiten) X<-.whiten_v2(as.matrix(X),scale = TRUE)
  
  if(weight=="dist"){
    Ytr = Y[train.id]
    Xtr = X[train.id,]
    n1 = length(train.id)
    cor = c()
    Bt = Bt = matrix(abs(rep(Ytr,n1)-rep(Ytr,each=n1)),n1)
    m1 = apply(Bt,1,mean,na.rm=TRUE)
    m2 = apply(Bt,2,mean,na.rm=TRUE)
    B = Bt - m1 - matrix(rep(m2,each=n1),n1)+mean(Bt,na.rm=TRUE)
    
    for(i in 1:p){
      At = matrix(abs(rep(Xtr[,i],n1)-rep(Xtr[,i],each=n1)),n1)
      m1 = apply(At,1,mean,na.rm=TRUE)
      m2 = apply(At,2,mean,na.rm=TRUE)
      A = At - m1 - matrix(rep(m2,each=n1),n1)+mean(At,na.rm=TRUE)
      cor[i] = mean(A*B,na.rm=TRUE)
    }
    X = t(t(X)*c(cor)/max(cor))
  }else if(weight=="max"){
    Ytr = Y[train.id]
    Xtr = X[train.id,]
    cor1 = cor(Xtr,Ytr,use = "pairwise.complete.obs")
    cor2 = cor(Xtr^2,Ytr,use = "pairwise.complete.obs")
    X = t(t(X)*c(pmax(abs(cor1),abs(cor2))))
  }
  
  print("stage 1 start")
  #stage 1
  res0 = stsne1_conv_cpp(X = X,Y = Y,W = W, k = k, train.id = train.id, niter = niter1, rho = rho,
                         epoch = epoch, perp.x = perp.x, max_iter = max_iter1, show_figure = show_figure) # 10s for n=100
  res1 = stsne1_conv_cpp(X = X,Y = Y,W = W, k = k, train.id = train.id, niter = niter1, rho = rho,
                         epoch = epoch, perp.x = perp.x, max_iter = max_iter1, show_figure = show_figure) # 10s for n=100
  if (res0$cost < res1$cost) {
    res = res0
  } else{
    res = res1
  }
  zdata = res$zdata
  print("stage 1 finish")
  ## at the end of stage 1, calculate D0, Ep0 and Zp0
  zdatatemp = rbind(zdata[train.id[1],],zdata[train.id,])
  D0 = as.matrix(stats::dist(zdatatemp)); 
  Ep0 =  1/(1 + D0^2) ## num = 1/(1+|z_i - z_j|^2) it is a matrix
  diag(Ep0)=0
  Zp0 = sum(Ep0)
  
  if(!is.null(pre.id)){
    print("stage 2 :")
    #stage 2
    for(pred.id in pre.id){
      print(pred.id)
      res2 = stage_2_tsne_cpp(X = X, Y = Y, W = W, zdata = zdata, k = k, 
                              train.id = train.id, pred.id = pred.id, 
                              D0 = D0, Ep0 = Ep0, Zp0 = Zp0, perp.x = perp.x, 
                              niter = niter2, epoch = epoch2, rho = rho, 
                              max_iter = max_iter2)
      zdata[pred.id,] = res2$zdata[1,]
    } #5s for n=100
    
    if(show_figure){
      plot(zdata,col=Y+1)
    }
    
    #stage 3, use a loop to update Y[pre.id]
    print("stage 3 :")
    ypred <- stage_3_tsne_cpp(Y, W, zdata, train.id, pre.id, k = k)
    if(repredict){
      ypred <- stage_3_tsne_cpp(ypred, W, zdata, train.id, pre.id, k = k)
      # for(pred.id in train.id){
      #   tr.id = setdiff(train.id,pred.id)
      #   ypred[pred.id] = stsne_pred(Y,W,zdata,tr.id,pred.id,k=2)
      # }
    }
  }
  
  ## store the results
  res_tsne = list()
  res_tsne$pre.id = pre.id
  res_tsne$zdata = zdata
  res_tsne$ypred = ypred
  return(res_tsne)
}