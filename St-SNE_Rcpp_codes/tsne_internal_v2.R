.Hbeta <- function(D, W, beta){
  ## this is to calcuate density in high dimensional space. P(ij)
  ## P is the P(ij)
  ## H is waht? the H in the paper. But the difference is in the paper, the entropy is on the 2 base
  ## what is beta? it is the inverse of sigmasq
	P = exp(-D * beta)*W
	sumP = sum(P)
	if (sumP == 0){
		H = 0
		P = D * 0
	} else {
		H = log(sumP) + beta * sum(D %*% P) /sumP
		P = P/sumP
	}
	r = {}
	r$H = H
	r$P = P
	r
}

.x2p <- function(X,W,perplexity = 15,tol = 1e-5){
  
  ## first check the input matrix X:
  ## If it is a dist matrix, then keep it
  ## If it is data matrix, then change it to dist matrix.
	if (class(X) == 'dist') {
		D = X;		
		#n = attr(D,'Size')
	} else{
		D = dist(X);		#n = attr(D,'Size')
	}
  n = nrow(X)
	D = as.matrix(D)
	P = matrix(0, n, n )		
	beta = rep(1, n)
	logU = log(perplexity)
	
	for (i in 1:n){
		betamin = -Inf
		betamax = Inf
		Di = D[i, -i]
		Wi = W[i, -i]
		hbeta = .Hbeta(Di,Wi, beta[i])
		H = hbeta$H; 
		thisP = hbeta$P
		Hdiff = H - logU;
		tries = 0;

		while(abs(Hdiff) > tol && tries < 50){
			if (Hdiff > 0){
				betamin = beta[i]
				if (is.infinite(betamax)) beta[i] = beta[i] * 2
				else beta[i] = (beta[i] + betamax)/2
			} else{
				betamax = beta[i]
				if (is.infinite(betamin))  beta[i] = beta[i]/ 2
				else beta[i] = ( beta[i] + betamin) / 2
			}
			
			hbeta = .Hbeta(Di, Wi, beta[i])
			H = hbeta$H
			thisP = hbeta$P
			Hdiff = H - logU
			tries = tries + 1
		}	
			P[i,-i]  = thisP	
	}	
	
	r = {}
	r$P = P
	r$beta = beta
	sigma = sqrt(1/beta)
	if(sum(sigma)<1e-2) sigma = .5
	
	#message('sigma summary: ', paste(names(summary(sigma)),':',summary(sigma),'|',collapse=''))

	r 
}


.Z2Q <- function(zdata){
  eps = 2^(-52) # typical machine precision
  D = as.matrix(stats::dist(zdata)); 
  Ep =  1/(1 + D^2) ## num = 1/(1+|z_i - z_j|^2) it is a matrix
  diag(Ep)=0
  Zp = sum(Ep)
  Q = Ep / Zp
  if (any(is.nan(Ep))) message ('NaN in grad. descent')
  Q[Q < eps] = eps
  
  r = {}
  r$D = D
  r$Q = Q
  r$Ep = Ep
  r$Zp = Zp
  
  return(r)
}

.Z2Q_one <- function(zdata,D0,Ep0,Zp0){
  n = nrow(zdata)
  eps = 2^(-52) # typical machine precision
  #D = as.matrix(dist(zdata)); ## change
  D = D0
  d = sqrt(apply((t(zdata[2:n,])-zdata[1,])^2,2,sum))
  D[1,-1] = d
  D[-1,1] = d

  Ep = Ep0
  ep =  1/(1 + d^2)
  Ep[1,-1] = ep
  Ep[-1,1] = ep
  diag(Ep)=0
  
  Zp = Zp0 + 2*sum(ep)
  Q = Ep / Zp
  if (any(is.nan(Ep))) message ('NaN in grad. descent')
  Q[Q < eps] = eps
  
  r = {}
  r$D = D
  r$Q = Q
  r$Ep = Ep
  r$Zp = Zp
  
  return(r)
}

.Y2O <- function(Y,W){
  eps = 2^(-52) # typical machine precision
  D = stats::dist(Y)
  D = as.matrix(D)
  E = exp(-D^2)*W
  diag(E) = 0
  Z = apply(E,1,sum)
  O = E/Z
  O = .5 * (O + t(O)); O[O < eps]<-eps;	O = O/sum(O)
  if (any(is.nan(E))) message ('NaN in grad. descent')
  O[O < eps] = eps
  
  r = {}
  r$O = O
  r$E = E
  r$D = D
  r$Z = Z
  
  return(r) 
}


.whiten <- function(X, row.norm=FALSE, verbose=FALSE, n.comp=ncol(X))
{  
  ## to standardize the columns
	n.comp; # forces an eval/save of n.comp
	if (verbose) message("Centering")
   n = nrow(X)
	p = ncol(X)
	X <- scale(X, scale = FALSE)
   X <- if (row.norm) 
       t(scale(X, scale = row.norm))
   else t(X)

   if (verbose) message("Whitening")
   V <- X %*% t(X)/n
   s <- La.svd(V)
   D <- diag(c(1/sqrt(s$d)))
   K <- D %*% t(s$u)
   K <- matrix(K[1:n.comp, ], n.comp, p)
   X = t(K %*% X)
	X
}

.whiten_v2 <- function(X, scale=FALSE){  
  ## to standardize the columns

  X <- scale(X, scale = scale)
  return(X)
}

.cost <- function(P,Y,W,zdata,rho=0.5,eps = 2^(-52)){
  O = .Y2O(Y,W)$O
  
  rz = .Z2Q(zdata)
  Q = rz$Q

  cost =  rho*sum(apply(P * log((P+eps)/(Q+eps)),1,sum)) + (1-rho)*sum(apply(O * log((O+eps)/(Q+eps)),1,sum)) 
  return(cost)
}

## cost2 does not depend on X
.cost2 <- function(Y,W,zdata,eps = 2^(-52)){
  O = .Y2O(Y,W)$O
  
  rz = .Z2Q(zdata)
  Q = rz$Q
  
  cost =  sum(apply(Q * log(1/(O+eps)),1,sum)) 
  return(cost)
}

