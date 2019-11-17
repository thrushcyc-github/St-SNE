simu_twosided = function(n,p){
  y = rbinom(n,size=1,p=.5)
  mux = y*1*(2*rbinom(n,size=1,p=.5)-1)
  Xo = matrix(rnorm(n*p),n)
  X = Xo 
  X[,1:10] = Xo[,1:10] + mux
  Y = matrix(y,length(y))
  data = list()
  data$X = X
  data$Y = Y
  return(data)
}
