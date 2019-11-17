// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
arma::mat dist(arma::mat X){
  // X : n-by-p matrix
  int n = X.n_rows;
  mat D = zeros(n, n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if(i != j){
        // D(i,j) = accu((X.row(i) - X.row(j)) % (X.row(i) - X.row(j)));
        // D(i,j) = accu(pow(X.row(i) - X.row(j), 2.0));
        D(i,j) = accu(square(X.row(i) - X.row(j)));
      }
    }
  }
  return sqrt(D);
  // return D;
}

// [[Rcpp::export]]
Rcpp::List Z2Q(arma::mat zdata){
  long double eps = 1.0 / (double) pow(2, 52); // typical machine precision
  // double eps = 1.0 / (double) pow(2, 20); // typical machine precision
  mat D = dist(zdata); // distance matrix
  mat Ep =  1.0 / (1.0 + square(D)); // num = 1/(1+|z_i - z_j|^2) it is a matrix
  Ep.diag().zeros();
  // // for(int j = 0; j < Ep.n_cols; j++){
  // //   Ep(j,j) = 0.0;
  // // }
  // mat Ep = zeros(D.n_rows, D.n_cols);
  // for(int i = 0; i < Ep.n_rows; i++){
  //   for(int j = 0; j < Ep.n_cols; j++){
  //     if(i != j){
  //       Ep(i,j) = 1.0 / (1.0 + D(i,j) * D(i,j));
  //     }
  //   }
  // }
  double Zp = accu(Ep);
  mat Q = Ep / Zp;
  // mat Q(Ep.n_rows, Ep.n_cols);
  // for(int i = 0; i < Q.n_rows; i++){
  //   for(int j = 0; j < Q.n_cols; j++){
  //       Q(i,j) = Ep(i,j) / Zp;
  //   }
  // }
  // Rcout << accu(Ep) << "\n";
  if (Ep.has_nan()){
    printf("NaN in grad. descent\n");
  }
  Q.elem(find(Q < eps)).fill(eps);
  // for(int i = 0; i < Q.n_rows; i++){
  //   for(int j = 0; j < Q.n_cols; j++){
  //     if(Q(i,j) < eps){
  //       Q(i,j) = eps;
  //     }
  //   }
  // }
  
  return List::create(
    Named("D") = D,
    Named("Q") = Q,
    Named("Ep") = Ep,
    Named("Zp") = Zp
  );
}
// [[Rcpp::export]]
Rcpp::List Z2Q_one(arma::mat zdata,
                   mat D0,
                   mat Ep0,
                   double Zp0){
  long double eps = 1.0 / (double) pow(2, 52); // typical machine precision
  // double eps = 1.0 / (double) pow(2, 20); // typical machine precision
  for(int i = 1; i < zdata.n_rows; i++){
    D0(0,i) = sqrt(accu(square(zdata.row(i) - zdata.row(0))));
    D0(i,i) = D0(0,i);
  }
  Ep0.row(0) = 1.0 / (1.0 + square(D0.row(0))); // using D0(0,0) = 0
  Ep0.col(0) = Ep0.row(0).t();
  Ep0.diag().zeros();
  // mat Ep =  1.0 / (1.0 + square(D)); // num = 1/(1+|z_i - z_j|^2) it is a matrix
  
  // Z2Q_one <- function(zdata,D0,Ep0,Zp0){
  // n = nrow(zdata)
  // eps = 2^(-52) # typical machine precision
  // #D = as.matrix(dist(zdata)); ## change
  // D = D0
  // d = sqrt(apply((t(zdata[2:n,])-zdata[1,])^2,2,sum))
  // D[1,-1] = d
  // D[-1,1] = d
  // 
  // Ep = Ep0
  // ep =  1/(1 + d^2)
  // Ep[1,-1] = ep
  // Ep[-1,1] = ep
  // diag(Ep)=0
  
  Zp0 += 2 * accu(Ep0.row(0));
  mat Q = Ep0 / Zp0;
  if (Ep0.has_nan()){
    printf("NaN in grad. descent\n");
  }
  Q.elem(find(Q < eps)).fill(eps);
  // Zp = Zp0 + 2*sum(ep)
  // Q = Ep / Zp
  // if (any(is.nan(Ep))) message ('NaN in grad. descent')
  // Q[Q < eps] = eps
  // 
  //   r = {}
  // r$D = D
  //   r$Q = Q
  //   r$Ep = Ep
  //   r$Zp = Zp
  // 
  // return(r)
  
  // Rcout << "size of D0.size()" << D0.size() << "\n";
  // Rcout << "size of Q.size()" << Q.size() << "\n";
  // Rcout << "size of Ep0.size()" << Ep0.size() << "\n";
  return List::create(
    Named("D") = D0,
    Named("Q") = Q,
    Named("Ep") = Ep0,
    Named("Zp") = Zp0
  );
}

// [[Rcpp::export]]
Rcpp::List Y2O(
    arma::mat Y,
    arma::mat W
){
  long double eps = 1.0 / (double) pow(2, 52); // typical machine precision
  mat D = dist(Y); // distance matrix
  //mat E =  exp(-square(D)) % W; // num = 1/(1+|z_i - z_j|^2) it is a matrix
  mat E =  exp(-square(D)); // num = 1/(1+|z_i - z_j|^2) it is a matrix
  E.diag().zeros();
  if (E.has_nan()){
    printf("NaN in grad. descent\n");
  }
  mat Z = sum(E, 1);
  mat O = E;
  for(int j = 0; j < O.n_cols; j++){
    O.col(j) = E.col(j) / Z;
  }
  O = (O + O.t()) / 2.0;
  O.elem(find(O < eps)).fill(eps);
  O = O / accu(O);
  O.elem(find(O < eps)).fill(eps);
  return List::create(
    Named("O") = O,
    Named("E") = E,
    Named("D") = D,
    Named("Z") = Z
  );
  // D = dist(Y)
  // D = as.matrix(D)
  // E = exp(-D^2)*W
  // diag(E) = 0
  // Z = apply(E,1,sum)
  // O = E/Z
  // O = .5 * (O + t(O)); 
  // O[O < eps]<-eps;	
  // O = O/sum(O)
  // if (any(is.nan(E))) message ('NaN in grad. descent')
  // O[O < eps] = eps
  
  //   r = {}
  // r$O = O
  //   r$E = E
  //   r$D = D
  //   r$Z = Z
  
  // return(r)   
}

// [[Rcpp::export]]
double cost2(
    arma::mat Y,
    arma::mat W,
    arma::mat zdata
){
  double eps = 1.0 / (double) pow(2, 52);
  // ## cost2 does not depend on X
  // .cost2 <- function(Y,W,zdata,eps = 2^(-52)){
  List res = Y2O(Y, W);
  mat O = as<mat>(res["O"]);
  
  res = Z2Q(zdata);
  mat Q = as<mat>(res["Q"]);
  
  double cost = accu(Q % log(1.0 / (O + eps)));
  return cost;
  // O = .Y2O(Y,W)$O
  
  // rz = .Z2Q(zdata)
  // Q = rz$Q
  
  // cost =  sum(apply(Q * log(1/(O+eps)),1,sum)) 
  // return(cost)
}


// [[Rcpp::export]]
Rcpp::List stage1_iteration(mat P_t,
                            double rho,
                            mat O_t,
                            
                            mat grads_t,
                            mat zdata_t,
                            mat gains_t,
                            mat incs_t,
                            
                            double min_gain,
                            double momentum,
                            double epsilon)
  // mat Q_t,
  // mat Ep_t,
  // double final_momentum,
  // int max_iter,
  // double eps
{
  int nr = zdata_t.n_rows;
  int nc = zdata_t.n_cols;
  mat stiffnesses(P_t.n_rows, P_t.n_cols);
  // zdata_t, gains_t, incs_t, grads_t = n_train-by-k matrix
  List rz_t;
  mat Q_t;
  mat Ep_t;
  // ###########################################################################
  // ## calculate grad for z train.
  // ###########################################################################
  rz_t = Z2Q(zdata_t);
  Q_t = as<mat>(rz_t["Q"]);
  Ep_t = as<mat>(rz_t["Ep"]);
  stiffnesses = 4.0 * (P_t * rho + O_t * (1.0 - rho) - Q_t) % Ep_t;
  // for(int i = 0; i < stiffnesses.n_rows; i++){
  //   for(int j = 0; j < stiffnesses.n_cols; j++){
  //     stiffnesses(i,j) = 4.0 * (P_t(i,j) * rho + O_t(i,j) * (1.0 - rho) - Q_t(i,j)) * Ep_t(i,j);
  //   }
  // }
  
  // what below loop does =
  // grads.t[i,] = apply(sweep(-zdata.t, 2, -zdata.t[i,]) * stiffnesses[,i],2,sum)
  
  // for(int i = 0; i < grads_t.n_rows; i++){
  //   for(int j = 0; j < grads_t.n_cols; j++){
  for(int i = 0; i < nr; i++){
    for(int j = 0; j < nc; j++){
      grads_t(i,j) = accu((-zdata_t.col(j) + zdata_t(i,j)) % stiffnesses.col(i));
    }
  }
  // ###########################################################################
  // ## update zdata_t
  // ###########################################################################
  // mat gains_t_1 = (gains_t + 0.2) % arma::conv_to<arma::mat>::from(arma::sign(grads_t) != arma::sign(incs_t));
  // mat gains_t_2 = 0.8 * gains_t % arma::conv_to<arma::mat>::from(arma::sign(grads_t) == arma::sign(incs_t));
  // mat gains_t_1 = (gains_t + 0.2) % arma::conv_to<arma::mat>::from(sign_cpp(grads_t) != sign_cpp(incs_t));
  // mat gains_t_2 = 0.8 * gains_t % arma::conv_to<arma::mat>::from(sign_cpp(grads_t) == sign_cpp(incs_t));
  // mat gains_t_1 = (gains_t + 0.2) % arma::conv_to<arma::mat>::from((grads_t) != (incs_t));
  // mat gains_t_2 = 0.8 * gains_t % arma::conv_to<arma::mat>::from((grads_t) == (incs_t));
  // gains_t = gains_t_1 + gains_t_2;
  
  //--------------------------- problems must be here!!!---------------------------
  // for(int i = 0; i < nr; i++){
  //   for(int j = 0; j < nc; j++){
  //     if(sign(grads_t(i,j)) == sign(incs_t(i,j))){
  //     // if((grads_t(i,j) * incs_t(i,j) > 0.0) | (abs(grads_t(i,j)) < 0.00000001 & abs(incs_t(i,j)) < 0.00000001)){
  //       gains_t(i,j) *= 0.8;
  //     } else{
  //       gains_t(i,j) += 0.2;
  //     }
  //   }
  // }
  gains_t = (gains_t + 0.2) % arma::conv_to<arma::mat>::from(arma::sign(grads_t) != arma::sign(incs_t)) +
    0.8 * gains_t % arma::conv_to<arma::mat>::from(arma::sign(grads_t) == arma::sign(incs_t));
  gains_t.elem(find(gains_t < min_gain)).fill(min_gain); // NOT A PROBLEM
  // for(int i = 0; i < nr; i++){
  //   for(int j = 0; j < nc; j++){
  //     if(gains_t(i,j) < min_gain){
  //       gains_t(i,j) = min_gain;
  //     }
  //   }
  // }
  // Rcout << "gains_t=\n" << gains_t << "\n";
  incs_t *= momentum; // NOT A PROBLEM
  incs_t -= epsilon * gains_t % grads_t;
  // incs_t = incs_t * momentum - epsilon * (gains_t % grads_t);
  // for(int i = 0; i < nr; i++){
  //   for(int j = 0; j < nc; j++){
  //     incs_t(i,j) *= momentum;
  //     incs_t(i,j) -= epsilon * gains_t(i,j) * grads_t(i,j);
  //     // incs_t(i,j) = incs_t(i,j) * momentum - epsilon * gains_t(i,j) * grads_t(i,j);
  //   }
  // }
  // incs_t -= epsilon * (gains_t * 1);
  // incs_t -= epsilon * (grads_t * 1);
  //---------------------------------------------------------------------------------
  zdata_t += incs_t;
  
  // what next line does; sweep(zdata_t,2,apply(zdata_t,2,mean))
  // we should center zdata at stage 1, but no need to center zdata at stage 2, since at stage 2, only part of zdata is updated.
  for(int j = 0; j < nc; j++){
    zdata_t.col(j) -= mean(zdata_t.col(j));
  }
  
  // Named("stiffnesses") = stiffnesses,
  // Named("P_t") = P_t,
  // Named("O_t") = O_t
  
  return List::create(
    Named("stiffnesses") = stiffnesses,
    Named("gains_t") = gains_t,
    // Named("gains_t_1") = gains_t_1,
    // Named("gains_t_2") = gains_t_2,
    Named("grads_t") = grads_t,
    Named("zdata_t") = zdata_t,
    Named("incs_t") = incs_t,
    Named("Q_t") = Q_t,
    Named("Ep_t") = Ep_t
  );
}

// [[Rcpp::export]]
Rcpp::List stage2_iteration(mat zdatanow,
                            mat D0,
                            mat Ep0,
                            double Zp0,
                            
                            mat P,
                            mat grads_p,
                            mat gains_p,
                            mat incs_p,
                            
                            double min_gain,
                            double momentum,
                            double epsilon)
{
  // ###########################################################################
  // ## calculate gradient for z pred
  // ###########################################################################
  List rz = Z2Q_one(zdatanow, D0, Ep0, Zp0);
  mat Q = as<mat>(rz["Q"]);
  mat Ep = as<mat>(rz["Ep"]);
  
  mat stiffnesses = 4 * (P - Q) % Ep; // stiffness = 4 * (P-Q) / (1+|z_i - z_j|^2)
  // Rcout << "here\n";
  for(int j = 0; j < grads_p.n_cols; j++){
    grads_p(0,j) = accu((-zdatanow.col(j) + zdatanow(0,j)) % stiffnesses.col(0));
    // Rcout << "here2\n";
  }
  
  // ###########################################################################
  // ## update zdata.t
  // ###########################################################################
  gains_p = (gains_p + 0.2) % arma::conv_to<arma::mat>::from(arma::sign(grads_p) != arma::sign(incs_p)) +
    0.8 * gains_p % arma::conv_to<arma::mat>::from(arma::sign(grads_p) == arma::sign(incs_p));
  gains_p.elem(find(gains_p < min_gain)).fill(min_gain);
  incs_p *= momentum; // NOT A PROBLEM
  incs_p -= epsilon * gains_p % grads_p;
  
  zdatanow.row(0) += incs_p.row(0); // using incs_p is a matrix with a single row
  
  return List::create(
    Named("stiffnesses") = stiffnesses,
    Named("gains_p") = gains_p,
    Named("grads_p") = grads_p,
    Named("incs_p") = incs_p,
    Named("zdatanow") = zdatanow,
    Named("Q") = Q,
    Named("Ep") = Ep
  );
}

// [[Rcpp::export]]
arma::vec stage_3_tsne_cpp(
    arma::vec Y,
    arma::mat W,
    arma::mat zdata,
    arma::vec train_id,
    arma::vec pre_id,
    int k = 2
){
  // long double eps = 1.0 / (double) pow(2, 52); // typical machine precision
  arma::vec ypred(pre_id.size());
  uvec tot_id(train_id.size() + 1);
  for(int i = 0; i < tot_id.size() - 1; i++){
    tot_id(i + 1) = train_id(i);
  }
  tot_id(0) = pre_id(0); // temporary
  vec Ynow(tot_id.size());
  // Rcout << "tot_id.size() = " << tot_id.size() << "\n";
  for(int i = 0; i < tot_id.size() - 1; i++){
    // Rcout << "i=" <<i << "\n";
    // Rcout << "tot_id(i)=" <<tot_id(i) << "\n";
    Ynow(i + 1) = Y(tot_id(i + 1) - 1);
  }
  // Ynow(0) will be determined in the for loop
  
  mat Wnow = W.submat(tot_id - 1, tot_id - 1);
  mat Wsub = W.cols(tot_id - 1);
  // Rcout << "Wnow.n_rows = " << Wnow.n_rows << "\n";
  mat zdatanow = zdata.rows(tot_id - 1);
  // Rcout << "zdatanow.n_rows = " << zdatanow.n_rows << "\n";
  for(int i = 0; i < pre_id.size(); i++){
    Rcout << pre_id(i) << "\n";
    tot_id(0) = pre_id(i);
    // print(pred.id)
    // stsne_pred <- 
    // eps = 2^(-52) # typical machine precision
    Ynow(0) = Y(pre_id(i) - 1);
    Wnow.row(0) = Wsub.row(pre_id(i) - 1);
    Wnow.col(0) = Wnow.row(0).t();
    zdatanow.row(0) = zdata.row(pre_id(i) - 1);
    // List ry = Y2O(Ynow, Wnow);
    // mat O = as<mat>(ry["O"]);
    // mat E = as<mat>(ry["E"]);
    // E.elem(find(E < eps)).fill(eps);
    // mat D = as<mat>(ry["D"]);
    // mat Z = as<mat>(ry["Z"]);
    
    Ynow(0) = 0;
    double c0 = cost2(Ynow, Wnow, zdatanow);
    Ynow(0) = 1;
    double c1 = cost2(Ynow, Wnow, zdatanow);
    if(c1 > c0){
      ypred(i) = 0;
    } else{
      ypred(i) = 1;
    }
  }
  return ypred;
  
  // List rz = Z2Q(zdatanow);
  // mat D = as<mat>(rz["D"]);
  // mat Q = as<mat>(rz["Q"]);
  // mat Ep = as<mat>(rz["Ep"]);
  // mat Zp = as<mat>(ry["Zp"]);
  
  // tot.id = c(pred.id,train.id)
  // Ynow = Y[tot.id]
  // Wnow = W[tot.id,tot.id]
  // zdatanow = zdata[tot.id,]
  
  
  // ry = .Y2O(Ynow,Wnow)
  // O = ry$O;
  // E = ry$E; E[E < eps]<-eps
  // D = ry$D;
  // Z = ry$Z
  
  // rz = .Z2Q(zdatanow)
  // Dp = rz$D;
  // Q = rz$Q
  // Ep = rz$Ep;
  // Zp = rz$Zp
  // int y0 = Ynow;
  // y0 = Ynow; y0[1] = 0; c0 = .cost2(y0,Wnow,zdatanow)
  // y1 = Ynow; y1[1] = 1; c1 = .cost2(y1,Wnow,zdatanow)
  // if(c1>c0){Ynow = y0;Y[pred.id]=0}else{Ynow = y1;Y[pred.id]=1}
  
  // #cost2 = .cost2(Ynow,zdatanow)
  // #cost2 =  sum(apply(Q * log(1/(O+eps)),1,sum)) 
  // #message("Epoch: prediction error is: ",cost2)
  // #print(Y[pred.id])
  
  // return(Y[pred.id])
}