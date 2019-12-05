#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
double obj_func_cpp(arma::mat sigma, arma::mat sigma_hat){
  arma::mat sigma_inv = arma::inv_sympd(sigma);
  return arma::accu( sigma_inv % sigma_hat ) + std::log(arma::det(sigma));
}
// [[Rcpp::export]]
arma::mat calcu_sigma_cmle_cpp(arma::mat theta, double tol){
  int N = theta.n_rows;
  arma::mat sigma_hat = theta.t() * theta / N;
  arma::mat sigma0 = arma::cor(theta);
  arma::mat sigma1 = sigma0;
  arma::mat tmp = sigma0;
  double eps = 1;
  double step = 1;
  while(eps > tol){
    step = 1;
    tmp = arma::inv_sympd(sigma0);
    sigma1 = sigma0 - step * ( - tmp * sigma_hat * tmp + tmp );
    sigma1.diag().ones();
    eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    // Rprintf(" eps ==0: %s ", (eps==0) ? "true" : "false");
    // Rprintf(" eps < -1e-7: %s ", (eps < -1e-7) ? "true" : "false");
    // -1e-7 here for preventing -0.0 fall into the while loop
    while(eps < -1e-7 || min(arma::eig_sym(sigma1)) < 0){
      step *= 0.5;
      if(step < 1e-5){
        Rprintf("Possible error in sigma estimation\n");
        break;
      }
      sigma1 = sigma0 - step * ( - tmp * sigma_hat * tmp + tmp );
      sigma1.diag().ones();
      eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    }
    // Rprintf("eps= %f\n", eps);
    sigma0 = sigma1;
  }
  return sigma0;
}

