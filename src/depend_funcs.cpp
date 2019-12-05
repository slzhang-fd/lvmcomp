// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "lvmcomp_omp.h"
#include "arms_ori.h"

struct log_pos_theta_mirt_param{
  arma::vec theta_minus_k;
  unsigned int k;
  arma::vec y_i;
  arma::mat inv_sigma;
  arma::mat A;
  arma::vec d;
};
double log_pos_theta_mirt(double x, void* params){
  struct log_pos_theta_mirt_param *d_params;
  d_params = static_cast<struct log_pos_theta_mirt_param *> (params);
  arma::vec theta = d_params->theta_minus_k;
  theta(d_params->k) =  x;
  arma::vec tmp = d_params->A * theta + d_params->d;
  return -0.5 * arma::as_scalar(theta.t() * d_params->inv_sigma * theta) + 
    arma::accu(d_params->y_i % tmp - arma::log(1+arma::exp(tmp)));
}

arma::vec sample_theta_i_arms(arma::vec theta0_i, arma::vec y_i,
                              arma::mat inv_sigma, arma::mat A, arma::vec d) {
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-3.0, -1.0, 0.0, 1.0, 3.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 0;
  double xprev = 0.0;
  
  log_pos_theta_mirt_param pos_theta_mirt_data;
  pos_theta_mirt_data.A = A;
  pos_theta_mirt_data.d = d;
  pos_theta_mirt_data.inv_sigma = inv_sigma;
  pos_theta_mirt_data.theta_minus_k = theta0_i;
  pos_theta_mirt_data.y_i = y_i;
  
  int K = theta0_i.n_elem;
  for(unsigned int k=0;k<K;++k){
    pos_theta_mirt_data.k = k;
    err = arms(xinit,ninit,&xl,&xr,log_pos_theta_mirt,&pos_theta_mirt_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    pos_theta_mirt_data.theta_minus_k(k) = xsamp[0];
  }
  return pos_theta_mirt_data.theta_minus_k;
}

struct log_pos_theta_pcirt_param{
  arma::vec theta_minus_k;
  unsigned int k;
  arma::uvec response_i;
  arma::mat inv_sigma;
  arma::mat A;
  arma::mat D;
};
double log_pos_theta_pcirt(double x, void* params){
  struct log_pos_theta_pcirt_param *d_params;
  d_params = static_cast<struct log_pos_theta_pcirt_param *> (params);
  
  int M = d_params->D.n_cols;
  int J = d_params->A.n_rows;
  arma::vec theta_i = d_params->theta_minus_k;
  theta_i(d_params->k) = x;
  arma::vec thetaA_i = d_params->A * theta_i;
  arma::mat tmp = thetaA_i * arma::linspace(0, M-1, M).t() + d_params->D;
  arma::vec tmp_res(J);
  for(unsigned int i=0;i<J;++i){
    tmp_res(i) = tmp(i, d_params->response_i(i)) - std::log(sum(arma::exp(tmp.row(i))));
  }
  return arma::accu(tmp_res) - 0.5 * arma::as_scalar(theta_i.t() * d_params->inv_sigma * theta_i);
}
arma::vec sample_theta_i_arms_partial_credit(arma::vec theta0_i, arma::uvec y_i,
                                             arma::mat inv_sigma, arma::mat A, arma::mat D){
  int err, ninit = 5, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-1.0, -0.5, 0.0, 0.5, 1.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 0;
  double xprev = 0.0;
  
  log_pos_theta_pcirt_param log_pos_theta_pcirt_data;
  log_pos_theta_pcirt_data.A = A;
  log_pos_theta_pcirt_data.D = D;
  log_pos_theta_pcirt_data.inv_sigma = inv_sigma;
  log_pos_theta_pcirt_data.response_i = y_i;
  log_pos_theta_pcirt_data.theta_minus_k = theta0_i;
  
  int K = theta0_i.n_elem;
  for(unsigned int k=0;k<K;++k){
    log_pos_theta_pcirt_data.k = k;
    err = arms(xinit,ninit,&xl,&xr,log_pos_theta_pcirt,&log_pos_theta_pcirt_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    log_pos_theta_pcirt_data.theta_minus_k(k) = xsamp[0];
  }
  return log_pos_theta_pcirt_data.theta_minus_k;
}


