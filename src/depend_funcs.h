#ifndef __DEPEND_FUNCS__
#define __DEPEND_FUNCS__
#include <RcppArmadillo.h>

arma::vec sample_theta_i_arms(arma::vec theta0_i, arma::vec y_i,
                              arma::mat inv_sigma, arma::mat A, arma::vec d);
arma::vec sample_theta_i_arms_partial_credit(arma::vec theta0_i, arma::uvec y_i,
                                             arma::mat inv_sigma, arma::mat A, arma::mat D);
double my_rand_unif();

arma::mat calcu_sigma_cmle_cpp(const arma::mat &theta, double tol);
arma::mat calcu_sigma_cmle_cpp1(const arma::mat &theta);

double log_full_likelihood(const arma::mat &response, const arma::mat &A,
                           const arma::vec &d, const arma::mat &theta, const arma::mat &inv_sigma);

Rcpp::List Update_init_sa(const arma::mat &theta0,const arma::mat &response,const arma::mat &A0,
                          const arma::mat &Q,const arma::vec &d0,const arma::mat &inv_sigma0);

arma::mat update_sigma_one_step(const arma::mat &theta0, double step_sigma);
arma::mat update_sigma_one_step1(const arma::mat &theta0,
                                 const arma::mat &B0,
                                 double step_B);
arma::mat update_sigma_one_step2(const arma::mat &theta0,
                                 const arma::mat &B0, arma::mat& B_res1, arma::mat& B_res2,
                                 arma::mat & B_mis_info,
                                 int mcn, double alpha, double step, double c1, double c2);
arma::mat update_sigma_one_step3(const arma::mat &theta0,
                                 const arma::mat &B0,
                                 double step_B);

arma::vec s_func(arma::mat theta, arma::mat response, arma::mat Q, arma::mat Sigma, arma::mat A, arma::vec d);

arma::mat h_func(arma::mat theta, arma::mat response, arma::mat Q, arma::mat Sigma, arma::mat A, arma::vec d);

#endif