#ifndef __DEPEND_FUNCS__
#define __DEPEND_FUNCS__
#include <RcppArmadillo.h>

arma::vec sample_theta_i_arms(arma::vec theta0_i, arma::vec y_i,
                              arma::mat inv_sigma, arma::mat A, arma::vec d);
arma::vec sample_theta_i_arms_partial_credit(arma::vec theta0_i, arma::uvec y_i,
                                             arma::mat inv_sigma, arma::mat A, arma::mat D);
#endif