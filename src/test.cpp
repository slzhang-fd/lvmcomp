#include <RcppArmadillo.h>
#include <random>
#include "depend_funcs.h"
// [[Rcpp::plugins(cpp11)]]

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

//' @export
// [[Rcpp::export]]
arma::mat test_r(long int rows, int cols){
  // configuration of multitreading
  arma::mat res(rows, cols);
#pragma omp parallel num_threads(cols)
{
#pragma omp for
  for(auto j=0;j<cols;++j){
    for(auto i=0;i<rows;++i){
      // res(i,j) = R::unif_rand();
      res(i,j) = my_rand_unif();
    }
  }
}
  return res;
}

//' @export
// [[Rcpp::export]]
arma::mat test_r1(long int rows, int cols){
  // configuration of multitreading
  arma::mat res(rows, cols);
  
  
  // Uniform Distribution
  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_real_distribution<double> dist(0.0, 1.0);
#pragma omp parallel num_threads(cols)
{
  
#pragma omp for
  for(auto j=0;j<cols;++j){
    for(auto i=0;i<rows;++i){
      res(i,j) = R::unif_rand();
      //res(i,j) = dist(mt);
    }
  }
}
return res;
}