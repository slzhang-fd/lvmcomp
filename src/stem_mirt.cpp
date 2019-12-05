#include <RcppArmadillo.h>
#include "lvmcomp_omp.h"
#include "calcu_sigma_cmle.h"
#include "my_Logistic.h"
#include "depend_funcs.h"
#include <ctime>

//' @export
// [[Rcpp::export]]
Rcpp::List stem_mirtc(const arma::mat &response, const arma::mat &Q,
                     arma::mat A0, arma::vec d0, arma::mat theta0, 
                     arma::mat sigma0, int T, bool parallel){
  // configuration of multitreading
  if(!parallel)
    omp_set_num_threads(1);
  else
    omp_set_num_threads(omp_get_num_procs());
  
  // obtain dimensions from input data
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A0.n_cols;
  //arma::vec x = arma::linspace(-5,5,10);
  arma::mat inv_sigma(K, K);
  arma::mat res = arma::zeros(T, K*K+J*K+J);
  arma::uvec rv(1), cv;
  // constraint on diagnal
  sigma0.diag().ones();
  for(unsigned int mcn=0;mcn<T;++mcn){
    Rcpp::checkUserInterrupt();
    Rprintf("\rStep  %d\t| ", mcn+1);
    // E step
    if(K==1){
      inv_sigma(0,0) = 1.0;
    }
    else{
      inv_sigma = arma::inv_sympd(sigma0);
    }
#pragma omp parallel for
    for(unsigned int i=0;i<N;++i){
      // theta0.row(i) = sample_theta_i_myars(x, theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
    }
    // M step
    if(K > 1){
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
    }
    for(unsigned int j=0;j<J;++j){
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
      d0(j) = glm_res(cv.n_rows);
    }
    res.row(mcn) = arma::join_cols(arma::join_cols(arma::vectorise(sigma0),arma::vectorise(A0)),d0).t();
  }
  return Rcpp::List::create(Rcpp::Named("res") = res,
                            Rcpp::Named("A0") = A0,
                            Rcpp::Named("d0") = d0,
                            Rcpp::Named("sigma0") = sigma0,
                            Rcpp::Named("theta0") = theta0);
}
// [[Rcpp::export]]
Rcpp::List stem_pcirtc(const arma::umat &response, const arma::mat &Q,  
                                    arma::mat A0, arma::mat D0, arma::mat theta0,
                                    arma::mat sigma0, int T, bool parallel){
  if(!parallel)
    omp_set_num_threads(1);
  else
    omp_set_num_threads(omp_get_num_procs());
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A0.n_cols;
  int M = D0.n_cols;
  sigma0.diag().ones();
  arma::mat inv_sigma(K, K);
  arma::mat res = arma::zeros(T, K*K+J*K+J*M);
  // struct timeval t1, t2;
  // double elapsedTime;
  // clock_t tt;
  for(unsigned int mcn=0;mcn<T;++mcn){
    Rprintf("\rStep  %d\t| ", mcn+1);
    Rcpp::checkUserInterrupt();
    // E step
    if(K==1){
      inv_sigma(0,0) = 1.0;
    }
    else{
      if(!sigma0.is_sympd()) Rcpp::Rcout << "sigma error: " << sigma0;
      inv_sigma = arma::inv_sympd(sigma0);
    }
    // tt = clock();
#pragma omp parallel for
    for(unsigned int i=0;i<N;++i){
      //Rprintf("i=%d\n",i);
      //theta0.row(i) = sample_theta_i_myars_partial_credit(x, theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, D0).t();
      theta0.row(i) = sample_theta_i_arms_partial_credit(theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, D0).t();
    }
    // Rprintf("sampling elaps time is:%f ms | ", (clock()-tt)*1000.0 / CLOCKS_PER_SEC);
    // M step
    // tt = clock();
    if(K > 1){
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
    }
    for(unsigned int j=0;j<J;++j){
      arma::uvec rv(1), cv;
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp_partial(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), D0.row(j).t());
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1);
      D0.row(j) = glm_res.subvec(cv.n_rows, cv.n_rows+M-1).t();
    }
    // Rprintf("optimization elaps time is:%f ms", (clock()-tt)*1000.0 / CLOCKS_PER_SEC);
    res.row(mcn) = arma::join_cols(arma::join_cols(arma::vectorise(sigma0),arma::vectorise(A0)),arma::vectorise(D0)).t();
  }
  return Rcpp::List::create(Rcpp::Named("res") = res,
                            Rcpp::Named("A0") = A0,
                            Rcpp::Named("D0") = D0,
                            Rcpp::Named("sigma0") = sigma0,
                            Rcpp::Named("theta0") = theta0);
}

double log_full_likelihood(const arma::mat &response, const arma::mat &A,
                           const arma::vec &d, const arma::mat &theta){
  int N = theta.n_rows;
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  return 1.0 / N * (arma::accu(M % response) - arma::accu(arma::log(1.0 + arma::exp(M)))-
                    0.5 * arma::accu(arma::square(theta)));
}

//' @export
// [[Rcpp::export]]
Rcpp::List stem_simu(const arma::mat &response, const arma::mat &Q,
                      arma::mat A0, arma::vec d0, arma::mat theta0, 
                      arma::mat A_true, int max_steps, 
                      int window_size=50, int block_size=20, double tol=0.05,
                      bool print_proc=false){
  // timer for computation
  // clock_t begin = clock(); 
  // double elapsed_secs0, elapsed_secs = 0;
  // 
  // obtain dimensions from input data
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A0.n_cols;
  int burn_in = max_steps;
  
  arma::mat inv_sigma = arma::eye(K, K);
  arma::mat res = arma::zeros(max_steps, J*K);
  arma::uvec rv(1), cv;
  arma::vec log_lik = arma::zeros(max_steps);
  arma::vec sin_angle = arma::zeros(max_steps);
  
  // averaged results as an estimator
  arma::mat A_hat = A0;
  // int time_record_num = (int)max_time / 1;
  // arma::vec time_points = arma::zeros(time_record_num);
  // arma::vec sin_angle = arma::zeros(time_record_num);
  // int time_ind = 0;
  
  // obtain left sigular vectors for true A
  arma::mat U1, V1, U2, V2;
  arma::vec s1, s2;
  arma::svd(U1, s1, V1, A_true);
  U1 = U1.cols(0, K-1);
  // constraint on diagnal

  for(unsigned int mcn=0;mcn<max_steps;++mcn){
    Rcpp::checkUserInterrupt();
    // E step
    for(unsigned int i=0;i<N;++i){
      // theta0.row(i) = sample_theta_i_myars(x, theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
    }
    // M step
    // if(K > 1){
    //   sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
    // }
    for(unsigned int j=0;j<J;++j){
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
      d0(j) = glm_res(cv.n_rows);
    }
    res.row(mcn) = arma::vectorise(A0).t();
    
    // check if burn in finished
    double log_lik_tmp = log_full_likelihood(response, A0, d0, theta0);
    log_lik(mcn) = log_lik_tmp;
    // if( !(mcn%block_size) && mcn <= burn_in && mcn>window_size){
    //   double m1 = arma::mean(log_lik.subvec(mcn-window_size, mcn-window_size+(int)(0.1*window_size)));
    //   double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
    //   if(std::abs(m1-m2) < tol){
    //     //burn_in = mcn;
    //     burn_in = mcn - window_size;
    //     arma::rowvec A_hat_tmp = arma::mean(res.rows(mcn-window_size, mcn-1), 1);
    //     A_hat = A_hat_tmp.reshape(A0.n_rows, A0.n_cols);
    //     max_steps = std::min(max_steps, (int)(1.5*mcn));
    //     Rcpp::Rcout << "burn in finished!\n" << std::endl;
    //   }
    //   else{
    //     Rcpp::Rcout << "log likelihood mean diff: " << std::abs(m1-m2) << std::endl;
    //   }
    // }
      
    // calculate A_hat based on burnin length
    if(mcn <= burn_in){
      if(!(mcn%block_size) && mcn>window_size){
        double m1 = arma::mean(log_lik.subvec(mcn-window_size, mcn-window_size+(int)(0.1*window_size)));
        double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
        if(std::abs(m1-m2) < tol){
          //burn_in = mcn;
          burn_in = mcn - window_size;
          arma::rowvec A_hat_tmp = arma::mean(res.rows(mcn-window_size, mcn-1), 0);
          A_hat = arma::reshape(A_hat_tmp, arma::size(A_hat));
          max_steps = std::min(max_steps, (int)(mcn+window_size));
          if(print_proc) Rcpp::Rcout << "burn in finished!\n" << std::endl;
        }
        else{
          if(print_proc) Rcpp::Rcout << "log likelihood mean diff: " << std::abs(m1-m2) << std::endl;
          A_hat = A0;
        }
      }
      else{
        A_hat = A0;
      }
    }
    else{
      double mcn_count = mcn - burn_in;
      A_hat = mcn_count / (mcn_count + 1.0) * A_hat + 1.0 / (mcn_count + 1.0) * A0;
    }
    // calculate cosine between A_hat and A_true
    arma::svd(U2, s2, V2, A_hat);
    arma::vec s_tmp = arma::svd(U1.t() * U2.cols(0, K-1));
    //Rcpp::Rcout << s_tmp << std::endl;
    
    // print estimation accuracy every 10 seconds
    // clock_t end = clock();
    // elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // if(elapsed_secs - elapsed_secs0 > 1){
    //   Rcpp::Rcout << "iter: " << mcn << " elapsed time: " << elapsed_secs << 
    //     " angle A_true, A_hat: " << std::sqrt(1 - s_tmp(K-1)) << std::endl;
    //   time_ind = (int)elapsed_secs / 1;
    //   time_points(time_ind-1) = elapsed_secs;
    //   sin_angle(time_ind -1) = std::sqrt(1 - s_tmp(K-1));
    //   elapsed_secs0 = elapsed_secs;
    // }

    double sin_angle_tmp = std::sqrt(1.0 - s_tmp(K-1));
    sin_angle(mcn) = sin_angle_tmp;
    
    if(print_proc)
      Rcpp::Rcout << "iter: " << mcn << " sin<: " << sin_angle_tmp <<
      "log_likelihood: " << log_lik_tmp << std::endl;
  }
  return Rcpp::List::create(Rcpp::Named("res") = res.rows(0, max_steps-1),
                            Rcpp::Named("A0") = A0,
                            Rcpp::Named("d0") = d0,
                            Rcpp::Named("theta0") = theta0,
                            Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("log_likelihood") = log_lik.subvec(0, max_steps-1),
                            Rcpp::Named("sin_angle") = sin_angle.subvec(0, max_steps-1));
}

//' @export
// [[Rcpp::export]]
arma::mat test_func(arma::mat A, int i, int j){
  arma::rowvec tmp = arma::mean(A, 0);
  return arma::reshape(tmp, i,j);
}