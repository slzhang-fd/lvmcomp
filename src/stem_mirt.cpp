#include <RcppArmadillo.h>
#include "lvmcomp_omp.h"
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



//' @export
// [[Rcpp::export]]
Rcpp::List stem_simu(const arma::mat &response, const arma::mat &Q,
                      arma::mat A0, arma::vec d0, arma::mat sigma0,
                      arma::mat theta0, 
                      double tol, int window_size=40, 
                      int block_size=20, int max_steps=2000,
                      bool print_proc=true){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A0.n_cols;
  bool arrive_flag = true;
  
  arma::mat inv_sigma0 = arma::inv_sympd(sigma0);
  arma::uvec rv(1), cv;
  // arma::vec log_lik = arma::zeros(max_steps);
  
  // averaged results as an estimator
  arma::mat A_hat = A0;
  arma::vec d_hat = d0;
  arma::mat sigma_hat = sigma0;
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat sigma_hat_all = arma::zeros(K*K, max_steps);

  long int mcn0 = 0;
  long int mcn = 0;
  for(mcn=0;mcn<max_steps;++mcn){
    Rcpp::checkUserInterrupt();
    // if(print_proc) Rprintf("\rStep  %d\t| ", mcn+1);
    
    // E step
    for(unsigned long int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma0, A0, d0).t();
    }
    if(theta0.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // M step
    if(K > 1){
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
      if(!sigma0.is_sympd()){
        Rcpp::stop("error after calcu_sigma\n");
      }
      inv_sigma0 = arma::inv_sympd(sigma0);
    }
    for(unsigned int j=0;j<J;++j){
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
      d0(j) = glm_res(cv.n_rows);
    }
    
    // calculate A_hat based on burnin length
    if(arrive_flag){
      double mcn_count = mcn - mcn0 + 1.0;
      A_hat = mcn_count / (mcn_count + 1.0) * A_hat + 1.0 / (mcn_count + 1.0) * A0;
      d_hat = mcn_count / (mcn_count + 1.0) * d_hat + 1.0 / (mcn_count + 1.0) * d0;
      sigma_hat = mcn_count / (mcn_count + 1.0) * sigma_hat + 1.0 / (mcn_count + 1.0) * sigma0;
    }
    // else{
    //   A_hat = A0;
    //   d_hat = d0;
    //   sigma_hat = sigma0;
    //   
    //   log_lik(mcn) = log_full_likelihood(response, A0, d0, theta0, inv_sigma0);
    //   if(!(mcn%block_size) && mcn>=(window_size-1)){
    //     double m1 = arma::mean(log_lik.subvec(mcn-window_size+1, mcn-window_size+(int)(0.1*window_size)));
    //     double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
    //     if(std::abs(m1-m2) < tol){
    //       arrive_flag = true;
    //       mcn0 = mcn;
    //       max_steps = std::min(max_steps, (int)(mcn+window_size));
    //       if(print_proc) Rcpp::Rcout << "stem burn in finish at " << mcn << std::endl;
    //     }
    //     else{
    //       if(print_proc) Rcpp::Rcout << "log likelihood mean diff: " << std::abs(m1-m2) << std::endl;
    //     }
    //   }
    // }
    
    A_hat_all.col(mcn) = arma::vectorise(A_hat);
    d_hat_all.col(mcn) = d_hat;
    sigma_hat_all.col(mcn) = arma::vectorise(sigma_hat);

  }
  return Rcpp::List::create(Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("d_hat") = d_hat,
                            Rcpp::Named("sigma_hat") = sigma_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("sigma_hat_all") = sigma_hat_all.cols(0, max_steps-1).t());
}
arma::mat soft_thre(arma::mat x, arma::mat mu){
  arma::mat y = arma::abs(x) - mu;
  y.elem( arma::find(y < 0)).zeros();
  y.elem( arma::find(x < 0)) *= -1;
  return y;
}

// Original first-order stochastic proximal gradient descent
void RM_update(const arma::mat &response, const arma::mat &theta, 
               arma::mat &A, arma::vec &d, arma::mat &B, 
               double lambda, arma::mat zero, arma::mat anchor,
               double alpha, double step_n){
  int N = response.n_rows;
  int J = response.n_cols;
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= (1 - zero);
  arma::vec d_deriv = arma::sum(response - M, 0).t() / J;


  A += step_n * A_deriv;
  A = soft_thre(A, (1-anchor) * lambda * step_n);
  d += step_n * d_deriv;
  B = update_sigma_one_step1(theta, B, step_n/N);
}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_penmirt(const arma::mat &response, arma::mat theta, arma::mat A, arma::vec d, arma::mat B, 
                      double lambda, arma::mat zero, arma::mat anchor, double alpha,
                      double step=1, int max_steps=200){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = B.n_cols;
  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  for(int mcn=1; mcn <= max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    
    // stochastic E step
    for(unsigned long int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // Robbins-Monro update
    double factor = std::pow(mcn, -alpha);
    double step_n = factor * step;
    RM_update(response, theta, A, d, B, lambda, zero, anchor, alpha, step_n);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    A_hat = A_hat * mcn / (mcn+1.0) + A / (mcn+1.0);
    //A_hat = soft_thre(A_hat, (1-anchor) * lambda * factor);
    d_hat = d_hat * mcn / (mcn+1.0) + d / (mcn+1.0);
    B_hat = arma::normalise(B_hat * mcn / (mcn+1.0) + B / (mcn+1.0));
    
    A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    d_hat_all.col(mcn-1) = d_hat;
    B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
  }
  return Rcpp::List::create(Rcpp::Named("A_hat")=soft_thre(A_hat, (1-anchor) * lambda),
                            Rcpp::Named("d_hat")=d_hat,
                            Rcpp::Named("Sigma_hat")=B_hat.t() * B_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("B_hat_all") = B_hat_all.cols(0, max_steps-1).t());
}

// stochastic proximal gradient descent with pseudo second-order information
void RM_update1(const arma::mat &response, const arma::mat &theta, 
               arma::mat &A, arma::vec &d, arma::mat &B, 
               double lambda, arma::mat zero, arma::mat anchor,
               double alpha, double step_n, double scale_bound){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A.n_cols;
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= (1 - zero);
  arma::mat A_scale = (M % (1-M)).t() * (theta % theta) / N;
  //Rcpp::Rcout << A_scale << std::endl;
  A_scale = arma::max(1.0/A_scale, scale_bound * arma::ones(J,K));

  A += step_n * (A_deriv % A_scale);
  A = soft_thre(A, (1-anchor) % A_scale * lambda * step_n);
  
  arma::vec d_deriv = arma::sum(response - M, 0).t() / J;
  arma::vec d_scale = arma::sum(M % (1-M), 0).t() / J;
  //Rcpp::Rcout << d_scale << std::endl;
  d_scale = arma::max(1.0 / d_scale, scale_bound * arma::ones(J));
  d += step_n * (d_deriv % d_scale);
  B = update_sigma_one_step1(theta, B, step_n/N);
}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_penmirt1(const arma::mat &response, arma::mat theta, arma::mat A, arma::vec d, arma::mat B, 
                      double lambda, arma::mat zero, arma::mat anchor, double alpha,
                      double step=1, int max_steps=200, double scale_bound=1){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = B.n_cols;
  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  for(int mcn=1; mcn <= max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    
    // stochastic E step
    for(unsigned long int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // Robbins-Monro update
    double factor = std::pow(mcn, -alpha);
    double step_n = factor * step;
    RM_update1(response, theta, A, d, B, lambda, zero, anchor, alpha, step_n, scale_bound);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    A_hat = A_hat * mcn / (mcn+1.0) + A / (mcn+1.0);
    //A_hat = soft_thre(A_hat, (1-anchor) * lambda * factor);
    d_hat = d_hat * mcn / (mcn+1.0) + d / (mcn+1.0);
    B_hat = arma::normalise(B_hat * mcn / (mcn+1.0) + B / (mcn+1.0));
    
    A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    d_hat_all.col(mcn-1) = d_hat;
    B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
  }
  return Rcpp::List::create(Rcpp::Named("A_hat")=soft_thre(A_hat, (1-anchor) * lambda),
                            Rcpp::Named("d_hat")=d_hat,
                            Rcpp::Named("Sigma_hat")=B_hat.t() * B_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("B_hat_all") = B_hat_all.cols(0, max_steps-1).t());
}
// stochastic proximal gradient descent with acceleration
void RM_update2(const arma::mat &response, const arma::mat &theta, 
               arma::mat &A, arma::vec &d, arma::mat &B, 
               double lambda, arma::mat zero, arma::mat anchor,
               double alpha, double step_n){
  int N = response.n_rows;
  int J = response.n_cols;
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= (1 - zero);
  arma::vec d_deriv = arma::sum(response - M, 0).t() / J;
  
  
  A += step_n * A_deriv;
  A = soft_thre(A, (1-anchor) * lambda * step_n);
  d += step_n * d_deriv;
  B = update_sigma_one_step1(theta, B, step_n/N);
}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_penmirt2(const arma::mat &response, arma::mat theta, arma::mat A, arma::vec d, arma::mat B, 
                      double lambda, arma::mat zero, arma::mat anchor, double alpha,
                      double step=1, int max_steps=200){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = B.n_cols;
  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  for(int mcn=1; mcn <= max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    
    // stochastic E step
    for(unsigned long int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // Robbins-Monro update
    double factor = std::pow(mcn, -alpha);
    double step_n = factor * step;
    RM_update2(response, theta, A, d, B, lambda, zero, anchor, alpha, step_n);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    A_hat = A_hat * mcn / (mcn+1.0) + A / (mcn+1.0);
    //A_hat = soft_thre(A_hat, (1-anchor) * lambda * factor);
    d_hat = d_hat * mcn / (mcn+1.0) + d / (mcn+1.0);
    B_hat = arma::normalise(B_hat * mcn / (mcn+1.0) + B / (mcn+1.0));
    
    A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    d_hat_all.col(mcn-1) = d_hat;
    B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
  }
  return Rcpp::List::create(Rcpp::Named("A_hat")=soft_thre(A_hat, (1-anchor) * lambda),
                            Rcpp::Named("d_hat")=d_hat,
                            Rcpp::Named("Sigma_hat")=B_hat.t() * B_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("B_hat_all") = B_hat_all.cols(0, max_steps-1).t());
}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_mirt_conf(const arma::mat &response, const arma::mat &Q,
                        arma::mat A0, arma::vec d0, arma::mat sigma0, arma::mat theta0,
                        double alpha, double tol,
                        int window_size=40, int block_size = 20, 
                        int max_steps = 2000, bool print_proc = true){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A0.n_cols;
  
  arma::mat inv_sigma0 = arma::inv_sympd(sigma0);
  arma::mat B0 = arma::chol(sigma0);
  
  
  arma::mat A_hat = A0;
  arma::vec d_hat = d0;
  arma::mat B_hat = B0;
  arma::mat A_deriv = arma::zeros(J,K);
  arma::vec d_deriv = arma::zeros(J);
  arma::mat A_deriv_all = arma::zeros(J*K, max_steps);
  arma::mat d_deriv_all = arma::zeros(J, max_steps);
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  
  // arma::vec log_lik = arma::zeros(max_steps);
  // arma::uvec rv(1), cv;
  long int mcn0 = 0;
  long int mcn = 0;
  bool arrive_flag = true;
  int param_num = arma::accu(Q) + J + K*(K-1)/2;
  arma::vec s;
  arma::vec res2 = arma::zeros(param_num);
  arma::mat res1 = arma::zeros(param_num, param_num);
  // arma::vec s = s_func(theta0, response, Q, sigma0, A0, d0);
  // arma::mat res1 = (h_func(theta0, response, Q, sigma0, A0, d0) - s * s.t());
  // arma::vec res2 = s;
  
  arma::vec mis_info2 = arma::zeros(param_num);
  arma::mat mis_info1 = arma::zeros(param_num, param_num);
  
  arma::vec check_info1 = arma::zeros(max_steps);
  arma::vec check_info2 = arma::zeros(max_steps);
  // arma::uvec A_free_ind = arma::find(arma::vectorise(Q) > 0);
  
  // for(unsigned int i=0;i<N;++i){
  //   theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma0, A0, d0).t();
  // }
  // if(theta0.has_nan()){
  //   Rcpp::stop("nan in theta0\n");
  // }
  // Rcpp::List init_res = Update_init_sa(theta0, response, A0, Q, d0, inv_sigma0);
  // double step_A = init_res[0];
  // double step_d = init_res[1];
  // arma::mat A1 = init_res[2];
  // arma::vec d1 = init_res[3];
  // A0 = A1;
  // d0 = d1;
  double step_B = 1.0 / (double)N;
  double step_a = 1.0 / (double)N;
  double step_d = 1.0 / (double)N;
  for(mcn=0; mcn < max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    for(unsigned long int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma0, A0, d0).t();
    }
    if(theta0.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
  
    if(arrive_flag){
      arma::mat M = theta0 * A0.t();
      M.each_row() += d0.t();
      M = 1.0 / (1.0 + arma::exp(-M));
      A_deriv = (response - M).t() * theta0; 
      A_deriv = A_deriv % Q;
      d_deriv = arma::sum(response - M, 0).t();
      
      double mcn_count = mcn - mcn0 + 1.0;
      double factor = std::pow(mcn_count, -alpha);
      A0 += step_a * factor * A_deriv;
      d0 += step_d * factor * d_deriv;
      B0 = update_sigma_one_step1(theta0, B0, step_B * factor);
      //sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
      sigma0 = B0.t() * B0; 
      inv_sigma0 = arma::inv_sympd(sigma0);
      
      
      double portion_old = mcn_count / (mcn_count + 1.0);
      double portion_new = 1.0 / (mcn_count + 1.0);
      A_hat = portion_old * A_hat + portion_new * A0;
      d_hat = portion_old * d_hat + portion_new * d0;
      B_hat = arma::normalise(portion_old * B_hat + portion_new * B0);
      
      // calculate missing information
      s = s_func(theta0, response, Q, sigma0, A0, d0);
      res1 += factor * (h_func(theta0, response, Q, sigma0, A0, d0) - s * s.t() - res1);
      res2 += factor * (s - res2);
      mis_info1 = portion_old * mis_info1 + portion_new * res1;
      mis_info2 = portion_old * mis_info2 + portion_new * res2;
      
      // if(mcn > 500){
      //   
      //   double mcn_count00 = mcn - 500;
      //   double factor00 = std::pow(mcn_count00, -alpha);
      //   double portion_old00 = mcn_count00 / (mcn_count00 + 1.0);
      //   double portion_new00 = 1.0 / (mcn_count00 + 1.0);
      //   
      //   s = s_func(theta0, response, Q, sigma0, A0, d0);
      //   res1 += factor00 * (h_func(theta0, response, Q, sigma0, A0, d0) - s * s.t() - res1);
      //   res2 += factor00 * (s - res2);
      //   
      //   mis_info1 = portion_old00 * mis_info1 + portion_new00 * res1;
      //   mis_info2 = portion_old00 * mis_info2 + portion_new00 * res2;
      // }
      
    }
    // else{
    //   if(K > 1){
    //     sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
    //     if(!sigma0.is_sympd()){
    //       Rcpp::stop("error after calcu_sigma\n");
    //     }
    //     inv_sigma0 = arma::inv_sympd(sigma0);
    //   }
    //   for(unsigned int j=0;j<J;++j){
    //     rv(0) = j;
    //     cv = arma::find(Q.row(j));
    //     arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
    //     A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
    //     d0(j) = glm_res(cv.n_rows);
    //   }
    //     
    //   A_hat = A0;
    //   d_hat = d0;
    //   sigma_hat = sigma0;
    //   log_lik(mcn) = log_full_likelihood(response, A0, d0, theta0, inv_sigma0);
    //   if(!(mcn%block_size) && mcn>=(window_size-1)){
    //     double m1 = arma::mean(log_lik.subvec(mcn-window_size+1, mcn-window_size+(int)(0.1*window_size)));
    //     double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
    //     if(std::abs(m1-m2) < tol){
    //       arrive_flag = true;
    //       mcn0 = mcn;
    //       max_steps = std::min(max_steps, (int)(mcn+window_size));
    //       if(print_proc) Rcpp::Rcout << "sa burn in finish at " << mcn << std::endl;
    //     }
    //     else{
    //       if(print_proc) Rcpp::Rcout << "log likelihood mean diff: " << std::abs(m1-m2) << std::endl;
    //     }
    //   }
    //   // log_lik(mcn) = log_full_likelihood(response, A0, d0, theta0, inv_sigma0);
    //   // if(!(mcn%block_size) && mcn>=(window_size-1) && !flag){
    //   //   //Rprintf("mcn = %d | ", mcn);
    //   //   // double m1 = arma::mean(log_lik.subvec(mcn-window_size, mcn-window_size+(int)(0.1*window_size)));
    //   //   // double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
    //   //   arma::uvec col_ind = arma::regspace<arma::uvec>(mcn-window_size, mcn-1);
    //   //   
    //   //   arma::mat A_hat_all_tmp = arma::diff(A_hat_all.submat(A_free_ind,col_ind), 1,1);
    //   //   arma::vec A_hat_tmp_std = arma::mean(A_hat_all_tmp, 1) / arma::stddev(A_hat_all_tmp, 1, 1);
    //   //   double m1 = arma::max(arma::abs(A_hat_tmp_std));
    //   //   
    //   //   arma::mat d_hat_tmp = arma::diff(d_hat_all.cols(mcn - window_size, mcn-1),1,1);
    //   //   arma::vec d_deriv_tmp_std = arma::mean(d_hat_tmp,1) / arma::stddev(d_hat_tmp,1,1);
    //   //   double m2 = arma::max(arma::abs(d_deriv_tmp_std));
    //   //   // if(std::abs(m1-m2) < tol){
    //   //   if(m1 < tol_a && m2 < tol_d){
    //   //     flag = true;
    //   //     mcn0 = mcn;
    //   //     Rcpp::Rcout << "arrive at " << mcn <<"\n";
    //   //     max_steps = std::min(max_steps, (int)(mcn + window_size));
    //   //   }
    //   //   else{
    //   //     if(print_proc) Rcpp::Rcout << "converge criterion: " <<" m1: "<< m1 << " m2: " << m2 << std::endl;
    //   //   }
    //   // }// end of if(!mcn)
    // }// end of if(arrive_flag)
    A_hat_all.col(mcn) = arma::vectorise(A_hat);
    d_hat_all.col(mcn) = d_hat;
    B_hat_all.col(mcn) = arma::vectorise(B_hat);
    A_deriv_all.col(mcn) = arma::vectorise(A_deriv);
    d_deriv_all.col(mcn) = d_deriv;
    
    check_info1(mcn) = mis_info1(0,0);
    check_info2(mcn) = mis_info2(0);
  }// end of mcn for loop

  return Rcpp::List::create(Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("d_hat") = d_hat,
                            Rcpp::Named("sigma_hat") = B_hat.t() * B_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("A_deriv_all") = A_deriv_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("d_deriv_all") = d_deriv_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("B_hat_all") = B_hat_all.cols(0, max_steps-1).t(),
                            Rcpp::Named("check_info1") = check_info1.subvec(0, max_steps-1),
                            Rcpp::Named("check_info2") = check_info2.subvec(0, max_steps-1),
                            Rcpp::Named("theta0") = theta0,
                            Rcpp::Named("oakes") = mis_info1 + mis_info2*mis_info2.t());
  
}
//' @export
// [[Rcpp::export]]
arma::mat test(arma::mat A, double c){
  return arma::max(A, c * arma::ones(arma::size(A)));
}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_mirt_arma(const arma::mat &response, const arma::mat &Q,
                        arma::mat A0, arma::vec d0, arma::mat theta0,
                        const arma::mat &A_true, const arma::vec d_true,
                        double alpha, double C0, int window_size=100,
                        int block_size=20, int max_steps=2000, bool print_proc=false){
  // deriv dimensions
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A0.n_cols;
  
  arma::mat U1, U2, V1, V2;
  arma::vec s1,s2;
  arma::svd(U1, s1, V1, A_true);
  U1 = U1.cols(0, K-1);
  
  
  //buffer for estimators
  arma::mat A_hat = A0;
  arma::vec d_hat = d0;
  arma::mat A_deriv = A0;
  arma::vec d_deriv = d0;
  
  arma::vec sin_angle = arma::zeros(max_steps);
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat A_deriv_all = arma::zeros(J*K, max_steps);
  arma::mat d_deriv_all = arma::zeros(J, max_steps);
  
  arma::vec std_deriv_mean_diff_all = arma::zeros(max_steps);
  int slide_window_count = 0;
  double std_deriv_mean_diff = 0;
  
  arma::mat inv_sigma = arma::eye(K, K);
  bool arrive_flag = false;
  int mcn = 0, mcn0 = 0;
  arma::uvec A_free_ind = arma::find(arma::vectorise(Q) > 0);
  
  double alpha_n = C0;
  for(mcn=0;mcn<max_steps;++mcn){
    Rcpp::checkUserInterrupt();
    // sample theta from posterior
    for(int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
    }
    
    // calculate deriv
    arma::mat M = theta0 * A0.t();
    M.each_row() += d0.t();
    M = 1.0 / (1.0 + arma::trunc_exp(-M));
    A_deriv = (response - M).t() * theta0; 
    A_deriv = A_deriv % Q;
    d_deriv = arma::sum(response - M, 0).t();
    A_deriv_all.col(mcn) = A_deriv.as_col();
    d_deriv_all.col(mcn) = d_deriv;
    
    if(mcn > window_size && !(mcn % block_size)){
      // calcu standardized deriv diff mean
      arma::uvec col_ind = arma::regspace<arma::uvec>(mcn-window_size+1, mcn);
      
      arma::mat d_deriv_tmp = d_deriv_all.cols(mcn - window_size+1, mcn);
      //Rcpp::Rcout << "size of mean: " << arma::size(arma::mean(d_deriv_tmp,1)) << " size of sd: " << arma::size(arma::stddev(d_deriv_tmp,1,1)) << std::endl;
      arma::vec d_deriv_tmp_sd = arma::mean(d_deriv_tmp,1) / arma::stddev(d_deriv_tmp,1,1);
      
      arma::mat A_deriv_tmp = A_deriv_all.submat(A_free_ind,col_ind);
      arma::vec A_deriv_tmp_sd = arma::mean(A_deriv_tmp, 1) / arma::stddev(A_deriv_tmp, 1, 1);
      
      std_deriv_mean_diff = (arma::accu(d_deriv_tmp_sd) + arma::accu(A_deriv_tmp_sd)) / double(J+A_free_ind.n_rows);
      std_deriv_mean_diff_all(slide_window_count++) = std_deriv_mean_diff;
      if(print_proc) Rcpp::Rcout << "stand deriv mean = " << std_deriv_mean_diff << std::endl;
      
      if(!arrive_flag){
        if(std::abs(std_deriv_mean_diff) < 0.05){
          mcn0 = mcn - 1;
          arrive_flag = true;
          Rcpp::Rcout << "arrive at " << mcn << std::endl;
          max_steps = mcn + window_size;
        }
      }
    }
    
    if(arrive_flag){
      int mcn_count = mcn - mcn0;
      alpha_n = C0 / std::pow(double(mcn_count), alpha);
      A0 += alpha_n / double(N) * A_deriv;
      d0 += alpha_n / double(N) * d_deriv;
      A_hat = double(mcn_count) / (mcn_count + 1.0) * A_hat + A0 / (mcn_count + 1.0);
      d_hat = double(mcn_count) / (mcn_count + 1.0) * d_hat + d0 / (mcn_count + 1.0);
    }
    else{
      A0 += alpha_n / double(N) * A_deriv;
      d0 += alpha_n / double(N) * d_deriv;
      A_hat = A0;
      d_hat = d0;
    }
    A_hat_all.col(mcn) = A_hat.as_col();
    d_hat_all.col(mcn) = d_hat;
    
    arma::svd(U2, s2, V2, A_hat);
    arma::vec s_tmp = arma::svd(U1.t() * U2.cols(0, K-1));
    
    if(print_proc) Rcpp::Rcout << "iter: " << mcn << "sin angle: " << std::sqrt(1.0 - s_tmp(K-1)) << std::endl;
    sin_angle(mcn) = std::sqrt(1.0 - s_tmp(K-1));
  }
  
  return Rcpp::List::create(Rcpp::Named("A0") = A0,
                            Rcpp::Named("d0") = d0,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-1).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-1).t(),
                            Rcpp::Named("A_deriv_all") = A_deriv_all.cols(0, mcn-1).t(),
                            Rcpp::Named("d_deriv_all") = d_deriv_all.cols(0, mcn-1).t(),
                            Rcpp::Named("sin_angle") = sin_angle.subvec(0, mcn-1),
                            Rcpp::Named("std_deriv_mean_diff_all") = std_deriv_mean_diff_all.subvec(0,slide_window_count-1));
}
//' @export
// [[Rcpp::export]]
Rcpp::List stem_init(const arma::mat &response, const arma::mat &Q,
                     arma::mat A0, arma::vec d0, arma::mat theta0, 
                     int max_steps, int window_size=40, 
                     int block_size=20, double tol=0.05){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A0.n_cols;
  bool arrive_flag = false;
  
  arma::mat sigma0 = arma::eye(K,K);
  arma::mat inv_sigma0 = arma::eye(K,K);
  arma::uvec rv(1), cv;
  arma::vec log_lik = arma::zeros(max_steps);
  
  long int mcn0=0, mcn = 0;
  for(mcn=0;mcn<max_steps;++mcn){
    Rcpp::checkUserInterrupt();
    // if(print_proc) Rprintf("\rStep  %d\t| ", mcn+1);
    
    // E step
    for(unsigned int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma0, A0, d0).t();
    }
    if(theta0.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // M step
    if(K > 1){
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-5);
      if(!sigma0.is_sympd()){
        Rcpp::stop("error after calcu_sigma\n");
      }
      inv_sigma0 = arma::inv_sympd(sigma0);
    }
    for(unsigned int j=0;j<J;++j){
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
      d0(j) = glm_res(cv.n_rows);
    }

    log_lik(mcn) = log_full_likelihood(response, A0, d0, theta0, inv_sigma0);
    if(!(mcn%block_size) && mcn>=(window_size-1)){
      double m1 = arma::mean(log_lik.subvec(mcn-window_size+1, mcn-window_size+(int)(0.1*window_size)));
      double m2 = arma::mean(log_lik.subvec(mcn-(int)(0.5*window_size), mcn-1));
      if(std::abs(m1-m2) < tol){
        arrive_flag = true;
        mcn0 = mcn;
        max_steps = std::min(max_steps, (int)(mcn+window_size));
        Rcpp::Rcout << "stem burn in finish at " << mcn << std::endl;
        break;
      }
      else{
        Rcpp::Rcout << "log likelihood mean diff: " << std::abs(m1-m2) << std::endl;
      }
    }
    
  }
  return Rcpp::List::create(Rcpp::Named("A0") = A0,
                            Rcpp::Named("d0") = d0,
                            Rcpp::Named("sigma0") = sigma0,
                            Rcpp::Named("theta0") = theta0,
                            Rcpp::Named("log_lik") = log_lik.subvec(0, mcn0-1));
}