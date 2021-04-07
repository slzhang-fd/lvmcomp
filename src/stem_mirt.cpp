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
  // if(!parallel)
  //   omp_set_num_threads(1);
  // else
  //   omp_set_num_threads(omp_get_num_procs());
  
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
    for(unsigned int i=0;i<N;++i){
      // theta0.row(i) = sample_theta_i_myars(x, theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma, A0, d0).t();
    }
    // M step
    if(K > 1){
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-8);
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
      sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-8);
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
                      arma::mat A0, arma::vec d0, arma::mat B0,
                      arma::mat theta0, int ave_from = 100,
                      int max_steps=200,
                      bool print_proc=true){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A0.n_cols;
  
  arma::mat sigma0 = B0.t() * B0;
  arma::mat inv_sigma0 = arma::inv_sympd(sigma0);
  arma::uvec rv(1), cv;
  // arma::vec log_lik = arma::zeros(max_steps);
  
  // averaged results as an estimator
  arma::mat A_hat = A0;
  arma::vec d_hat = d0;
  arma::mat sigma_hat = sigma0;
  // arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  // arma::mat d_hat_all = arma::zeros(J, max_steps);
  // arma::mat B_hat_all = arma::zeros(K*K, max_steps);

  int mcn = 1;
  for(mcn=1;mcn<max_steps;++mcn){
    Rcpp::checkUserInterrupt();
    if(print_proc) Rprintf("\rStep  %d\t| ", mcn+1);
    
    // E step
    for(unsigned long int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma0, A0, d0).t();
    }
    if(theta0.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    
    // M step
    sigma0 = calcu_sigma_cmle_cpp(theta0, 1e-8);
    inv_sigma0 = arma::inv_sympd(sigma0);
    // B0 = calcu_sigma_cmle_cpp1(theta0);
    // sigma0 = B0.t() * B0;
    // inv_sigma0 = arma::inv_sympd(sigma0);
    
    for(unsigned int j=0;j<J;++j){
      rv(0) = j;
      cv = arma::find(Q.row(j));
      arma::vec glm_res = my_Logistic_cpp(theta0.cols(cv), response.col(j), A0.submat(rv, cv).t(), d0(j));
      A0.submat(rv, cv) = glm_res.subvec(0, cv.n_rows-1).t();
      d0(j) = glm_res(cv.n_rows);
    }
    
    // calculate A_hat based on burnin length
    if(mcn<ave_from){
      A_hat = A0;
      d_hat = d0;
      sigma_hat = sigma0;
    }
    else{
      double mcn0 = mcn - ave_from;
      A_hat = A_hat * mcn0 / (mcn0+1.0) + A0 / (mcn0+1.0);
      d_hat = d_hat * mcn0 / (mcn0+1.0) + d0 / (mcn0+1.0);
      sigma_hat = sigma_hat * mcn0 / (mcn0+1.0) + sigma0 / (mcn0+1.0);
    }
    // A_hat_all.col(mcn) = arma::vectorise(A_hat);
    // d_hat_all.col(mcn) = d_hat;
    // B_hat_all.col(mcn) = arma::vectorise(B_hat);

  }
  return Rcpp::List::create(Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("d_hat") = d_hat,
                            Rcpp::Named("sigma_hat") = sigma_hat);
                            // Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-1).t(),
                            // Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-1).t(),
                            // Rcpp::Named("B_hat_all") = B_hat_all.cols(0, mcn-1).t());

}
arma::mat soft_thre(const arma::mat &x, const arma::mat &mu){
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
  arma::vec d_deriv = arma::sum(response - M, 0).t() / N;


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
                    arma::mat &A, arma::mat &A_res1, arma::mat &A_res2, arma::mat &A_mis_info,
                    arma::vec &d, arma::vec &d_res1, arma::vec &d_res2, arma::mat &d_mis_info,
                    arma::mat &B, arma::mat &B_res1, arma::mat &B_res2, arma::mat &B_mis_info,
                    double lambda, arma::mat zero, arma::mat anchor,
                    int mcn, double alpha, double step, double c1, double c2){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A.n_cols;
  double step_n = step * std::pow(mcn, -alpha);
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  
  // update A
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= (1 - zero);
  A_res1 += step_n * (A_deriv - A_res1);
  A_res2 += step_n * ((M % (1-M)).t() * (theta % theta) / N - arma::square(A_deriv) - A_res2);
  A_mis_info = A_mis_info * (mcn-1.0)/mcn +
    arma::min(arma::max(A_res2 + arma::square(A_res1), c1*arma::ones(J,K)), c2*arma::ones(J,K))/mcn;
  A += step_n * (A_deriv / A_mis_info);
  A = soft_thre(A, (1-anchor) / A_mis_info * lambda * step_n);
  
  // update d
  arma::vec d_deriv = arma::sum(response - M, 0).t() / N;
  d_res1 += step_n * (d_deriv - d_res1);
  d_res2 += step_n * (arma::sum(M % (1-M), 0).t() / N - arma::square(d_deriv) - d_res2);
  d_mis_info = d_mis_info * (mcn-1.0)/mcn + 
    arma::min(arma::max(d_res2 + arma::square(d_res1), c1*arma::ones(J)),c2*arma::ones(J))/mcn;
  // arma::vec d_scale = arma::min(arma::max(1.0 / d_mis_info, c1 * arma::ones(J)),c2*arma::ones(J));
  d += step_n * (d_deriv / d_mis_info);
  
  B = update_sigma_one_step2(theta, B, B_res1, B_res2, B_mis_info, mcn, alpha, step, c1, c2);
}
// void RM_update1(const arma::mat &response, const arma::mat &theta, 
//                arma::mat &A, arma::vec &d, arma::mat &B, 
//                double lambda, arma::mat zero, arma::mat anchor,
//                double alpha, double step_n, double scale_bound){
//   int N = response.n_rows;
//   int J = response.n_cols;
//   int K = A.n_cols;
//   arma::mat M = theta * A.t();
//   M.each_row() += d.t();
//   M = 1.0 / (1.0 + arma::exp(-M));
//   arma::mat A_deriv = (response - M).t() * theta / N; 
//   A_deriv %= (1 - zero);
//   arma::mat A_scale = (M % (1-M)).t() * (theta % theta) / N;
//   //Rcpp::Rcout << A_scale << std::endl;
//   A_scale = arma::max(1.0/A_scale, scale_bound * arma::ones(J,K));
// 
//   A += step_n * (A_deriv % A_scale);
//   A = soft_thre(A, (1-anchor) % A_scale * lambda * step_n);
//   
//   arma::vec d_deriv = arma::sum(response - M, 0).t() / J;
//   arma::vec d_scale = arma::sum(M % (1-M), 0).t() / J;
//   //Rcpp::Rcout << d_scale << std::endl;
//   d_scale = arma::max(1.0 / d_scale, scale_bound * arma::ones(J));
//   d += step_n * (d_deriv % d_scale);
//   B = update_sigma_one_step1(theta, B, step_n/N);
// }
//' @export
// [[Rcpp::export]]
Rcpp::List sa_penmirt1(const arma::mat &response, arma::mat theta, arma::mat A, arma::vec d, arma::mat B, 
                      double lambda, arma::mat zero, arma::mat anchor, double alpha,
                      double step=1, int ave_from = 50,int max_steps=200, double tol = 0.01){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = B.n_cols;
  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  arma::mat A_all = arma::zeros(J*K, max_steps);
  arma::mat d_all = arma::zeros(J, max_steps);
  arma::mat B_all = arma::zeros(K*K, max_steps);
  arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  arma::mat d_hat_all = arma::zeros(J, max_steps);
  arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  
  arma::mat A_res1 = arma::zeros(J,K);
  arma::mat A_res2 = arma::zeros(J,K);
  arma::mat A_mis_info = arma::zeros(J,K);
  arma::vec d_res1 = arma::zeros(J);
  arma::vec d_res2 = arma::zeros(J);
  arma::vec d_mis_info = arma::zeros(J);
  arma::mat B_res1 = arma::zeros(K,K);
  arma::mat B_res2 = arma::zeros(K,K);
  arma::mat B_mis_info = arma::zeros(K,K);
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  int mcn=1;
  double eps = 1.0;
  for(mcn=1; mcn < max_steps && eps > tol; mcn++){
    Rcpp::checkUserInterrupt();
    Rcpp::Rcout << "\r iter: " << mcn <<" eps: " << eps;
    // stochastic E step
    for(unsigned long int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    // RM_update1(response, theta, A, d, B, lambda, zero, anchor, alpha, step_n, scale_bound);
    RM_update1(response, theta, A, A_res1, A_res2, A_mis_info,
                   d, d_res1, d_res2, d_mis_info,
                   B, B_res1, B_res2, B_mis_info,
                   lambda, zero, anchor,
                   mcn, alpha, step, 1e-3, 1000);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    if(mcn<ave_from){
      A_hat = A;
      d_hat = d;
      B_hat = B;
    }
    else{
      double mcn0 = mcn - ave_from;
      A_hat = A_hat * mcn0 / (mcn0+1.0) + A / (mcn0+1.0);
      d_hat = d_hat * mcn0 / (mcn0+1.0) + d / (mcn0+1.0);
      B_hat = arma::normalise(B_hat * mcn0 / (mcn0+1.0) + B / (mcn0+1.0));
    }
    A_all.col(mcn-1) = arma::vectorise(A_hat);
    d_all.col(mcn-1) = d_hat;
    B_all.col(mcn-1) = arma::vectorise(B_hat);
    A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    d_hat_all.col(mcn-1) = d_hat;
    B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
    if(mcn>=ave_from){
      double eps_A = std::max(arma::max(arma::abs(A_all.col(mcn-1) - A_all.col(mcn-2))),
                              arma::max(arma::abs(A_all.col(mcn-2) - A_all.col(mcn-3))));
      double eps_d = std::max(arma::max(arma::abs(d_all.col(mcn-1) - d_all.col(mcn-2))),
                              arma::max(arma::abs(d_all.col(mcn-2) - d_all.col(mcn-3))));
      double eps_B = std::max(arma::max(arma::abs(B_all.col(mcn-1) - B_all.col(mcn-2))),
                              arma::max(arma::abs(B_all.col(mcn-2) - B_all.col(mcn-3))));
      eps = std::max({eps_A, eps_d, eps_B});
    }

  }
  return Rcpp::List::create(Rcpp::Named("A_hat")=soft_thre(A_hat, (1-anchor) * lambda),
                            Rcpp::Named("d_hat")=d_hat,
                            Rcpp::Named("Sigma_hat")=B_hat.t() * B_hat,
                            Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-2).t(),
                            Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-2).t(),
                            Rcpp::Named("B_hat_all") = B_hat_all.cols(0, mcn-2).t());
}
// stochastic proximal gradient descent with acceleration
// void RM_update2(const arma::mat &response, const arma::mat &theta, 
//                const arma::mat &A, const arma::vec &d, const arma::mat &B, 
//                arma::mat &A_s, arma::vec &d_s, arma::mat &B_s,
//                double lambda, arma::mat zero, arma::mat anchor,
//                double alpha, double step_n){
//   int N = response.n_rows;
//   int J = response.n_cols;
//   arma::mat M = theta * A.t();
//   M.each_row() += d.t();
//   M = 1.0 / (1.0 + arma::exp(-M));
//   arma::mat A_deriv = (response - M).t() * theta / N; 
//   A_deriv %= (1 - zero);
//   arma::vec d_deriv = arma::sum(response - M, 0).t() / J;
//   
//   
//   A_s = A + step_n * A_deriv;
//   A_s = soft_thre(A_s, (1-anchor) * lambda * step_n);
//   d_s = d + step_n * d_deriv;
//   B_s = update_sigma_one_step1(theta, B, step_n/N);
// }
// //' @export
// // [[Rcpp::export]]
// Rcpp::List sa_penmirt2(const arma::mat &response, arma::mat theta, arma::mat A, arma::vec d, arma::mat B, 
//                       double lambda, arma::mat zero, arma::mat anchor, double alpha,
//                       double step=1, int max_steps=200){
//   int N = response.n_rows;
//   int J = response.n_cols;
//   int K = B.n_cols;
//   arma::mat A_s = A;
//   arma::vec d_s = d;
//   arma::mat B_s = B;
//   arma::mat A_s_old = A;
//   arma::vec d_s_old = d;
//   arma::mat B_s_old = B;
//   
//   arma::mat A_hat_all = arma::zeros(J*K, max_steps);
//   arma::mat d_hat_all = arma::zeros(J, max_steps);
//   arma::mat B_hat_all = arma::zeros(K*K, max_steps);
//   
//   arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
//   for(int mcn=1; mcn <= max_steps; mcn++){
//     Rcpp::checkUserInterrupt();
//     
//     // stochastic E step
//     for(unsigned long int i=0;i<N;++i){
//       theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
//     }
//     if(theta.has_nan()){
//       Rcpp::stop("nan in theta0\n");
//     }
//     // Robbins-Monro update
//     double factor = std::pow(mcn, -alpha);
//     double step_n = factor * step;
//     RM_update2(response, theta, A, d, B, A_s, d_s, B_s, lambda, zero, anchor, alpha, step_n);
//     
//     A = A_s + (mcn-1)/(mcn+2) * (A_s - A_s_old);
//     d = d_s + (mcn-1)/(mcn+2) * (d_s - d_s_old);
//     B = B_s + (mcn-1)/(mcn+2) * (B_s - B_s_old);
//     
//     A_s_old = A_s;
//     d_s_old = d_s;
//     B_s_old = B_s;
//     
//     inv_sigma = arma::inv_sympd(B_s.t() * B_s);
//     
//     // // Ruppert-averaging
//     // A_hat = A_hat * mcn / (mcn+1.0) + A / (mcn+1.0);
//     // //A_hat = soft_thre(A_hat, (1-anchor) * lambda * factor);
//     // d_hat = d_hat * mcn / (mcn+1.0) + d / (mcn+1.0);
//     // B_hat = arma::normalise(B_hat * mcn / (mcn+1.0) + B / (mcn+1.0));
//     
//     A_hat_all.col(mcn-1) = arma::vectorise(A_s);
//     d_hat_all.col(mcn-1) = d_s;
//     B_hat_all.col(mcn-1) = arma::vectorise(B_s);
//   }
//   return Rcpp::List::create(Rcpp::Named("A_hat")=soft_thre(A_s, (1-anchor) * lambda),
//                             Rcpp::Named("d_hat")=d_s,
//                             Rcpp::Named("Sigma_hat")=B_s.t() * B_s,
//                             Rcpp::Named("A_hat_all") = A_hat_all.cols(0, max_steps-1).t(),
//                             Rcpp::Named("d_hat_all") = d_hat_all.cols(0, max_steps-1).t(),
//                             Rcpp::Named("B_hat_all") = B_hat_all.cols(0, max_steps-1).t());
// }

void RM_update_conf(const arma::mat &response, const arma::mat &theta, 
                arma::mat &A, arma::mat &A_res1, arma::mat &A_res2, arma::mat &A_mis_info,
                const arma::mat &Q, 
                arma::vec &d, arma::vec &d_res1, arma::vec &d_res2, arma::vec &d_mis_info,
                arma::mat &B, arma::mat &B_res1, arma::mat &B_res2, arma::mat &B_mis_info,
                int mcn, double alpha, double step, double c1, double c2){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A.n_cols;
  
  double step_n = step * std::pow(mcn, -alpha);
  
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  
  // update A
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= Q;
  A_res1 += step_n * (A_deriv - A_res1);
  A_res2 += step_n * ((M % (1-M)).t() * (theta % theta) / N - arma::square(A_deriv) - A_res2);
  A_mis_info = A_mis_info * (mcn-1.0)/mcn + 
    arma::min(arma::max(A_res2 + arma::square(A_res1), c1 * arma::ones(J,K)), c2*arma::ones(J,K))/mcn;
  // Rcpp::Rcout << A_scale[0] << std::endl;
  A += step_n * (A_deriv / A_mis_info);
  
  // update d
  arma::vec d_deriv = arma::sum(response - M, 0).t() / N;
  d_res1 += step_n * (d_deriv - d_res1);
  d_res2 += step_n * (arma::sum(M % (1-M), 0).t() / N - arma::square(d_deriv) - d_res2);
  d_mis_info = d_mis_info * (mcn-1.0)/mcn + 
    arma::min(arma::max(d_res2 + arma::square(d_res1), c1*arma::ones(J)),c2*arma::ones(J))/mcn;
  // Rcpp::Rcout << d_scale[0] << std::endl;
  d += step_n * (d_deriv / d_mis_info);
  B = update_sigma_one_step2(theta, B, B_res1, B_res2, B_mis_info, mcn, alpha, step, c1, c2);
}

//' @export
// [[Rcpp::export]]
Rcpp::List sa_mirt_conf(const arma::mat &response, 
                        arma::mat A, const arma::mat &Q, arma::vec d,
                        arma::mat B, arma::mat theta,
                        double alpha, double step = 1.0,
                        int ave_from = 100, int max_steps = 200){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A.n_cols;

  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);

  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  // arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  // arma::mat d_hat_all = arma::zeros(J, max_steps);
  // arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  // arma::mat A_all = arma::zeros(J*K, max_steps);
  // arma::mat d_all = arma::zeros(J, max_steps);
  // arma::mat B_all = arma::zeros(K*K, max_steps);
  
  arma::mat A_res1 = arma::zeros(J,K);
  arma::mat A_res2 = arma::zeros(J,K);
  arma::mat A_mis_info = arma::zeros(J,K);
  arma::vec d_res1 = arma::zeros(J);
  arma::vec d_res2 = arma::zeros(J);
  arma::vec d_mis_info = arma::zeros(J);
  arma::mat B_res1 = arma::zeros(K,K);
  arma::mat B_res2 = arma::zeros(K,K);
  arma::mat B_mis_info = arma::zeros(K,K);
  // arma::vec log_lik = arma::zeros(max_steps);

  // int param_num = arma::accu(Q) + J + K*(K-1)/2;
  // arma::vec s;
  // arma::vec res2 = arma::zeros(param_num);
  // arma::mat res1 = arma::zeros(param_num, param_num);
  // arma::vec mis_info2 = arma::zeros(param_num);
  // arma::mat mis_info1 = arma::zeros(param_num, param_num);
  int mcn = 1;
  for(mcn=1; mcn < max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    // sampling process
    for(int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }

    // calculate missing information
    // s = s_func(theta, response, Q, inv_sigma, A, d);
    // res1 += factor * (h_func(theta, response, Q, inv_sigma, A, d) - s * s.t() - res1);
    // res2 += factor * (s - res2);
    // mis_info1 = mis_info1 * mcn / (mcn+1.0) + res1 / (mcn+1.0);
    // mis_info2 = mis_info2 * mcn / (mcn+1.0) + res2 / (mcn+1.0);
    
    // Robbin-Monro update
    RM_update_conf(response, theta, A, A_res1, A_res2, A_mis_info, Q,
                   d, d_res1, d_res2, d_mis_info,
                   B, B_res1, B_res2, B_mis_info,
                   mcn, alpha, step, 1e-3, 1000);
    inv_sigma = arma::inv_sympd(B.t() * B);

    // Ruppert-averaging
    // epsA = arma::abs(A_hat-A).max()/(mcn+1.0);
    // epsd = arma::abs(d_hat-d).max()/(mcn+1.0);
    // epsB = arma::abs(B_hat-B).max()/(mcn+1.0);
    // eps = std::max({epsA,epsd,epsB});
    // Rcpp::Rcout << "\r iter: " << mcn << " eps: " << eps;
    if(mcn<ave_from){
      A_hat = A;
      d_hat = d;
      B_hat = B;
    }
    else{
      double mcn0 = mcn - ave_from;
      A_hat = A_hat * mcn0 / (mcn0+1.0) + A / (mcn0+1.0);
      d_hat = d_hat * mcn0 / (mcn0+1.0) + d / (mcn0+1.0);
      B_hat = arma::normalise(B_hat * mcn0 / (mcn0+1.0) + B / (mcn0+1.0));
    }
    
    // A_all.col(mcn-1) = arma::vectorise(A);
    // d_all.col(mcn-1) = d;
    // B_all.col(mcn-1) = arma::vectorise(B);
    // A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    // d_hat_all.col(mcn-1) = d_hat;
    // B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
    
  }// end of mcn for loop

  return Rcpp::List::create(Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("d_hat") = d_hat,
                            Rcpp::Named("sigma_hat") = B_hat.t() * B_hat,
                            // Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_hat_all") = B_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("A_all") = A_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_all") = d_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_all") = B_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("oakes") = mis_info1 + mis_info2*mis_info2.t(),
                            Rcpp::Named("theta") = theta);

}
//' @export
// [[Rcpp::export]]
Rcpp::List sa_mirt_conf_no_ave(const arma::mat &response, 
                        arma::mat A, const arma::mat &Q, arma::vec d,
                        arma::mat B, arma::mat theta,
                        double alpha, double step = 1.0, int max_steps = 200){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A.n_cols;
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  
  arma::mat A_res1 = arma::zeros(J,K);
  arma::mat A_res2 = arma::zeros(J,K);
  arma::mat A_mis_info = arma::zeros(J,K);
  arma::vec d_res1 = arma::zeros(J);
  arma::vec d_res2 = arma::zeros(J);
  arma::vec d_mis_info = arma::zeros(J);
  arma::mat B_res1 = arma::zeros(K,K);
  arma::mat B_res2 = arma::zeros(K,K);
  arma::mat B_mis_info = arma::zeros(K,K);
  // arma::vec log_lik = arma::zeros(max_steps);
  
  // int param_num = arma::accu(Q) + J + K*(K-1)/2;
  // arma::vec s;
  // arma::vec res2 = arma::zeros(param_num);
  // arma::mat res1 = arma::zeros(param_num, param_num);
  // arma::vec mis_info2 = arma::zeros(param_num);
  // arma::mat mis_info1 = arma::zeros(param_num, param_num);
  int mcn = 1;
  for(mcn=1; mcn < max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    // sampling process
    for(int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    
    // calculate missing information
    // s = s_func(theta, response, Q, inv_sigma, A, d);
    // res1 += factor * (h_func(theta, response, Q, inv_sigma, A, d) - s * s.t() - res1);
    // res2 += factor * (s - res2);
    // mis_info1 = mis_info1 * mcn / (mcn+1.0) + res1 / (mcn+1.0);
    // mis_info2 = mis_info2 * mcn / (mcn+1.0) + res2 / (mcn+1.0);
    
    // Robbin-Monro update
    RM_update_conf(response, theta, A, A_res1, A_res2, A_mis_info, Q,
                   d, d_res1, d_res2, d_mis_info,
                   B, B_res1, B_res2, B_mis_info,
                   mcn, alpha, step, 1e-3, 1000);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
  }// end of mcn for loop
  
  return Rcpp::List::create(Rcpp::Named("A_hat") = A,
                            Rcpp::Named("d_hat") = d,
                            Rcpp::Named("sigma_hat") = B.t() * B,
                            // Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_hat_all") = B_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("A_all") = A_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_all") = d_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_all") = B_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("oakes") = mis_info1 + mis_info2*mis_info2.t(),
                            Rcpp::Named("theta") = theta);
  
}
//' @export
// [[Rcpp::export]]
Rcpp::List MHRM_conf(const arma::mat &response, 
                        arma::mat A, const arma::mat &Q, arma::vec d,
                        arma::mat B, arma::mat theta, double step = 1.0,
                        int max_steps = 200){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A.n_cols;
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  
  // arma::mat A_all = arma::zeros(J*K, max_steps);
  // arma::mat d_all = arma::zeros(J, max_steps);
  // arma::mat B_all = arma::zeros(K*K, max_steps);
  
  arma::mat A_res1 = arma::zeros(J,K);
  arma::mat A_res2 = arma::zeros(J,K);
  arma::mat A_mis_info = arma::zeros(J,K);
  arma::vec d_res1 = arma::zeros(J);
  arma::vec d_res2 = arma::zeros(J);
  arma::vec d_mis_info = arma::zeros(J);
  arma::mat B_res1 = arma::zeros(K,K);
  arma::mat B_res2 = arma::zeros(K,K);
  arma::mat B_mis_info = arma::zeros(K,K);
  // arma::vec log_lik = arma::zeros(max_steps);
  
  // int param_num = J*K + J + K*(K-1)/2;
  // arma::vec s;
  // arma::vec res2 = arma::zeros(param_num);
  // arma::mat res1 = arma::zeros(param_num, param_num);
  // arma::vec mis_info2 = arma::zeros(param_num);
  // arma::mat mis_info1 = arma::zeros(param_num, param_num);
  // double epsA, epsd, epsB;
  // double eps = 1.0;
  int mcn = 1;
  for(mcn=1; mcn < max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    // sampling process
    for(unsigned long int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    
    // calculate missing information
    // double step_n = std::pow(mcn, -1);
    // s = s_func(theta, response, arma::ones(J,K), inv_sigma, A, d);
    // res1 += step_n * (h_func(theta, response, arma::ones(J,K), inv_sigma, A, d) - s * s.t() - res1);
    // res2 += step_n * (s - res2);
    // mis_info1 = mis_info1 * (mcn-1.0) / mcn + res1 / mcn;
    // mis_info2 = mis_info2 * (mcn-1.0) / mcn + res2 / mcn;
    // 
    // arma::vec tmp = arma::diagvec(arma::inv_sympd(mis_info1 + mis_info2*mis_info2.t()));
    // Rcpp::Rcout << tmp.n_rows <<std::endl;
    // arma::mat tmpsub = tmp.head(J*K);
    // tmpsub.reshape(J,K);
    // Rcpp::Rcout << tmpsub.n_rows << std::endl;
    // Rcpp::Rcout << arma::size(tmpsub);
    // A_scale_hat = tmpsub;
    // d_scale_hat = tmp.subvec(J*K, J*K+J-1);
    // // Robbin-Monro update
    // MHRM_update(response, theta, A, A_scale_hat, Q, d, d_scale_hat,
    //                B, mcn, 1, 1e-3);
    RM_update_conf(response, theta, A, A_res1, A_res2, A_mis_info, Q,
                   d, d_res1, d_res2, d_mis_info,
                   B, B_res1, B_res2, B_mis_info,
                   mcn, 1, step, 1e-3, 1000);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    // epsA = arma::abs(A_hat-A).max()/(mcn+1.0);
    // epsd = arma::abs(d_hat-d).max()/(mcn+1.0);
    // epsB = arma::abs(B_hat-B).max()/(mcn+1.0);
    // eps = std::max({epsA,epsd,epsB});
    // Rcpp::Rcout << "iter: " << mcn << " eps: " << eps << std::endl;
    // Rcpp::Rcout << "\r iter: " << mcn;
    
    // A_all.col(mcn-1) = arma::vectorise(A);
    // d_all.col(mcn-1) = d;
    // B_all.col(mcn-1) = arma::vectorise(B);
    
  }// end of mcn for loop
  
  return Rcpp::List::create(Rcpp::Named("A") = A,
                            Rcpp::Named("d") = d,
                            Rcpp::Named("sigma") = B.t() * B,
                            // Rcpp::Named("A_all") = A_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_all") = d_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_all") = B_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("oakes") = mis_info1 + mis_info2*mis_info2.t(),
                            Rcpp::Named("theta") = theta);
  
}
// use only first order information
void RM_update_conf1(const arma::mat &response, const arma::mat &theta, 
                    arma::mat &A, const arma::mat &Q, 
                    arma::vec &d,
                    arma::mat &B,
                    int mcn, double alpha, double step){
  int N = response.n_rows;
  int J = response.n_cols;
  int K = A.n_cols;
  
  double step_n = step * std::pow(mcn, -alpha);
  
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  M = 1.0 / (1.0 + arma::exp(-M));
  
  // update A
  arma::mat A_deriv = (response - M).t() * theta / N; 
  A_deriv %= Q;
  A += step_n * A_deriv;
  
  // update d
  arma::vec d_deriv = arma::sum(response - M, 0).t() / N;
  d += step_n * d_deriv;
  B = update_sigma_one_step3(theta, B, step_n);
}

//' @export
// [[Rcpp::export]]
Rcpp::List sa_mirt_conf1(const arma::mat &response, 
                        arma::mat A, const arma::mat &Q, arma::vec d,
                        arma::mat B, arma::mat theta,
                        double alpha, double step = 1.0,
                        int ave_from = 100, int max_steps = 200){
  long int N = response.n_rows;
  long int J = response.n_cols;
  long int K = A.n_cols;
  
  arma::mat inv_sigma = arma::inv_sympd(B.t() * B);
  
  arma::mat A_hat = A;
  arma::vec d_hat = d;
  arma::mat B_hat = B;
  // arma::mat A_hat_all = arma::zeros(J*K, max_steps);
  // arma::mat d_hat_all = arma::zeros(J, max_steps);
  // arma::mat B_hat_all = arma::zeros(K*K, max_steps);
  // arma::mat A_all = arma::zeros(J*K, max_steps);
  // arma::mat d_all = arma::zeros(J, max_steps);
  // arma::mat B_all = arma::zeros(K*K, max_steps);
  
  // arma::vec log_lik = arma::zeros(max_steps);
  
  // int param_num = arma::accu(Q) + J + K*(K-1)/2;
  // arma::vec s;
  // arma::vec res2 = arma::zeros(param_num);
  // arma::mat res1 = arma::zeros(param_num, param_num);
  // arma::vec mis_info2 = arma::zeros(param_num);
  // arma::mat mis_info1 = arma::zeros(param_num, param_num);
  int mcn = 1;
  for(mcn=1; mcn < max_steps; mcn++){
    Rcpp::checkUserInterrupt();
    // sampling process
    for(int i=0;i<N;++i){
      theta.row(i) = sample_theta_i_arms(theta.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    if(theta.has_nan()){
      Rcpp::stop("nan in theta0\n");
    }
    
    // calculate missing information
    // s = s_func(theta, response, Q, inv_sigma, A, d);
    // res1 += factor * (h_func(theta, response, Q, inv_sigma, A, d) - s * s.t() - res1);
    // res2 += factor * (s - res2);
    // mis_info1 = mis_info1 * mcn / (mcn+1.0) + res1 / (mcn+1.0);
    // mis_info2 = mis_info2 * mcn / (mcn+1.0) + res2 / (mcn+1.0);
    
    // Robbin-Monro update
    RM_update_conf1(response, theta, A, Q,
                   d, B, mcn, alpha, step);
    inv_sigma = arma::inv_sympd(B.t() * B);
    
    // Ruppert-averaging
    // epsA = arma::abs(A_hat-A).max()/(mcn+1.0);
    // epsd = arma::abs(d_hat-d).max()/(mcn+1.0);
    // epsB = arma::abs(B_hat-B).max()/(mcn+1.0);
    // eps = std::max({epsA,epsd,epsB});
    // Rcpp::Rcout << "\r iter: " << mcn << " eps: " << eps;
    if(mcn<ave_from){
      A_hat = A;
      d_hat = d;
      B_hat = B;
    }
    else{
      double mcn0 = mcn - ave_from;
      A_hat = A_hat * mcn0 / (mcn0+1.0) + A / (mcn0+1.0);
      d_hat = d_hat * mcn0 / (mcn0+1.0) + d / (mcn0+1.0);
      B_hat = arma::normalise(B_hat * mcn0 / (mcn0+1.0) + B / (mcn0+1.0));
    }
    
    // A_all.col(mcn-1) = arma::vectorise(A);
    // d_all.col(mcn-1) = d;
    // B_all.col(mcn-1) = arma::vectorise(B);
    // A_hat_all.col(mcn-1) = arma::vectorise(A_hat);
    // d_hat_all.col(mcn-1) = d_hat;
    // B_hat_all.col(mcn-1) = arma::vectorise(B_hat);
    
  }// end of mcn for loop
  
  return Rcpp::List::create(Rcpp::Named("A_hat") = A_hat,
                            Rcpp::Named("d_hat") = d_hat,
                            Rcpp::Named("sigma_hat") = B_hat.t() * B_hat,
                            // Rcpp::Named("A_hat_all") = A_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_hat_all") = d_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_hat_all") = B_hat_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("A_all") = A_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("d_all") = d_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("B_all") = B_all.cols(0, mcn-2).t(),
                            // Rcpp::Named("oakes") = mis_info1 + mis_info2*mis_info2.t(),
                            Rcpp::Named("theta") = theta);
  
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