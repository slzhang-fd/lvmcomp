// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "lvmcomp_omp.h"
#include "arms_ori.h"
#include <random>
#include <thread>

//' @export
// [[Rcpp::export]]
double my_rand_unif(){
  //static thread_local std::random_device rd;
  static thread_local std::mt19937 mt(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  // Uniform Distribution
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  
  return dist(mt);
}
//' @export
//[[Rcpp::export]]
double obj_func_cpp(arma::mat sigma, arma::mat sigma_hat){
  arma::mat sigma_inv;
  if(sigma.is_sympd()){
    sigma_inv = arma::inv_sympd(sigma);
  }
  else{
    Rcpp::stop("error in obj_func");
  }
  return arma::accu( sigma_inv % sigma_hat ) + std::log(arma::det(sigma));
}
//' @export
// [[Rcpp::export]]
arma::mat calcu_sigma_cmle_cpp(const arma::mat &theta, double tol){
  int N = theta.n_rows;
  arma::mat sigma_hat = theta.t() * theta / (double)N;
  arma::mat sigma0 = arma::cor(theta);
  arma::mat sigma1 = sigma0;
  arma::mat tmp = sigma0;
  double eps = 1;
  double step = 0.1;
  while(eps < 0 || eps > tol){
    step = 0.1;
    if(!sigma0.is_sympd()){
      Rcpp::Rcout << sigma0 << "\n step=" << step <<std::endl;
      Rcpp::Rcout << "size of theta" << arma::size(theta) <<std::endl;
      Rcpp::stop("error in while eps\n");
    }
    tmp = arma::inv_sympd(sigma0);
    arma::mat sigma_deriv = - tmp * sigma_hat * tmp + tmp;
    sigma1 = sigma0 - step * sigma_deriv;
    sigma1.diag().ones();

    // -1e-7 here for preventing -0.0 fall into the while loop
    eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    while(!sigma1.is_sympd() || 
          eps < -1e-7 ||
          min(arma::eig_sym(sigma1)) < 1e-5){
      step *= 0.5;
      if(step < 1e-7){
        Rcpp::stop("Possible error in sigma estimation\n");
        // return sigma0;
        // return sigma0;
      }
      sigma1 = sigma0 - step * sigma_deriv;
      sigma1.diag().ones();
      eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    }
    sigma0 = sigma1;
  }
  return sigma0;
}
struct log_pos_theta_mirt_param{
  arma::vec theta_minus_k;
  unsigned int k;
  arma::vec y_i;
  arma::mat inv_sigma;
  arma::mat A;
  arma::vec d;
};
//' @export
//[[Rcpp::export]]
double log_sum_exp2(const arma::vec &tmp){
  arma::vec max_tmp_0 = arma::max(tmp, arma::zeros(tmp.n_rows));
  return arma::accu(max_tmp_0 + arma::log((arma::exp(- max_tmp_0) + arma::exp(tmp - max_tmp_0))));
}
double log_pos_theta_mirt(double x, void* params){
  struct log_pos_theta_mirt_param *d_params;
  d_params = static_cast<struct log_pos_theta_mirt_param *> (params);
  arma::vec theta = d_params->theta_minus_k;
  theta(d_params->k) =  x;
  arma::vec tmp = d_params->A * theta + d_params->d;
  return -0.5 * arma::as_scalar(theta.t() * d_params->inv_sigma * theta) + 
    arma::accu(d_params->y_i % tmp) - log_sum_exp2(tmp);
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

double log_sum_exp1(arma::mat tmp){
  arma::mat tmp_zeros = arma::zeros(tmp.n_rows, tmp.n_cols);
  arma::mat max_tmp_0 = arma::max(tmp_zeros, tmp);
  return arma::accu(max_tmp_0 + arma::log(arma::exp(- max_tmp_0) + arma::exp(tmp - max_tmp_0)));
}

double log_full_likelihood(const arma::mat &response, const arma::mat &A,
                           const arma::vec &d, const arma::mat &theta, const arma::mat &inv_sigma){
  long int N = theta.n_rows;
  arma::mat M = theta * A.t();
  M.each_row() += d.t();
  double res = 0.5 * N * std::log(arma::det(inv_sigma));
  for(long int i=0;i<N;++i){
    res -= 0.5 * arma::as_scalar(theta.row(i) * inv_sigma * theta.row(i).t());
  }
  res += arma::accu(M % response) - log_sum_exp1(M);
  return res / (double)N;
}

Rcpp::List Update_init_sa(const arma::mat &theta0,const arma::mat &response,const arma::mat &A0,
                          const arma::mat &Q,const arma::vec &d0,const arma::mat &inv_sigma0){
  double step_A = 10.0;
  double step_d = 10.0;
  
  arma::mat M = theta0 * A0.t();
  M.each_row() += d0.t();
  M = 1.0 / (1.0 + arma::trunc_exp(-M));
  arma::mat A_deriv = (response - M).t() * theta0; 
  A_deriv = A_deriv % Q;
  arma::mat d_deriv = arma::sum(response - M, 0).t();
  
  arma::mat A1 = A0 + step_A * A_deriv;
  while(log_full_likelihood(response, A0, d0, theta0, inv_sigma0) > 
          log_full_likelihood(response, A1, d0, theta0, inv_sigma0) && step_A > 1e-5){
    step_A *= 0.5;
    A1 = A0 + step_A * A_deriv;
  }
  
  arma::vec d1 = d0 + step_d * d_deriv;
  while(log_full_likelihood(response, A1, d0, theta0, inv_sigma0) > 
          log_full_likelihood(response, A1, d1, theta0, inv_sigma0) && step_d > 1e-5){
    step_d *= 0.5;
    d1 = d0 + step_d * d_deriv;
  }
  
  return Rcpp::List::create(Rcpp::Named("step_A")=step_A,
                            Rcpp::Named("step_d")=step_d,
                            Rcpp::Named("A1") = A1, 
                            Rcpp::Named("d1") = d1);
}

arma::mat update_sigma_one_step(const arma::mat &theta0, double step_sigma){
  long int N = theta0.n_rows;
  arma::mat sigma_hat = theta0.t() * theta0 / (double)N;
  
  arma::mat sigma0 = arma::cor(theta0);
  arma::mat sigma0_inv = arma::inv_sympd(sigma0);
  
  arma::mat sigma1 = sigma0;
  arma::mat sigma_deriv = - sigma0_inv * sigma_hat * sigma0_inv + sigma0_inv;
  sigma1 = sigma0 - step_sigma * sigma_deriv;
  sigma1.diag().ones();
  
  while(!sigma1.is_sympd() || 
    (obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat) < -1e-7) || 
    min(arma::eig_sym(sigma1)) < 1e-5){
    step_sigma *= 0.5;
    if(step_sigma < 1e-5){
      Rprintf("Possible error in sigma estimation\n");
      break;
    }
    sigma1 = sigma0 - step_sigma * sigma_deriv;
    sigma1.diag().ones();
  }
  return sigma1;
}
arma::mat update_sigma_one_step1(const arma::mat &theta0,
                                 const arma::mat &B0,
                                 double step_B){
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = (B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) - theta0.n_rows * B_inv;
  B_deriv(0,0) = 0;
  B_deriv = arma::trimatu(B_deriv);
  // return B_deriv;
  // one step and proximal operation
  // arma::mat B1 = arma::normalise(B0 + step_B * B_deriv);
  // Rcpp::Rcout << B_scale[2] << std::endl;
  arma::mat B1 = arma::normalise(B0 + step_B * B_deriv);
  if(!(B1.t()*B1).is_sympd()){
    Rcpp::Rcout << "B0" << B0;
    Rcpp::Rcout << "step" << step_B;
    Rcpp::Rcout << "B_deriv" << B_deriv / theta0.n_rows;
    Rcpp::Rcout << "B=" << B1;
    Rcpp::stop("B not sympd\n");
  }
  return B1;
}
double lambda_func(double lambda, const arma::vec& bk, const arma::vec& dk){
  int K = bk.n_rows;
  return arma::norm(bk / (arma::ones(K) + lambda * dk),2)-1;
}
arma::mat lagrange_opt(arma::mat B1, arma::mat B_scale){
  int K = B1.n_rows;
  for(int k=1;k<K;++k){
    arma::vec bk = B1.col(k);
    arma::vec dk = B_scale.col(k);
    double lambda1 = -0.5/dk(k);
    double lambda2 = 1.0/dk(k);
    for(int i=0;i<100;++i){
      if(lambda_func(lambda1, bk,dk)>0) break;
      lambda1 -= 0.1/dk(k);
    }
    for(int i=0;i<100;++i){
      if(lambda_func(lambda2, bk,dk)<0) break;
      lambda2 += 1.0/dk(k);
    }
    if(lambda_func(lambda1, bk,dk)<0 || lambda_func(lambda2, bk,dk)>0){
      Rcpp::Rcout << std::endl << "lambda1=" << lambda_func(lambda1, bk,dk) << " lambda2= "<<lambda_func(lambda2, bk,dk)<< bk;
      Rcpp::stop("lambda range error\n");
    }
    double eps = lambda_func((lambda1+lambda2)/2, bk,dk);
    while(eps < -1e-5 || eps > 1e-5){
      if(eps>1e-5){
        lambda1 = (lambda1+lambda2)/2;
      }
      else{
        lambda2 = (lambda1+lambda2)/2;
      }
      eps = lambda_func((lambda1+lambda2)/2, bk,dk);
    }
    B1.col(k) = bk / (arma::ones(K) + (lambda1+lambda2)/2 * dk);
  }
  return B1;
}

//' @export
// [[Rcpp::export]]
arma::mat update_sigma_one_step2(const arma::mat &theta0,
                                 const arma::mat &B0, arma::mat& B_res1, arma::mat& B_res2,
                                 arma::mat & B_mis_info,
                                 int mcn, double alpha, double step, double c1, double c2){
  double step_B = step * std::pow(mcn, -alpha);
  int N = theta0.n_rows;
  int K = B0.n_rows;
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = ((B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) - theta0.n_rows * B_inv)/N;
  arma::mat B_deriv2 = arma::zeros(K,K);
  for(int j=1;j<K;++j){
    for(int i=0;i<=j;++i){
      arma::mat B01 = B0;
      B01(i,j) += 1e-7;
      arma::mat B_inv1 = arma::inv(B01.t());
      arma::mat B_deriv1 = ((B_inv1 * theta0.t() * theta0 * B_inv1.t() * B_inv1) - theta0.n_rows * B_inv1)/N;
      B_deriv2(i,j) = -(B_deriv1(i,j) - B_deriv(i,j))/1e-7;
    }
  }

  B_res1 += step_B * (B_deriv - B_res1);
  B_res2 += step_B * (B_deriv2 - arma::square(B_deriv) - B_res2);
  B_mis_info = B_mis_info * (mcn-1.0)/mcn + 
    arma::min(arma::max(B_res2 + arma::square(B_res1), c1 * arma::ones(K,K)),c2*arma::ones(K,K))/mcn;
  arma::mat B_scale = 1.0 / B_mis_info;
  // Rcpp::Rcout << B_deriv2 <<std::endl;
  // arma::mat B_scale = arma::min(arma::max(1.0/B_mis_info,1e-3*arma::ones(K,K)),1000*arma::ones(K,K));
  // Rcpp::Rcout << B_scale<<std::endl;
  B_deriv(0,0) = 0;
  B_deriv = arma::trimatu(B_deriv);
  // return B_deriv;
  // one step and proximal operation
  // arma::mat B1 = arma::normalise(B0 + step_B * B_deriv);
  // Rcpp::Rcout << B_scale[2] << std::endl;
  arma::mat B1 = lagrange_opt(B0 + step_B * (B_deriv % B_scale), B_scale);
  // arma::mat B1 = arma::normalise(B0 + step_B * (B_deriv % B_scale));
  if(!(B1.t()*B1).is_sympd()){
    Rcpp::Rcout << "B0" << B0;
    Rcpp::Rcout << "step" << step_B;
    Rcpp::Rcout << "B_deriv" << B_deriv / theta0.n_rows;
    Rcpp::Rcout << "B=" << B1;
    Rcpp::stop("B not sympd\n");
  }
  return B1;
}
// use only first order information and Lagrangian
arma::mat update_sigma_one_step3(const arma::mat &theta0,
                                 const arma::mat &B0,
                                 double step_B){
  int N = theta0.n_rows;
  int K = B0.n_rows;
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = ((B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) - theta0.n_rows * B_inv)/N;
  B_deriv(0,0) = 0;
  B_deriv = arma::trimatu(B_deriv);
  arma::mat B1 = lagrange_opt(B0 + step_B * B_deriv, arma::ones(K,K));
  // arma::mat B1 = arma::normalise(B0 + step_B * (B_deriv % B_scale));
  if(!(B1.t()*B1).is_sympd()){
    Rcpp::Rcout << "B0" << B0;
    Rcpp::Rcout << "step" << step_B;
    Rcpp::Rcout << "B_deriv" << B_deriv / theta0.n_rows;
    Rcpp::Rcout << "B=" << B1;
    Rcpp::stop("B not sympd\n");
  }
  return B1;
}
//' @export
// [[Rcpp::export]]
arma::mat calcu_sigma_cmle_cpp1(const arma::mat &theta){
  int N = theta.n_rows;
  int K = theta.n_cols;
  arma::mat sigma_hat = theta.t() * theta / (double)N;
  arma::mat sigma0 = arma::cor(theta);
  arma::mat B0 = arma::chol(sigma0);
  arma::mat B_scale = arma::ones(K,K);
  arma::mat B1 = B0;
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = ((B_inv * theta.t() * theta * B_inv.t() * B_inv) - theta.n_rows * B_inv)/N;
  for(int j=1;j<K;++j){
    for(int i=0;i<=j;++i){
      arma::mat B01 = B0;
      B01(i,j) += 1e-7;
      arma::mat B_inv1 = arma::inv(B01.t());
      arma::mat B_deriv1 = ((B_inv1 * theta.t() * theta * B_inv1.t() * B_inv1) - theta.n_rows * B_inv1)/N;
      B_scale(i,j) = 1e-7 / (B_deriv(i,j) - B_deriv1(i,j));
    }
  }
  B_deriv(0,0) = 0;
  B_deriv = arma::trimatu(B_deriv);
  B1 = lagrange_opt(B0 + 1.0 * (B_deriv % B_scale), B_scale);
    // eps = obj_func_cpp(B0.t() * B0, sigma_hat) - obj_func_cpp(B1.t() * B1, sigma_hat);
    // if(eps < 0){
    //   Rcpp::stop("stem update sigma error\n");
    // }
  return B1;
}
//' @export
// [[Rcpp::export]]
arma::mat test_deriv(arma::mat B0, arma::mat theta0){
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = (B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) - theta0.n_rows * B_inv ;
  return B_deriv;
}
arma::vec s_func(arma::mat theta, arma::mat response, arma::mat Q, arma::mat Sigma_inv, arma::mat A, arma::vec d){
  int K = Sigma_inv.n_cols;
  int N = theta.n_rows;
  arma::uvec Q_ind = arma::find(arma::vectorise(Q.t())>0);
  arma::vec res1 = arma::zeros(K*(K-1)/2);
  int kk = 0;
  for(int i=0;i<(K-1);++i){
    for(int j=(i+1);j<K;++j){
      res1(kk) = -N*Sigma_inv(i,j) + arma::accu(Sigma_inv.row(i) * theta.t() * theta * Sigma_inv.col(j));
      kk += 1;
    }
  }
  arma::mat tmp = theta * A.t() + arma::ones(N) * d.t();
  tmp = response - 1/(1+exp(-tmp));
  arma::vec res2 = arma::vectorise(theta.t() * tmp);
  arma::vec res3 = sum(tmp, 0).t();
  return arma::join_cols(arma::join_cols(res1,res2.elem(Q_ind)),res3);
}

arma::mat h_func(arma::mat theta, arma::mat response, arma::mat Q, arma::mat Sigma_inv, arma::mat A, arma::vec d){
  int K = Sigma_inv.n_cols;
  int J = A.n_rows;
  int N = theta.n_rows;
  arma::uvec Q_ind = arma::find(arma::vectorise(Q.t())>0);
  arma::mat res = arma::zeros(K*(K-1)/2+Q_ind.n_elem+J,K*(K-1)/2+Q_ind.n_elem+J);
  arma::mat res1 = arma::zeros(K*(K-1)/2,K*(K-1)/2);
  int kk1 = 0;
  int kk2 = 0;
  arma::mat tmp1(K,K);
  arma::mat tmp2(K,K);
  arma::mat temp(N,N);
  for(int i1=0;i1<(K-1);++i1){
    for(int j1=(i1+1);j1<K;++j1){
      kk2 = 0;
      for(int i2=0;i2<(K-1);++i2){
        for(int j2=(i2+1);j2<K;++j2){
          tmp1 = arma::zeros(K, K);
          tmp2 = arma::zeros(K, K);
          tmp1(i1,j1) = 1;
          tmp1(j1,i1) = 1;
          tmp2(i2,j2) = 1;
          tmp2(j2,i2) = 1;
          temp = theta * Sigma_inv * tmp1 * Sigma_inv * tmp2 * Sigma_inv * theta.t();
          res(kk1, kk2) = N* arma::accu(Sigma_inv.row(i1) * tmp2 * Sigma_inv.col(j1)) - arma::accu(temp.diag());
          kk2 += 1;
        }
      }
      kk1 += 1;
    }
  }
  arma::mat res2 = arma::zeros(J*K, J*K);
  arma::mat res23 = arma::zeros(J*K, J);
  arma::mat tmp = theta * A.t() + arma::ones(N) * d.t();
  tmp = -exp(tmp)/square(1+exp(tmp));
  for(int i=0;i<J;++i){
    res2.submat((i*K), (i*K), (i+1)*K-1, (i+1)*K-1) = theta.t() * arma::diagmat(tmp.col(i)) * theta;
    res23.submat(i*K,i,(i+1)*K-1,i) = theta.t() * tmp.col(i);
  }
  res.submat(K*(K-1)/2,K*(K-1)/2,K*(K-1)/2+Q_ind.n_elem-1, K*(K-1)/2+Q_ind.n_elem-1) = res2.submat(Q_ind, Q_ind);
  res.submat(K*(K-1)/2+Q_ind.n_elem, K*(K-1)/2+Q_ind.n_elem, K*(K-1)/2+Q_ind.n_elem+J-1,K*(K-1)/2+Q_ind.n_elem+J-1) = arma::diagmat(sum(tmp,0).t());
  res.submat(K*(K-1)/2, K*(K-1)/2+Q_ind.n_elem, K*(K-1)/2 + Q_ind.n_elem-1, K*(K-1)/2+Q_ind.n_elem+J-1) = res23.rows(Q_ind);
  res.submat(K*(K-1)/2+Q_ind.n_elem,K*(K-1)/2, K*(K-1)/2+Q_ind.n_elem +J-1, K*(K-1)/2 +Q_ind.n_elem-1) = res23.rows(Q_ind).t();
  return -res;
}
//' @export
// [[Rcpp::export]]
arma::mat oakes(arma::mat theta0, arma::mat response, arma::mat Q, arma::mat Sigma, 
                arma::mat A, arma::vec d, int M=400){
  int N = theta0.n_rows;
  int K = theta0.n_cols;
  arma::vec x(3);
  x(0) = -5;
  x(1) = 0;
  x(2) = 5;
  arma::mat inv_sigma = arma::inv(Sigma);
  arma::vec s = s_func(theta0, response, Q, inv_sigma, A, d);
  arma::mat res1 = (h_func(theta0, response, Q, inv_sigma, A, d) - s * s.t());
  arma::vec res2 = s;
  for(int m=1;m<M;++m){
    Rcpp::checkUserInterrupt();
    Rprintf("m=%d\n",m);
    // for(int i=0;i<N;++i){
    //   for(int k=0;k<K;++k){
    //     theta0(i,k) = my_ars_cpp(x, theta0.row(i).t(), k+1, response.row(i).t(), inv_sigma, A, d);
    //   }
    // }
    for(unsigned long int i=0;i<N;++i){
      theta0.row(i) = sample_theta_i_arms(theta0.row(i).t(), response.row(i).t(), inv_sigma, A, d).t();
    }
    s = s_func(theta0, response, Q, inv_sigma, A, d);
    res1 += (h_func(theta0, response, Q, inv_sigma, A, d) - s * s.t());
    res2 += s;
  }
  res1 /= M;
  res2 /= M;
  return res1 + res2*res2.t();
}
//' @export
// [[Rcpp::export]]
double obj_det(arma::mat theta, arma::mat B){
  if(!B.is_sympd()){
    Rcpp::Rcout <<B;
  }
  arma::mat sigma_inv = arma::inv_sympd(B.t() * B);
  long int N = theta.n_rows;
  double res = -0.5 * N * std::log(arma::det(B.t() * B));
  for(long int i=0;i<N;++i){
    res -= 0.5 * arma::as_scalar(theta.row(i) * sigma_inv * theta.row(i).t());
  }
  return res;
}
arma::vec deriv_func(const arma::mat &B1, const arma::mat &theta0){
  arma::mat B_inv = arma::inv(B1.t());
  arma::mat B_deriv = (B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) / theta0.n_rows - B_inv;
  return B_deriv.as_col();
}
arma::mat update_H(arma::mat H_old, arma::vec s_k, arma::vec y_k){
  double rho_k = 1.0 / arma::dot(s_k, y_k);
  arma::mat V_k = arma::eye(s_k.n_rows,s_k.n_rows) - rho_k * s_k * y_k.t();
  return  (V_k * H_old * V_k.t() + rho_k * s_k * s_k.t());
}
//' @export
// [[Rcpp::export]]
arma::mat bfgs_update(const arma::mat &theta0,
                      const arma::mat &B0, bool flag, double tol = 1e-4){
  arma::mat H_0 = arma::eye(4,4);
  arma::mat B_inv = arma::inv(B0.t());
  arma::mat B_deriv = (B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) / theta0.n_rows - B_inv;
  arma::vec param_deriv0 = B_deriv.as_col();
  arma::vec param_deriv1(4);
  arma::vec x_0 = B0.as_col();
  arma::vec x_1(4);
  
  arma::vec p_0(4);
  arma::vec s_0 = arma::ones(4);
  arma::vec y_0 = arma::ones(4);
  arma::mat B1 = B0;
  
  double obj0 = obj_det(theta0, B0);
  double obj1 = obj0;
  double eps = 1;
  // one step and proximal operation
  while(eps > tol){
    double step_B = 1;
    p_0 = H_0 * param_deriv0;
    Rcpp::Rcout << "curvature condition: " << arma::dot(s_0, y_0) <<  " cos(p_k, g_k): " << arma::dot(p_0, param_deriv0) << std::endl;
    obj1 = obj0 - 1;
    while(obj0 > obj1){
      if(step_B < 1e-10){
        Rcpp::Rcout << "step_B low" << std::endl;
        return reshape(x_0,2,2);
      }
      param_deriv1 = deriv_func(B1, theta0);
      while(arma::dot(param_deriv1, p_0) <= arma::dot(param_deriv1, p_0)){
        step_B *= 1.1;
        x_1 = x_0 + step_B * p_0;
        B1 = arma::reshape(x_1,2,2);
        B1 = arma::normalise(B1);
        param_deriv1 = deriv_func(B1, theta0);
      }
      obj1 = obj_det(theta0, B1);
      step_B *= 0.5;
    }
    eps = obj1 - obj0;
    Rprintf("step size: %f, eps: %f\n", step_B, eps);
    obj0 = obj1;
    x_1 = B1.as_col();
    
    // calcu new deriv
    // B_inv = arma::inv(B1.t());
    // B_deriv = (B_inv * theta0.t() * theta0 * B_inv.t() * B_inv) / theta0.n_rows - B_inv;
    // param_deriv1 = B_deriv.as_col();
    param_deriv1 = deriv_func(B1, theta0);
    
    s_0 = x_1 - x_0;
    y_0 = param_deriv1 - param_deriv0;
    
    // update
    if(flag) H_0 = update_H(H_0, s_0, y_0);
    param_deriv0 = param_deriv1;
    x_0 = x_1;
  }
  
  return arma::reshape(x_0, 2, 2);
}
