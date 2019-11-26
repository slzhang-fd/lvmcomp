// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// calcu_sigma_cmle_cpp
arma::mat calcu_sigma_cmle_cpp(arma::mat theta, double tol);
RcppExport SEXP _lvmcomp_calcu_sigma_cmle_cpp(SEXP thetaSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(calcu_sigma_cmle_cpp(theta, tol));
    return rcpp_result_gen;
END_RCPP
}
// deriv_based_ARS
double deriv_based_ARS(arma::vec x, arma::vec theta_minus_k, int k, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::vec d);
RcppExport SEXP _lvmcomp_deriv_based_ARS(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(deriv_based_ARS(x, theta_minus_k, k, y_i, inv_sigma, A, d));
    return rcpp_result_gen;
END_RCPP
}
// F_prime_theta_y_eta_cpp_partial_credit1
arma::vec F_prime_theta_y_eta_cpp_partial_credit1(arma::vec x, arma::vec theta_minus_k, int k, arma::vec response_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_F_prime_theta_y_eta_cpp_partial_credit1(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP response_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(F_prime_theta_y_eta_cpp_partial_credit1(x, theta_minus_k, k, response_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// F_prime_theta_y_eta_cpp_partial_credit
double F_prime_theta_y_eta_cpp_partial_credit(double x, arma::vec theta_minus_k, int k, arma::vec response_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_F_prime_theta_y_eta_cpp_partial_credit(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP response_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(F_prime_theta_y_eta_cpp_partial_credit(x, theta_minus_k, k, response_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// F_theta_y_eta_cpp_partial_credit1
arma::vec F_theta_y_eta_cpp_partial_credit1(arma::vec x, arma::vec theta_minus_k, int k, arma::vec response_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_F_theta_y_eta_cpp_partial_credit1(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP response_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(F_theta_y_eta_cpp_partial_credit1(x, theta_minus_k, k, response_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// F_theta_y_eta_cpp_partial_credit
double F_theta_y_eta_cpp_partial_credit(double x, arma::vec theta_minus_k, int k, arma::vec response_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_F_theta_y_eta_cpp_partial_credit(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP response_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_i(response_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(F_theta_y_eta_cpp_partial_credit(x, theta_minus_k, k, response_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// deriv_based_ARS_partial_credit
double deriv_based_ARS_partial_credit(arma::vec x, arma::vec theta_minus_k, int k, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_deriv_based_ARS_partial_credit(SEXP xSEXP, SEXP theta_minus_kSEXP, SEXP kSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_minus_k(theta_minus_kSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(deriv_based_ARS_partial_credit(x, theta_minus_k, k, y_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// sample_theta_i_myars
arma::vec sample_theta_i_myars(arma::vec x, arma::vec theta0_i, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::vec d);
RcppExport SEXP _lvmcomp_sample_theta_i_myars(SEXP xSEXP, SEXP theta0_iSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta0_i(theta0_iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_theta_i_myars(x, theta0_i, y_i, inv_sigma, A, d));
    return rcpp_result_gen;
END_RCPP
}
// sample_theta_i_arms
arma::vec sample_theta_i_arms(arma::vec theta0_i, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::vec d);
RcppExport SEXP _lvmcomp_sample_theta_i_arms(SEXP theta0_iSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type theta0_i(theta0_iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_theta_i_arms(theta0_i, y_i, inv_sigma, A, d));
    return rcpp_result_gen;
END_RCPP
}
// sample_theta_i_myars_partial_credit
arma::vec sample_theta_i_myars_partial_credit(arma::vec x, arma::vec theta0_i, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_sample_theta_i_myars_partial_credit(SEXP xSEXP, SEXP theta0_iSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta0_i(theta0_iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_theta_i_myars_partial_credit(x, theta0_i, y_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// sample_theta_i_arms_partial_credit
arma::vec sample_theta_i_arms_partial_credit(arma::vec theta0_i, arma::vec y_i, arma::mat inv_sigma, arma::mat A, arma::mat D);
RcppExport SEXP _lvmcomp_sample_theta_i_arms_partial_credit(SEXP theta0_iSEXP, SEXP y_iSEXP, SEXP inv_sigmaSEXP, SEXP ASEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type theta0_i(theta0_iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_i(y_iSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_sigma(inv_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_theta_i_arms_partial_credit(theta0_i, y_i, inv_sigma, A, D));
    return rcpp_result_gen;
END_RCPP
}
// neg_loglik_logi_partial_credit
double neg_loglik_logi_partial_credit(arma::mat theta, arma::vec response_j, arma::vec A_j, arma::vec D_j);
RcppExport SEXP _lvmcomp_neg_loglik_logi_partial_credit(SEXP thetaSEXP, SEXP response_jSEXP, SEXP A_jSEXP, SEXP D_jSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_j(response_jSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type A_j(A_jSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D_j(D_jSEXP);
    rcpp_result_gen = Rcpp::wrap(neg_loglik_logi_partial_credit(theta, response_j, A_j, D_j));
    return rcpp_result_gen;
END_RCPP
}
// neg_loglik_deri_partial_credit
arma::vec neg_loglik_deri_partial_credit(arma::mat theta, arma::vec response_j, arma::vec A_j, arma::vec D_j);
RcppExport SEXP _lvmcomp_neg_loglik_deri_partial_credit(SEXP thetaSEXP, SEXP response_jSEXP, SEXP A_jSEXP, SEXP D_jSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type response_j(response_jSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type A_j(A_jSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D_j(D_jSEXP);
    rcpp_result_gen = Rcpp::wrap(neg_loglik_deri_partial_credit(theta, response_j, A_j, D_j));
    return rcpp_result_gen;
END_RCPP
}
// my_Logistic_cpp
arma::vec my_Logistic_cpp(arma::mat XX, arma::vec YY, arma::vec beta0, double d0);
RcppExport SEXP _lvmcomp_my_Logistic_cpp(SEXP XXSEXP, SEXP YYSEXP, SEXP beta0SEXP, SEXP d0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type XX(XXSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type YY(YYSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< double >::type d0(d0SEXP);
    rcpp_result_gen = Rcpp::wrap(my_Logistic_cpp(XX, YY, beta0, d0));
    return rcpp_result_gen;
END_RCPP
}
// my_Logistic_cpp_partial
arma::vec my_Logistic_cpp_partial(arma::mat XX, arma::vec YY, arma::vec beta0, arma::vec D0);
RcppExport SEXP _lvmcomp_my_Logistic_cpp_partial(SEXP XXSEXP, SEXP YYSEXP, SEXP beta0SEXP, SEXP D0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type XX(XXSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type YY(YYSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D0(D0SEXP);
    rcpp_result_gen = Rcpp::wrap(my_Logistic_cpp_partial(XX, YY, beta0, D0));
    return rcpp_result_gen;
END_RCPP
}
// stem_mirtc
Rcpp::List stem_mirtc(const arma::mat& response, const arma::mat& Q, arma::mat A0, arma::vec d0, arma::mat theta0, arma::mat sigma0, int T, bool parallel);
RcppExport SEXP _lvmcomp_stem_mirtc(SEXP responseSEXP, SEXP QSEXP, SEXP A0SEXP, SEXP d0SEXP, SEXP theta0SEXP, SEXP sigma0SEXP, SEXP TSEXP, SEXP parallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type response(responseSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d0(d0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma0(sigma0SEXP);
    Rcpp::traits::input_parameter< int >::type T(TSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    rcpp_result_gen = Rcpp::wrap(stem_mirtc(response, Q, A0, d0, theta0, sigma0, T, parallel));
    return rcpp_result_gen;
END_RCPP
}
// stem_pcirtc
Rcpp::List stem_pcirtc(const arma::mat& response, const arma::mat& Q, arma::mat A0, arma::mat D0, arma::mat theta0, arma::mat sigma0, int T, bool parallel);
RcppExport SEXP _lvmcomp_stem_pcirtc(SEXP responseSEXP, SEXP QSEXP, SEXP A0SEXP, SEXP D0SEXP, SEXP theta0SEXP, SEXP sigma0SEXP, SEXP TSEXP, SEXP parallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type response(responseSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D0(D0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma0(sigma0SEXP);
    Rcpp::traits::input_parameter< int >::type T(TSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    rcpp_result_gen = Rcpp::wrap(stem_pcirtc(response, Q, A0, D0, theta0, sigma0, T, parallel));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_lvmcomp_calcu_sigma_cmle_cpp", (DL_FUNC) &_lvmcomp_calcu_sigma_cmle_cpp, 2},
    {"_lvmcomp_deriv_based_ARS", (DL_FUNC) &_lvmcomp_deriv_based_ARS, 7},
    {"_lvmcomp_F_prime_theta_y_eta_cpp_partial_credit1", (DL_FUNC) &_lvmcomp_F_prime_theta_y_eta_cpp_partial_credit1, 7},
    {"_lvmcomp_F_prime_theta_y_eta_cpp_partial_credit", (DL_FUNC) &_lvmcomp_F_prime_theta_y_eta_cpp_partial_credit, 7},
    {"_lvmcomp_F_theta_y_eta_cpp_partial_credit1", (DL_FUNC) &_lvmcomp_F_theta_y_eta_cpp_partial_credit1, 7},
    {"_lvmcomp_F_theta_y_eta_cpp_partial_credit", (DL_FUNC) &_lvmcomp_F_theta_y_eta_cpp_partial_credit, 7},
    {"_lvmcomp_deriv_based_ARS_partial_credit", (DL_FUNC) &_lvmcomp_deriv_based_ARS_partial_credit, 7},
    {"_lvmcomp_sample_theta_i_myars", (DL_FUNC) &_lvmcomp_sample_theta_i_myars, 6},
    {"_lvmcomp_sample_theta_i_arms", (DL_FUNC) &_lvmcomp_sample_theta_i_arms, 5},
    {"_lvmcomp_sample_theta_i_myars_partial_credit", (DL_FUNC) &_lvmcomp_sample_theta_i_myars_partial_credit, 6},
    {"_lvmcomp_sample_theta_i_arms_partial_credit", (DL_FUNC) &_lvmcomp_sample_theta_i_arms_partial_credit, 5},
    {"_lvmcomp_neg_loglik_logi_partial_credit", (DL_FUNC) &_lvmcomp_neg_loglik_logi_partial_credit, 4},
    {"_lvmcomp_neg_loglik_deri_partial_credit", (DL_FUNC) &_lvmcomp_neg_loglik_deri_partial_credit, 4},
    {"_lvmcomp_my_Logistic_cpp", (DL_FUNC) &_lvmcomp_my_Logistic_cpp, 4},
    {"_lvmcomp_my_Logistic_cpp_partial", (DL_FUNC) &_lvmcomp_my_Logistic_cpp_partial, 4},
    {"_lvmcomp_stem_mirtc", (DL_FUNC) &_lvmcomp_stem_mirtc, 8},
    {"_lvmcomp_stem_pcirtc", (DL_FUNC) &_lvmcomp_stem_pcirtc, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_lvmcomp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
