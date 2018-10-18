#' Stochastic EM algorithm for solving generalized partial credit model
#' 
#' @param response Matrix contains {0,1,...,M-1} responses for N rows of examinees and J columns of items.
#' @param Q J(# of items) by K(# of latent traits) matrix, contains 0/1 indicate each item measures which latent abilities.
#' @param A0 J by K matrix, the initial value of loading matrix.
#' @param D0 J by M(# of categories of responses) matrix, the initial value of intercept parameters.
#' @param theta0 N(# of examinees) by J matrix, the initial value of latent traits for each subjects.
#' @param sigma0 K by K matrix, the initial value of the correlation matrix of latent traits.
#' @param m The initial length of Markov chain window, default value is 200.
#' @param TT The length of block size, default value is 20.
#' @param max.attempt The maximum attampt times if the precision criterion is not meet.
#' @param tol The tolerance of geweke statistic used for determining burn in size, default value is 1.5.
#' @param precision The pre-set precision value for determining the length of Markov chain, default value is 0.015.
#' @param frac1_value The first frac1_value percent and last 50 percent of Markov chain window to 
#'     be used to calculate geweke statistic.
#' 
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{A_hat}{The estimated loading matrix}
#'   \item{D_hat}{The estimated value of intercept parameters.}
#'   \item{sigma_hat}{The estimated value of correlation matrix of latent traits.}
# #'   \item{theta0}{N by J matrix, the estimated posterior mean of latent traits.}
#'   \item{burn_in_T}{The length of burn in size.}
#' }
#' 
#' @references 
#' An Improved Stochastic EM Algorithm for Large-Scale Full-information Item Factor Analysis
#' 
#' @examples
#' \dontrun{
#' # 2018-08-10
#' # run example of StEM of generalized partial credit mode
#' 
#' # generate response data from generalized partial credit model
#' M <- 3
#' N <- 2000
#' J <- 10
#' K <- 2
#' 
#' A <- matrix(0, J, K)
#' for(i in 1:K){
#'   A[((i-1)*J/K+1):(i*J/K),i] <- 1
#' }
#' A[which(A>0)] <- runif(J, 0.5, 1.5)
#' 
#' D <- matrix(0, J, M)
#' D[,2:M] <- rnorm(J*(M-1), 0, 0.3)
#' D <- t(apply(D, 1, cumsum))
#' theta <- matrix(rnorm(N*K), N, K)
#' thetaA <- theta %*% t(A)
#' response <- matrix(0, N, J)
#' for(i in 1:N){
#'   for(j in 1:J){
#'     tmp <- seq(0, M-1) * thetaA[i,j] + D[j,]
#'     prob <- exp(tmp)/sum(exp(tmp))
#'     response[i,j] <- which(rmultinom(1,1,prob)>0) - 1
#'   }
#' }
#' 
#' Q <- A != 0
#' A0 <- Q
#' D0 <- matrix(1, J, M)
#' D0[,1] <- 0
#' theta0 <- matrix(rnorm(N*K), N, K)
#' sigma0 <- diag(K)
#' 
#' pcirt_res <- StEM_pcirt(response, Q, A0, D0, theta0, sigma0)
#' }   
#' @importFrom coda geweke.diag mcmc
#' @importFrom stats sd cor
#' @export StEM_pcirt
StEM_pcirt <- function(response, Q, A0, D0, theta0, sigma0, m = 200, TT = 20, max.attempt = 40,
                      tol = 1.5, precision = 0.015, frac1_value = 0.2){
  M <- max(response) + 1
  N <- nrow(response)
  K <- ncol(A0)
  J <- nrow(A0)
  if(!all(Q %in% c(0,1)))
    stop("input Q should only contains 0/1 (True/False)! \n")
  if(ncol(response)!=J || nrow(Q)!=J || ncol(Q)!=K || nrow(theta0)!=N ||
     ncol(theta0)!=K || nrow(sigma0)!=K || ncol(sigma0)!=K)
    stop("The input argument dimension is not correct, please check!\n")
  burn_in_T <- 0
  temp <- stem_pcirtc(response, Q, A0, D0, theta0, sigma0, m)
  full_window <- result.window <- temp$res
  flag1 <- FALSE
  flag2 <- FALSE
  seg_all <- m / TT
  for(attempt.i in 1:max.attempt){
    cat("attemp: ",attempt.i,", ")
    if(!flag1){
      z.value <- abs(geweke.diag(mcmc(result.window), frac1 = frac1_value)$z)
      z.value <- (sum(z.value[1:(K^2)]^2, na.rm = T)/2 + sum(z.value[(K^2+1):length(z.value)]^2, na.rm = T)) / ((K^2-K)/2+sum(Q) + (M-1)*J)
      cat("Geweke stats: ", z.value,"\n")
      if(z.value < tol){
        flag1 <- TRUE
        cat("Burn in finished.\n")
      }
      else{
        A0 <- temp$A0
        D0 <- temp$D0
        theta0 <- temp$theta0
        sigma0 <- temp$sigma0
        temp <- stem_pcirtc(response, Q, A0, D0, theta0, sigma0, TT)
        result.window <- rbind(result.window[(TT+1):m,], temp$res)
        full_window <- rbind(full_window, temp$res)
        burn_in_T <- burn_in_T + 1
      }
    }
    if(flag1 && !flag2){
      res.sd <- NULL
      for(seg_i in 1:seg_all){
        res.sd <- rbind(res.sd,colMeans(result.window[((seg_i-1)*TT+1):(seg_i*TT),]))
      }
      prop.precision <- apply(res.sd, 2, sd) / sqrt(seg_all)
      cat("max sd: ", max(prop.precision), "\n")
      if(max(prop.precision) < precision*50/sqrt(N)){
        flag2 <- TRUE
        cat("Precision reached!\n")
      }
      else{
        seg_all <- seg_all + 1
        A0 <- temp$A0
        D0 <- temp$D0
        theta0 <- temp$theta0
        sigma0 <- temp$sigma0
        temp <- stem_pcirtc(response, Q, A0, D0, theta0, sigma0, TT)
        result.window <- rbind(result.window, temp$res)
        full_window <- rbind(full_window, temp$res)
      }
    }
    if(flag1 && flag2){
      res_tmp <- colMeans(full_window)
      sigma_hat <- matrix(res_tmp[1:(K^2)], K)
      A_hat <- matrix(res_tmp[(K^2+1):(K^2+J*K)], J)
      D_hat <- matrix(res_tmp[(K^2+J*K+1):(K^2+J*K+M*J)], J)
      return(list("A_hat"= A_hat,
                  "D_hat"= D_hat,
                  "sigma_hat" = sigma_hat,
                  "burn_in_T" = burn_in_T))
    }
  }
  cat("No feasible result.window founded\n")
  res_tmp <- colMeans(full_window)
  sigma_hat <- matrix(res_tmp[1:(K^2)], K)
  A_hat <- matrix(res_tmp[(K^2+1):(K^2+J*K)], J)
  D_hat <- matrix(res_tmp[(K^2+J*K+1):(K^2+J*K+M*J)], J)
  return(list("A_hat"= A_hat,
              "D_hat"= D_hat,
              "sigma_hat" = sigma_hat,
              "M" = nrow(full_window)/TT,
              "burn_in_T" = burn_in_T))
}