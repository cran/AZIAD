#' Maximum likelihood estimate for general discrete and continuous distributions
#' @description calculate the Maximum likelihood estimate and the corresponding negative log likelihood value for
#'  general Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential distributions.
#' @usage new.mle(x,r,p,alpha1,alpha2,n,lambda,mean,sigma,dist,lowerbound=0.01,upperbound=10000)
#' @param x A vector of count data which should non-negative integers for discrete cases. Real valued random generation for continuous cases.
#' @param r An initial value for the number of success before which m failures are observed, where m is the element of x. Must be a positive number, but not required to be an integer.
#' @param p An initial value for the probability of success, should be a positive value within (0,1).
#' @param alpha1 An initial value for the first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 An initial value for the second shape parameter of beta distribution. Should be a positive number.
#' @param n An initial value for the number of trials. Must be a positive number, but not required to be an integer.
#' @param lambda An initial value for the rate. Must be a positive real number.
#' @param mean An initial value of the mean or expectation. A real number
#' @param sigma An initial value of the standard deviation. Must be a positive real number.
#' @param lowerbound A lower searching bound used in the optimization of likelihood function. Should be a small positive number. The default is 1e-2.
#' @param upperbound An upper searching bound used in the optimization of likelihood function. Should be a large positive number. The default is 1e4.
#' @param dist  The distribution used to calculate the maximum likelihood estimate. Can
#'  be one of {'poisson', 'geometric', 'nb', 'nb1', 'bb', 'bb1', 'bnb', 'bnb1', 'normal', 'halfnormal',
#' 'lognormal', 'exponential'}, which corresponds to Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential  distributions.
#'
#' @details new.mle calculate Maximum likelihood estimate and corresponding negative log likelihood
#'  of general Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential distributions.
#' @return A row vector containing the maximum likelihood estimate of the unknown parameters and the corresponding value of negative log likelihood.
#'
#'   If dist = poisson, the following values are returned:
#' \itemize{
#' \item lambda: the maximum likelihood estimate of \eqn{\lambda}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimate plugged-in.}
#' If dist = geometric, the following values are returned:
#' \itemize{
#' \item p: the maximum likelihood estimate of p.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimate plugged-in.}
#' If dist = nb, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of r.
#' \item p: the maximum likelihood estimate of p.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = nb1, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of rounded r (returns integer estimate).
#' \item p: the maximum likelihood estimate of p.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bb, the following values are returned:
#' \itemize{
#' \item n: the maximum likelihood estimate of n.
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bb1, the following values are returned:
#' \itemize{
#' \item n: the maximum likelihood estimate of rounded n (returns integer estimate).
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bnb, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of r.
#' \item a lpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bnb1, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of rounded r (returns integer estimate).
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = normal, the following values are returned:
#' \itemize{
#' \item mean: the maximum likelihood estimate of \eqn{\mu}.
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = lognormal, the following values are returned:
#' \itemize{
#' \item mean: the maximum likelihood estimate of \eqn{\mu}.
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#'\item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = halfnormal, the following values are returned:
#' \itemize{
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = exponential, the following values are returned:
#' \itemize{
#' \item lambda: the maximum likelihood estimate of \eqn{\lambda}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#'
#' @references \itemize{\item Dousti Mousavi et al. (2023) <doi:10.1080/00949655.2023.2207020>}
#' @export
#' @examples
#' set.seed(001)
#' x1=stats::rpois(1000,lambda=10)
#' new.mle(x1,lambda=3,dist="poisson")                         #9.776 -2611.242
#' x2=stats::rgeom(1000,prob=0.2)
#' new.mle(x2,p=0.5,dist="geometric")                          #0.1963865 -2522.333
#' x3=stats::rnbinom(1000,size=5,prob=0.3)
#' new.mle(x3,r=2,p=0.6,dist="nb")                             #5.113298 0.3004412 -3186.163
#' new.mle(x3,r=2,p=0.6,dist="nb1")                            #5 0.299904 -3202.223
#' x4=extraDistr::rbbinom(1000,size=4,alpha=2,beta=3)
#' new.mle(x4,n=10,alpha1=3,alpha2=4,dist="bb")                #3.99 1.78774 2.680009 -1533.982
#' new.mle(x4,n=10,alpha1=3,alpha2=4,dist="bb1")               #4 1.800849 2.711264 -1534.314
#' x5=extraDistr::rbnbinom(1000, size=5, alpha=3,beta=3)
#' new.mle(x5,r=5,alpha1=3,alpha2=4,dist="bnb")                #5.472647 3.008349 2.692704 -3014.372
#' new.mle(x5,r=5,alpha1=3,alpha2=4,dist="bnb1")               #5 2.962727 2.884826 -3014.379
#' x6=stats::rnorm(1000,mean=10,sd=2)
#' new.mle(x6,mean=2,sigma=1,dist="normal")                    #9.976704 2.068796 -2145.906
#' x7=stats::rlnorm(1000, meanlog = 1, sdlog = 4)
#' new.mle(x7,mean=2,sigma=2,dist="lognormal")                 #0.9681913 3.299503 -3076.156
#' x8=extraDistr::rhnorm(1000, sigma = 3)
#' new.mle(x8,sigma=2,dist="halfnormal")                       #3.103392 -1858.287
#' x9=stats::rexp(1000,rate=1.5)
#' new.mle(x9,lambda=3,dist="exponential")                     #1.454471 -625.3576
new.mle <- function (x, r , p , alpha1 , alpha2 , n, lambda, mean, sigma, dist, lowerbound = 0.01, upperbound = 10000 )
{
  if(dist == "poisson")
  {
    N = length(x)
    neg.log.lik<-function(y){
      l1 = y[1]
      ans = - sum(x) * log(l1) + N * l1 + sum(lgamma(x + 1))
      return(ans) }
    gp<-function(y){
      l2 = y[1]
      dl = - sum(x) / l2 + N
      return(dl)}
    estimate= stats::optim(par = lambda, fn = neg.log.lik, gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = upperbound)
    mle = matrix(c(estimate$par[1], -estimate$value), nrow=1)
    colnames(mle) = c("lambda", "loglik")
    return(mle)
  }
  if(dist == "geometric")
  {
    N = length(x)
    neg.log.lik<-function(y){
      l1 = y[1]
      ans = -sum(x) * log(1 - l1) - N * log(l1)
      return(ans) }
    gp<-function(y){
      l2 = y[1]
      dl = sum(x)/(1 - l2) - N/l2
      return(dl)}
    estimate = stats::optim(par = p, fn = neg.log.lik, gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = 1-lowerbound)
    mle = matrix(c(estimate$par[1], -estimate$value), nrow=1)
    colnames(mle) = c("p" , "loglik")
    return(mle)
  }
  if(dist=="nb")
  {
    N = length(x)
    neg.log.lik <- function(y) {
      r1 = y[1]
      p1 = y[2]
      ans = -N * r1 * log(p1) + N * lgamma(r1) -
        sum(x) * log(1 -p1) - sum(lgamma(x + r1)) + sum(lgamma(x + 1))
      return(ans)
    }
    gp <- function(y) {
      r2 = y[1]
      p2 = y[2]
      dr = -N * log(p2) + N * digamma(r2) - sum(digamma(x + r2))
      dp = sum(x)/(1 - p2) - N * r2/p2
      return(c(dr, dp))
    }
    estimate = stats::optim(par = c(r,p), fn = neg.log.lik, gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, 1 - lowerbound))
    mle = matrix(c(estimate$par[1], estimate$par[2], -estimate$value),nrow = 1)
    colnames(mle) = c("r","p", "loglik")
    return(mle)
  }
  if(dist=="nb1")
  { #return integer r
    N = length(x)
    neg.log.lik <- function(y) {
      r1 = y[1]
      p1 = y[2]
      ans = -N * r1 * log(p1) + N * lgamma(r1) -
        sum(x) * log(1 -p1) - sum(lgamma(x + r1)) + sum(lgamma(x + 1))
      return(ans)
    }
    gp <- function(y) {
      r2 = y[1]
      p2 = y[2]
      dr = -N * log(p2) + N * digamma(r2) - sum(digamma(x + r2))
      dp = sum(x)/(1 - p2) - N * r2/p2
      return(c(dr, dp))
    }
    estimate = stats::optim(par = c(r,p), fn = neg.log.lik,gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, 1 - lowerbound))
    rnew = max (1,round(estimate$par[1]))
    neg.log.lik1 <- function(y) {
      p1=y[1]
      ans = -N * rnew * log(p1) + N * lgamma(rnew) - sum(x) * log(1 -p1) - sum(lgamma(x + rnew)) + sum(lgamma(x + 1))
      return(ans)
    }
    gp1 <- function(y) {
      p2=y[1]
      dp = sum(x)/(1 - p2) - N * rnew/p2
      return(dp)
    }
    estimate1 = stats::optim(par = p, fn = neg.log.lik1,gr = gp1, method = "L-BFGS-B", lower = lowerbound,upper = 1-lowerbound)
    mle = matrix(c(rnew, estimate1$par[1], -estimate1$value),nrow = 1)
    colnames(mle) = c("r","p", "loglik")
    return(mle)
  }
  if(dist == "bb")
  {
    N = length(x)
    neg.log.lik <- function(y) {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans = -N * lgamma(n1 + 1) - N * lgamma(a1 + b1) + N * lgamma(a1) +
        N * lgamma(b1) + N * lgamma(n1 + a1 + b1) - sum(lgamma(x + a1)) -
        sum(lgamma(n1 - x + b1)) + sum(lgamma(x + 1)) + sum(lgamma(n1 - x + 1))
      return(ans)
    }
    gp <- function(y) {
      n2 = y[1]
      a2 = y[2]
      b2 = y[3]
      dn = -N * digamma(n2 + 1) + N * digamma(a2 + n2 + b2) - sum(digamma(n2 - x + b2)) + sum(digamma(n2 - x +  1))
      da = -N * digamma(a2 + b2) + N * digamma(a2) + N * digamma(n2 + a2 + b2) - sum(digamma(x + a2))
      db = -N * digamma(a2 + b2) + N * digamma(b2) + N * digamma(n2 + a2 + b2) - sum(digamma(n2 - x + b2))
      return(c(dn, da, db))
    }
    estimate = stats::optim(par = c(n,alpha1,alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(max(x) - lowerbound,lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    mle = matrix(c(estimate$par[1], estimate$par[2], estimate$par[3], -estimate$value), nrow = 1)
    colnames(mle) = c("n", "Alpha", "Beta", "loglik")
    return(mle)
  }
  if(dist == "bb1")
  { #return integer n
    N = length(x)
    neg.log.lik <- function(y) {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans = -N * lgamma(n1 + 1) - N * lgamma(a1 + b1) + N * lgamma(a1) +
        N * lgamma(b1) + N * lgamma(n1 + a1 + b1) - sum(lgamma(x + a1))-
        sum(lgamma(n1 - x + b1)) + sum(lgamma(x + 1)) +sum(lgamma(n1 - x + 1))
      return(ans)
    }
    gp <- function(y) {
      n2 = y[1]
      a2 = y[2]
      b2 = y[3]
      dn = -N * digamma(n2 + 1) + N * digamma(a2 + n2 + b2) - sum(digamma(n2 - x + b2)) + sum(digamma(n2 - x +  1))
      da = -N * digamma(a2 + b2) + N * digamma(a2) + N * digamma(n2 + a2 + b2) - sum(digamma(x + a2))
      db = -N * digamma(a2 + b2) + N * digamma(b2) + N * digamma(n2 + a2 + b2) - sum(digamma(n2 - x + b2))
      return(c(dn, da, db))
    }
    estimate = stats::optim(par = c(n,alpha1,alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(max(x) ,
                                                                    lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    nnew = max(1,round(estimate$par[1]))
    neg.log.lik1 <- function(y) {
      a1 = y[1]
      b1 = y[2]
      ans = -N * lgamma(nnew + 1) - N * lgamma(a1 + b1) + N * lgamma(a1) +
        N * lgamma(b1) + N * lgamma(nnew + a1 + b1) - sum(lgamma(x + a1)) -
        sum(lgamma(nnew - x + b1)) + sum(lgamma(x + 1)) +sum(lgamma(nnew - x + 1))
      return(ans)
    }
    gp1 <- function(y) {
      a2 = y[1]
      b2 = y[2]
      da = -N * digamma(a2 + b2) + N * digamma(a2) +N * digamma(nnew + a2 + b2) - sum(digamma(x + a2))
      db = -N * digamma(a2 + b2) + N * digamma(b2)+N * digamma(nnew + a2 + b2) - sum(digamma(nnew - x + b2))
      return(c(da, db))
    }
    estimate1 = stats::optim(par = c(alpha1,alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound, lowerbound), upper = c(upperbound, upperbound))
    mle = matrix(c(nnew, estimate1$par[1], estimate1$par[2], -estimate1$value), nrow = 1)
    colnames(mle) = c("n", "Alpha", "Beta", "loglik")
    return(mle)
  }
  if(dist == "bnb")
  {
    N = length(x)
    neg.log.lik <- function(y) {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans = -N * lgamma(a1 + r1) - N * lgamma(a1 + b1) +
        N * lgamma(r1) + N * lgamma(a1) + N * lgamma(b1) -
        sum(lgamma(x + r1)) - sum(lgamma(x + b1)) + sum(lgamma(x + 1)) +
        sum(lgamma(a1 + r1 + b1 + x))
      return(ans)
    }
    gp <- function(y) {
      r2 = y[1]
      a2 = y[2]
      b2 = y[3]
      dr = -N * digamma(a2 + r2) + N * digamma(r2) -
        sum(digamma(r2 + x)) + sum(digamma(a2 + r2 + b2 + x))
      da = -N * digamma(a2 + r2) - N * digamma(a2 + b2) +
        N * digamma(a2) + sum(digamma(a2 + r2 + b2 + x))
      db = -N * digamma(a2 + b2) + N * digamma(b2) -
        sum(digamma(b2 + x)) + sum(digamma(a2 + r2 + b2 + x))
      return(c(dr, da, db))
    }
    estimate = stats::optim(par = c(r,alpha1,alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound,
                                                                    lowerbound), upper = c(upperbound, upperbound, upperbound))
    mle = matrix(c(estimate$par[1], estimate$par[2], estimate$par[3], -estimate$value), nrow = 1)
    colnames(mle) = c("r", "Alpha", "Beta", "loglik")
    return(mle)
  }
  if(dist == "bnb1")
  { #return integer r
    N = length(x)
    neg.log.lik <- function(y) {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans = -N * lgamma(a1 + r1) - N * lgamma(a1 + b1) +
        N * lgamma(r1) + N * lgamma(a1) + N * lgamma(b1) -
        sum(lgamma(x + r1)) - sum(lgamma(x + b1)) + sum(lgamma(x + 1)) +
        sum(lgamma(a1 + r1 + b1 + x))
      return(ans)
    }
    gp <- function(y) {
      r2 = y[1]
      a2 = y[2]
      b2 = y[3]
      dr = -N * digamma(a2 + r2) + N * digamma(r2) -
        sum(digamma(r2 + x)) + sum(digamma(a2 + r2 + b2 + x))
      da = -N * digamma(a2 + r2) - N * digamma(a2 + b2) +
        N * digamma(a2) + sum(digamma(a2 + r2 + b2 + x))
      db = -N * digamma(a2 + b2) + N * digamma(b2) -
        sum(digamma(b2 + x)) + sum(digamma(a2 + r2 + b2 + x))
      return(c(dr, da, db))
    }
    estimate = stats::optim(par = c(r,alpha1,alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound,
                                                                    lowerbound), upper = c(upperbound, upperbound, upperbound))
    rnew = max (1, round(estimate$par[1]))
    neg.log.lik1 <- function(y) {
      a1 = y[1]
      b1 = y[2]
      ans1 = -N * lgamma(a1 + rnew) - N * lgamma(a1 + b1) +
        N * lgamma(rnew) + N * lgamma(a1) + N * lgamma(b1) -
        sum(lgamma(x + rnew)) - sum(lgamma(x + b1)) + sum(lgamma(x + 1)) +
        sum(lgamma(a1 + rnew + b1 + x))
      return(ans1)
    }
    gp1 <- function(y) {
      a2 = y[1]
      b2 = y[2]
      da1 = -N * digamma(a2 + rnew) - N * digamma(a2 + b2) +
        N * digamma(a2) + sum(digamma(a2 + rnew + b2 + x))
      db1 = -N * digamma(a2 + b2) + N * digamma(b2) -
        sum(digamma(b2 +  x)) + sum(digamma(a2 + rnew + b2 + x))
      return(c(da1, db1))
    }
    estimate1 = stats::optim(par = c(alpha1,alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound,lowerbound), upper = c( upperbound, upperbound))
    mle = matrix(c(rnew, estimate1$par[1], estimate1$par[2], -estimate1$value), nrow = 1)
    colnames(mle) = c("r", "Alpha", "Beta", "loglik")
    return(mle)
  }
  if(dist == "normal")
  {
    N = length(x)
    neg.log.lik <- function(y) {
      m1 = y[1]
      s1 = y[2]
      ans = 0.5*N*log(2*pi) + N*log(s1) + 0.5*sum(x^2)/(s1^2) + 0.5*N*(m1^2)/(s1^2) - m1*sum(x)/(s1^2)
      return(ans)
    }
    gp <- function(y) {
      m2 = y[1]
      s2 = y[2]
      dm = (N*m2)/(s2^2) - sum(x)/(s2^2)
      ds = N/s2 - sum(x^2)/(s2^3) - (N*(m2^2))/(s2^3) + 2*m2*sum(x)/(s2^3)
      return(c(dm,ds))
    }
    estimate = stats::optim( par = c(mean,sigma), fn = neg.log.lik,
                             gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                             upper = c(upperbound, upperbound))
    mle = matrix(c(estimate$par[1], estimate$par[2], -estimate$value), nrow = 1)
    colnames(mle) = c("mean", "sigma", "loglik")
    return(mle)
  }
  if(dist == "lognormal")
  {
    N = length(x)
    x=x[x>0]
    neg.log.lik <- function(y) {
      m1 = y[1]
      s1 = y[2]
      ans = 0.5*N*log(2*pi) + sum(log(x)) + N*log(s1) + (sum(log(x)-m1)^2)/(2*s1^2)
      return(ans)
    }
    gp <- function(y) {
      m2 = y[1]
      s2 = y[2]
      dm = (N*m2)/(s2^2) - sum(log(x))/(s2^2)
      ds = N/s2 - sum(log(x)^2)/(s2)^3 - N*(m2)^2/(s2)^3 + 2*m2*sum(log(x))/(s2^3)
      #ds = N/s2 - (sum(log(x)-m1)^2)/(s1^3)
      return(c(dm,ds))
    }
    estimate = stats::optim( par = c(mean,sigma), fn = neg.log.lik,
                             gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                             upper = c(upperbound, upperbound))
    mle = matrix(c(estimate$par[1], estimate$par[2], -estimate$value), nrow = 1)
    colnames(mle) = c("mean", "sigma", "loglik")
    return(mle)
  }
  if(dist == "halfnormal")
  {
    N = length(x)
    neg.log.lik <- function(y) {
      s1 = y[1]
      ans = -0.5*N*log(2)+N*log(s1)+0.5*N*log(pi)+sum(x^2)/(2*(s1^2))
      return(ans)
    }
    gp <- function(y) {
      s1 = y[1]
      ds = N/s1 - (sum(x^2))/(s1^3)
      return(ds)
    }
    estimate= stats::optim(par=sigma,fn=neg.log.lik, gr=gp, method="L-BFGS-B",lower=lowerbound,upper=upperbound)
    mle = matrix(c(estimate$par[1], -estimate$value), nrow = 1)
    colnames(mle) = c("sigma", "loglik")
    return(mle)
  }
  if(dist == "exponential")
  {
    N = length(x)
    x=x[x>=0]
    neg.log.lik <- function(y) {
      l1 = y[1]
      ans = -N*log(l1)+l1*sum(x)
      return(ans)
    }
    gp <- function(y) {
      l2 = y[1]
      dl = -N/l2+sum(x)
      return(dl)
    }
    estimate= stats::optim(par=lambda,fn=neg.log.lik, gr=gp, method="L-BFGS-B",lower=lowerbound,upper=upperbound)
    mle = matrix(c(estimate$par[1], -estimate$value), nrow = 1)
    colnames(mle) = c("lambda", "loglik")
    return(mle)
  }
}
library(foreach);library(extraDistr);library(base);library(corpcor);library(Rcpp);

#' Maximum likelihood estimate for Zero-Inflated or Zero-Altered discrete and continuous distributions.
#' @description Calculate the Maximum likelihood estimate and the corresponding negative log likelihood value for
#'  Zero-Inflated or Zero-Altered Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential distributions.
#' @usage zih.mle(x,r,p,alpha1,alpha2,n,lambda,mean,sigma,
#' type=c("zi","h"),dist,lowerbound=0.01,upperbound = 10000 )
#' @param x A vector of count data which should non-negative integers for discrete cases. Real-valued random generation for continuous cases.
#' @param r An initial value of the number of success before which m failures are observed, where m is the element of x. Must be a positive number, but not required to be an integer.
#' @param p An initial value of the probability of success, should be a positive value within (0,1).
#' @param alpha1 An initial value for the first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 An initial value for the second shape parameter of beta distribution. Should be a positive number.
#' @param n An initial value of the number of trials. Must be a positive number, but not required to be an integer.
#' @param lambda An initial value of the rate. Must be a postive real number.
#' @param mean An initial value of the mean or expectation.
#' @param sigma An initial value of the standard deviation. Must be a positive real number.
#' @param type the type of distribution used to calculate the sample estimate, where 'zi' stand for zero-inflated and 'h'  stands for hurdle distributions.
#' @param lowerbound A lower searching bound used in the optimization of likelihood function. Should be a small positive number. The default is 1e-2.
#' @param upperbound An upper searching bound used in the optimization of likelihood function. Should be a large positive number. The default is 1e4.
#' @param dist The distribution used to calculate the maximum likelihood estimate. Can be one
#' of {'poisson.zihmle', 'geometric.zihmle', 'nb.zihmle', 'nb1.zihmle', 'bb.zihmle', 'bb1.zihmle',
#'  'bnb.zihmle', 'bnb1.zihmle', 'normal.zihmle', 'halfnorm.zihmle', 'lognorm.zimle', 'exp.zihmle'} which
#'  corresponds to Zero-Inflated or Zero-Hurdle Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, log normal, half normal, and exponential distributions.
#' @details zih.mle calculate the Maximum likelihood estimate and the corresponding negative log likelihood
#'  of Zero-Inflated or Zero-Hurdle Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, log normal, half normal, and exponential distributions.
#'
#'   If dist = poisson.zihmle, the following values are returned:
#' \itemize{
#' \item lambda: the maximum likelihood estimate of \eqn{\lambda}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimate plugged-in.}
#' If dist = geometric.zihmle, the following values are returned:
#' \itemize{
#' \item p: the maximum likelihood estimate of p.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimate plugged-in.}
#' If dist = nb.zihmle, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of r.
#' \item p: the maximum likelihood estimate of p.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = nb1.zihmle, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of rounded r (returns integer estimate).
#' \item p: the maximum likelihood estimate of p.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bb.zihmle, the following values are returned:
#' \itemize{
#' \item n: the maximum likelihood estimate of n.
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bb1.zihmle, the following values are returned:
#' \itemize{
#' \item n: the maximum likelihood estimate of rounded n (returns integer estimate).
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bnb.zihmle, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of r.
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = bnb1.zihmle, the following values are returned:
#' \itemize{
#' \item r: the maximum likelihood estimate of rounded r (returns integer estimate).
#' \item alpha1: the maximum likelihood estimate of \eqn{\alpha_1}.
#' \item alpha2: the maximum likelihood estimate of \eqn{\alpha_2}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = normal.zihmle, the following values are returned:
#' \itemize{
#' \item mean: the maximum likelihood estimate of \eqn{\mu}.
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = lognorm.zihmle, the following values are returned:
#' \itemize{
#' \item mean: the maximum likelihood estimate of \eqn{\mu}.
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = halfnorm.zihmle, the following values are returned:
#' \itemize{
#' \item sigma: the maximum likelihood estimate of \eqn{\sigma}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' If dist = exp.zihmle, the following values are returned:
#' \itemize{
#' \item lambda: the maximum likelihood estimate of \eqn{\lambda}.
#' \item phi: the maximum likelihood estimate of \eqn{\phi}.
#' \item loglik: the value of negative log likelihood with maximum likelihood estimates plugged-in.}
#' @return A row vector containing the maximum likelihood estimate of the unknown parameters and the corresponding value of negative log likelihood.
#' @references \itemize{\item H. Aldirawi, J. Yang (2019). Model Selection and Regression Analysis for Zero-altered or Zero-inflated Data, Statistical Laboratory Technical Report, no.2019-01, University of Illinois at Chicago.}
#' @export
#' @examples
#' set.seed(007)
#' x1=sample.zi1(2000,phi=0.3,dist='poisson',lambda=2)
#' zih.mle(x1,lambda=10,dist="poisson.zihmle",type="zi")
#' #2.00341 0.3099267 -3164.528
#' x2=sample.zi1(2000,phi=0.3,dist='geometric',p=0.2)
#' zih.mle(x2,p=0.3,dist="geometric.zihmle",type="zi")
#' #0.1976744 0.2795942 -4269.259
#' x3=sample.zi1(2000,phi=0.3,dist='nb',r=10,p=0.3)
#' zih.mle(x3,r=2,p=0.2,dist="nb.zihmle",type="zi")
#' #10.18374 0.3033975 0.2919962 -6243.002
#' zih.mle(x3,r=2,p=0.2,dist="nb1.zihmle",type="zi")
#' #10 0.2995633 0.2919959 -6243.059
#' x4=sample.zi1(2000,phi=0.3,dist='bb',n=10,alpha1=2,alpha2=4)
#' zih.mle(x4,n=10,alpha1=3,alpha2=4,dist="bb.zihmle",type="zi")
#' #9.99 1.862798 3.756632 0.2643813 -3982.646
#' zih.mle(x4,n=10,alpha1=3,alpha2=4,dist="bb1.zihmle",type="zi")
#' #10 1.866493 3.76888 0.2644992 -3982.682
#' x5=sample.zi1(2000,phi=0.3,dist='bnb',r=5,alpha=3,alpha2=3)
#' zih.mle(x5,r=10,alpha1=3,alpha2=4,dist="bnb.zihmle",type="zi")
#' #6.936502 3.346791 2.32905 0.285682 -5088.173
#' zih.mle(x5,r=10,alpha1=3,alpha2=4,dist="bnb1.zihmle",type="zi")
#' #7 3.353377 2.313633 0.2855203 -5088.173
#' x6=sample.zi1(2000,phi=0.3,dist="normal",mean=10,sigma=2)
#' zih.mle(x6,mean=2,sigma=2,dist="normal.zihmle",type="zi")
#' #9.988447 2.015987 0.28 -4242.18
#' x7=sample.zi1(2000,phi=0.3,dist="lognormal",mean=1,sigma=4)
#' zih.mle(x7,mean=4,sigma=2,dist="lognorm.zihmle",type="zi")
#' #1.003887 3.945388 0.2985 -6544.087
#' x8=sample.zi1(2000,phi=0.3,dist="halfnormal",sigma=4)
#' zih.mle(x8,sigma=1,dist="halfnorm.zihmle",type="zi")
#' #1.292081 0.294 -8573.562
#' x9=sample.zi1(2000,phi=0.3,dist="exponential",lambda=20)
#' zih.mle(x9,lambda=10,dist="exp.zihmle",type="zi")
#' #20.1165 0.294 1614.786
#'
#' set.seed(008)
#' y1=sample.h1(2000,phi=0.3,dist='poisson',lambda=10)
#' zih.mle(y1,lambda=10,dist="poisson.zihmle",type="h")
#' #10.11842 0.3015 -4826.566
#' y2=sample.h1(2000,phi=0.3,dist='geometric',p=0.3)
#' zih.mle(y2,p=0.2,dist="geometric.zihmle",type="h")
#' #0.3050884 0.2925 -4061.65
#' y3=sample.h1(2000,phi=0.3,dist='nb',r=10,p=0.3)
#' zih.mle(y3,r=2,p=0.2,dist="nb.zihmle",type="h")
#' #9.50756 0.2862545 0.297 -6261.479
#' zih.mle(y3,r=2,p=0.2,dist="nb1.zihmle",type="h")
#' #10 0.2966819 0.297 -6261.932
#' y4=sample.h1(2000,phi=0.3,dist='bb',n=10,alpha1=2,alpha2=4)
#' zih.mle(y4,n=10,alpha1=3,alpha2=4,dist="bb.zihmle",type="h")
#' #9.99 1.894627 3.851142 0.293 -4092.983
#' zih.mle(y4,n=10,alpha1=3,alpha2=4,dist="bb1.zihmle",type="h")
#' #10 1.898415 3.863768 0.293 -4093.004
#' y5=sample.h1(2000,phi=0.3,dist='bnb',r=5,alpha=3,alpha2=3)
#' zih.mle(y5,r=10,alpha1=3,alpha2=4,dist="bnb.zihmle",type="h")
#' #3.875685 3.026982 3.874642 0.328 -5274.091
#' zih.mle(y5,r=10,alpha1=3,alpha2=4,dist="bnb1.zihmle",type="h")
#' #4 3.028185 3.756225 0.328 -5274.092
#' y6=sample.h1(2000,phi=0.3,dist="normal",mean=10,sigma=2)
#' zih.mle(y6,mean=2,sigma=2,dist="normal.zihmle",type="h")
#' #10.01252 1.996997 0.29 -4201.334
#' y7=sample.h1(2000,phi=0.3,dist="lognormal",mean=1,sigma=4)
#' zih.mle(y7,mean=4,sigma=2,dist="lognorm.zihmle",type="h")
#' #0.9305549 3.891624 0.287 -6486.92
#' y8=sample.h1(2000,phi=0.3,dist="halfnormal",sigma=4)
#' zih.mle(y8,sigma=1,dist="halfnorm.zihmle",type="h")
#' #1.26807 0.3 -8863.063
#' y9=sample.h1(2000,phi=0.3,dist="exponential",lambda=20)
#' zih.mle(y9,lambda=10,dist="exp.zihmle",type="h")
#' #20.26938 0.2905 1645.731
zih.mle <- function (x, r , p , alpha1 , alpha2 , n, lambda, mean, sigma, type = c("zi", "h"), dist, lowerbound = 0.01, upperbound = 10000 )
{
  if (dist == "poisson.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    m = length(t)
    neg.log.lik <- function(y){
      l=y[1]
      ans = -sum(t)*log(l) + l*m + sum (lgamma(t+1)) + m*log (1-exp(-l))
      return(ans)
    }
    gp <- function(y)
    {
      l = y[1]
      dl = -sum(t)/l + m + m*exp(-l)/(1-exp(-l))
      return(dl)
    }
    estimate = stats::optim(par = lambda, fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = upperbound)
    ans = estimate$par
    fvalue = estimate$value
    p0 = exp(-ans[1])
    neg.log.lik1 <- function(y)
    {
      l=y[1]
      ans1 = -sum(t)*log(l) + l*m + sum (lgamma(t+1)) - (N-m)*l
      return(ans1)
    }
    gp1 <- function(y)
    {
      l = y[1]
      dl1 = -sum(t)/l + 2*m -N
      return(dl1)
    }
    estimate1 = stats::optim(par = lambda, fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = lowerbound, upper = upperbound)
    ans1 = estimate1$par
    fvalue1 = estimate1$value
    p1 = exp(-ans1[1])
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("lambda", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1-p0) && type=="zi")
    {
      phi=1-m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("lambda", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1-p0) && type=="zi"){
      psi=min(m/N,1-p1)
      phi=1- psi/(1-p1)
      lik=-fvalue1+(N-m)*log(1-psi)+m*log(psi)
      mle = matrix(c(ans1[1], phi, lik), nrow = 1)
      colnames(mle) = c("lambda", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h")
    {
      warning("cannot obtain mle with the current model type, the output estimate is derived from general poisson distribution.")
      ans = new.mle(x, lambda=lambda,lowerbound, upperbound,dist = "poisson")
      mle = matrix(c(ans[1], 0, ans[2]), nrow = 1)
      colnames(mle) = c("lambda", "phi","loglik")
      return(mle)
    }
  }
  if (dist == "geometric.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y){
      p2=y[1]
      ans = -tsum * log(1-p2) - m * log(p2) + m*log(1-p2)
      return(ans)
    }
    gp <- function(y)
    {
      p2 = y[1]
      dl = tsum/(1-p2) - m/p2 - m/(1-p2)
      return(dl)
    }
    estimate = stats::optim(par = p, fn = neg.log.lik, gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = 1-lowerbound)
    ans = estimate$par
    fvalue = estimate$value
    p0 = ans[1]
    neg.log.lik1 <- function(y)
    {
      p2=y[1]
      ans1 = -tsum * log(1-p2) - m * log(p2)  - (N-m)*log(p2)
      return(ans1)
    }
    gp1 <- function(y)
    {
      p2 = y[1]
      dl1 = (tsum-m)/(1-p2) - m/p2 - (N-m)/p2
      return(dl1)
    }
    estimate1 = stats::optim(par = p, fn = neg.log.lik1, gr = gp1, method = "L-BFGS-B", lower = lowerbound, upper = 1-lowerbound)
    ans1 = estimate1$par
    fvalue1 = estimate1$value
    p1 = ans[1]
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1-p0) && type=="zi")
    {
      phi=1-m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1-p0) && type=="zi"){
      psi=min(m/N,1-p1)
      phi=1- psi/(1-p1)
      lik=-fvalue1+(N-m)*log(1-psi)+m*log(psi)
      mle = matrix(c(ans1[1], phi, lik), nrow = 1)
      colnames(mle) = c("p", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h")
    {
      warning("cannot obtain mle with the current model type, the output estimate is derived from general geometric distribution.")
      ans = new.mle(x, p=p,lowerbound, upperbound,dist = "geometric")
      mle = matrix(c(ans[1], 0, ans[2]), nrow = 1)
      colnames(mle) = c("p", "phi","loglik")
      return(mle)
    }
  }
  if (dist == "nb.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      r1 = y[1]
      p1 = y[2]
      ans = -sum(lgamma(t + r1)) + sum(lgamma(t + 1)) + m * lgamma(r1) -
        tsum * log(1 - p1) - m * r1 * log(p1) + m * log(1 - p1^r1)
      return(ans)
    }
    gp <- function(y) {
      r1 = y[1]
      p1 = y[2]
      dr = -sum(digamma(t + r1)) + m * digamma(r1) - m * log(p1) -
        m * log(p1)/(1 - p1^r1)* ((p1)^r1)
      dp = tsum/(1 - p1) - m * r1/p1 - m * r1/(1 - p1^r1) * ((p1)^(r1 - 1))
      return(c(dr, dp))
    }
    estimate = stats::optim(par = c(r, p), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, 1 - lowerbound))
    ans = estimate$par
    fvalue = estimate$value
    p0 = (ans[2])^(ans[1])
    neg.log.lik1 <- function(y)
    {
      r1 = y[1]
      p1 = y[2]
      ans1= -sum(lgamma(t+r1))+sum(lgamma(t+1))+m*lgamma(r1)-
        tsum*log(1-p1)-m*r1*log(p1)-(N-m)*r1*log(p1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      r1 = y[1]
      p1 = y[2]
      dr1= -sum(digamma(t+r1))+m*digamma(r1)-N*log(p1)
      dp1= tsum/(1-p1)  - N*r1/p1
      return(c(dr1,dp1))
    }
    estimate1 = stats::optim(par = c(r, p), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound,lowerbound), upper = c(upperbound,1 - lowerbound))
    ans1 = estimate1$par
    fvalue1 = estimate1$value
    p1 = (ans1[2])^(ans1[1])
    #estimation part
    if (!is.na(p0) && type == "h"){
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1-p0) && type=="zi")
    {
      phi=1-m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1-p0) && type=="zi")
    {
      psi=min(m/N,1-p1)
      phi=1-(psi/(1-p1))
      lik=-fvalue1+(N-m)*log(1-psi)+m*log(psi)
      mle = matrix(c(ans1[1], ans1[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general negative binomial distribution.")
      ans = new.mle(x,r=r,p=p,lowerbound, upperbound,dist="nb")
      mle = matrix(c(ans[1], ans[2], 0, ans[3]), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "nb1.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      r1 = y[1]
      p1 = y[2]
      ans = -sum(lgamma(t + r1)) + sum(lgamma(t + 1)) + m * lgamma(r1) -
        tsum * log(1 - p1) - m * r1 * log(p1) + m * log(1 - p1^r1)
      return(ans)
    }
    gp <- function(y) {
      r1 = y[1]
      p1 = y[2]
      dr = -sum(digamma(t + r1)) + m * digamma(r1) - m * log(p1) -
        m * log(p1)/(1 - p1^r1)* ((p1)^r1)
      dp = tsum/(1 - p1) - m * r1/p1 - m * r1/(1 - p1^r1) * ((p1)^(r1 - 1))
      return(c(dr, dp))
    }
    estimate = stats::optim(par = c(r, p), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, 1 - lowerbound))
    ans = estimate$par
    rnew = max (1, round(ans[1]))
    neg.log.liknew <- function(y) {
      p1 = y[1]
      ans = -sum(lgamma(t + rnew)) + sum(lgamma(t + 1)) + m * lgamma(rnew) -
        tsum * log(1 - p1) - m * rnew * log(p1) + m * log(1 - (p1)^rnew)
      return(ans)
    }
    gpnew <- function(y) {
      p2 = y[1]
      dp = tsum/(1 - p2) - m * rnew/p2 - m * rnew/(1 - p2^rnew) * ((p2)^(rnew - 1))
      return(dp)
    }
    estimatenew = stats::optim(par = p, fn = neg.log.liknew, gr = gpnew, method = "L-BFGS-B", lower = lowerbound,upper = 1-lowerbound)
    ansnew = estimatenew$par
    fvalue = estimatenew$value
    p0 = (ansnew[1])^(rnew)
    neg.log.lik1 <- function(y)
    {
      r1 = y[1]
      p1 = y[2]
      ans1= -sum(lgamma(t+r1))+sum(lgamma(t+1))+m*lgamma(r1)-
        tsum*log(1-p1)-m*r1*log(p1)-(N-m)*r1*log(p1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      r1 = y[1]
      p1 = y[2]
      dr1= -sum(digamma(t+r1))+m*digamma(r1)-N*log(p1)
      dp1= tsum/(1-p1)  - N*r1/p1
      return(c(dr1,dp1))
    }
    estimate1 = stats::optim(par = c(r, p), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound,lowerbound), upper = c(upperbound,1 - lowerbound))
    ansb = estimate1$par
    rnew1=max (1, round(ansb[1]))
    neg.log.lik1new <- function(y)
    {
      p1 = y[1]
      ans1= -sum(lgamma(t+rnew1))+sum(lgamma(t+1))+m*lgamma(rnew1)-
        tsum*log(1-p1)-m*rnew1*log(p1)-(N-m)*rnew1*log(p1)
      return(ans1)
    }
    gp1new <- function(y)
    {
      p1 = y[1]
      dp1= tsum/(1-p1)  - N*rnew1/p1
      return(dp1)
    }
    estimate1new = stats::optim(par = p, fn = neg.log.lik1new, gr = gp1new, method = "L-BFGS-B", lower = lowerbound, upper = 1 - lowerbound)
    ans1 = estimate1new$par
    fvalue1 = estimate1new$value
    p1 =(ans1[1])^(rnew1)
    #estimation part
    if (!is.na(p0) && type == "h"){
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(rnew, ansnew[1], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1-p0) && type=="zi")
    {
      phi=1-m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(rnew, ansnew[1], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1-p0) && type=="zi")
    {
      psi=min(m/N,1-p1)
      phi=1-(psi/(1-p1))
      lik=-fvalue1+(N-m)*log(1-psi)+m*log(psi)
      mle = matrix(c(rnew1, ans1[1], phi, lik), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general negative binomial 1 distribution.")
      ans = new.mle(x,r=r,p=p,lowerbound, upperbound,dist="nb1")
      mle = matrix(c(ans[1], ans[2], 0, ans[3]), nrow = 1)
      colnames(mle) = c("r", "p", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "bb.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    m = length(t) # m is the number of nonzero observations
    neg.log.lik <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + n1 + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(n1 + b1)
      ans = m * log(1 - exp(logB - logA)) + m * logA-
        m *lgamma(n1 + 1) - sum(lgamma(t + a1)) - sum(lgamma(n1 - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(n1 - t + 1)) + sum(lgamma(t + 1)) +m * lgamma(a1)
      return(ans)
    }
    gp <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + n1 + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(n1 + b1)
      dn = -m * exp(logB - logA) * (digamma(n1 + b1) - digamma(a1 + n1 + b1))/(1 - exp(logB - logA))+m * digamma(a1 + n1 + b1) -
        m * digamma(n1 + 1) - sum(digamma(n1 + b1 - t)) + sum(digamma(n1 - t + 1))
      da = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + n1 + b1))/(1 - exp(logB - logA)) + m * digamma(a1 + n1 + b1) -
        sum(digamma(t + a1)) - m * digamma(a1 + b1)+ m * digamma(a1)
      db = -m * exp(logB - logA) * (digamma(a1 + b1) + digamma(n1 + b1) - digamma(a1 + n1 + b1) - digamma(b1))/(1 - exp(logB - logA)) +
        m * digamma(b1) + m * digamma(a1 + n1 + b1)- sum(digamma(n1 - t + b1))-m * digamma(a1 + b1)
      return(c(dn, da, db))
    }
    estimate = stats::optim(par = c(n, alpha1, alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(max(x) - lowerbound,
                                                                    lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans = estimate$par
    fvalue = estimate$value
    p0 = beta(ans[2], ans[1] + ans[3])/beta(ans[2], ans[3])
    neg.log.lik1 <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans1 = -m * lgamma(n1 + 1) - sum(lgamma(t + a1)) - sum(lgamma(n1 - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(n1 - t + 1)) + sum(lgamma(t + 1)) +
        m * lgamma(n1 + a1 + b1) + m * lgamma(a1) + m * lgamma(b1) -
        (N - m) *lgamma(n1+b1) - (N - m) * lgamma(a1 + b1) + (N - m) * lgamma(n1 + a1 + b1) + (N - m) * lgamma(b1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      dn1 = -m * digamma(n1 + 1) - sum(digamma(n1 - t + b1)) + sum(digamma(n1 - t + 1)) + m*digamma(n1 + a1 + b1) +
        (N - m) * digamma(n1 + a1 + b1) - (N - m) * digamma(n1 + b1)
      da1 = -sum(digamma(t + a1)) - m * digamma(a1 + b1) + m * digamma(n1 + a1 + b1) + m * digamma(a1) -
        (N - m) * digamma(a1 + b1) + (N - m)*digamma(n1 + a1 + b1)
      db1 = -sum(digamma(n1 - t + b1)) - m* digamma(a1 + b1) + m * digamma(a1 + n1 + b1) + m * digamma(b1) +
        (N - m) * digamma(n1 + a1 + b1) + (N - m) * digamma(b1) - (N - m)* digamma(a1 + b1) - (N - m)* digamma(n1 + b1)
      return(c(dn1, da1, db1))
    }
    estimate1 = stats::optim(par = c(n, alpha1, alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(max(x) - lowerbound,
                                                                      lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans1 = estimate1$par
    fvalue1 = estimate1$value
    p1 = beta(ans1[2], ans1[1] + ans1[3])/beta(ans1[2], ans1[3])
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], ans[3], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1-p0) && type == "zi")
    {
      phi=1 - m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], ans[3], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    }else if (!is.na(p0) && (m/N) > (1-p0) && type == "zi")
    {
      psi = min(m/N, 1 - p1)
      phi = 1 - psi/(1 - p1)
      lik = -fvalue1 + (N - m) * log(1 - psi) + m * log(psi)
      mle = matrix(c(ans1[1], ans1[2], ans1[3], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (type != "zi" && type != "h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general beta binomial distribution.")
    ans = new.mle(x, n = n, alpha1 = alpha1, alpha2 = alpha2, lowerbound, upperbound, dist = "bb")
    mle = matrix(c(ans[1], ans[2], ans[3], 0, ans[4]), nrow = 1)
    colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
    return(mle)
    }
  }
  if (dist == "bb1.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    m = length(t)
    neg.log.lik <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + n1 + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(n1 + b1)
      ans = m * log(1 - exp(logB - logA)) + m * logA -
        m *lgamma(n1 + 1) - sum(lgamma(t + a1)) - sum(lgamma(n1 - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(n1 - t + 1)) + sum(lgamma(t + 1)) + m * lgamma(a1)
      return(ans)
    }
    gp <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + n1 + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(n1 + b1)
      dn = -m * exp(logB - logA) * (digamma(n1 + b1) -digamma(a1 +n1 + b1))/(1 - exp(logB - logA)) + m * digamma(a1 + n1 + b1) -
        m * digamma(n1 + 1) - sum(digamma(n1 - t + b1)) + sum(digamma(n1 -t + 1))
      da = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + n1 + b1))/(1 - exp(logB - logA)) + m * digamma(a1 + n1 + b1) -
        sum(digamma(t + a1)) - m * digamma(a1 + b1) + m * digamma(a1)
      db = -m * exp(logB - logA) * (digamma(a1 + b1) + digamma(n1 + b1) - digamma(a1 + n1 + b1) - digamma(b1))/(1 - exp(logB -logA)) +
        m * digamma(b1) + m * digamma(a1 + n1 + b1) - sum(digamma(n1 - t + b1)) - m * digamma(a1 + b1)
      return(c(dn, da, db))
    }
    estimate = stats::optim(par = c(n, alpha1, alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(max(x),
                                                                    lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans = estimate$par
    nnew = max(1, round(ans[1]))
    neg.log.liknew <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      logA = lgamma(a1 + nnew + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(nnew + b1)
      ans = m * log(1 - exp(logB - logA)) + m * logA -
        m *lgamma(nnew + 1) - sum(lgamma(t + a1)) - sum(lgamma(nnew - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(nnew - t + 1)) + sum(lgamma(t + 1)) + m * lgamma(a1)
      return(ans)
    }
    gpnew <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      logA = lgamma(a1 + nnew + b1) + lgamma(b1)
      logB = lgamma(a1 + b1) + lgamma(nnew + b1)
      da = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + nnew + b1))/(1 - exp(logB - logA)) + m * digamma(a1 + nnew + b1) -
        sum(digamma(t + a1)) - m * digamma(a1 + b1)+ m * digamma(a1)
      db = -m * exp(logB - logA) * (digamma(a1 + b1) + digamma(nnew + b1) - digamma(a1 + nnew + b1) - digamma(b1))/(1 - exp(logB -logA)) +
        m * digamma(b1) + m * digamma(a1 + nnew + b1) - sum(digamma(nnew - t + b1)) - m * digamma(a1 + b1)
      return(c(da, db))
    }
    estimatenew = stats::optim(par = c(alpha1, alpha2), fn = neg.log.liknew,
                               gr = gpnew, method = "L-BFGS-B", lower = c(lowerbound, lowerbound), upper = c(upperbound, upperbound))
    ansnew=estimatenew$par
    fvalue = estimatenew$value
    p0 = exp(base::lbeta(ansnew[1], nnew + ansnew[2]) - base::lbeta(ansnew[1], ansnew[2]))
    neg.log.lik1 <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      ans1 = -m *lgamma(n1 + 1) - sum(lgamma(t + a1)) - sum(lgamma(n1 - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(n1 - t + 1)) + sum(lgamma(t + 1)) +
        m*lgamma(n1 + a1 + b1) + m * lgamma(a1) + m*lgamma(b1) -
        (N - m) * lgamma(n1+b1) - (N - m) * lgamma(a1 + b1) + (N - m) * lgamma(n1 + a1 + b1) + (N - m) * lgamma(b1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      n1 = y[1]
      a1 = y[2]
      b1 = y[3]
      dn1 = -m * digamma(n1 + 1) - sum(digamma(n1 - t + b1)) + sum(digamma(n1 - t + 1)) + m*digamma(n1 + a1 + b1) +
        (N - m) * digamma(n1 + a1 + b1) - (N - m) * digamma(n1 + b1)
      da1 = -sum(digamma(t + a1)) - m * digamma(a1 + b1) + m*digamma(n1 + a1 + b1) + m * digamma(a1) -
        (N-m)* digamma(a1 + b1) + (N - m)*digamma(n1 + a1 + b1)
      db1 = -sum(digamma(n1 - t+ b1)) - m * digamma(a1+b1) + m * digamma(a1 + n1 + b1) + m * digamma(b1) +
        (N - m)*digamma(n1 + a1 + b1) + (N - m) * digamma(b1) -(N - m) * digamma(a1 + b1) - (N - m) * digamma(n1 + b1)
      return(c(dn1, da1, db1))
    }
    estimate1 = stats::optim(par = c(n, alpha1, alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(max(x) ,
                                                                      lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans1 = estimate1$par
    nnew1 = max (1, round (ans1[1]))
    neg.log.lik1new <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      ans1 = -m *lgamma(nnew1 + 1) - sum(lgamma(t + a1)) - sum(lgamma(nnew1 - t + b1)) -
        m * lgamma(a1 + b1) + sum(lgamma(nnew1 - t + 1)) + sum(lgamma(t + 1)) +
        m * lgamma(nnew1 + a1 + b1) + m * lgamma(a1) + m * lgamma(b1) -
        (N - m) * lgamma(nnew1 + b1) - (N - m) * lgamma(a1 + b1) + (N - m) * lgamma(nnew1 + a1 + b1) + (N - m) * lgamma(b1)
      return(ans1)
    }
    gp1new <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      da1 = -sum(digamma(t + a1)) - m * digamma(a1 + b1) + m * digamma(nnew1 + a1 + b1) + m * digamma(a1) -
        (N - m) * digamma(a1 + b1) + (N - m) * digamma(nnew1 + a1 + b1)
      db1 = -sum(digamma(nnew1 - t + b1)) - m* digamma(a1 + b1) + m * digamma(a1 + nnew1 + b1) + m * digamma(b1) +
        (N - m) * digamma(nnew1 + a1 + b1) + (N - m) * digamma(b1) - (N - m) * digamma(a1 + b1) - (N - m) * digamma(nnew1 + b1)
      return(c(da1, db1))
    }
    estimate1new = stats::optim(par = c(alpha1, alpha2), fn = neg.log.lik1new,
                                gr = gp1new, method = "L-BFGS-B", lower = c(lowerbound, lowerbound), upper = c(upperbound, upperbound))
    ansnew1=estimate1new$par
    fvalue1 = estimate1new$value
    p1 = exp(base::lbeta(ansnew1[1], nnew1 + ansnew1[2])-base::lbeta(ansnew1[1], ansnew1[2]))
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(nnew, ansnew[1], ansnew[2], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1 - p0) && type == "zi")
    {
      phi=1-m/N/(1-p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(nnew, ansnew[1], ansnew[2], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1 - p0) && type == "zi")
    {
      psi = min(m/N , (1 - p1))
      phi = 1 - psi/(1 - p1)
      lik = -fvalue1 + (N - m) * log(1 - psi) + m * log(psi)
      mle = matrix(c(nnew1, ansnew1[1], ansnew1[2], phi, lik), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (type != "zi" && type != "h")
    {
      warning("cannot obtain mle with the current model type, the output estimate is derived from general beta binomial 1 distribution.")
      ans = new.mle(x, n = n, alpha1 = alpha1, alpha2 = alpha2, lowerbound, upperbound, dist = "bb1")
      mle = matrix(c(ans[1], ans[2], ans[3], 0, ans[4]), nrow = 1)
      colnames(mle) = c("n", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "bnb.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    m = length(t)
    neg.log.lik <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      ans = m * log(1 - exp(logB - logA)) + m * lgamma(a1) -
        m * lgamma(a1 + r1) - sum(lgamma(r1 + t)) - sum(lgamma(t + b1 )) -
        m * lgamma(a1 + b1) + sum(lgamma(t + 1))+ m * lgamma(r1) + sum(lgamma(a1 + r1 + b1 + t)) + m *lgamma(b1)
      return(ans)
    }
    gp <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      dr = -m * exp(logB - logA) * (digamma(a1 + r1) - digamma(a1 + r1 + b1))/(1 - exp(logB - logA)) -
        sum(digamma(r1 + t)) - m * digamma(a1 + r1) + m * digamma(r1) + sum(digamma(a1 + r1 + b1 + t))
      da = -m * exp(logB - logA) * (digamma(a1 + r1) + digamma(a1 + b1) - digamma(a1 + r1 + b1)-digamma(a1))/(1 - exp(logB -logA)) +
        m * digamma(a1) - m * digamma(a1 + r1) - m * digamma(a1 + b1) + sum(digamma(a1 + r1 + b1 + t))
      db = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + r1 + b1))/(1 - exp(logB - logA))-
        sum(digamma(t + b1)) - m * digamma(a1 + b1) + sum(digamma(a1 + r1 +t+ b1)) + m * digamma(b1)
      return(c(dr, da, db))
    }
    estimate = stats::optim(par = c(r, alpha1, alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound,
                                                                    lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans = estimate$par
    fvalue = estimate$value
    p0 = exp(base::lbeta(ans[1] + ans[2], ans[3]) - (base::lbeta(ans[2], ans[3])))
    neg.log.lik1 <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      ans1 = m * log(1 - exp(logB - logA)) + m * lgamma(a1) -
        m * lgamma(a1 + r1) - sum(lgamma(r1 + t)) - sum(lgamma(t + b1 )) -
        m * lgamma(a1 + b1) + sum(lgamma(t + 1)) + m * lgamma(r1) + sum(lgamma(a1 + r1 + b1 + t)) + m *lgamma(b1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      dr1 = - sum(digamma(r1 + t)) + m * digamma(r1) + sum(digamma(r1 + a1 + b1 + t)) -
        N * (digamma(r1 + a1) + (N-m) * digamma(r1 + a1 + b1))
      da1 = sum(digamma(r1 + a1 + b1 + t)) - N * digamma(r1 + a1) - N * digamma(a1 + b1) +
        (N - m) * digamma(r1 + a1 + b1) + N * digamma(a1)
      db1 = -sum(digamma(t + b1)) + m * digamma(b1) + sum(digamma(r1 + a1 + b1 + t)) -
        N * digamma(a1 + b1) + (N - m) * digamma(r1 + a1 + b1)
      return(c(dr1, da1, db1))
    }
    estimate1 = stats::optim(par = c(r, alpha1, alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound,lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans1 = estimate1$par
    fvalue1 = estimate1$value
    p1 = exp(base::lbeta(ans1[1] + ans1[2], ans1[3]) - base::lbeta(ans1[2], ans1[3]))
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], ans[3], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1 - p0) && type == "zi")
    {
      phi = 1 - m/N/(1 - p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], ans[3], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1 - p0) && type == "zi")
    {
      psi = min(m/N , 1 - p1)
      phi = 1 - psi/(1 - p1)
      lik = -fvalue1 + (N - m) * log(1 - psi) + m * log(psi)
      mle = matrix(c(ans1[1], ans1[2], ans1[3], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (type != "zi" && type != "h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general beta negative binomial distribution.")
      ans = new.mle(x, r = r, alpha1 = alpha1, alpha2 = alpha2, lowerbound, upperbound, dist = "bnb")
      mle = matrix(c(ans[1], ans[2], ans[3], 0, ans[4]), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "bnb1.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    m = length(t)
    neg.log.lik <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      ans = m * log(1 - exp(logB - logA)) + m * lgamma(a1) -
        m * lgamma(a1 + r1) - sum(lgamma(r1 + t)) - sum(lgamma(t + b1 )) -
        m * lgamma(a1 + b1) + sum(lgamma(t + 1)) + m * lgamma(r1) + sum(lgamma(a1 + r1 + b1 + t)) + m * lgamma(b1)
      return(ans)
    }
    gp <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      dr = -m * exp(logB - logA) * (digamma(a1 + r1) - digamma(a1 + r1 + b1))/(1 - exp(logB - logA)) -
        sum(digamma(r1 + t)) - m * digamma(a1 + r1) + m * digamma(r1) + sum(digamma(a1 + r1 + b1 + t))
      da = -m * exp(logB - logA) * (digamma(a1 + r1) + digamma(a1 + b1) - digamma(a1 + r1 + b1) - digamma(a1))/(1 - exp(logB - logA)) +
        m * digamma(a1) - m * digamma(a1 + r1) - m * digamma(a1 + b1) + sum(digamma(a1 + r1 + b1 + t))
      db = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + r1 + b1))/(1 - exp(logB - logA)) -
        sum(digamma(t + b1)) - m * digamma(a1 + b1) + sum(digamma(a1 + r1 +t+ b1)) + m * digamma(b1)
      return(c(dr, da, db))
    }
    estimate = stats::optim(par = c(r, alpha1, alpha2), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound,
                                                                    lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans = estimate$par
    rnew = max(1,round(ans[1]))
    neg.log.liknew <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      logA = lgamma(a1 + rnew + b1) + lgamma(a1)
      logB = lgamma(a1 + rnew) + lgamma(a1 + b1)
      ans = m * log(1 - exp(logB - logA)) + m * lgamma(a1) - m * lgamma(a1 + rnew) - sum(lgamma(rnew + t)) - sum(lgamma(t + b1 )) -
        m * lgamma(a1 + b1) + sum(lgamma(t + 1)) + m * lgamma(rnew) + sum(lgamma(a1 + rnew + b1 + t)) + m *lgamma(b1)
      return(ans)
    }
    gpnew <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      logA = lgamma(a1 + rnew + b1) + lgamma(a1)
      logB = lgamma(a1 + rnew) + lgamma(a1 + b1)
      da = -m * exp(logB - logA) * (digamma(a1 + rnew) + digamma(a1 + b1) - digamma(a1 + rnew + b1) - digamma(a1))/(1 - exp(logB - logA)) +
        m * digamma(a1) - m * digamma(a1 + rnew) - m * digamma(a1 + b1) + sum(digamma(a1 + rnew + b1 + t))
      db = -m * exp(logB - logA) * (digamma(a1 + b1) - digamma(a1 + rnew + b1))/(1 - exp(logB - logA)) -
        sum(digamma(t + b1)) - m * digamma(a1 + b1) + sum(digamma(a1 + rnew + t + b1)) + m * digamma(b1)
      return(c(da, db))
    }
    estimatenew = stats::optim(par = c(alpha1, alpha2), fn = neg.log.liknew, gr = gpnew, method = "L-BFGS-B", lower = c(lowerbound, lowerbound), upper = c(upperbound, upperbound))
    ansnew = estimatenew$par
    fvalue = estimatenew$value
    p0 = exp(base::lbeta(rnew + ansnew[1], ansnew[2]) - (base::lbeta(ansnew[1], ansnew[2])))
    neg.log.lik1 <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      # logA = lgamma(a1 + r1 + b1) + lgamma(a1)
      # logB = lgamma(a1 + r1) + lgamma(a1 + b1)
      # ans1 = m * log(1 - exp(logB - logA)) + m * lgamma(a1) -
      #   m * lgamma(a1 + r1) - sum(lgamma(r1 + t)) - sum(lgamma(t + b1 )) -
      #   m * lgamma(a1 + b1) + sum(lgamma(t + 1)) + m * lgamma(r1) + sum(lgamma(a1 + r1 + b1 + t)) + m * lgamma(b1)
      ans1 = -sum(lgamma(r1 + t)) - sum(lgamma(t + b1)) + m * lgamma(r1) + sum(lgamma(t + 1)) + m * lgamma(b1) +
        sum(lgamma(r1 + a1 + b1 + t)) - N * lgamma(r1 + a1) - N * lgamma(a1 + b1) + N * lgamma(r1 + a1 + b1) +
        N * lgamma(a1) - m * lgamma(r1 + a1 + b1)
      return(ans1)
    }
    gp1 <- function(y)
    {
      r1 = y[1]
      a1 = y[2]
      b1 = y[3]
      dr1 = - sum(digamma(r1 + t)) + m * digamma(r1) + sum(digamma(r1 + a1 + b1 + t)) -
        N * (digamma(r1 + a1) + (N-m) * digamma(r1 + a1 + b1))
      da1 = sum(digamma(r1 + a1 + b1 + t)) - N * digamma(r1 + a1) - N * digamma(a1 + b1) +
        (N - m) * digamma(r1 + a1 + b1) + N * digamma(a1)
      db1 = -sum(digamma(t + b1)) + m * digamma(b1) + sum(digamma(r1 + a1 + b1 + t)) -
        N * digamma(a1 + b1) + (N - m) * digamma(r1 + a1 + b1)
      return(c(dr1, da1, db1))
    }
    estimate1 = stats::optim(par = c(r, alpha1, alpha2), fn = neg.log.lik1,
                             gr = gp1, method = "L-BFGS-B", lower = c(lowerbound, lowerbound, lowerbound), upper = c(upperbound, upperbound, upperbound))
    ans1 = estimate1$par
    rnew1 = max (1, round(ans1[1]))
    neg.log.lik1new <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      # logA = lgamma(a1 + rnew1 + b1) + lgamma(a1)
      # logB = lgamma(a1 + rnew1) + lgamma(a1 + b1)
      # ans1 = m * log(1 - exp(logB - logA)) + m * lgamma(a1) -
      #   m * lgamma(a1 + rnew1) - sum(lgamma(rnew1 + t)) - sum(lgamma(t + b1 )) -
      #   m * lgamma(a1 + b1) + sum(lgamma(t + 1)) + m * lgamma(rnew1) + sum(lgamma(a1 + rnew1 + b1 + t)) + m *lgamma(b1)
      ans1 = -sum(lgamma(rnew1 + t)) - sum(lgamma(t + b1)) + m * lgamma(rnew1) + sum(lgamma(t + 1)) + m * lgamma(b1) +
        sum(lgamma(rnew1 + a1 + b1 + t)) - N * lgamma(rnew1 + a1) - N * lgamma(a1 + b1) + N * lgamma(rnew1 + a1 + b1) +
        N * lgamma(a1) - m * lgamma(rnew1 + a1 + b1)
      return(ans1)
    }
    gp1new <- function(y)
    {
      a1 = y[1]
      b1 = y[2]
      da1 = sum(digamma(rnew1 + a1 + b1 + t)) - N * digamma(rnew1 + a1) - N * digamma(a1 + b1) +
        N * digamma(rnew1 + a1 + b1) + N * digamma(a1) - m * digamma(rnew1 + a1 + b1)
      db1 = -sum(digamma(t + b1)) + m * digamma(b1) + sum(digamma(rnew1 + a1 + b1 + t)) -
        N * digamma(a1 + b1) + N * digamma(rnew1 + a1 + b1) - m * digamma(rnew1 + a1 + b1)
      return(c(da1, db1))
    }
    estimate1new = stats::optim(par = c(alpha1, alpha2), fn = neg.log.lik1new, gr = gp1new, method = "L-BFGS-B", lower = c(lowerbound, lowerbound), upper = c(upperbound, upperbound))
    ansnew1 = estimate1new$par
    fvalue1 = estimate1new$value
    p1 = exp(base::lbeta(rnew1 + ansnew1[1], ansnew1[2]) - base::lbeta(ansnew1[1], ansnew1[2]))
    #estimation part
    if (!is.na(p0) && type == "h")
    {
      phi = 1 - (m/N)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(rnew, ansnew[1], ansnew[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) <= (1 - p0) && type == "zi")
    {
      phi = 1 - m/N/(1 - p0)
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(rnew, ansnew[1], ansnew[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (!is.na(p0) && (m/N) > (1 - p0) && type == "zi")
    {
      psi = min(m/N , (1 - p1))
      phi = 1 - psi/(1 - p1)
      lik = -fvalue1 + (N - m) * log(1 - psi) + m * log(psi)
      mle = matrix(c(rnew1, ansnew1[1], ansnew1[2], phi, lik), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    } else if (type != "zi" && type != "h")
    {
      warning("cannot obtain mle with the current model type, the output estimate is derived from general beta negative binomial 1 distribution.")
      ans = new.mle(x, r = r, alpha1 = alpha1, alpha2 = alpha2, lowerbound, upperbound,dist = "bnb1")
      mle = matrix(c(ans[1], ans[2], ans[3], 0, ans[4]), nrow = 1)
      colnames(mle) = c("r", "alpha1", "alpha2", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "normal.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      m1 = y[1]
      s1 = y[2]
      ans = 0.5*m*log(2*pi) + m*log(s1) + 0.5*sum(t^2)/(s1^2)+0.5*m*(m1^2)/(s1^2)-m1*sum(t)/(s1^2)
      return(ans)
    }
    gp <- function(y) {
      m2 = y[1]
      s2 = y[2]
      dm = (m*m2)/(s2^2) - sum(t)/(s2^2)
      ds = m/s2 - sum(t^2)/(s2^3) - (m*(m2^2))/(s2^3) + 2*m2*sum(t)/(s2^3)
      return(c(dm, ds))
    }
    estimate = stats::optim(par = c(mean, sigma), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, upperbound))
    ans = estimate$par
    fvalue = estimate$value
    #estimation part
    if (type == "h" | type=="zi"){
      phi = 1 - m/N
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], phi, lik), nrow = 1)
      colnames(mle) = c("mean", "sigma", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general normal(Gaussian) distribution.")
      ans = new.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="normal")
      mle = matrix(c(ans[1], ans[2], 0, ans[3]), nrow = 1)
      colnames(mle) = c("mean", "sigma", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "lognorm.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      m1 = y[1]
      s1 = y[2]
      ans = 0.5*m*log(2*pi) + sum(log(t)) + m*log(s1) + 0.5*sum(log(t)^2)/(s1^2) + 0.5*m*(m1^2)/(s1^2) - m1*sum(log(t))/(s1^2)
      return(ans)
    }
    gp <- function(y) {
      m2 = y[1]
      s2 = y[2]
      dm = m*m2/(s2^2) - sum(log(t))/(s2^2)
      ds = m/s2 - sum(log(t)^2)/(s2^3) - m*(m2^2)/(s2^3) + 2*m2*sum(log(t))/(s2^3)
      return(c(dm, ds))
    }
    estimate = stats::optim(par = c(mean, sigma), fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = c(lowerbound, lowerbound),
                            upper = c(upperbound, upperbound))
    ans = estimate$par
    fvalue = estimate$value
    #estimation part
    if (type == "h" | type=="zi"){
      phi = 1 - m/N
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], ans[2], phi, lik), nrow = 1)
      colnames(mle) = c("mean", "sigma", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general lognormal distribution.")
      ans = new.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="lognormal")
      mle = matrix(c(ans[1], ans[2], 0, ans[3]), nrow = 1)
      colnames(mle) = c("mean", "sigma", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "halfnorm.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      s1 = y[1]
      ans =  -0.5*m*log(2) + m*log(s1) + 0.5*m*log(pi) + sum(t^2)/(2*(s1^2))
      return(ans)
    }
    gp <- function(y) {
      s2 = y[1]
      ds = m/s2 - sum(t)^2/(s2^3)
      return(ds)
    }
    estimate = stats::optim(par = sigma, fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = upperbound)
    ans = estimate$par
    fvalue = estimate$value
    #estimation part
    if (type == "h" | type=="zi"){
      phi = 1 - m/N
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("sigma", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general half normal distribution.")
      ans = new.mle(x, sigma=sigma, lowerbound, upperbound, dist="halfnormal")
      mle = matrix(c(ans[1], 0, ans[2]), nrow = 1)
      colnames(mle) = c("sigma", "phi", "loglik")
      return(mle)
    }
  }
  if (dist == "exp.zihmle")
  {
    N = length(x)
    t = x[x > 0]
    tsum = sum(t)
    m = length(t)
    neg.log.lik <- function(y) {
      l1 = y[1]
      ans =  -m*log(l1)+l1*sum(t)
      return(ans)
    }
    gp <- function(y) {
      l2 = y[1]
      dl = -m/l2+sum(t)
      return(dl)
    }
    estimate = stats::optim(par = lambda, fn = neg.log.lik,
                            gr = gp, method = "L-BFGS-B", lower = lowerbound, upper = upperbound)
    ans = estimate$par
    fvalue = estimate$value
    #estimation part
    if (type == "h" | type=="zi"){
      phi = 1 - m/N
      lik = -fvalue + (N - m) * log(1 - m/N) + m * log(m/N)
      mle = matrix(c(ans[1], phi, lik), nrow = 1)
      colnames(mle) = c("lambda", "phi", "loglik")
      return(mle)
    } else if (type!="zi" && type!="h"){
      warning("cannot obtain mle with the current model type, the output estimate is derived from general exponential distribution.")
      ans = new.mle(x, lambda=lambda, lowerbound, upperbound, dist="exponential")
      mle = matrix(c(ans[1], 0, ans[2]), nrow = 1)
      colnames(mle) = c("lambda", "phi", "loglik")
      return(mle)
    }
  }
}

#' Generate random deviates from zero-inflated
#' @description Generate random deviates from zero-inflated Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal, half normal, and exponential models.
#' @usage sample.zi1(N,phi,dist="poisson",lambda=NA,r=NA,p=NA,
#' alpha1=NA,alpha2=NA,n=NA,mean=NA,sigma=NA)
#' @param N The sample size. Should be a positive number. If it is not an integer, N will be automatically rounded up to the smallest integer that no less than N.
#' @param phi The structural parameter \eqn{\phi}, should be a positive value within (0,1).
#' @param dist The corresponding standard distribution. Can be one of {'poisson', 'geometric','nb','bb', 'bnb','normal', 'lognormal', 'halfnormal','exponential'}, which corresponds to Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal, hal fnormal, and exponential distributions respectively.
#' @param lambda A value for the parameter of Poisson distribution. Should be a positive number.
#' @param r the number of success before which m failures are observed, where m is a random variable from negative binomial or beta negative binomial distribution. Must be a positive number. If it is not an integer, r will be automatically rounded up to the smallest integer that no less than r.
#' @param p The probability of success, should be a positive value within (0,1).
#' @param alpha1 The first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 The second shape parameter of beta distribution. Should be a positive number.
#' @param n The number of trials. Must be a positive number. If it is not an integer, n will be automatically rounded up to the smallest integer that no less than n.
#' @param mean A value for parameter of the mean or expectation.
#' @param sigma A value of parameter for standard deviation. Must be a positive real number.
#'
#' @return A vector of length N containing non-negative integers from the zero-inflated version of distribution determined by dist.
#'
#' @details \itemize{\item Setting dist=poisson and lambda, sample.zi1 simulates N random deviates from zero-inflated Poisson distribution, respectively, and so on forth.
#'
#' \item Setting the dist=geometric and the argument p is for the use of zero-inflated geometric distributions.
#' \item ASetting the dist=nb and the arguments r and p are for the use of zero-inflated negative binomial distributions.
#' \item Setting the dist=bb and the arguments n, alpha1, and alpha2 are for zero-inflated beta binomial distributions.
#' \item Setting the dist=bnb and the arguments r, alpha1, and alpha2 are used in zero-inflated beta negative binomial distributions.
#' \item Setting the dist=normal and the arguments mean and sigma are used in zero-inflated normal distributions.
#' \item Setting the dist=lognormal and the arguments mean and sigma are used in zero-inflated log normal distributions.
#' \item Setting the dist=halfnormal and the argument sigma is used in zero-inflated half normal distributions.
#' \item Setting the dist=exponential and the argument lambda is used in zero-inflated exponential distributions.}
#'
#' Random deviates from standard Poisson, geometric, negative binomial, normal, log normal, and exponential distributions can be generated by basic R
#' function rpois, rgeom, rnbinom, rnorm, rlnorm, and rexp in R package stats.
#'
#' Functions rbbinom and rbnbinom, and rhnorm are available for standard beta binomial, beta negative binomial, and half normal distributions in R package extraDistr.
#'
#' @references \itemize{\item H. Aldirawi, J. Yang, A. A. Metwally, Identifying Appropriate Probabilistic Models for Sparse Discrete Omics Data, accepted for publication in 2019 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI) (2019).
#' \item T. Wolodzko, extraDistr: Additional Univariate and Multivariate Distributions, R package version 1.8.11 (2019), https://CRAN.R-project.org/package=extraDistr.}
#' @export
#' @examples
#' x1=sample.zi1(2000,phi=0.3,dist='poisson',lambda=10)         #zero-inflated Poisson
#' x2=sample.zi1(2000,phi=0.2,dist='geometric',p=0.2)           #zero-inflated geometric
#' x3=sample.zi1(2000,phi=0.3,dist='bb',n=10,alpha1=2,alpha2=4) #zero-inflated beta binomial
#' x4=sample.zi1(2000,phi=0.3,dist="normal",mean=10,sigma=2)    #zero-inflated normal
#' x5=sample.zi1(2000,phi=0.3,dist="exponential",lambda=20)     #zero-inflated exponential
sample.zi1<-function (N, phi, dist = "poisson", lambda = NA, r = NA, p = NA, alpha1 = NA, alpha2 = NA, n = NA, mean = NA, sigma = NA)
{
  dist = tolower(dist)[1]
  N = ceiling(N)[1]
  phi = phi[1]
  if (N <= 0)
    stop("The sample size N is too small.")
  if ((phi >= 1) | (phi < 0))
    stop("phi is not between 0 and 1 (not including 0 and 1).")
  if (!(dist %in% c("poisson","geometric", "nb", "bb", "bnb","normal","halfnormal","lognormal","exponential")))
    stop("please input a distribution name among poisson,geometric,nb,bb,bnb,normal,lognormal,halfnormal,exponential.")
  if (dist == "poisson") {
    lambda = lambda[1]
    if (lambda <= 0)
      stop("lambda is less or equal to 0.")
  }
  if (dist=="geometric"){
    p=p[1]
    if ((p <= 0) | (p >= 1))
      stop("p is not between 0 and 1 (not including 0 and 1).")
  }
  if (dist == "nb") {
    r = ceiling(r[1])
    p = p[1]
    if (r <= 0)
      stop("r is too small.")
    if ((p <= 0) | (p >= 1))
      stop("p is not between 0 and 1 (not including 0 and 1).")
  }
  if ((dist == "bb") | (dist == "bnb")) {
    alpha1 = alpha1[1]
    alpha2 = alpha2[1]
    if (alpha1 <= 0)
      stop("alpha1 is less or equal to 0.")
    if (alpha2 <= 0)
      stop("alpha2 is less or equal to 0.")
  }
  if (dist == "bb") {
    n = ceiling(n[1])
    if (n <= 0)
      stop("n is too small.")
  }
  if (dist == "bnb") {
    r = ceiling(r[1])
    if (r <= 0)
      stop("r is too small.")
  }
  if (dist == "normal") {
    mean = mean[1]
    sigma = sigma[1]
    if (mean <= 0)
      stop("mean is too small.")
    if (sigma<=0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "lognormal") {
    mean = ceiling(mean[1])
    sigma = sigma[1]
    if (mean <= 0)
      stop("mean is too small.")
    if (sigma<=0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "halfnormal") {
    sigma = sigma[1]
    if (sigma <= 0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "exponential") {
    lambda = lambda[1]
    if (lambda <= 0)
      stop("lambda is less or equal to 0.")
  }
  ans = stats::rbinom(N, size = 1, prob = phi)
  m = sum(ans == 0)
  temp1 = NULL
  if (m > 0) {
    temp1 = switch(dist, geometric=stats::rgeom(m,prob=p),
                   nb = stats::rnbinom(m, size = ceiling(r), prob = p),
                   bb = extraDistr::rbbinom(m, size=ceiling(n), alpha1, alpha2),
                   bnb = extraDistr::rbnbinom(m, size=ceiling(r), alpha1, alpha2),
                   poisson=stats::rpois(m, lambda),
                   normal = stats::rnorm(m, mean, sigma),
                   lognormal = stats::rlnorm(m, mean, sigma),
                   halfnormal = extraDistr::rhnorm(m, sigma),
                   exponential = stats::rexp(m, lambda))
  }
  temp2 = ans
  temp2[ans == 1] = 0
  temp2[ans == 0] = temp1
  return(temp2)
}

#' Generate random deviates from hurdle models
#' @description Generate random deviates from hurdle Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal, half normal, and exponential models.
#' @usage sample.h1(N,phi,dist="poisson",lambda=NA,r=NA,p=NA,
#' alpha1=NA,alpha2=NA,n=NA,mean=NA,sigma=NA)
#' @param N The sample size. Should be a positive number. If it is not an integer, N will be automatically rounded up to the smallest integer that no less than N.
#' @param phi The structural parameter \eqn{\phi}, should be a positive value within (0,1).
#' @param dist The corresponding standard distribution. Can be one of {'poisson', 'geometric','nb','bb', 'bnb','normal', 'lognormal', 'halfnormal','exponential'}, which corresponds to Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal, hal fnormal, and exponential distributions respectively.
#' @param lambda A value for the parameter of Poisson distribution. Should be a positive number.
#' @param r the number of success before which m failures are observed, where m is a random variable from negative binomial or beta negative binomial distribution. Must be a positive number. If it is not an integer, r will be automatically rounded up to the smallest integer that no less than r.
#' @param p The probability of success, should be a positive value within (0,1).
#' @param alpha1 The first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 The second shape parameter of beta distribution. Should be a positive number.
#' @param n The number of trials. Must be a positive number. If it is not an integer, n will be automatically rounded up to the smallest integer that no less than n.
#' @param mean A value for parameter of the mean or expectation.
#' @param sigma A value of parameter for standard deviation. Must be a positive real number.
#'
#' @return A vector of length N containing non-negative integers from the hurdle version of distribution determined by dist.
#'
#' @details \itemize{\item Setting dist=poisson and lambda, sample.h1 simulates N random deviates from hurdle  Poisson distribution, respectively, and so on forth.
#'
#' \item Setting the dist=geometric and the argument p is for the use of hurdle geometric distributions.
#' \item ASetting the dist=nb and the arguments r and p are for the use of and hurdle negative binomial distributions.
#' \item Setting the dist=bb and the arguments n, alpha1, and alpha2 are for and hurdle beta binomial distributions.
#' \item Setting the dist=bnb and the arguments r, alpha1, and alpha2 are used in   hurdle beta negative binomial distributions.
#' \item Setting the dist=normal and the arguments mean and sigma are used in and hurdle normal distributions.
#' \item Setting the dist=lognormal and the arguments mean and sigma are used in and hurdle log normal distributions.
#' \item Setting the dist=halfnormal and the argument sigma is used in and hurdle half normal distributions.
#' \item Setting the dist=exponential and the argument lambda is used in and hurdle exponential distributions.}
#'
#' Random deviates from standard Poisson, geometric, negative binomial, normal, log normal, and exponential distributions can be generated by basic R
#' function rpois, rgeom, rnbinom, rnorm, rlnorm, and rexp in R package stats.
#'
#' Functions rbbinom and rbnbinom, and rhnorm are available for standard beta binomial, beta negative binomial, and half normal distributions in R package extraDistr.
#'
#' @references \itemize{\item H. Aldirawi, J. Yang, A. A. Metwally, Identifying Appropriate Probabilistic Models for Sparse Discrete Omics Data, accepted for publication in 2019 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI) (2019).
#' \item T. Wolodzko, extraDistr: Additional Univariate and Multivariate Distributions, R package version 1.8.11 (2019), https://CRAN.R-project.org/package=extraDistr.}
#' @export
#' @examples
#' x6=sample.h1(2000,phi=0.3,dist='nb',r=10,p=0.3)              #hurdle negative binomial
#' x7=sample.h1(2000,phi=0.3,dist='bnb',r=5,alpha=3,alpha2=3)   #hurdle beta negative binomial
#' x8=sample.h1(2000,phi=0.3,dist="halfnormal",sigma=4)         #hurdle half normal
#' x9=sample.h1(2000,phi=0.3,dist="lognormal",mean=1,sigma=4)   #hurdle log normal
sample.h1<-function (N, phi, dist = "poisson", lambda = NA, r = NA, p = NA, alpha1 = NA, alpha2 = NA, n = NA, mean = NA, sigma = NA)
{
  dist = tolower(dist)[1]
  N = ceiling(N)[1]
  phi = phi[1]
  if (N <= 0)
    stop("the sample size N is too small.")
  if ((phi >= 1) | (phi < 0))
    stop("phi is not between 0 and 1 (not including 0 and 1).")
  if (!(dist %in% c("poisson", "geometric","nb", "bb", "bnb","normal","halfnormal","lognormal","exponential")))
    stop("please input a distribution name among poisson,geometric,nb,bb,bnb,normal,lognormal,halfnormal,exponential.")
  if (dist == "poisson") {
    lambda = lambda[1]
    if (lambda <= 0)
      stop("lambda is less or equal to 0.")
  }
  if (dist=="geometric"){
    p=p[1]
    if ((p <= 0) | (p >= 1))
      stop("p is not between 0 and 1 (not including 0 and 1).")
  }
  if (dist == "nb") {
    r = ceiling(r[1])
    p = p[1]
    if (r <= 0)
      stop("r is too small.")
    if ((p <= 0) | (p >= 1))
      stop("p is not between 0 and 1 (not including 0 and 1).")
  }
  if ((dist == "bb") | (dist == "bnb")) {
    alpha1 = alpha1[1]
    alpha2 = alpha2[1]
    if (alpha1 <= 0)
      stop("alpha1 is less or equal to 0.")
    if (alpha2 <= 0)
      stop("alpha2 is less or equal to 0.")
  }
  if (dist == "bb") {
    n = ceiling(n[1])
    if (n <= 0)
      stop("n is too small.")
  }
  if (dist == "bnb") {
    r = ceiling(r[1])
    if (r <= 0)
      stop("r is too small.")
  }
  if (dist == "normal") {
    mean = mean[1]
    sigma = sigma[1]
    if (mean <= 0)
      stop("mean is too small.")
    if (sigma<=0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "lognormal") {
    mean = ceiling(mean[1])
    sigma = sigma[1]
    if (mean <= 0)
      stop("mean is too small.")
    if (sigma<=0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "halfnormal") {
    sigma = sigma[1]
    if (sigma <= 0)
      stop("sigma is less or equal to 0.")
  }
  if (dist == "exponential") {
    lambda = lambda[1]
    if (lambda <= 0)
      stop("lambda is less or equal to 0.")
  }
  ans = stats::rbinom(N, size = 1, prob = 1 - phi)
  m = sum(ans == 1)
  p0 = switch(dist, geometric = p,
              nb = p^r,
              bb = beta(alpha1, n + alpha2)/beta(alpha1,alpha2),
              bnb = beta(alpha1 + r, alpha2)/beta(alpha1,alpha2),
              poisson = exp(-lambda),
              normal=0,
              lognormal=0,
              halfnormal=0,
              exponential=0)
  #normal = stats::rnorm(m, mean=mean, sd=sigma),
  #lognormal = stats::rlnorm(m, meanlog=mean, sdlog=sigma),
  #halfnormal = extraDistr::rhnorm(m, sigma=sigma),
  #exponential = stats::rexp(m, rate=lambda))
  M = ceiling((m + 2 * sqrt(p0 * (m + p0)) + 2 * p0)/(1 - p0))
  z = switch(dist, geometric = stats::rgeom(M,prob=p),
             nb = stats::rnbinom(M, size = ceiling(r), prob = p),
             bb = extraDistr::rbbinom(M, size=ceiling(n), alpha=alpha1, beta=alpha2),
             bnb = extraDistr::rbnbinom(M,size=ceiling(r), alpha=alpha1, beta=alpha2),
             poisson = stats::rpois(M, lambda),
             normal = stats::rnorm(m, mean=mean, sd=sigma),
             lognormal = stats::rlnorm(m, meanlog=mean, sdlog=sigma),
             halfnormal = extraDistr::rhnorm(m, sigma=sigma),
             exponential = stats::rexp(m, rate=lambda))
  u = z[z > 0]
  t = length(u)
  if (t < m) {
    u1 = NULL
    itemp = m - t
    while (itemp > 0) {
      temp = switch(dist, geometric=stats::rgeom(itemp,prob=p),
                    nb = stats::rnbinom(itemp,size = ceiling(r), prob = p),
                    bb = extraDistr::rbbinom(itemp, size=ceiling(n), alpha=alpha1, beta=alpha2),
                    bnb = extraDistr::rbnbinom(itemp,size=ceiling(r), alpha=alpha1, beta=alpha2),
                    poisson = stats::rpois(itemp, lambda),
                    normal = stats::rnorm(itemp, mean=mean, sd=sigma),
                    lognormal = stats::rlnorm(itemp, meanlog=mean, sdlog=sigma),
                    halfnormal = extraDistr::rhnorm(itemp, sigma=sigma),
                    exponential = stats::rexp(itemp, rate=lambda))
      u1 = c(u1, temp[temp > 0])
      itemp = m - t - length(u1)
    }
    u = c(u, u1)
  }
  ans[ans == 1] = u[1:m]
  return(ans)
}

#' The Monte Carlo estimate for the p-value of a discrete KS Test based on zih.mle estimates.
#' @description Computes the Monte Carlo estimate for the p-value of a discrete one-sample Kolmogorov-Smirnov (KS) Test
#'  based on zih.mle function estimates for Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal,
#'  halfnormal, and exponential distributions and their zero-inflated as well as hurdle versions.
#' @usage kstest.A(x,nsim=200,bootstrap=TRUE,dist='poisson',r=NULL,p=NULL,alpha1=NULL,
#' alpha2=NULL,n=NULL,lambda=NULL,mean=NULL,sigma=NULL,
#' lowerbound=1e-2,upperbound=1e4,parallel=FALSE)
#' @param x A vector of count data. Should be non-negative integers for discrete cases.  Random generation for continuous cases.
#' @param nsim The number of bootstrapped samples or simulated samples generated to compute p-value. If it is not an integer, nsim will be automatically rounded up to the smallest integer that is no less than nsim. Should be greater than 30. Default is 200.
#' @param bootstrap Whether to generate bootstrapped samples or not. See Details. 'TRUE' or any numeric non-zero value indicates the generation of bootstrapped samples. The default is 'TRUE'.
#' @param dist The distribution used as the null hypothesis. Can be one of {'poisson', 'geometric',
#' 'nb', 'nb1', 'bb', 'bb1', 'bnb', 'bnb1', 'normal', 'lognormal', 'halfnormal', 'exponential',
#' 'zip', 'zigeom', 'zinb', 'zibb',  zibnb', 'zinormal', 'zilognorm', 'zihalfnorm', 'ziexp',
#' 'ph', 'geomh','nbh','bbh','bnbh', 'normalh', 'lognormh', 'halfnormh', and 'exph' }, which corresponds to Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential distributions and their zero-inflated as well as hurdle version, respectively.
#'  Defult is 'poisson'.
#' @param r An initial value of the number of success before which m failures are observed, where m is the element of x. Must be a positive number, but not required to be an integer.
#' @param p An initial value of the probability of success, should be a positive value within (0,1).
#' @param alpha1 An initial value for the first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 An initial value for the second shape parameter of beta distribution. Should be a positive number.
#' @param n An initial value of the number of trials. Must be a positive number, but not required to be an integer.
#' @param lambda An initial value of the rate. Must be a positive real number.
#' @param mean An initial value of the mean or expectation.
#' @param sigma An initial value of the standard deviation. Must be a positive real number.
#' @param lowerbound A lower searching bound used in the optimization of likelihood function. Should be a small positive number. The default is 1e-2.
#' @param upperbound An upper searching bound used in the optimization of likelihood function. Should be a large positive number. The default is 1e4.
#' @param parallel whether to use multiple threads for paralleling computation. Default is FALSE. Please aware that it may take longer time to execute the program with parallel=FALSE.
#'
#' @details In arguments nsim, bootstrap, dist, if the length is larger than 1, only the first element will be used. For other arguments except for x, the first valid value will be used if the input is not NULL, otherwise some naive sample estimates will be fed into the algorithm.
#' Note that only the initial values that is used in the null distribution dist are needed. For example, with dist=poisson, user should provide a value for lambda but not for other parameters.
#' With an output p-value less than some user-specified significance level, x is very likely from a distribution other than the dist, given the current data.
#' If p-values of more than one distributions are greater than the pre-specified significance level, user may consider a following likelihood ratio test to select a 'better' distribution.
#' The methodology of computing Monte Carlo p-value is taken from Aldirawi et al. (2019) except changing the zih.mle function and have accurate estimates and adding new discrete and continuous distributions.
#' When bootstrap=TRUE, nsim bootstrapped samples will be generated by resampling x without replacement. Otherwise, nsim samples are simulated from the null distribution with the maximum likelihood estimate of original data x.
#' Then compute the maximum likelihood estimates of nsim bootstrapped or simulated samples, based on which nsim new samples are generated under the null distribution. nsim KS statistics are calculated for the nsim new samples, then the Monte Carlo p-value is resulted from comparing the nsim KS statistics and the statistic of original data x.
#' During the process of computing maximum likelihood estimates, the negative log likelihood function is minimized via basic R function optim with the searching interval decided by lowerbound and upperbound.
#' For large sample sizes we may use kstest.A and for small sample sizes (less that 50 or 100), kstest.B is preferred.
#' @return An object of class 'kstest.A' including the following elements:
#' \itemize{
#' \item x: x used in computation.
#' \item nsim: nsim used in computation.
#' \item bootstrap: bootstrap used in computation.
#' \item dist: dist used in computation.
#' \item lowerbound: lowerbound used in computation.
#' \item upperbound: upperboound used in computation.
#' \item mle_new: A matrix of the maximum likelihood estimates of unknown parameters under the null distribution, using nsim bootstrapped or simulated samples.
#' \item mle_ori: A row vector of the maximum likelihood estimates of unknown parameters under the null distribution, using the original data x.
#' \item pvalue: Monte Carlo p-value of the one-sample KS test.
#' \item N: length of x.
#' \item r: initial value of r used in computation.
#' \item p: initial value of p used in computation.
#' \item alpha1: initial value of alpha1 used in computation.
#' \item alpha2: initial value of alpha2 used in computation.
#' \item lambda: initial value of lambda used in computation.
#' \item n: initial value of n used in computation.
#' \item mean: initial value of mean used in computation.
#' \item sigma: initial value of sigma used in computation.
#' }
#'
#' @seealso \link[AZIAD]{lrt.A}
#'
#' @references \itemize{\item H. Aldirawi, J. Yang, A. A. Metwally (2019). Identifying Appropriate Probabilistic Models for Sparse Discrete Omics Data, accepted for publication in 2019 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI).}
#' @export
#' @examples
#' set.seed(007)
#' x1=sample.zi1(2000,phi=0.3,dist='bnb',r=5,alpha=3,alpha2=3)
#' \donttest{kstest.A(x1,nsim=200,bootstrap = TRUE,dist= 'zinb')$pvalue}      #0
#' \donttest{kstest.A(x1,nsim=200,bootstrap = TRUE,dist= 'zibnb')$pvalue}     #1
#' \donttest{kstest.A(x1,nsim=100,bootstrap = TRUE,dist= 'zibb')$pvalue}      #0.03
#' \donttest{x2=sample.h1(2000,phi=0.3,dist="normal",mean=10,sigma=2)}
#' \donttest{kstest.A(x2,nsim=100,bootstrap = TRUE,dist= 'normalh')$pvalue}   #1
#' \dontrun{kstest.A(x2,nsim=100,bootstrap = TRUE,dist= 'halfnormh')$pvalue} #0.04
kstest.A <- function(x, nsim=200, bootstrap=TRUE, dist='poisson', r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL,
                     lowerbound=0.01,upperbound=10000,parallel=FALSE)
{
  `%do%` <- foreach::`%do%`
  x=ceiling(as.vector(x))
  nsim=ceiling(nsim[1])
  bootstrap=bootstrap[1]
  dist=tolower(dist)[1]
  check.input(x,nsim,dist,lowerbound,upperbound)
  init=init.para1(x,lambda,r,p,alpha1,alpha2,n,mean,sigma)
  r = init$r
  p = init$p
  alpha1 = init$alpha1
  alpha2 = init$alpha2
  n = init$n
  lambda=init$lambda
  N=length(x)
  mean=init$mean
  sigma=init$sigma
  if(parallel){
    cl_cores=parallel::detectCores()
    cl=parallel::makeCluster(cl_cores-2)
  }else{
    cl=parallel::makeCluster(1)
  }
  doParallel::registerDoParallel(cl)
  j=0
  if(dist=='poisson')
  {
    mle_ori=new.mle(x,lambda=lambda,dist='poisson')
    probs_ori=stats::ppois(0:max(x),lambda=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),lambda=mle_ori[1],dist="poisson")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='poisson',type='general')
    temp=list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if(dist=='geometric')
  {
    mle_ori=new.mle(x,p=p,dist='geometric')
    probs_ori=stats::pgeom(0:max(x),prob=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),p=mle_ori[1],dist="geometric")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,p=mle_new[1,j],dist='geometric',type='general')
    temp=list(r = NULL, p = p, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='nb')
  {
    mle_ori=new.mle(x,r=r,p=p,dist="nb",lowerbound,upperbound)
    probs_ori=stats::pnbinom(0:max(x),size=ceiling(mle_ori[1]),prob=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),r=mle_ori[1],p=mle_ori[2],lowerbound,upperbound,dist="nb")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,r=mle_new[1,j],p=mle_new[2,j],dist='nb',type='general')
    temp=list(r=r, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='nb1')
  {
    mle_ori=new.mle(x,r=r,p=p,dist="nb1",lowerbound,upperbound)
    probs_ori=stats::pnbinom(0:max(x),size=ceiling(mle_ori[1]),prob=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),r=mle_ori[1],p=mle_ori[2],lowerbound,upperbound,dist="nb1")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,r=mle_new[1,j],p=mle_new[2,j],dist='nb',type='general')
    temp=list(r=r, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bb')
  {
    mle_ori=new.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,dist="bb",lowerbound,upperbound)
    probs_ori=extraDistr::pbbinom(0:max(x), size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),n=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],dist="bb",lowerbound,upperbound)
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,n=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist='bb',type='general')

    temp=list(r=NULL,p=NULL,alpha1=alpha1,alpha2=alpha2,n=n,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bb1')
  {
    mle_ori=new.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,dist="bb1",lowerbound,upperbound)
    probs_ori=extraDistr::pbbinom(0:max(x), size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),n=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],dist="bb1",lowerbound,upperbound)
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,n=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist='bb',type='general')

    temp=list(r=NULL, p=NULL, alpha1=alpha1, alpha2=alpha2, n=n,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bnb')
  {
    mle_ori=new.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,dist="bnb",lowerbound,upperbound)
    probs_ori=extraDistr::pbnbinom(0:max(x),size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],dist="bnb",lowerbound,upperbound)
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,r=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist='bnb',type='general')
    temp=list(r=r, p=NULL, alpha1=alpha1, alpha2=alpha2, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bnb1')
  {
    mle_ori=new.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,dist="bnb1",lowerbound,upperbound)
    probs_ori=extraDistr::pbnbinom(0:max(x),size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],dist="bnb1",lowerbound,upperbound)
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,r=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist='bnb',type='general')
    temp=list(r=r, p=NULL, alpha1=alpha1, alpha2=alpha2, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='normal')
  {
    mle_ori=new.mle(x, mean=mean, sigma=sigma, dist="normal",lowerbound,upperbound)
    probs_ori=stats::pnorm(0:max(x), mean=ceiling(mle_ori[1]), sd=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T), mean=mle_ori[1], sigma=mle_ori[2],lowerbound,upperbound,dist="normal")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N, mean=mle_new[1,j], sigma=mle_new[2,j], dist='normal', type='general')
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='lognormal')
  {
    mle_ori=new.mle(x, mean=mean, sigma=sigma, dist="lognormal",lowerbound,upperbound)
    probs_ori=EnvStats::plnorm3(0:max(x), meanlog=ceiling(mle_ori[1]), sdlog=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T), mean=mle_ori[1], sigma=mle_ori[2],lowerbound,upperbound,dist="lognormal")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N, mean=mle_new[1,j], sigma=mle_new[2,j], dist='lognormal', type='general')
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='halfnormal')
  {
    mle_ori=new.mle(x, sigma=sigma, dist='halfnormal')
    probs_ori=extraDistr::rhnorm(0:max(x),sigma=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),sigma=mle_ori[1],dist="halfnormal")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,sigma=mle_new[1,j],dist='halfnormal',type='general')
    temp=list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL, mean=NULL, sigma=sigma)
  }
  if(dist=='exponential')
  {
    mle_ori=new.mle(x,lambda=lambda,dist='exponential')
    probs_ori=stats::rexp(0:max(x),rate=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        new.mle(sample(x, size=N, replace=T),lambda=mle_ori[1],dist="exponential")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='exponential',type='general')
    temp=list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if(dist=='zip')
  {
    mle_ori=zih.mle(x,type='zi',lambda=lambda,lowerbound,upperbound,dist="poisson.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*stats::ppois(0:max(x),lambda=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),lambda=lambda,type='zi',lowerbound,upperbound,dist="poisson.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='poisson',type='zi',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if(dist=='zigeom')
  {
    mle_ori=zih.mle(x, p=p, type='zi',lowerbound,upperbound,dist="geometric.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*stats::pgeom(0:max(x),prob=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),p=p,type='zi',lowerbound,upperbound,dist="geometric.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,p=mle_new[1,j],dist='geometric',type='zi',phi=mle_new[2,j])
    temp=list(r=NULL, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='zinb')
  {
    mle_ori=zih.mle(x,r=r,p=p,type='zi',lowerbound,upperbound,dist="nb1.zihmle")
    probs_ori=mle_ori[3]+(1-mle_ori[3])*stats::pnbinom(0:max(x),size=mle_ori[1],prob=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),r=mle_ori[1],p=mle_ori[2],type='zi',lowerbound,upperbound,dist="nb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,r=mle_new[1,j],p=mle_new[2,j],dist='nb',type='zi',phi=mle_new[3,j])
    temp=list(r=r, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='zibb')
  {
    mle_ori=zih.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,type="zi",lowerbound,upperbound,dist="bb1.zihmle")
    probs_ori=mle_ori[4]+(1-mle_ori[4])*extraDistr::pbbinom(0:max(x), size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),n=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],type="zi",lowerbound,upperbound,dist="bb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,n=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist="bb",type="zi",phi=mle_new[4,j])
    temp=list(r=NULL, p=NULL, alpha1=alpha1, alpha2=alpha2, n=n, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='zibnb')
  {
    mle_ori=zih.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,type="zi",lowerbound,upperbound,dist="bnb1.zihmle")
    probs_ori=mle_ori[4]+(1-mle_ori[4])*extraDistr::pbnbinom(0:max(x),size=mle_ori[1],alpha=mle_ori[2],beta=mle_ori[3])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],type="zi",lowerbound,upperbound,dist="bnb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('AZIAD','extraDistr')) %do%
      general.ks(N,r=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist="bnb",type="zi",phi=mle_new[4,j])
    temp=list(r=r, p=NULL, alpha1=alpha1, alpha2=alpha2, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='zinormal')
  {
    mle_ori=zih.mle(x,mean=mean,sigma=sigma,type='zi',lowerbound,upperbound,dist="normal.zihmle")
    probs_ori=mle_ori[3]+(1-mle_ori[3])*stats::rnorm(0:max(x),mean=mle_ori[1],sd=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),mean=mle_ori[1],sigma=mle_ori[2],type='zi',lowerbound,upperbound,dist="normal.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,mean=mle_new[1,j],sigma=mle_new[2,j],dist='normal',type='zi',phi=mle_new[3,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='zilognorm')
  {
    mle_ori=zih.mle(x,mean=mean,sigma=sigma,type='zi',lowerbound,upperbound,dist="lognorm.zihmle")
    probs_ori=mle_ori[3]+(1-mle_ori[3])*stats::rlnorm(0:max(x),meanlog=mle_ori[1],sdlog=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),mean=mle_ori[1],sigma=mle_ori[2],type='zi',lowerbound,upperbound,dist="lognorm.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,mean=mle_new[1,j],sigma=mle_new[2,j],dist='lognormal',type='zi',phi=mle_new[3,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='zihalfnorm')
  {
    mle_ori=zih.mle(x,type='zi',sigma=sigma,lowerbound,upperbound,dist="halfnorm.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*extraDistr::rhnorm(0:max(x),sigma=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages=c('AZIAD')) %do%
        zih.mle(sample(x, size=N, replace=T),sigma=sigma,type='zi',lowerbound,upperbound,dist="halfnorm.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,sigma=mle_new[1,j],dist='halfnormal',type='zi',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=sigma)
  }
  if(dist=='ziexp')
  {
    mle_ori=zih.mle(x,type='zi',lambda=lambda,lowerbound,upperbound,dist="exp.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*stats::rexp(0:max(x),rate=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages=c('AZIAD')) %do%
        zih.mle(sample(x, size=N, replace=T),lambda=lambda,type='zi',lowerbound,upperbound,dist="exp.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='exponential',type='zi',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if(dist=='ph')
  {
    mle_ori=zih.mle(x,lambda=lambda,type='h',lowerbound,upperbound,dist="poisson.zihmle")
    probs_ori=stats::ppois(0:max(x),lambda=mle_ori[1])
    probs_ori=mle_ori[2]+(1-mle_ori[2])*(probs_ori-probs_ori[1])/(1-probs_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),lambda=lambda,type='h',lowerbound,upperbound,dist="poisson.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='poisson',type='h',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if(dist=='geomh')
  {
    mle_ori=zih.mle(x,p=p,type='h',lowerbound,upperbound,dist="geometric.zihmle")
    probs_ori=stats::pgeom(0:max(x),prob=mle_ori[1])
    probs_ori=mle_ori[2]+(1-mle_ori[2])*(probs_ori-probs_ori[1])/(1-probs_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),p=p,type='h',lowerbound,upperbound,dist="geometric.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,p=mle_new[1,j],dist='geometric',type='h',phi=mle_new[2,j])
    temp=list(r=NULL, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='nbh')
  {
    mle_ori=zih.mle(x,r=r,p=p,type='h',lowerbound,upperbound,dist="nb1.zihmle")
    probs_ori=stats::pnbinom(0:max(x),size=ceiling(mle_ori[1]),prob=mle_ori[2])
    probs_ori=mle_ori[3]+(1-mle_ori[3])*(probs_ori-probs_ori[1])/(1-probs_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),r=mle_ori[1],p=mle_ori[2],type='h',lowerbound,upperbound,dist="nb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,r=mle_new[1,j],p=mle_new[2,j],dist='nb',type='h',phi=mle_new[3,j])
    temp=list(r=r, p=p, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bbh')
  {
    mle_ori=zih.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,type="h",lowerbound,upperbound,dist="bb1.zihmle")
    probs_ori=extraDistr::pbbinom(0:max(x), size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    probs_ori=mle_ori[4]+(1-mle_ori[4])*(probs_ori-probs_ori[1])/(1-probs_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),n=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],type='h',lowerbound,upperbound,dist="bb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('extraDistr','AZIAD')) %do%
      general.ks(N,n=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist="bb",type="h",phi=mle_new[4,j])
    temp=list(r=NULL, p=NULL, alpha1=alpha1, alpha2=alpha2, n=n, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='bnbh')
  {
    mle_ori=zih.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,type='h',lowerbound,upperbound,dist="bnb1.zihmle")
    probs_ori=extraDistr::pbnbinom(0:max(x),size=ceiling(mle_ori[1]),alpha=mle_ori[2],beta=mle_ori[3])
    probs_ori=mle_ori[4]+(1-mle_ori[4])*(probs_ori-probs_ori[1])/(1-probs_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind) %do%
        zih.mle(sample(x, size=N, replace=T),r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],type='h',lowerbound,upperbound,dist="bnb1.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages=c('extraDistr','AZIAD')) %do%
      general.ks(N,r=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],dist='bnb',type='h',phi=mle_new[4,j])
    temp=list(r=r, p=NULL, alpha1=alpha1, alpha2=alpha2, n=NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if(dist=='normalh')
  {
    mle_ori=zih.mle(x,mean=mean,sigma=sigma,type='h',lowerbound,upperbound,dist="normal.zihmle")
    probs_ori=mle_ori[3]+(1-mle_ori[3])*stats::pnorm(0:max(x),mean=mle_ori[1],sd=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),mean=mle_ori[1],sigma=mle_ori[2],type='h',lowerbound,upperbound,dist="normal.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,mean=mle_new[1,j],sigma=mle_new[2,j],dist='normal',type='h',phi=mle_new[3,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='lognormh')
  {
    mle_ori=zih.mle(x,mean=mean,sigma=sigma,type='h',lowerbound,upperbound,dist="lognorm.zihmle")
    probs_ori=mle_ori[3]+(1-mle_ori[3])*EnvStats::plnorm3(0:max(x),meanlog=mle_ori[1],sdlog=mle_ori[2])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages='AZIAD') %do%
        zih.mle(sample(x, size=N, replace=T),mean=mle_ori[1],sigma=mle_ori[2],type='h',lowerbound,upperbound,dist="lognorm.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,mean=mle_new[1,j],sigma=mle_new[2,j],dist='lognormal',type='h',phi=mle_new[3,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if(dist=='halfnormh')
  {
    mle_ori=zih.mle(x,type='h',sigma=sigma,lowerbound,upperbound,dist="halfnorm.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*extraDistr::rhnorm(0:max(x),sigma=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages=c('AZIAD')) %do%
        zih.mle(sample(x, size=N, replace=T),sigma=sigma,type='h',lowerbound,upperbound,dist="halfnorm.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,sigma=mle_new[1,j],dist='halfnormal',type='h',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=NULL, mean=NULL, sigma=sigma)
  }
  if(dist=='exph')
  {
    mle_ori=zih.mle(x,type='h',lambda=lambda,lowerbound,upperbound,dist="exp.zihmle")
    probs_ori=mle_ori[2]+(1-mle_ori[2])*stats::rexp(0:max(x),rate=mle_ori[1])
    step_ori=stats::stepfun(0:max(x),c(0,probs_ori))
    z=stats::knots(step_ori)
    dev=c(0, stats::ecdf(x)(z)-step_ori(z))
    Dn_ori=max(abs(dev))
    if(bootstrap){
      mle_new=foreach::foreach(j=1:nsim,.combine=rbind,.packages=c('AZIAD')) %do%
        zih.mle(sample(x, size=N, replace=T),lambda=lambda,type='h',lowerbound,upperbound,dist="exp.zihmle")
      mle_new=t(mle_new)
    }else{
      mle_new=matrix(rep(mle_ori,nsim),ncol=nsim)
    }
    D2=foreach::foreach(j=1:nsim,.combine=c,.packages='AZIAD') %do%
      general.ks(N,lambda=mle_new[1,j],dist='exponential',type='h',phi=mle_new[2,j])
    temp=list(r=NULL, p=NULL, alpha1=NULL, alpha2=NULL, n=NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  parallel::stopCluster(cl)
  pvalue=sum(D2>Dn_ori)/nsim
  ans=list(x=x,nsim=nsim,bootstrap=bootstrap,dist=dist,lowerbound=lowerbound,upperbound=upperbound,mle_new=mle_new,mle_ori=mle_ori,pvalue=pvalue, N=N)
  ans=c(ans,temp)
  class(ans)='kstest.A'
  return(ans)
}

#' The Monte Carlo estimate for the p-value of a discrete KS test based on nested bootstrapped samples
#' @description Computes the Monte Carlo estimate for the p-value of a discrete one-sample Kolmogorov-Smirnov (KS) test based on nested bootstrapped samples
#'  for Poisson, geometric, negative binomial, beta binomial, beta negative binomial, normal, log normal,
#'  halfnormal, and exponential distributions and their zero-inflated as well as hurdle versions.
#' @usage kstest.B(x,nsim=200,bootstrap=TRUE,dist="poisson",
#' r=NULL,p=NULL,alpha1=NULL,alpha2=NULL,n=NULL,lambda=NULL,mean=NULL,sigma=NULL,
#' lowerbound = 0.01, upperbound = 10000, parallel = FALSE)
#' @param x 	A vector of count data. Should be non-negative integers for discrete cases.  Random generation for continuous cases.
#' @param nsim The number of bootstrapped samples or simulated samples generated to compute p-value. If it is not an integer, nsim will be automatically rounded up to the smallest integer that is no less than nsim. Should be greater than 30. Default is 200.
#' @param bootstrap Whether to generate bootstrapped samples or not. See Details. 'TRUE' or any numeric non-zero value indicates the generation of bootstrapped samples. The default is 'TRUE'.
#' @param dist The distribution used as the null hypothesis. Can be one of {'poisson', 'geometric',
#' 'nb', 'nb1', 'bb', 'bb1', 'bnb', 'bnb1', 'normal', 'lognormal', 'halfnormal', 'exponential',
#' 'zip', 'zigeom', 'zinb', 'zibb',  zibnb', 'zinormal', 'zilognorm', 'zihalfnorm', 'ziexp',
#' 'ph', 'geomh','nbh','bbh','bnbh', 'normalh', 'lognormh', 'halfnormh', and 'exph' }, which corresponds to Poisson, geometric, negative binomial, negative binomial1,
#'  beta binomial, beta binomial1, beta negative binomial, beta negative binomial1,
#'  normal, half normal, log normal, and exponential distributions and their zero-inflated as well as hurdle version, respectively.
#'  Defult is 'poisson'.
#' @param r An initial value of the number of success before which m failures are observed, where m is the element of x. Must be a positive number, but not required to be an integer.
#' @param p An initial value of the probability of success, should be a positive value within (0,1).
#' @param alpha1 An initial value for the first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 An initial value for the second shape parameter of beta distribution. Should be a positive number.
#' @param n An initial value of the number of trials. Must be a positive number, but not required to be an integer.
#' @param lambda An initial value of the rate. Must be a positive real number.
#' @param mean An initial value of the mean or expectation.
#' @param sigma An initial value of the standard deviation. Must be a positive real number.
#' @param lowerbound A lower searching bound used in the optimization of likelihood function. Should be a small positive number. The default is 1e-2.
#' @param upperbound An upper searching bound used in the optimization of likelihood function. Should be a large positive number. The default is 1e4.
#' @param parallel whether to use multiple threads paralleling for computation. Default is FALSE. Please aware that it may take longer time to execute the program with parallel=FALSE.
#'
#' @details In arguments nsim, bootstrap, dist, if the length is larger than 1, the first element will be used.
#' For other arguments except for x, the first valid value will be used if the input is not NULL, otherwise some naive sample estimates will be fed into the algorithm.
#' Note that only the initial values that is used in the null distribution dist are needed. For example, with dist=poisson, user should provide a value for lambda and not the other parameters.
#' With an output p-value less than some user-specified significance level, x is probably coming from a distribution other than the dist, given the current data.
#' If p-values of more than one distributions are greater than the pre-specified significance level, user may consider a following likelihood ratio test to select a 'better' distribution.
#' The methodology of computing Monte Carlo p-value is when bootstrap=TRUE, nsim bootstrapped samples will be generated by re-sampling x without replacement. Otherwise, nsim samples are simulated from the null distribution with the maximum likelihood estimate of original data x.
#' Then compute the maximum likelihood estimates of nsim bootstrapped or simulated samples, based on which nsim new samples are generated under the null distribution. nsim KS statistics are calculated for the nsim new samples, then the Monte Carlo p-value is resulted from comparing the nsim KS statistics and the statistic of original data x.
#' During the process of computing maximum likelihood estimates, the negative log likelihood function is minimized via basic R function optim with the searching interval decided by lowerbound and upperbound. Next simulate i.i.d. simulates from the estimated parameters and calculate a new
#' mle based on the bootstrapped samples. Then calculate the KS statistic and the p-value.
#' For large sample sizes we may use kstest.A and for small sample sizes (less that 50 or 100), kstest.B is preferred.
#' @return An object of class 'kstest.A' including the following elements:
#' \itemize{
#' \item x: x used in computation.
#' \item nsim: nsim used in computation.
#' \item bootstrap: bootstrap used in computation.
#' \item dist: dist used in computation.
#' \item lowerbound: lowerbound used in computation.
#' \item upperbound: upperboound used in computation.
#' \item mle_new: A matrix of the maximum likelihood estimates of unknown parameters under the null distribution, using nsim bootstrapped or simulated samples.
#' \item mle_ori: A row vector of the maximum likelihood estimates of unknown parameters under the null distribution, using the original data x.
#' \item mle_c: A row vector of the maximum likelihood estimates of unknown parameters under the null distribution, using bootstrapped samples with parameters of mle_new.
#' \item pvalue: Monte Carlo p-value of the one-sample KS test.
#' \item N: length of x.
#' \item r: initial value of r used in computation.
#' \item p: initial value of p used in computation.
#' \item alpha1: initial value of alpha1 used in computation.
#' \item alpha2: initial value of alpha2 used in computation.
#' \item lambda: initial value of lambda used in computation.
#' \item n: initial value of n used in computation.
#' \item mean: initial value of mean used in computation.
#' \item sigma: initial value of sigma used in computation.
#' }
#' @references \itemize{\item H. Aldirawi, J. Yang, A. A. Metwally (2019). Identifying Appropriate Probabilistic Models for Sparse Discrete Omics Data, accepted for publication in 2019 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI).}
#' @seealso \link[AZIAD]{kstest.A}
#' @export
#'
#' @examples
#' set.seed(008)
#' x=sample.zi1(2000,phi=0.3,dist='bnb',r=5,alpha=3,alpha2=3)
#' \donttest{kstest.B(x,nsim=100,bootstrap = TRUE,dist= 'zinb')$pvalue}   #0.01
#' \donttest{kstest.B(x,nsim=100,bootstrap = TRUE,dist= 'zibb')$pvalue}   #0.02
#' \donttest{kstest.B(x,nsim=100,bootstrap = TRUE,dist= 'zibnb')$pvalue}  #0.67
#' \donttest{x2=sample.h1(2000,phi=0.3,dist="halfnormal",sigma=4)}
#' \donttest{kstest.B(x2,nsim=100,bootstrap = TRUE,dist= 'halfnormh')$pvalue}   #0.73
#' \donttest{kstest.B(x2,nsim=100,bootstrap = TRUE,dist= 'lognormh')$pvalue}    #0
kstest.B<-function (x, nsim = 200, bootstrap = TRUE, dist = "poisson",
                    r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL,lambda=NULL, mean=NULL,sigma=NULL,
                    lowerbound = 0.01, upperbound = 10000, parallel = FALSE)
{
  `%do%` <- foreach::`%do%`
  x = ceiling(as.vector(x))
  nsim = ceiling(nsim[1])
  bootstrap = bootstrap[1]
  dist = tolower(dist)[1]
  check.input(x,nsim,dist,lowerbound,upperbound)
  init = init.para1(x,lambda,r,p,alpha1,alpha2,n,mean,sigma)
  r = init$r
  p = init$p
  alpha1 = init$alpha1
  alpha2 = init$alpha2
  n = init$n
  lambda=init$lambda
  N = length(x)
  mean=init$mean
  sigma=init$sigma
  if (parallel) {
    cl_cores = parallel::detectCores()
    cl = parallel::makeCluster(cl_cores - 2)
  }
  else {
    cl = parallel::makeCluster(1)
  }
  doParallel::registerDoParallel(cl)
  j = 0
  if (dist == "poisson")
  {
    mle_ori = new.mle(x, lambda = lambda, dist = 'poisson')
    probs_ori = stats::ppois(0:max(x), lambda = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N, replace = T), lambda=mle_ori[1], dist = 'poisson')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rpois(N,lambda=mle_new[1,j]),lambda=mle_new[1,j],dist = 'poisson')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, lambda = mle_c[1,j], dist = "poisson",type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL,lambda=lambda, mean=NULL, sigma=NULL)
  }
  if (dist == "geometric")
  {
    mle_ori = new.mle(x, p = p, dist = 'geometric')
    probs_ori = stats::pgeom(0:max(x), prob = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N, replace = T), p=mle_ori[1], dist = 'geometric')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rgeom(N,p=mle_new[1,j]),p=mle_new[1,j],dist = 'geometric')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, p = mle_c[1,j], dist = "geometric",type = "general")
    temp = list(r = NULL, p = p, alpha1 = NULL, alpha2 = NULL, n = NULL,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "nb")
  {
    mle_ori = new.mle(x,r=r,p=p,dist = 'nb', lowerbound, upperbound)
    probs_ori = stats::pnbinom(0:max(x), size = ceiling(mle_ori[1]),prob = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T), r=mle_ori[1], p=mle_ori[2], dist= 'nb')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rnbinom(N,size=mle_new[1,j],prob=mle_new[2,j]),
                                                                   r=mle_new[1,j], p=mle_new[2,j], dist = 'nb')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, r = mle_c[1, j], p = mle_c[2, j], dist = "nb", type = "general")
    temp = list(r = r, p = p, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "nb1")
  {
    mle_ori = new.mle(x,r=r,p=p,dist = 'nb1', lowerbound, upperbound)
    probs_ori = stats::pnbinom(0:max(x), size = ceiling(mle_ori[1]),prob = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T),
                                                                     r=mle_ori[1],p=mle_ori[2],dist= 'nb1')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rnbinom(N,size=mle_new[1,j],prob=mle_new[2,j]),
                                                                   r=mle_new[1,j], p=mle_new[2,j], dist = 'nb1')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, r = mle_c[1, j], p = mle_c[2, j], dist = "nb", type = "general")
    temp = list(r = r, p = p, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "bb")
  {
    mle_ori = new.mle(x, n=n,alpha1=alpha1,alpha2=alpha2,dist = "bb", lowerbound, upperbound)
    probs_ori = extraDistr::pbbinom(0:max(x), size = ceiling(mle_ori[1]),alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    mle_new<-NULL;
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "AZIAD") %do% new.mle(sample(x, size = N,replace = T),
                                                                  n=mle_ori[1], alpha1=mle_ori[2], alpha2=mle_ori[3], dist = 'bb')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-extraDistr::pbbinom(0:max(x),size=ceiling(mle_new[1,i]),alpha=mle_new[2,i],beta=mle_new[3,i])
        temp1<-new.mle(xc[[i]],n=mle_new[1,i], alpha1=mle_new[2,i], alpha2=mle_new[3,i],dist='bb')
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)}

    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = c("AZIAD",
                                                                  "extraDistr")) %do% general.ks(N, n = mle_c[1,i], alpha1 = mle_c[2, i], alpha2 = mle_c[3, i],
                                                                                                 dist = "bb", type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = alpha1, alpha2 = alpha2, n = n, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "bb1")
  {
    mle_ori = new.mle(x, n=n,alpha1=alpha1,alpha2=alpha2,dist = "bb1", lowerbound, upperbound)
    probs_ori = extraDistr::pbbinom(0:max(x), size = ceiling(mle_ori[1]),alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    mle_new<-NULL;
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "AZIAD") %do% new.mle(sample(x, size = N,replace = T),
                                                                  n=mle_ori[1], alpha1=mle_ori[2], alpha2=mle_ori[3], dist = 'bb1')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "AZIAD") %do% new.mle(extraDistr::rbbinom(N,size=mle_new[1,j],alpha=mle_new[2,j],beta=mle_new[3,j]),
                                                                n=mle_new[1,j], alpha1=mle_new[2,j], alpha2=mle_new[3,j],dist='bb1')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = c("AZIAD",
                                                                  "extraDistr")) %do% general.ks(N, n = mle_c[1,i], alpha1 = mle_c[2, i], alpha2 = mle_c[3, i],
                                                                                                 dist = "bb", type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = alpha1, alpha2 = alpha2, n = n, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "bnb")
  {
    mle_ori = new.mle(x, r=r,alpha1=alpha1,alpha2=alpha2, lowerbound, upperbound,dist = "bnb")
    probs_ori = extraDistr::pbnbinom(0:max(x), size = ceiling(mle_ori[1]),
                                     alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T),
                                                                     r=mle_ori[1], alpha1=mle_ori[2], alpha2=mle_ori[3],dist = "bnb")
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "AZIAD") %do% new.mle(extraDistr::rbnbinom(N, size=mle_new[1,j], alpha=mle_new[2,j],beta=mle_new[3,j]),
                                                                r=mle_new[1,j], alpha1=mle_new[2,j], alpha2=mle_new[3,j],dist="bnb")
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = c("AZIAD",
                                                                  "extraDistr")) %do% general.ks(N, r = mle_new[1,j], alpha1 = mle_new[2, j], alpha2 = mle_new[3, j],
                                                                                                 dist = "bnb", type = "general")
    temp = list(r = r, p = NULL, alpha1 = alpha1, alpha2 = alpha2, n = NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "bnb1")
  {
    mle_ori = new.mle(x, r=r,alpha1=alpha1,alpha2=alpha2, lowerbound, upperbound,dist = "bnb1")
    probs_ori = extraDistr::pbnbinom(0:max(x), size = ceiling(mle_ori[1]),
                                     alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T),
                                                                     r=mle_ori[1], alpha1=mle_ori[2], alpha2=mle_ori[3],dist = "bnb1")
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "AZIAD") %do% new.mle(extraDistr::rbnbinom(N, size=mle_new[1,j], alpha=mle_new[2,j],beta=mle_new[3,j]),
                                                                r=mle_new[1,j], alpha1=mle_new[2,j], alpha2=mle_new[3,j],dist="bnb1")
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = c("AZIAD",
                                                                  "extraDistr")) %do% general.ks(N, r = mle_new[1,j], alpha1 = mle_new[2, j], alpha2 = mle_new[3, j],
                                                                                                 dist = "bnb", type = "general")
    temp = list(r = r, p = NULL, alpha1 = alpha1, alpha2 = alpha2, n = NULL, lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "normal")
  {
    mle_ori = new.mle(x, mean=mean, sigma=sigma,dist = 'normal', lowerbound, upperbound)
    probs_ori = stats::pnorm(0:max(x), mean = ceiling(mle_ori[1]),sd = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T),
                                                                     mean=mle_ori[1], sigma=mle_ori[2],dist= 'normal')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rnorm(N,mean=mle_new[1,j],sd=mle_new[2,j]),
                                                                   mean=mle_new[1,j], sigma=mle_new[2,j], dist = 'normal')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, mean = mle_c[1, j], sigma = mle_c[2, j], dist = "normal", type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if (dist == "lognormal")
  {
    mle_ori = new.mle(x, mean=mean, sigma=sigma,dist = 'lognormal', lowerbound, upperbound)
    probs_ori = EnvStats::plnorm3(0:max(x), meanlog = ceiling(mle_ori[1]),sdlog = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N,replace = T),
                                                                     mean=mle_ori[1], sigma=mle_ori[2],dist= 'lognormal')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rlnorm(N, meanlog = mle_new[1,j], sdlog = mle_new[2,j]),
                                                                   mean=mle_new[1,j], sigma=mle_new[2,j], dist = 'lognormal')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, mean = mle_c[1, j], sigma = mle_c[2, j], dist = "lognormal", type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, mean=mean, sigma=sigma)
  }
  if (dist == "halfnormal")
  {
    mle_ori = new.mle(x, sigma=sigma, dist = 'halfnormal', lowerbound, upperbound)
    probs_ori = extraDistr::phnorm(0:max(x), sigma = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N, replace = T), sigma=sigma,dist = 'halfnormal')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(extraDistr::rhnorm(N, sigma = mle_new[1,j]),
                                                                   sigma=mle_new[1,j],dist = 'halfnormal')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, sigma = mle_c[1, j], dist = "halfnormal",type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL,lambda=NULL, mean=NULL, sigma=sigma)
  }
  if (dist == "exponential")
  {
    mle_ori = new.mle(x, lambda = lambda, dist = 'exponential')
    probs_ori = stats::pexp(0:max(x), rate = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% new.mle(sample(x, size = N, replace = T), lambda=lambda,dist = 'exponential')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if (bootstrap) {
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% new.mle(stats::rexp(100,rate=mle_new[1,j]),
                                                                   lambda=mle_new[1,j],dist = 'exponential')
      mle_c = t(mle_c)
    }else {
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, lambda = mle_c[1, j], dist = "exponential",type = "general")
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL,lambda=lambda, mean=NULL, sigma=NULL)
  }
  if (dist == "zip")
  {
    mle_ori = zih.mle(x,lambda=lambda,type="zi",lowerbound,upperbound,dist="poisson.zihmle")
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * stats::ppois(0:max(x),lambda = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],lambda=mle_ori[1],
                      type = "zi",lowerbound, upperbound,dist="poisson.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N,lambda=mle_new[1,i],phi=mle_new[2,i],dist = "poisson")
        temp1<-zih.mle(xc[[i]],lambda=mle_new[1,i],type="zi",lowerbound,upperbound,dist = "poisson.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, lambda = mle_c[1, i], dist = "poisson",type = "zi", phi = mle_c[2, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=lambda,mean=NULL,sigma=NULL)
  }
  if (dist == "zigeom")
  {
    mle_ori = zih.mle(x,p=p,type="zi",lowerbound,upperbound,dist="geometric.zihmle")
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * stats::pgeom(0:max(x),prob = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],p=mle_ori[1],type = "zi",lowerbound, upperbound,dist="geometric.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N,p=mle_new[1,i],phi=mle_new[2,i],dist = "geometric")
        temp1<-zih.mle(xc[[i]],p=mle_new[1,i],type="zi",lowerbound,upperbound,dist = "geometric.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, p = mle_c[1, i], dist = "geometric",type = "zi", phi = mle_c[2, i])
    temp = list(r = NULL, p = p, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=NULL,sigma=NULL)
  }
  if (dist == "zinb")
  {
    mle_ori = zih.mle(x,r=r,p=p,type="zi",lowerbound, upperbound,dist = "nb1.zihmle")
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * stats::pnbinom(0:max(x), size = mle_ori[1], prob = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if(bootstrap){
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],r=mle_ori[1],p=mle_ori[2],
                      type = "zi",lowerbound, upperbound,dist="nb1.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N,r=mle_new[1,i],p = mle_new[2,i],phi=mle_new[3,i],dist = "nb")
        temp1<-zih.mle(xc[[i]],r=mle_new[1,i],p=mle_new[2,i],type="zi",lowerbound,upperbound,dist = "nb1.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, r= mle_c[1, i], p=mle_c[2,i], dist = "nb",type = "zi", phi = mle_c[3, i])
    temp = list(r = r, p = p, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=NULL,sigma=NULL)
  }
  if (dist == "zibb")
  {
    mle_ori = zih.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,type = "zi",lowerbound,upperbound,dist="bb1.zihmle")
    probs_ori = mle_ori[4] + (1 - mle_ori[4]) * extraDistr::pbbinom(0:max(x), size = ceiling(mle_ori[1]), alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if(bootstrap){
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim)
      {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],n=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],type = "zi",lowerbound, upperbound,dist ="bb1.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N,n=mle_new[1,i],alpha1 = mle_new[2,i],alpha2=mle_new[3,i],phi=mle_new[4,i],dist = "bb")
        temp1<-zih.mle(xc[[i]],n=mle_new[1,i],alpha1=mle_new[2,i],alpha2=mle_new[3,i],type="zi",lowerbound,upperbound,dist ="bb1.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)}
    D2<-NULL;
    for (i in 1:nsim){
      temp2<- general.ks(N, n = mle_c[1, i], alpha1 = mle_c[2, i],alpha2=mle_c[3,i],dist = "bb", type = "zi", phi = mle_c[4,i])
      D2<-rbind(D2,temp2)}
    temp = list(r = NULL, p = NULL, alpha1 = alpha1, alpha2 = alpha2,n = n,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "zibnb")
  {
    mle_ori = zih.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, type = "zi",lowerbound, upperbound,dist="bnb1.zihmle")
    probs_ori = mle_ori[4] + (1 - mle_ori[4]) * extraDistr::pbnbinom(0:max(x),size = mle_ori[1], alpha = mle_ori[2], beta = mle_ori[3])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],
                      type = "zi",lowerbound, upperbound,dist="bnb1.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N,r=mle_new[1,i],alpha1=mle_new[2,i],alpha2=mle_new[3,i],phi=mle_new[4,i],dist = "bnb")
        temp1<-zih.mle(xc[[i]],r=mle_new[1,i],alpha1=mle_new[2,i],alpha2=mle_new[3,i],type="zi",lowerbound,upperbound,dist = "bnb1.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2<-NULL;
    for (i in 1:nsim){
      temp2<- general.ks(N, r = mle_c[1, i], alpha1 = mle_c[2, i],alpha2=mle_c[3,i],dist = "bnb", type = "zi", phi = mle_c[4,i])
      D2<-rbind(D2,temp2)}
    temp = list(r = r, p = NULL, alpha1 = alpha1, alpha2 = alpha2,n = NULL,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "zinormal")
  {
    mle_ori = zih.mle(x, mean=mean, sigma=sigma, type="zi", lowerbound, upperbound, dist = "normal.zihmle")
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * stats::rnorm(0:max(x), mean = mle_ori[1], sd = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if(bootstrap){
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x, length(x), replace = TRUE)
        temp<-zih.mle(xb[[i]], mean=mle_ori[1], sigma=mle_ori[2], type = "zi", lowerbound, upperbound, dist="normal.zihmle")
        mle_new<-rbind(mle_new, temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N, mean=mle_new[1,i], sigma = mle_new[2,i], phi=mle_new[3,i], dist = "normal")
        temp1<-zih.mle(xc[[i]], mean=mle_new[1,i], sigma=mle_new[2,i], type="zi", lowerbound, upperbound, dist = "normal.zihmle")
        mle_c<-rbind(mle_c, temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, mean = mle_c[1, i], sigma=mle_c[2,i], dist = "normal",type = "zi", phi = mle_c[3, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=mean,sigma=sigma)
  }
  if (dist == "zilognorm")
  {
    mle_ori = zih.mle(x, mean=mean, sigma=sigma, type="zi", lowerbound, upperbound, dist = "lognorm.zihmle")
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * stats::rlnorm(0:max(x), meanlog = mle_ori[1], sdlog = mle_ori[2])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if(bootstrap){
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x, length(x), replace = TRUE)
        temp<-zih.mle(xb[[i]], mean=mle_ori[1], sigma=mle_ori[2], type = "zi", lowerbound, upperbound, dist="lognorm.zihmle")
        mle_new<-rbind(mle_new, temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N, mean=mle_new[1,i], sigma = mle_new[2,i], phi=mle_new[3,i], dist = "lognormal")
        temp1<-zih.mle(xc[[i]], mean=mle_new[1,i], sigma=mle_new[2,i], type="zi", lowerbound, upperbound, dist = "lognorm.zihmle")
        mle_c<-rbind(mle_c, temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, mean = mle_c[1, i], sigma=mle_c[2,i], dist = "lognormal",type = "zi", phi = mle_c[3, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda = NULL, mean = mean, sigma = sigma)
  }
  if (dist == "zihalfnorm")
  {
    mle_ori = zih.mle(x, sigma=sigma, type="zi", lowerbound, upperbound, dist="halfnorm.zihmle")
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * extraDistr::rhnorm(0:max(x),sigma = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]], sigma=mle_ori[1], type = "zi", lowerbound, upperbound, dist="halfnorm.zihmle")
        mle_new<-rbind(mle_new, temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N, sigma=mle_new[1,i], phi=mle_new[2,i], dist = "halfnormal")
        temp1<-zih.mle(xc[[i]], sigma=mle_new[1,i], type="zi", lowerbound, upperbound, dist = "halfnorm.zihmle")
        mle_c<-rbind(mle_c, temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, sigma = mle_c[1, i], dist = "halfnormal", type = "zi", phi = mle_c[2, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, sigma=sigma, mean=NULL)
  }
  if (dist == "ziexp")
  {
    mle_ori = zih.mle(x, lambda=lambda, type="zi", lowerbound, upperbound, dist="exp.zihmle")
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * stats::rexp(0:max(x), rate = mle_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x, length(x), replace = TRUE)
        temp<-zih.mle(xb[[i]], lambda=mle_ori[1], type = "zi", lowerbound, upperbound, dist="exp.zihmle")
        mle_new<-rbind(mle_new, temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.zi1(N, lambda=mle_new[1,i], phi=mle_new[2,i], dist = "exponential")
        temp1<-zih.mle(xc[[i]], lambda=mle_new[1,i], type="zi", lowerbound, upperbound, dist = "exp.zihmle")
        mle_c<-rbind(mle_c, temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "AZIAD") %do%
      general.ks(N, lambda = mle_c[1, i], dist = "exponential",type = "zi", phi = mle_c[2, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=lambda, mean=NULL, sigma=NULL)
  }
  if (dist == "ph")
  {
    mle_ori = zih.mle(x,lambda=lambda, type = "h", lowerbound, upperbound,dist='poisson.zihmle')
    probs_ori = stats::ppois(0:max(x), lambda = mle_ori[1])
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages ="foreach") %do% zih.mle(sample(x,size = N, replace = T), lambda=mle_ori[1],
                                                                    type = "h", lowerbound,upperbound,dist = 'poisson.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% zih.mle(sample.h1(N,phi=mle_new[2,j],lambda=mle_new[1,j]),
                                                                   type = "h", lambda=mle_new[1,j],lowerbound,upperbound,dist = 'poisson.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, lambda = mle_c[1, j], dist = "poisson",type = "h", phi = mle_c[2, j])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=lambda,mean=NULL, sigma=NULL)
  }
  if (dist == "geomh")
  {
    mle_ori = zih.mle(x,p=p, type = "h", lowerbound, upperbound,dist='geometric.zihmle')
    probs_ori = stats::pgeom(0:max(x), prob = mle_ori[1])
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],p=mle_ori[1],type = "h",lowerbound, upperbound,dist="geometric.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.h1(N,p=mle_new[1,i],phi=mle_new[2,i],dist = "geometric")
        temp1<-zih.mle(xc[[i]],p=mle_new[1,i],type="h",lowerbound,upperbound,dist = "geometric.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, p = mle_c[1, i], dist = "geometric",type = "h", phi = mle_c[2, i])
    temp = list(r = NULL, p = p, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=NULL,sigma=NULL)
  }
  if (dist == "nbh")
  {
    mle_ori = zih.mle(x, r=r, p=p, type = "h", lowerbound,upperbound,dist = 'nb1.zihmle')
    probs_ori = stats::pnbinom(0:max(x), size = mle_ori[1],prob = mle_ori[2])
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% zih.mle(sample(x,length(x), replace = T), r=mle_ori[1], p=mle_ori[2],
                                                                     type = "h", lowerbound, upperbound,dist = 'nb1.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c=foreach::foreach(j = 1:nsim, .combine = rbind,
                             .packages = "foreach") %do% zih.mle(sample.h1(N,dist ="nb",r=mle_new[1,j],
                                                                           p=mle_new[2,j],phi=mle_new[3,j]), r=mle_new[1,j], p=mle_new[2,j],type = "h",dist='nb1.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c,.packages = "foreach") %do%
      general.ks(N, r = mle_c[1, j], p = mle_c[2, j],dist = "nb", type = "h", phi = mle_c[3, j])
    temp = list(r = r, p = p, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=NULL, sigma=NULL)
  }
  if (dist == "bbh")
  {
    mle_ori = zih.mle(x, n=n, alpha1=alpha1, alpha2=alpha2, type = "h",lowerbound, upperbound,dist = 'bb1.zihmle')
    probs_ori = extraDistr::pbbinom(0:max(x), size = ceiling(mle_ori[1]),alpha = mle_ori[2], beta = mle_ori[3])
    probs_ori = mle_ori[4] + (1 - mle_ori[4]) * (probs_ori - probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% zih.mle(sample(x,size = N, replace = T), n=mle_ori[1],
                                                                     alpha1=mle_ori[2],alpha2=mle_ori[3], type = "h", lowerbound, upperbound,dist='bb1.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c= foreach::foreach(j = 1:nsim, .combine = rbind,
                              .packages = "foreach") %do% zih.mle(sample.h1(N,n=mle_new[1,j],alpha1=mle_new[2,j],alpha2=mle_new[3,j],
                                                                            phi=mle_new[4,j],dist = "bb"),n=mle_new[1,j], alpha1=mle_new[2,j],
                                                                  alpha2=mle_new[3,j], type = "h", lowerbound, upperbound,dist='bb1.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do% general.ks(N, n = mle_c[1,j],
                                                                                           alpha1 = mle_c[2, j], alpha2 = mle_c[3, j],dist = "bb", type = "h", phi = mle_c[4, j])
    temp = list(r = NULL, p = NULL, alpha1 = alpha1, alpha2 = alpha2,n = n,lambda=NULL,mean=NULL,sigma=NULL)
  }
  if (dist == "bnbh")
  {
    mle_ori = zih.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, type = "h",lowerbound, upperbound,dist = 'bnb1.zihmle')
    probs_ori = extraDistr::pbnbinom(0:max(x), size = mle_ori[1],alpha = mle_ori[2], beta = mle_ori[3])
    probs_ori = mle_ori[4] + (1 - mle_ori[4]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]],r=mle_ori[1],alpha1=mle_ori[2],alpha2=mle_ori[3],
                      type = "h",lowerbound, upperbound,dist="bnb1.zihmle")
        mle_new<-rbind(mle_new,temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.h1(N,r=mle_new[1,i],alpha1=mle_new[2,i],alpha2=mle_new[3,i],phi=mle_new[4,i],dist = "bnb")
        temp1<-zih.mle(xc[[i]],r=mle_new[1,i],alpha1=mle_new[2,i],alpha2=mle_new[3,i],type="h",lowerbound,upperbound,dist = "bnb1.zihmle")
        mle_c<-rbind(mle_c,temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2<-NULL;
    for (i in 1:nsim){
      temp2<- general.ks(N, r = mle_c[1, i], alpha1 = mle_c[2, i],alpha2=mle_c[3,i],dist = "bnb", type = "h", phi = mle_c[4,i])
      D2<-rbind(D2,temp2)}
    temp = list(r = r, p = NULL, alpha1 = alpha1, alpha2 = alpha2,n = NULL,lambda=NULL, mean=NULL, sigma=NULL)
  }
  if (dist == "normalh")
  {
    mle_ori = zih.mle(x, mean=mean, sigma=sigma, type = "h", lowerbound,upperbound,dist = 'normal.zihmle')
    probs_ori = stats::rnorm(0:max(x), mean = mle_ori[1],sd = mle_ori[2])
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% zih.mle(sample(x,length(x), replace = T), mean=mle_ori[1], sigma=mle_ori[2],
                                                                     type = "h", lowerbound, upperbound,dist = 'normal.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c=foreach::foreach(j = 1:nsim, .combine = rbind,
                             .packages = "foreach") %do% zih.mle(sample.h1(N,dist ="normal",mean=mle_new[1,j],
                                                                           sigma=mle_new[2,j],phi=mle_new[3,j]), mean=mle_new[1,j], sigma=mle_new[2,j],type = "h",dist='normal.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c,.packages = "foreach") %do%
      general.ks(N, mean = mle_c[1, j], sigma = mle_c[2, j],dist = "normal", type = "h", phi = mle_c[3, j])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=mean, sigma=sigma)
  }
  if (dist == "lognormh")
  {
    mle_ori = zih.mle(x, mean=mean, sigma=sigma, type = "h", lowerbound,upperbound,dist = 'lognorm.zihmle')
    probs_ori = stats::rlnorm(0:max(x), meanlog = mle_ori[1],sdlog = mle_ori[2])
    probs_ori = mle_ori[3] + (1 - mle_ori[3]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages = "foreach") %do% zih.mle(sample(x,length(x), replace = T), mean=mle_ori[1], sigma=mle_ori[2],
                                                                     type = "h", lowerbound, upperbound,dist = 'lognorm.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c=foreach::foreach(j = 1:nsim, .combine = rbind,
                             .packages = "foreach") %do% zih.mle(sample.h1(N,dist ="lognormal",mean=mle_new[1,j],
                                                                           sigma=mle_new[2,j],phi=mle_new[3,j]), mean=mle_new[1,j], sigma=mle_new[2,j],type = "h",dist='lognorm.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_new, nsim), ncol = nsim)
    }

    D2 = foreach::foreach(j = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, mean = mle_c[1, j], sigma = mle_c[2, j],dist = "lognormal", type = "h", phi = mle_c[3, j])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=NULL,mean=mean, sigma=sigma)
  }
  if (dist == "halfnormh")
  {
    mle_ori = zih.mle(x,sigma=sigma, type = "h", lowerbound, upperbound,dist='halfnorm.zihmle')
    probs_ori = extraDistr::rhnorm(0:max(x), sigma = mle_ori[1])
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new<-NULL;
      xb <- list()
      for(i in 1:nsim) {
        xb[[i]] <- sample(x,length(x),replace = TRUE)
        temp<-zih.mle(xb[[i]], sigma=mle_ori[1], type = "h", lowerbound, upperbound, dist="halfnorm.zihmle")
        mle_new<-rbind(mle_new, temp)
      }
      mle_new<-t(mle_new)
    }else{
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)}
    if(bootstrap){
      mle_c<-NULL;
      xc<-list()
      for (i in 1:nsim){
        xc[[i]]<-sample.h1(N, sigma=mle_new[1,i], phi=mle_new[2,i], dist = "halfnormal")
        temp1<-zih.mle(xc[[i]], sigma=mle_new[1,i], type="h", lowerbound, upperbound, dist = "halfnorm.zihmle")
        mle_c<-rbind(mle_c, temp1)
      }
      mle_c<-t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)}
    D2 = foreach::foreach(i = 1:nsim, .combine = c, .packages = "foreach") %do%
      general.ks(N, sigma = mle_c[1, i], dist = "halfnormal", type = "h", phi = mle_c[2, i])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda=NULL, sigma=sigma, mean=NULL)
  }
  if (dist == "exph")
  {
    mle_ori = zih.mle(x,lambda=lambda, type = "h", lowerbound, upperbound,dist='exp.zihmle')
    probs_ori = stats::pexp(0:max(x), rate = mle_ori[1])
    probs_ori = mle_ori[2] + (1 - mle_ori[2]) * (probs_ori -probs_ori[1])/(1 - probs_ori[1])
    step_ori = stats::stepfun(0:max(x), c(0, probs_ori))
    z = stats::knots(step_ori)
    dev = c(0, (stats::ecdf(x))(z) - step_ori(z))
    Dn_ori = max(abs(dev))
    if (bootstrap) {
      mle_new = foreach::foreach(j = 1:nsim, .combine = rbind,
                                 .packages ="foreach") %do% zih.mle(sample(x,size = N, replace = T), lambda=mle_ori[1],
                                                                    type = "h", lowerbound,upperbound,dist = 'exp.zihmle')
      mle_new = t(mle_new)
    }else {
      mle_new = matrix(rep(mle_ori, nsim), ncol = nsim)
    }
    if(bootstrap){
      mle_c = foreach::foreach(j = 1:nsim, .combine = rbind,
                               .packages = "foreach") %do% zih.mle(sample.h1(1000,phi=mle_new[2,j],lambda=mle_new[1,j]),
                                                                   type = "h", lambda=mle_new[1,j],lowerbound,upperbound,dist = 'exp.zihmle')
      mle_c = t(mle_c)
    }else{
      mle_c = matrix(rep(mle_c, nsim), ncol = nsim)
    }
    D2 = foreach::foreach(j = 1:nsim, .combine = c,.packages = "foreach") %do%
      general.ks(N, lambda = mle_c[1, j], dist = "exponential",type = "h", phi = mle_c[2, j])
    temp = list(r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL,n = NULL,lambda=lambda,mean=NULL, sigma=NULL)
  }
  parallel::stopCluster(cl)
  pvalue = sum(D2 > Dn_ori)/nsim
  ans = list(x = x, nsim = nsim, bootstrap = bootstrap, dist = dist,
             lowerbound = lowerbound, upperbound = upperbound, mle_new = mle_new,
             mle_ori = mle_ori,mle_c=mle_c, pvalue = pvalue, N = N)
  ans = c(ans, temp)
  class(ans) = "kstest.B"
  return(ans)
}

check.input<-function(x,nsim,dist,lowerbound,upperbound)
{
  if(min(x)<0)
    stop('x should be nonnegative.')
  if(length(x)<=30)
    warning('Small sample size may lead to biased or inaccurate results.')
  if(length(unique(x))==1)
    stop('There must be more than one unique values in x.')
  if(nsim<=30)
    stop('nsim is too small: please input nsim larger than 30.')
  if(!(dist%in%c('poisson','geometric','nb','nb1','bb','bb1','bnb','bnb1','normal','halfnormal','lognormal','exponential',
                 'zip','zigeom','zinb','zibb','zibnb','zinormal','zilognorm','zihalfnorm','ziexp',
                 'ph','geomh','nbh','bbh','bnbh','normalh','lognormh','halfnormh','exph')))
    stop('please input a distribution name among poisson,geometric,nb,nb1,bb,bb1,bnb,bnb1,normal,halfnormal,lognormal,exponential,
                  zip,zigeom,zinb,zibb,zibnb,zinormal,zilognorm,zihalfnrom,ziexp,
                  ph,geomh,nbh,bbh,bnbh,normalh,lohgnormh,halfnormh,exph.')
  if((lowerbound<0)|(lowerbound>0.1))
    stop('lowerbound is negative or larger than 0.1')
  if(upperbound<1)
    stop('upperbound is too small.')
}

init.para1<-function(x,lambda,r,p,alpha1,alpha2,n,mean,sigma)
{
  lambda=lambda[lambda>0]
  r=r[r>0]
  p=p[(p>0)&(p<1)]
  alpha1=alpha1[alpha1>0]
  alpha2=alpha2[alpha2>0]
  n=n[n>0]
  mean=mean[mean>0]
  sigma=sigma[sigma>0]
  if(length(lambda)>0){
    lambda=lambda[1]
  }else{
    lambda=max(x)+2
  }
  if(length(r)>0){
    r=r[1]  ##only use the first value if user inputs more than one applicable values
  }else{
    r=max(x)  ##if there is no r input, use max(x) as initial value.
  }
  if(length(p)>0){
    p=p[1]  ##only use the first value if user inputs more than one applicable values
  }else{
    p=sum(x>0)/length(x)  ##if there is no p input, use sum(x>0)/length(x) as initial value.
  }
  if(length(n)>0){
    n=n[1]  ##only use the first value if user inputs more than one applicable values
  }else{
    n=max(x)+1  ##if there is no n input, use max(x)+1 as initial value.
  }
  temp1=mean(x)
  temp2=stats::var(x)
  if(length(alpha1)>0){
    alpha1=alpha1[1]  ##only use the first value if user inputs more than one applicable values
  }else{
    #alpha1=max(x)
    alpha1=abs(temp1*(temp1*(1-temp1)/temp2-1))  ##the initial value derives from the moment estimate of beta distribution mean(mean(1-mean)/var-1).
  }
  if(length(alpha2)>0){
    alpha2=alpha2[1]  ##only use the first value if user inputs more than one applicable values
  }else{
    #alpha2=max(x)+3
    alpha2=abs((1-temp1)*(temp1*(1-temp1)/temp2-1))  ##the initial value derives from the moment estimate of beta distribution (1-mean)(mean(1-mean)/var-1).
  }
  if(length(mean)>0){
    mean=mean[1]
  }else{
    mean=sum(x>0)/length(x)
  }
  if(length(sigma)>0){
    sigma=sigma[1]
  }else{
    sigma=stats::var(x)
  }
  return(list(r=r,lambda=lambda,p=p,alpha1=alpha1,alpha2=alpha2,n=n,mean=mean,sigma=sigma))
}

general.ks<-function(N,lambda,r,p,n,alpha1,alpha2,mean,sigma,dist,type=c('general','zi','h'),phi=0)
{
  if((type=='general')|(phi==0)){
    x_new=switch(dist,poisson=stats::rpois(N,lambda=lambda),
                 geometric=stats::rgeom(N,prob=p),
                 nb=stats::rnbinom(N,size=ceiling(r),prob=p),
                 bb=extraDistr::rbbinom(N,size=ceiling(n),alpha=alpha1,beta=alpha2),
                 bnb=extraDistr::rbnbinom(N,size=ceiling(r),alpha=alpha1,beta=alpha2),
                 normal=stats::rnorm(N,mean=mean,sd=sigma),
                 lognormal=stats::plnorm(N,meanlog=mean,sdlog=sigma),
                 halfnormal=extraDistr::rhnorm(N,sigma=sigma),
                 exponential=stats::rexp(N,rate=lambda))
    probs_new=switch(dist,poisson=stats::ppois(0:max(x_new),lambda=lambda),
                     geometric=stats::pgeom(0:max(x_new),prob=p),
                     nb=stats::pnbinom(0:max(x_new),size=ceiling(r),prob=p),
                     bb=extraDistr::pbbinom(0:max(x_new),size=ceiling(n),alpha=alpha1,beta=alpha2),
                     bnb=extraDistr::pbnbinom(0:max(x_new),size=ceiling(r),alpha=alpha1,beta=alpha2),
                     normal=stats::rnorm(0:max(x_new),mean=mean,sd=sigma),
                     lognormal=stats::plnorm(0:max(x_new),meanlog=mean,sdlog=sigma),
                     halfnormal=extraDistr::rhnorm(0:max(x_new),sigma=sigma),
                     exponential=stats::rexp(0:max(x_new),rate=lambda))
  }else{
    if(type=='zi')
    {
      x_new=switch(dist,poisson=sample.zi1(N,phi,dist='poisson',lambda=lambda),
                   geometric=sample.zi1(N,phi,dist='geometric',p=p),
                   nb=sample.zi1(N,phi,dist='nb',r=r,p=p),
                   bb=sample.zi1(N,phi,dist='bb',alpha1=alpha1,alpha2=alpha2,n=n),
                   bnb=sample.zi1(N,phi,dist='bnb',r=r,alpha1=alpha1,alpha2=alpha2),
                   normal=sample.zi1(N,phi,dist='normal',mean=mean,sigma=sigma),
                   lognormal=sample.zi1(N,phi,dist='lognormal',mean=mean,sigma=sigma),
                   halfnormal=sample.zi1(N,phi,dist='halfnormal',sigma=sigma),
                   exponential=sample.zi1(N,phi,dist='exponential',lambda=lambda))
      probs_new=switch(dist,poisson=stats::ppois(0:max(x_new),lambda=lambda),
                       geometric=stats::pgeom(0:max(x_new),prob=p),
                       nb=stats::pnbinom(0:max(x_new),size=ceiling(r),prob=p),
                       bb=extraDistr::pbbinom(0:max(x_new), size=ceiling(n),alpha=alpha1,beta=alpha2),
                       bnb=extraDistr::pbnbinom(0:max(x_new),size=ceiling(r),alpha=alpha1,beta=alpha2),
                       normal=stats::rnorm(0:max(x_new),mean=mean,sd=sigma),
                       lognormal=stats::plnorm(0:max(x_new),meanlog=mean,sdlog=sigma),
                       halfnormal=extraDistr::rhnorm(0:max(x_new),sigma=sigma),
                       exponential=stats::rexp(0:max(x_new),rate=lambda))
      probs_new=phi+(1-phi)*probs_new
    }else{
      if(type=='h'){
        x_new=switch(dist,poisson=sample.h1(N,phi,dist='poisson',lambda=lambda),
                     geometric=sample.h1(N,phi,dist='geometric',p=p),
                     nb=sample.h1(N,phi,dist='nb',r=r,p=p),
                     bb=sample.h1(N,phi,dist='bb',n=n,alpha1=alpha1,alpha2=alpha2),
                     bnb=sample.h1(N,phi,dist='bnb',r=r,alpha1=alpha1,alpha2=alpha2),
                     normal=sample.h1(N,phi,dist='normal',mean=mean,sigma=sigma),
                     lognormal=sample.h1(N,phi,dist='lognormal',mean=mean,sigma=sigma),
                     halfnormal=sample.h1(N,phi,dist='halfnormal',sigma=sigma),
                     exponential=sample.h1(N,phi,dist='exponential',lambda=lambda))
        probs_new=switch(dist,poisson=stats::ppois(0:max(x_new),lambda=lambda),
                         geometric=stats::pgeom(0:max(x_new),prob=p),
                         nb=stats::pnbinom(0:max(x_new),size=ceiling(r),prob=p),
                         bb=extraDistr::pbbinom(0:max(x_new),size=ceiling(n),alpha=alpha1,beta=alpha2),
                         bnb=extraDistr::pbnbinom(0:max(x_new),size=ceiling(r),alpha=alpha1,beta=alpha2),
                         normal=stats::rnorm(0:max(x_new),mean=mean,sd=sigma),
                         lognormal=stats::plnorm(0:max(x_new),meanlog=mean,sdlog=sigma),
                         halfnormal=extraDistr::rhnorm(0:max(x_new),sigma=sigma),
                         exponential=stats::rexp(0:max(x_new),rate=lambda))
        probs_new=phi+(1-phi)*(probs_new-probs_new[1])/(1-probs_new[1])
      }
    }
  }
  step_new=stats::stepfun(0:max(x_new),c(0,probs_new))  ##step function for the new samples
  z=stats::knots(step_new)
  dev=c(0, stats::ecdf(x_new)(z)-step_new(z))
  Dn_new=max(abs(dev)) ##KS statistic for the new samples
  return(Dn_new)
}

#' likelihood ratio test for two models based on kstest.A
#' @description Conduct likelihood ratio test for comparing two different models.
#' @usage lrt.A(d1, d2, parallel = FALSE)
#' @param d1 	An object of class 'kstest.A'.
#' @param d2  An object of class 'kstest.A'.
#' @param parallel Whether to use multiple threads to paralleling computation. Default is FALSE. Please aware that it may take longer time to execute the program with parallel=FALSE.
#'
#' @details If the pvalue of d1 and d2 are greater than the user-specified significance level, which indicates that the original data x may come from the two distributions in d1 and d2, a likelihood ratio test is desired to choose a more 'possible' distribution based on the current data.
#' NOTE that the x in d1 and d2 must be IDENTICAL! Besides, NOTE that the dist in d1 and d2 must be DIFFERENT!
#' The dist inherited from d1 is the null distribution and that from d2 is used as the alternative distribution.
#'
#' If the output p-value smaller than the user-specified significance level, the dist of d2 is more appropriate for modeling x.
#' Otherwise, There is no significant difference between dist of d1 and dist of d2, given the current data.
#' @return The p-value of the likelihood ratio test.
#'
#' @export
#' @examples
#' set.seed(1005)
#' x=sample.h1(2000,phi=0.3,dist='poisson',lambda=10)
#' d1=kstest.A(x,nsim=100,bootstrap = TRUE,dist= 'ph',lowerbound = 1e-10, upperbound = 100000)
#' d2=kstest.A(x,nsim=100,bootstrap = TRUE,dist= 'geomh',lowerbound = 1e-10, upperbound = 100000)
#' lrt.A(d1,d2, parallel = FALSE) #0.28
lrt.A<-function (d1, d2, parallel = FALSE)
{
  `%do%` <- foreach::`%do%`
  if (!(methods::is(d1, "kstest.A") & methods::is(d2, "kstest.A")))
    stop("d1, d2 must be objects from class kstest.A.")
  if (sum((d1$x - d2$x) == 0) < length(d1$x))
    stop("d1$x must be identical to d2$x.")
  lik1_ori = d1$mle_ori
  length1 = length(lik1_ori)
  lik2_ori = d2$mle_ori
  length2 = length(lik2_ori)
  t_ori = lik2_ori[length2] - lik1_ori[length1]
  mle_new = d1$mle_new
  dist1 = d1$dist
  N = d1$N
  lambda1 = d1$lambda
  r1 = d1$r
  p1 = d1$p
  alpha11 = d1$alpha1
  alpha12 = d1$alpha2
  n1 = d1$n
  mean1 = d1$mean
  sigma1 = d1$sigma
  phi1 = d1$phi
  nsim = d1$nsim
  lowerbound1 = d1$lowerbound
  upperbound1 = d1$upperbound
  dist2 = d2$dist
  lambda2 = d2$lambda
  r2 = d2$r
  p2 = d2$p
  alpha21 = d2$alpha1
  alpha22 = d2$alpha2
  n2 = d2$n
  mean2 = d2$mean
  sigma2 = d2$sigma
  phi2 = d2$phi
  lowerbound2 = d2$lowerbound
  upperbound2 = d2$upperbound
  f1 <- function(dist, para)
  {
    temp1 = switch(dist,
                   poisson = stats::rpois(N, lambda = para[1]),
                   geometric=stats::rgeom(N, prob=para[1]),
                   nb = stats::rnbinom(N, size = ceiling(para[1]), prob = para[2]),
                   nb1 = stats::rnbinom(N, size = ceiling(para[1]), prob = para[2]),
                   bb = extraDistr::rbbinom(N, size = ceiling(para[1]), alpha = para[2], beta = para[3]),
                   bb1 = extraDistr::rbbinom(N, size = ceiling(para[1]), alpha = para[2], beta = para[3]),
                   bnb = extraDistr::rbnbinom(N, size = ceiling(para[1]), alpha = para[2], beta = para[3]),
                   bnb1 = extraDistr::rbnbinom(N, size = ceiling(para[1]), alpha = para[2], beta = para[3]),
                   normal = stats::rnorm(N, mean=para[1], sd=para[2]),
                   lognormal = stats::rlnorm(N, meanlog = para[1], sdlog = para[2]),
                   halfnormal = extraDistr::rhnorm(N, sigma = para[1]),
                   exponential = stats::rexp(N,rate=para[1]),
                   zip = sample.zi1(N, phi = para[2], lambda = para[1],dist ="poisson"),
                   zigeom = sample.zi1(N, phi = para[2], p=para[1], dist="geometric"),
                   zinb = sample.zi1(N, phi=para[3], dist = "nb", r = para[1], p = para[2]),
                   zibb = sample.zi1(N, phi=para[4], dist = "bb", alpha1 = para[2], alpha2 = para[3], n = para[1]),
                   zibnb = sample.zi1(N, phi=para[4], dist = "bnb", r = para[1], alpha1 = para[2],  alpha2 = para[3]),
                   zinormal = sample.zi1(N, phi=para[3], dist="normal", mean=para[1], sigma=para[2]),
                   zilognorm = sample.zi1(N, phi=para[3], dist="lognormal", mean=para[1], sigma=para[2]),
                   zihalfnorm = sample.zi1(N, phi=para[2], dist="halfnormal", sigma=para[1]),
                   ziexp = sample.zi1(N,phi=para[2],dist="exponential", lambda=para[1]),
                   ph = sample.h1(N, phi=para[2], lambda = para[1], dist="poisson"),
                   geomh = sample.h1(N, phi=para[2], p=para[1], dist="geometric"),
                   nbh = sample.h1(N, phi=para[3], dist = "nb", r = para[1], p = para[2]),
                   bbh = sample.h1(N, phi=para[4], dist = "bb", alpha1 = para[2], alpha2 = para[3], n = para[1]),
                   bnbh = sample.h1(N, phi=para[4], dist = "bnb", r = para[1], alpha1 = para[2], alpha2 = para[3]),
                   normalh = sample.h1(N, phi=para[3], dist="normal", mean=para[1], sigma=para[2]),
                   lognormh = sample.h1(N, phi=para[3], dist="lognormal", mean=para[1], sigma=para[2]),
                   halfnormh = sample.h1(N, phi=para[2], dist="halfnormal", sigma=para[1]),
                   exph = sample.h1(N, phi=para[2], dist="exponential", lambda=para[1]))
    return(temp1)
  }
  f2 <- function(x, dist, lambda, r, p, alpha1, alpha2, n, mean, sigma, phi, lowerbound, upperbound)
  {
    temp2 = switch(dist,
                   poisson = new.mle(x,lambda=lambda, dist="poisson"),
                   geometric = new.mle(x, p=p, dist="geometric"),
                   nb = new.mle(x, r=r, p=p, lowerbound, upperbound,dist="nb"),
                   nb1 = new.mle(x, r=r, p=p, lowerbound, upperbound,dist="nb1"),
                   bb = new.mle(x, n=n, alpha1=alpha1, alpha2=alpha2, lowerbound, upperbound,dist="bb"),
                   bb1 = new.mle(x, n=n, alpha1=alpha1, alpha2=alpha2, lowerbound, upperbound,dist="bb1"),
                   bnb = new.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, lowerbound, upperbound,dist="bnb"),
                   bnb1 = new.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, lowerbound, upperbound,dist="bnb1"),
                   normal = new.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="normal"),
                   lognormal = new.mle(x, mean = mean, sigma=sigma, lowerbound, upperbound, dist="lognormal"),
                   halfnormal = new.mle(x, sigma = sigma, lowerbound, upperbound, dist="halfnormal"),
                   exponential = new.mle(x, lambda=lambda, lowerbound, upperbound, dist="exponential"),
                   zip = zih.mle(x, lambda=lambda, type = "zi", lowerbound, upperbound, dist="poisson.zihmle"),
                   zigeom = zih.mle(x,p=p,type="zi",lowerbound, upperbound,dist="geometric.zihmle"),
                   zinb = zih.mle(x, r=r, p=p, type = "zi", lowerbound, upperbound,dist="nb1.zihmle"),
                   zibb = zih.mle(x, n=n, alpha1=alpha1, alpha2=alpha2, type = "zi", lowerbound, upperbound,dist="bb1.zihmle"),
                   zibnb = zih.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, type = "zi", lowerbound, upperbound,dist="bnb1.zihmle"),
                   zinormal = zih.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound,dist="normal.zihmle", type="zi"),
                   zilognorm = zih.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="lognorm.zihmle", type="zi"),
                   zihalfnorm = zih.mle(x, sigma=sigma, lowerbound, upperbound, dist="halfnorm.zihmle", type="zi"),
                   ziexp = zih.mle(x, lambda=lambda, lowerbound, upperbound, dist="exp.zihmle", type="zi"),
                   ph = zih.mle(x, lambda=lambda,type = "h", lowerbound, upperbound,dist="poisson.zihmle"),
                   geomh=zih.mle(x, p=p, type="h", lowerbound, upperbound, dist="geometric.zihmle"),
                   nbh = zih.mle(x, r=r, p=p, type = "h", lowerbound, upperbound,dist="nb1.zihmle"),
                   bbh = zih.mle(x, n=n, alpha1=alpha1, alpha2=alpha2, type = "h", lowerbound, upperbound,dist="bb1.zihmle"),
                   bnbh = zih.mle(x, r=r, alpha1=alpha1, alpha2=alpha2, type = "h", lowerbound, upperbound,dist="bnb1.zihmle"),
                   normalh = zih.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="normal.zihmle", type="h"),
                   lognormh = zih.mle(x, mean=mean, sigma=sigma, lowerbound, upperbound, dist="lognorm.zihmle", type="h"),
                   halfnormh = zih.mle(x, sigma=sigma, lowerbound, upperbound, dist="halfnorm.zihmle", type="h"),
                   exph = zih.mle(x, lambda=lambda, lowerbound, upperbound, dist="exp.zihmle", type="h"))
    return(temp2)
  }
  if (parallel) {
    cl_cores = parallel::detectCores()
    cl = parallel::makeCluster(cl_cores - 2)
  }
  else {
    cl = parallel::makeCluster(1)
  }
  doParallel::registerDoParallel(cl)
  j = 0
  t_new = foreach::foreach(j = 1:nsim, .combine = c,
                           .packages = c("extraDistr","foreach")) %do% {x = f1(dist1,mle_new[, j])
                           new1 = f2(x, dist1, lambda1, r1, p1, alpha11, alpha12, n1, mean1, sigma1, phi1, lowerbound1, upperbound1)
                           new2 = f2(x, dist2, lambda2, r2, p2, alpha21, alpha22, n2, mean2, sigma2, phi2, lowerbound2, upperbound2)
                           dif = new2[length2] - new1[length1]
                           }
  parallel::stopCluster(cl)
  pvalue = sum(t_new >= t_ori)/nsim
  return(pvalue)
}



check.input1<-function(x,dist,lowerbound,upperbound)
{
  if(min(x)<0)
    stop('x should be nonnegative.')
  if(length(x)<=30)
    warning('Small sample size may lead to biased or inaccurate results.')
  if(!(dist%in%c('poisson','geometric','nb','bb','bnb','normal','halfnormal','lognormal','exponential',
                 'zip','zigeom','zinb','zibb','zibnb','zinormal','zilognorm','zihalfnorm','ziexp',
                 'ph','geomh','nbh','bbh','bnbh')))
    stop('please input a distribution name among poisson,geometric,nb,bb,bnb,normal,halfnormal,lognormal,exponential,
                  zip,zigeom,zinb,zibb,zibnb,zinormal,zilognorm,zihalfnrom,ziexp,
                  ph,geomh,nbh,bbh,bnbh.')
  if((lowerbound<0)|(lowerbound>0.1))
    stop('lowerbound is negative or larger than 0.1')
  if(upperbound<1)
    stop('upperbound is too small.')
}


#' Inverse Fisher Information matrix and confidence intervals of the parameters for general, continuous, and discrete
#' zero-inflated or hurdle distributions.
#'
#' @description Computes the inverse of the fisher information matrix for Poisson, geometric, negative binomial, beta binomial,
#' beta negative binomial, normal, lognormal, half normal, and exponential distributions and their zero-inflated and hurdle versions along with the confidence intervals
#' of all parameters in the model.
#' @usage FI.ZI(x,dist="poisson",r=NULL,p=NULL,alpha1=NULL,alpha2=NULL,
#' n=NULL,lambda=NULL,mean=NULL,sigma=NULL,lowerbound=0.01,upperbound=10000)
#' @param x A vector of count data. Should be non-negative integers for discrete cases.  Random generation for continuous cases.
#' @param dist The distribution used to calculate the inverse of fisher information and confidence interval. It can be one of
#' {'poisson','geometric','nb','bb','bnb','normal','halfnormal','lognormal','exponential',
#' 'zip','zigeom','zinb','zibb','zibnb', 'zinormal','zilognorm','zohalfnorm','ziexp',
#' 'ph','geomh','nbh','bbh','bnbh'} which corresponds to general Poisson, geometric, negative binomial, beta binomial,
#'  beta negative binomial, normal, log normal, half normal, exponential, Zero-Inflated Poisson,
#'  Zero-Inflated geometric, Zero-Inflated negative binomial, Zero-Inflated beta binomial,
#'  Zero-Inflated beta negative binomial, Zero-Inflated/hurdle normal, Zero-Inflated/hurdle log normal,
#'  Zero-Inflated/hurdle half normal, Zero-Inflated/hurdle exponential, Zero-Hurdle Poisson,
#'  Zero-Hurdle geometric, Zero-Hurdle negative binomial, Zero-Hurdle beta binomial, and
#'  Zero-Hurdle beta negative binomial distributions, respectively.
#' @param r An initial value of the number of success before which m failures are observed, where m is the element of x. Must be a positive number, but not required to be an integer.
#' @param p An initial value of the probability of success, should be a positive value within (0,1).
#' @param alpha1 An initial value for the first shape parameter of beta distribution. Should be a positive number.
#' @param alpha2 An initial value for the second shape parameter of beta distribution. Should be a positive number.
#' @param n An initial value of the number of trials. Must be a positive number, but not required to be an integer.
#' @param lambda An initial value of the rate. Must be a positive real number.
#' @param mean An initial value of the mean or expectation.
#' @param sigma An initial value of the standard deviation. Must be a positive real number.
#' @param lowerbound A lower searching bound used in the optimization of likelihood function. Should be a small positive number. The default is 1e-2.
#' @param upperbound An upper searching bound used in the optimization of likelihood function. Should be a large positive number. The default is 1e4.
#'
#' @details FI.ZI calculate the inverse of the fisher information matrix and the corresponding confidence interval of
#' the parameter of general, Zero-Inflated, and Zero-Hurdle Poisson, geometric, negative binomial, beta binomial,
#' beta negative binomial, normal, log normal, half normal, and exponential distributions.
#' Note that zero-inflated and hurdle are the same in continuous distributions.
#'
#' @return A list containing the inverse of the fisher information matrix and the corresponding 95\% confidence interval for all the parameters in the model.
#'
#' @references \itemize{\item Aldirawi H, Yang J (2022). “Modeling Sparse Data Using MLE with Applications to Micro-
#' biome Data.” Journal of Statistical Theory and Practice, 16(1), 1–16.}
#' @export
#' @examples
#' set.seed(111)
#' N=1000;lambda=10;
#' x<-stats::rpois(N,lambda=lambda)
#' FI.ZI(x,lambda=5,dist="poisson")
#' #$inversefisher
#' #     lambda
#' #[1,]  9.896
#'
#' #$ConfidenceIntervals
#' #[1]  9.701025 10.090974
#' set.seed(111)
#'N=1000;lambda=10;phi=0.4;
#' x1<-sample.h1(N,lambda=lambda,phi=phi,dist="poisson")
#' FI.ZI(x1,lambda=4,dist="ph")
#' #$inversefisher
#' #       [,1]     [,2]
#' #[1,] 0.239799  0.00000
#' #[2,] 0.000000 16.64774
#'
#' #$ConfidenceIntervals
#' #              [,1]       [,2]
#' #CI of Phi    0.3686491  0.4293509
#' #CI of lambda 9.7483238 10.2540970
#' set.seed(289)
#' N=2000;mean=10;sigma=2;phi=0.4;
#' x<-sample.zi1(N,phi=phi,mean=mean,sigma=sigma,dist="lognormal")
#' FI.ZI(x, mean=1,sigma=1, dist="zilognorm")
#' # $inversefisher
#' #        [,1]     [,2]     [,3]
#' #[1,] 0.6313214 0.000000 0.000000
#' #[2,] 0.0000000 6.698431 0.000000
#' #[3,] 0.0000000 0.000000 3.349215
#'
#' #$ConfidenceIntervals
#' #              [,1]       [,2]
#' #CI of phi   0.3521776  0.4218224
#' #CI of mean  9.8860358 10.1128915
#' #CI of sigma 1.9461552  2.1065664

FI.ZI <- function (x, dist= "poisson",
                   r = NULL, p = NULL, alpha1 = NULL, alpha2 = NULL, n = NULL, lambda = NULL, mean = NULL, sigma = NULL,
                   lowerbound = 0.01, upperbound = 10000)
{
  x=ceiling(as.vector(x))
  dist=tolower(dist)[1]
  check.input1(x,dist,lowerbound,upperbound)
  init=init.para1(x,lambda,r,p,alpha1,alpha2,n,mean,sigma)
  r = init$r
  p = init$p
  alpha1 = init$alpha1
  alpha2 = init$alpha2
  n = init$n
  lambda=init$lambda
  N=length(x)
  mean=init$mean
  sigma=init$sigma
  ####### General Type ######
  if (dist=="poisson")
  {
    #y<-stats::rpois(N,lambda=lambda)
    mle<-new.mle(x,lambda=lambda,dist="poisson",lowerbound,upperbound)
    lambda=mle[1,1]
    lambdaest<-lambda
    N=length(x)
    phi=0
    A11 = (1 - exp(-lambda))/((phi  + (1-phi) * exp(-lambda)) * (1-phi))
    A12 = - exp(-lambda)/(phi + (1-phi) * exp(-lambda))
    A21 = A12
    A22 = (1-phi) * ((1/lambda) - (phi * exp(-lambda)/(phi + (1-phi) * exp(-lambda))))
    FZIP = N * matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(A22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIl = lambdaest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    return(list(inversefisher=inv.fish,ConfidenceIntervals=CIl))
  }
  if (dist=="geometric")
  {
    #y<-stats::rgeom(N,p=p)
    mle<-new.mle(x,p=p,dist="geometric",lowerbound,upperbound)
    p=mle[1,1]
    pest<-p
    N=length(x)
    phi=0
    p0=p/(1-p)
    cstar=p0/(1-p0)
    A22 = -(1-phi)/(p^2*(1-p))
    inv.fish<-solve(A22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    return(list(inversefisher=inv.fish,ConfidenceIntervals=CIp))
  }
  if (dist=="nb")
  {
    #
    mle<-new.mle(x,r=r,p=p,dist="nb",lowerbound,upperbound)
    r=mle[1,1]
    p=mle[1,2]
    rest<-r
    pest<-p
    phi=0
    N = length(x)
    m = 5000;
    y<-stats::rnbinom(m,r,p)
    p0 = (1-p)^r
    A11 = (1-p0)/(phi+(1-phi) * p0) * (1-phi)
    A12 = (p0/(phi+(1-phi) * p0)) * log(1-p)
    A21 = A12
    A13 = (p0/(phi+(1-phi) * p0)) * (-r/(1-p))
    A31 = A13
    Cstar=(phi*p0)/(phi+(1-phi) * p0)
    A22=-(1-phi) * (mean(trigamma(y+r)) - trigamma(r) + Cstar * (log(1-p))^2)
    A23=-(1-phi) * ((-1/(1-p)) - Cstar * (r * log(1-p)/(1-p)))
    A32=A23
    A33=-(1-phi) * ((-r/(p*(1-p)^2)) + Cstar * (r^2/(1-p)^2))
    FZINB =  matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZINB22 = matrix (c(A22,A23,A32,A33), ncol=2,nrow=2,byrow =TRUE)
    if(corpcor::is.positive.definite(FZINB22, tol=1e-8))
    {
      inv.fish<-solve(FZINB22)
    }
    else{
      a<-corpcor::make.positive.definite(FZINB22)
      inv.fish<-solve(a)
    }
    crit <- stats::qnorm((1 + 0.95)/2)
    CIr = rest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIr,CIp),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of r", "CI of p")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="bb")
  {
    mle1<-new.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,dist="bb",lowerbound,upperbound)
    N = length(x)
    n=mle1[1,1]
    alpha=mle1[1,2]
    beta=mle1[1,3]
    phi=0
    nest<-n;alphaest<-alpha;betaest<-beta
    m = 5000;
    y<-extraDistr::rbbinom(m, size=n, alpha=alpha, beta=beta)
    p0 = beta(alpha,n+beta)/beta(alpha,beta)
    A11 = (1-p0)/(phi+(1-phi) * p0) * (1-phi)
    A12 = (p0/(phi+(1-phi) * p0)) * (digamma(n+beta) - digamma(n+alpha+beta))
    A21 = A12
    A13 = (p0/(phi+(1-phi) * p0)) * (digamma(alpha+beta) - digamma(n+alpha+beta))
    A31 = A13
    A14 = (p0/(phi+(1-phi) * p0)) * (digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(alpha))
    A41=A14
    
    Cstar = (phi*p0)/(phi+(1-phi) * p0)
    
    A22 = -(1-phi) * (trigamma(n+1) - trigamma(n+alpha+beta) + mean(trigamma(n-y+beta)) - mean(trigamma(n-y+1))+
                        Cstar * (digamma(n+beta) - digamma(n+alpha+beta))^2)
    A23 = -(1-phi) * (-trigamma(n+alpha+beta) +
                        Cstar * (digamma(n+beta) - digamma(n+alpha+beta)) * (digamma(alpha+beta) - digamma(n+alpha+beta)))
    A32 = A23
    A24 = -(1-phi) * (mean(trigamma(n-y+beta)) - trigamma(n+alpha+beta) +
                        Cstar * (digamma(n+beta) - digamma(n+alpha+beta)) * (digamma(n+beta) + digamma(alpha+beta) -
                                                                               digamma(n+alpha+beta) - digamma(beta)))
    A42 = A24
    A33 = -(1-phi) * (trigamma(alpha+beta) - trigamma(n+alpha+beta) - trigamma(alpha) + mean(trigamma(y+alpha)) +
                        Cstar * (digamma(alpha+beta) - digamma(n+alpha+beta))^2)
    A34 = -(1-phi) * (trigamma(alpha+beta) - trigamma(n+alpha+beta) +
                        Cstar * (digamma(alpha+beta) - digamma(n+alpha+beta))*
                        (digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(beta)))
    A43 = A34
    A44 = -(1-phi) * (trigamma(alpha+beta) - trigamma(beta) - trigamma(n+alpha+beta) + mean(trigamma(n-y+beta)) +
                        Cstar * (digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(beta))^2)
    
    FZIBB = N * matrix(c(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,
                         A34,A41,A42,A43,A44),ncol=4,nrow=4,byrow=TRUE)
    FZIBB33 = matrix(c(A22,A23,A24,A32,A33,A34,A42,A43,A44), ncol=3,nrow=3,byrow =TRUE)
    if(corpcor::is.positive.definite(FZIBB33, tol=1e-8))
    {
      inv.fish<-solve(FZIBB33)
    }
    else{
      a<-corpcor::make.positive.definite(FZIBB33)
      inv.fish<-solve(a)
    }
    crit <- stats::qnorm((1 + 0.95)/2)
    CIn = nest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    
    CIa = alphaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    
    CIb = betaest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    
    ConfInt=matrix(c(CIn,CIa,CIb),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of n", "CI of alpha" , "CI of beta")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="bnb")
  {
    mle1<-new.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,dist="bnb",lowerbound,upperbound)
    r=mle1[1,1]
    alpha=mle1[1,2]
    beta=mle1[1,3]
    phi=0
    N = length(x)
    m = 5000;
    y<-extraDistr::rbnbinom(m, size=r, alpha=alpha, beta=beta)
    p0=beta(r+alpha,beta)/beta(alpha,beta)
    A11 = (1-p0)/(phi+(1-phi) * p0) * (1-phi)
    A12 = (p0/(phi+(1-phi) * p0)) * (digamma(r+alpha) - digamma(r+alpha+beta))
    A21 = A12
    A13 = (p0/(phi+(1-phi) * p0)) * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha))
    A31 = A13
    A14 = (p0/(phi+(1-phi) * p0)) * (digamma(alpha+beta) - digamma(r+alpha+beta))
    A41=A14
    Cstar = (phi*p0)/(phi+(1-phi) * p0)
    A22 = -(1-phi) * (trigamma(r+alpha) - trigamma(r) - mean(trigamma(r+y+alpha+beta)) + mean(trigamma(r+y)) +
                        Cstar * (digamma(r+alpha) - digamma(r+alpha+beta))^2)
    A23 = -(1-phi) * (trigamma(r+alpha) - mean(trigamma(r+y+alpha+beta)) +
                        Cstar * (digamma(r+alpha) - digamma(r+alpha+beta)) * (digamma(r+alpha) +
                                                                                digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha)))
    A32 = A23
    A24 = -(1-phi) * (-mean(trigamma(r+y+alpha+beta)) +
                        Cstar * (digamma(r+alpha) - digamma(r+alpha+beta)) * (digamma(alpha+beta) - digamma(r+alpha+beta)))
    A42 = A24
    A33 = -(1-phi) * (trigamma(alpha+r) - trigamma(alpha) + trigamma(alpha+beta) - mean(trigamma(r+y+alpha+beta)) +
                        Cstar * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha))^2)
    A34 = -(1-phi) * (-mean(trigamma(r+y+alpha+beta)) + trigamma(alpha+beta) +
                        Cstar * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha))*
                        (digamma(alpha+beta) - digamma(r+alpha+beta)))
    A43 = A34
    A44 = -(1-phi) * (trigamma(alpha+beta) - trigamma(beta) - mean(trigamma(r+y+alpha+beta)) + mean(trigamma(y+beta)) +
                        Cstar * (digamma(alpha+beta) - digamma(r+alpha+beta))^2)
    
    FZIBNB = matrix(c(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,
                      A34,A41,A42,A43,A44),ncol=4,nrow=4,byrow=TRUE)
    FZIBNB33= matrix (c(A22,A23,A24,A32,A33,A34,A42,A43,A44), ncol=3,nrow=3,byrow =TRUE)
    if(corpcor::is.positive.definite(FZIBNB33, tol=1e-8))
    {
      inv.fish<-solve(FZIBNB33)
    }
    else{
      a<-corpcor::make.positive.definite(FZIBNB33)
      inv.fish<-solve(a)
    }
    crit <- stats::qnorm((1 + 0.95)/2)
    CIr = r + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIa = alpha + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIb = beta + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    ConfInt=matrix(c(CIr,CIa,CIb),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of r", "CI of alpha" , "CI of beta")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="exponential")
  {
    mle<-new.mle(x,lambda=lambda,dist="exponential",lowerbound,upperbound)
    lambda=mle[1,1]
    N=length(x)
    phi=0
    A22=(1-phi)/(lambda^2)
    inv.fish<-solve(A22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIlambda = lambda + c(-1, 1) * crit * sqrt(inv.fish)/sqrt(N)
    return(list(inversefisher=inv.fish,ConfidenceIntervals=CIlambda))
  }
  if (dist=="normal")
  {
    #y<-stats::rnorm(N,mean=mean,sd=sigma)
    mle<-new.mle(x,mean=mean,sigma=sigma,dist="normal",lowerbound,upperbound)
    mean<-mle[1,1]
    sigma<-mle[1,2]
    phi=0
    meanest<-mean;sigmaest<-sigma;
    N=length(x)
    A11=1 /phi* (1-phi)
    A12=0
    A21=A12
    A13=0
    A31=A13
    A22=(1-phi)/(sigma^2)
    A23=0
    A32=0
    A33=(1-phi)*2/(sigma^2)
    FZIN = matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZIN22 = matrix(c(A22,A23,A32,A33),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIN22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIm = meanest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIs = sigmaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIm,CIs),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of mean", "CI of sigma")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="halfnormal")
  {
    #y<-extraDistr::rhnorm(N,sigma=sigma)
    mle<-new.mle(x,sigma=sigma,dist="halfnormal",lowerbound,upperbound)
    sigma<-mle[1,1]
    sigmaest<-sigma
    phi=0
    N=length(x)
    A11=1 /phi* (1-phi)
    A12=0
    A21=A12
    A22=(1-phi)*2/(sigma^2)
    FZIhf =  matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(A22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIs = sigma + c(-1, 1) * crit * sqrt(inv.fish)/sqrt(N)
    return(list(inversefisher=inv.fish,ConfidenceIntervals=CIs))
  }
  if (dist=="lognormal")
  {
    #y<-stats::rlnorm(N, meanlog =mean, sdlog = sigma)
    mle<-new.mle(x,mean=mean,sigma=sigma,dist="lognormal",lowerbound,upperbound)
    mean=mle[1,1]
    sigma=mle[1,2]
    phi=0
    N=length(x)
    A11=1 /phi* (1-phi)
    A12=0
    A21=A12
    A13=0
    A31=A13
    A22=(1-phi)/(sigma^2)
    A23=0
    A32=A23
    A33=(1-phi)*2/(sigma^2)
    FZIN=N*matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZIN22=matrix(c(A22,A23,A32,A33),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIN22)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIm = mean + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIs = sigma + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIm,CIs),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of mean", "CI of sigma")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  ######### ZI Type ##########
  if (dist=="zip")
  {
    #y<-sample.zi1(N,phi=phi,lambda=lambda,dist='poisson')
    mle<-zih.mle(x,lambda = lambda,dist="poisson.zihmle",type='zi',lowerbound,upperbound)
    lambda=mle[1,1]
    phi=mle[1,2]
    phiest<-phi
    lambdaest<-lambda
    N=length(x)
    A11 = (1 - exp(-lambda))/((phi  + (1-phi) * exp(-lambda)) * (1-phi))
    A12 = - exp(-lambda)/(phi + (1-phi) * exp(-lambda))
    A21 = A12
    A22 = (1-phi) * ((1/lambda) - (phi * exp(-lambda))/(phi + (1-phi) * exp(-lambda)))
    FZIP =  matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIP)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIl = lambdaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIl),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of lambda")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zigeom")
  {
    mle<-zih.mle(x,p=p,dist="geometric.zihmle",type='zi',lowerbound,upperbound)
    p=mle[1,1]
    phi=mle[1,2]
    pest<-p;phiest<-phi;
    N=length(x)
    A11 = (1-p) /(phi+(1-phi)*p)*(1-phi)
    A12 = 1/(phi+(1-phi)*p)
    A21 = A12
    A22 = 1/(phi+(1-phi)*p) * ((1-phi)*(phi*(1-p)+p*(1-phi)+phi*p^2))/(p^2*(1-p))
    FZIgeom = matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    
    if(corpcor::is.positive.definite(FZIgeom, tol=1e-8))
    {
      inv.fish<-solve(FZIgeom)
    }
    else{
      a<-QRM::eigenmeth(FZIgeom, delta = 0.001)
      inv.fish<-solve(a)
    }
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIp),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of p" )
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zinb")
  {
    mle<-zih.mle(x,r=r,p=p,dist="nb.zihmle",type="zi",lowerbound,upperbound)
    N = length(x)
    r=mle[1,1]
    p=mle[1,2]
    phi=mle[1,3]
    rest<-r;pest<-p;phiest<-phi;
    m = 1e5;
    y<-stats::rnbinom(m,r,p)
    p0 = (1-p)^r
    A11 = (1-p0)/((phi+(1-phi) * p0) * (1-phi))
    A12 = (p0/(phi+(1-phi) * p0)) * log(1-p)
    A21 = A12
    A13 = (p0/(phi+(1-phi) * p0)) * (-r/(1-p))
    A31 = A13
    Cstar=(phi*p0)/(phi+(1-phi) * p0)
    
    A22=-(1-phi) * (mean(trigamma(y+r)) - trigamma(r) + Cstar * log(1-p)^2)
    A23=(1-phi) * ((1/(1-p)) + Cstar * (r * log(1-p)/(1-p)))
    A32=A23
    A33=(1-phi) * ((r/(p*(1-p)^2)) - Cstar * (r/(1-p))^2)
    FZINB =  matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZINB22= matrix (c(A22,A23,A32,A33), ncol=2,nrow=2,byrow =TRUE)
    inv.fish<-solve(FZINB)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIr = rest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIr,CIp),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of r", "CI of p")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zibb")
  {
    mle<-zih.mle(x,r=r,p=p,dist="nb.zihmle",type="zi",lowerbound,upperbound)
    N = length(x)
    r=mle[1,1]
    p=mle[1,2]
    phi=mle[1,3]
    m = 1e5;
    y<-stats::rnbinom(m,r,p)
    rest<-r;pest<-p;phiest<-phi;
    p0 = (p)^r
    A11 = (1-p0)/((phi+(1-phi) * p0) * (1-phi))
    A12 = (p0/(phi+(1-phi) * p0)) * log(p)
    A21 = A12
    A13 = (p0/(phi+(1-phi) * p0)) * (r/p)
    A31 = A13
    Cstar=(phi*p0)/(phi+(1-phi) * p0)
    
    A22 = -(1-phi) * (mean(trigamma(y+r)) - trigamma(r) + Cstar * log(p)^2)
    A23 = (1-phi) * ((1/p) + Cstar * (r * log(p)/(p)))
    A32=A23
    A33=(1-phi) * ((r/(p^2*(1-p))) - Cstar * (r/p)^2)
    FZINB =  matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZINB22= matrix (c(A22,A23,A32,A33), ncol=2,nrow=2,byrow =TRUE)
    inv.fish<-solve(FZINB)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIr = rest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    CIs=matrix(c(CIphi,CIr,CIp),ncol=2,nrow=3,byrow=TRUE)
    rownames(CIs) = c("CI of phi","CI of r", "CI of p")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=CIs))
  }
  if (dist=="zibnb")
  {mle1 <- zih.mle(x, r = r, alpha1 = alpha1, alpha2 = alpha2, dist = "bnb.zihmle", type = "zi", lowerbound, upperbound)
  N = length(x)
  r = mle1[1, 1]
  alpha = mle1[1, 2]
  beta = mle1[1, 3]
  phi = mle1[1, 4]
  m = 2e5
  y <- extraDistr::rbnbinom(m, size = r, alpha = alpha, beta = beta)
  p0 = exp(base::lbeta(r + alpha, beta) - base::lbeta(alpha, beta))
  #p0 = (gamma(r + alpha) * gamma(beta) * gamma(alpha + beta)) / (gamma(alpha) * gamma(beta) * gamma(r + alpha + beta))
  A11 = (1 - p0)/((phi + ((1 - phi) * p0)) * (1 - phi))
  #denom = (phi * gamma(r + alpha + beta) * gamma(alpha)) + ((1-phi) * gamma(r + alpha) * gamma(alpha + beta))
  #A11 = ((gamma(r + alpha + beta) * gamma(alpha)) - (gamma(r + alpha) * gamma(alpha + beta)))/(denom * (1 - phi))
  A12 = (p0/(phi + ((1 - phi) * p0))) * (digamma(r + alpha) - digamma(r + alpha + beta))
  #A12 = (gamma(r + alpha) * gamma(alpha + beta)) * (digamma(r + alpha) - digamma(r + alpha + beta)) / denom
  A21 = A12
  A13 = (p0/(phi + ((1 - phi) * p0))) * (digamma(r + alpha) + digamma(alpha + beta) - digamma(r + alpha + beta) - digamma(alpha))
  #A13 = gamma(r + alpha) * gamma(alpha + beta) * (digamma(r + alpha) + digamma(alpha + beta) - digamma(r + alpha + beta) - digamma(alpha))/denom
  A31 = A13
  A14 = (p0/(phi + ((1 - phi) * p0))) * (digamma(alpha + beta) - digamma(r + alpha + beta))
  #A14 = gamma(r + alpha) * gamma(alpha + beta) * (digamma(alpha + beta) - digamma(r + alpha + beta)) /denom
  A41 = A14
  Cstar = (phi * p0)/(phi + ((1 - phi) * p0));
  A22 = -(1 - phi) * ((mean(trigamma(r + y)) - trigamma(r) + trigamma(r + alpha) - mean(trigamma(r + y + alpha + beta))) +
                        (Cstar * ((digamma(r + alpha) - digamma(r + alpha + beta))^2)))
  A23 = -(1 - phi) * ((trigamma(r + alpha) - mean(trigamma(r + y + alpha + beta))) +
                        (Cstar * (digamma(r + alpha) - digamma(r + alpha + beta)) * (digamma(r + alpha) + digamma(alpha + beta) - digamma(r + alpha + beta) - digamma(alpha))))
  A32 = A23
  A24 = -(1 - phi) * ((-mean(trigamma(r + y + alpha + beta))) + 
                        (Cstar * (digamma(r + alpha) - digamma(r + alpha + beta)) * (digamma(alpha + beta) - digamma(r + alpha + beta))))
  A42 = A24
  A33 = -(1 - phi) * ((trigamma(r + alpha) - mean(trigamma(r + y + alpha + beta)) + trigamma(alpha + beta) - trigamma(alpha)) + 
                        (Cstar * ((digamma(r + alpha) + digamma(alpha + beta) - digamma(r + alpha + beta) - digamma(alpha))^2)))
  A34 = -(1 - phi) * ((-mean(trigamma(r + y + alpha + beta)) + trigamma(alpha + beta)) +
                        (Cstar * (digamma(r + alpha) + digamma(alpha + beta) - digamma(r + alpha + beta) - digamma(alpha)) * (digamma(alpha + beta) - digamma(r + alpha + beta))))
  A43 = A34
  A44 = -(1 - phi) * ((mean(trigamma(y +  beta)) - mean(trigamma(r + y + alpha + beta)) + trigamma(alpha + beta) - trigamma(beta))  + 
                        (Cstar * ((digamma(alpha + beta) - digamma(r + alpha + beta))^2)))
  FZIBNB = matrix(c(A11, A12, A13, A14, A21, A22, A23, A24, A31, A32, A33, A34, A41, A42, A43, A44), ncol = 4, nrow = 4, byrow = TRUE)
  FZIBNB33 = matrix(c(A22, A23, A24, A32, A33, A34, A42, A43, A44), ncol = 3, nrow = 3, byrow = TRUE)
  if (corpcor::is.positive.definite(FZIBNB, tol = 1e-08)) {
    inv.fish <- solve(FZIBNB)
  }
  else {
    a <- QRM::eigenmeth(FZIBNB, delta = 0.001)
    inv.fish <- solve(a)
  }
  crit <- stats::qnorm((1 + 0.95)/2)
  CIphi = phi + c(-1, 1) * crit * sqrt(inv.fish[1, 1])/sqrt(N)
  CIr = r + c(-1, 1) * crit * sqrt(inv.fish[2, 2])/sqrt(N)
  CIa = alpha + c(-1, 1) * crit * sqrt(inv.fish[3, 3])/sqrt(N)
  CIb = beta + c(-1, 1) * crit * sqrt(inv.fish[4, 4])/sqrt(N)
  ConfInt = matrix(c(CIphi, CIr, CIa, CIb), ncol = 2, nrow = 4, byrow = TRUE)
  rownames(ConfInt) = c("CI of phi", "CI of r", "CI of alpha", "CI of beta")
  return(list(inversefisher = inv.fish, ConfidenceIntervals = ConfInt))
  }
  if (dist=="ziexp")
  {
    mle<-zih.mle(x, lambda = lambda, dist="exp.zihmle",type="zi",lowerbound,upperbound)
    lambda=mle[1,1]
    phi=mle[1,2]
    N=length(x)
    lambdaest<-lambda;phiest<-phi;
    A11=1 /(phi* (1-phi))
    A12=0
    A21=A12
    A22=(1-phi)/(lambda^2)
    FZIP =  matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIP)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIl = lambdaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIl),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of lambda" )
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zinormal")
  {
    #y<-sample.zi1(N,phi=phi,mean=mean,sigma=sigma,dist="normal")
    mle<-zih.mle(x,mean=mean,sigma=sigma,dist="normal.zihmle",type='zi',lowerbound,upperbound)
    mean<-mle[1,1]
    sigma<-mle[1,2]
    phi<-mle[1,3]
    meanest<-mean
    sigmaest<-sigma
    phiest<-phi
    N=length(x)
    A11=1 /(phi* (1-phi))
    A12=0
    A21=A12
    A13=0
    A31=A13
    A22=(1-phi)/(sigma^2)
    A23=0
    A32=0
    A33=(1-phi)*2/(sigma^2)
    FZIN = matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZIN22=matrix(c(A22,A23,A32,A33),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIN)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIm = meanest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIs = sigmaest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    
    ConfInt=matrix(c(CIphi,CIm,CIs),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of mean", "CI of sigma")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zihalfnorm")
  {
    #y<-sample.zi1(N,phi=phi,sigma=sigma,dist="halfnormal")
    mle<-zih.mle(x,sigma=sigma,dist="halfnorm.zihmle",type='zi',lowerbound,upperbound)
    sigma<-mle[1,1]
    phi=mle[1,2]
    sigmaest<-sigma
    phiest<-phi
    N=length(x)
    A11=1 /phi* (1-phi)
    A12=0
    A21=A12
    A22=(1-phi)*2/(sigma^2)
    FZIhf =  matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIhf)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIs = sigmaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIs),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of sigma" )
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="zilognorm")
  {
    #y<-sample.zi1(N,phi=phi,mean=mean,sigma=sigma,dist="lognormal")
    mle<-zih.mle(x,mean=mean,sigma=sigma,dist="lognorm.zihmle",type='zi',lowerbound,upperbound)
    mean<-mle[1,1]
    sigma<-mle[1,2]
    phi<-mle[1,3]
    meanest<-mean
    sigmaest<-sigma
    phiest<-phi
    N=length(x)
    A11=1 /phi* (1-phi)
    A12=0
    A21=A12
    A13=0
    A31=A13
    A22=(1-phi)/(sigma^2)
    A23=0
    A32=A23
    A33=(1-phi)*2/(sigma^2)
    FZIN=matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZIN22=matrix(c(A22,A23,A32,A33),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIN)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIm = meanest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIs = sigmaest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIm,CIs),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of mean", "CI of sigma")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  ######### Hurdle Type ########
  if (dist=="ph")
  {
    mle<-zih.mle(x,lambda = lambda,dist="poisson.zihmle",type='h',lowerbound,upperbound)
    lambda=mle[1,1]
    phi=mle[1,2]
    lambdaest<-lambda;phiest<-phi;
    N=length(x)
    A11 = 1 /(phi* (1-phi))
    A12 = 0
    A21 = A12
    A22 = (1-phi)/(1-exp(-lambda)) * ((1/lambda) - (exp(-lambda)/(1-exp(-lambda))))
    FZIP = matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    inv.fish<-solve(FZIP)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIl = lambdaest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIl),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of lambda" )
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="geomh")
  {
    mle<-zih.mle(x,p=p,dist="geometric.zihmle",type='h',lowerbound,upperbound)
    p=mle[1,1]
    phi=mle[1,2]
    pest<-p;phiest<-phi;
    N=length(x)
    A11 = 1 /(phi* (1-phi))
    A12 = 0
    A21 = A12
    A22 = (1-phi)/((p^2)*(1-p))
    FZIgeom = matrix(c(A11,A12,A21,A22),ncol=2,nrow=2,byrow=TRUE)
    
    if(corpcor::is.positive.definite(FZIgeom, tol=1e-8))
    {
      inv.fish<-solve(FZIgeom)
    }
    else{
      a<-QRM::eigenmeth(FZIgeom, delta = 0.001)
      inv.fish<-solve(a)
    }
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIp),ncol=2,nrow=2,byrow=TRUE)
    rownames(ConfInt) = c("CI of Phi","CI of p" )
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="nbh")
  {
    mle<-zih.mle(x,r=r,p=p,dist="nb.zihmle",type="h",lowerbound,upperbound)
    N = length(x)
    r=mle[1,1]
    p=mle[1,2]
    phi=mle[1,3]
    rest<-r;pest<-p;phiest<-phi;
    m = 1e5;
    y<-stats::rnbinom(m,r,p)
    p0 = exp(log((1-p)^r))
    A11 = 1/(phi * (1-phi))
    A12 = 0
    A21 = A12
    A13 = 0
    A31 = A13
    Cstar = p0/(1-p0)
    A22 = ((1-phi)/(1-p0)) * (trigamma(r) - mean(trigamma(y+r))  - Cstar * (log(1-p)^2))
    A23 = ((1-phi)/(1-p0)) * ((1/(1-p)) + Cstar * (r * log(1-p)/(1-p)))
    A32 = A23
    A33 = ((1-phi)/(1-p0)) * ((r/(p*(1-p)^2)) - Cstar * (r/(1-p))^2)
    FZINB =  matrix(c(A11,A12,A13,A21,A22,A23,A31,A32,A33),ncol=3,nrow=3,byrow=TRUE)
    FZINB22= matrix (c(A22,A23,A32,A33), ncol=2,nrow=2,byrow =TRUE)
    inv.fish<-solve(FZINB)
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIr = rest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIp = pest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIr,CIp),ncol=2,nrow=3,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of r", "CI of p")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="bbh")
  {
    mle1<-zih.mle(x,n=n,alpha1=alpha1,alpha2=alpha2,dist="bb.zihmle",type="h",lowerbound,upperbound)
    N = length(x)
    
    n=mle1[1,1]
    alpha=mle1[1,2]
    beta=mle1[1,3]
    phi=mle1[1,4]
    m= 1e5;
    y<-extraDistr::rbbinom(m, size=n, alpha=alpha, beta=alpha)
    p0 = beta(alpha,n+beta)/beta(alpha,beta)
    
    A11 = 1/(phi*(1-phi))
    A12 = 0
    A21 = A12
    A13 = 0
    A31 = A13
    A14 = 0
    A41=A14
    
    Bstar = -(1-phi)/(1-p0)
    Cstar = p0/(1-p0)
    
    A22 = Bstar * ((trigamma(n+1) - trigamma(n+alpha+beta) + mean(trigamma(n-y+beta)) - mean(trigamma(n-y+1))) +
                     (Cstar * ((digamma(n+beta) - digamma(n+alpha+beta))^2)))
    A23 = Bstar * (-trigamma(n+alpha+beta) +
                     (Cstar * (digamma(n+beta) - digamma(n+alpha+beta)) * (digamma(alpha+beta) - digamma(n+alpha+beta))))
    A32 = A23
    A24 = Bstar * ((mean(trigamma(n-y+beta)) - trigamma(n+alpha+beta)) +
                     (Cstar * (digamma(n+beta) - digamma(n+alpha+beta)) * (digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(beta))))
    A42 = A24
    A33 = Bstar * ((trigamma(alpha+beta) - trigamma(n+alpha+beta) - trigamma(alpha) + mean(trigamma(y+alpha))) +
                     (Cstar * ((digamma(alpha+beta) - digamma(n+alpha+beta))^2)))
    A34 = Bstar * ((trigamma(alpha+beta) - trigamma(n+alpha+beta)) +
                     (Cstar * ((digamma(alpha+beta) - digamma(n+alpha+beta))*
                                 (digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(beta)))))
    A43 = A34
    A44 = Bstar * ((trigamma(alpha+beta) - trigamma(beta) - trigamma(n+alpha+beta) + mean(trigamma(n-y+beta))) +
                     (Cstar * ((digamma(n+beta) + digamma(alpha+beta) - digamma(n+alpha+beta) - digamma(beta))^2)))
    
    FZIBB = matrix(c(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44),ncol=4,nrow=4,byrow=TRUE)
    FZIBB33 = matrix(c(A22,A23,A24,A32,A33,A34,A42,A43,A44), ncol=3,nrow=3,byrow =TRUE)
    
    if(corpcor::is.positive.definite(FZIBB, tol=1e-8))
    {
      inv.fish<-solve(FZIBB)
    }
    else{
      a<-QRM::eigenmeth(FZIBB, delta = 0.001)
      inv.fish<-solve(a)
    }
    
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phi+ c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIn = n + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIa = alpha + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    CIb = beta + c(-1, 1) * crit * sqrt(inv.fish[4,4])/sqrt(N)
    ConfInt = matrix(c(CIphi,CIn,CIa,CIb),ncol=2,nrow=4,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of n", "CI of alpha" , "CI of beta")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
  if (dist=="bnbh")
  {
    mle1<-zih.mle(x,r=r,alpha1=alpha1,alpha2=alpha2,dist="bnb.zihmle",type="h",lowerbound,upperbound)
    N = length(x)
    r=mle1[1,1]
    alpha=mle1[1,2]
    beta=mle1[1,3]
    phi=mle1[1,4]
    m = 1e5;
    y<-extraDistr::rbbinom(m, size=r, alpha=alpha, beta=alpha)
    rest<-r;alphaest<-alpha;betaest<-beta;phiest<-phi;
    p0=beta(r+alpha,beta)/beta(alpha,beta)
    A11 = 1/((1-phi) * phi)
    A12 = 0
    A21 = A12
    A13 = 0
    A31 = A13
    A14 = 0
    A41=A14
    Cstar = p0/(1-p0)
    A22 = -((1-phi)/(1-p0)) * (trigamma(r+alpha) - trigamma(r) - mean(trigamma(r+y+alpha+beta)) + mean(trigamma(r+y)) +
                                 Cstar * (digamma(r+alpha) - digamma(r+alpha+beta))^2)
    A23 = -((1-phi)/(1-p0)) * (trigamma(r+alpha) - mean(trigamma(r+y+alpha+beta)) +
                                 Cstar * (digamma(r+alpha) - digamma(r+alpha+beta)) * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha)))
    A32 = A23
    A24 = -((1-phi)/(1-p0)) * (-mean(trigamma(r+y+alpha+beta)) +
                                 Cstar * (digamma(r+alpha) - digamma(r+alpha+beta)) * (digamma(r+alpha) - digamma(r+alpha+beta)))
    A42 = A24
    A33 = -((1-phi)/(1-p0)) * (trigamma(alpha+r) - trigamma(alpha) + trigamma(alpha+beta) - mean(trigamma(r+y+alpha+beta)) +
                                 Cstar * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha))^2)
    A34 = -((1-phi)/(1-p0)) * (-mean(trigamma(r+y+alpha+beta)) + trigamma(alpha+beta) +
                                 Cstar * (digamma(r+alpha) + digamma(alpha+beta) - digamma(r+alpha+beta) - digamma(alpha))*
                                 (digamma(r+beta) - digamma(r+alpha+beta)))
    A43 = A34
    A44 = -((1-phi)/(1-p0)) * (trigamma(alpha+beta) - trigamma(beta) - mean(trigamma(r+y+alpha+beta)) + mean(trigamma(y+beta)) +
                                 Cstar * (digamma(r+alpha) - digamma(r+alpha+beta))^2)
    
    FZIBNB = matrix(c(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,
                      A34,A41,A42,A43,A44),ncol=4,nrow=4,byrow=TRUE)
    FZIBNB33= matrix (c(A22,A23,A24,A32,A33,A34,A42,A43,A44), ncol=3,nrow=3,byrow =TRUE)
    
    if(corpcor::is.positive.definite(FZIBNB, tol=1e-8))
    {
      inv.fish<-solve(FZIBNB)
    }
    else{
      a<-QRM::eigenmeth(FZIBNB, delta = 0.001)
      inv.fish<-solve(a)
    }
    
    crit <- stats::qnorm((1 + 0.95)/2)
    CIphi = phiest + c(-1, 1) * crit * sqrt(inv.fish[1,1])/sqrt(N)
    CIr = rest + c(-1, 1) * crit * sqrt(inv.fish[2,2])/sqrt(N)
    CIa = alphaest + c(-1, 1) * crit * sqrt(inv.fish[3,3])/sqrt(N)
    CIb = betaest + c(-1, 1) * crit * sqrt(inv.fish[4,4])/sqrt(N)
    ConfInt=matrix(c(CIphi,CIr,CIa,CIb),ncol=2,nrow=4,byrow=TRUE)
    rownames(ConfInt) = c("CI of phi","CI of r", "CI of alpha" , "CI of beta")
    return(list(inversefisher=inv.fish,ConfidenceIntervals=ConfInt))
  }
}


#' Number of physician office visits.
#'
#'A data set containing ofp (number of physician office visit) of 4406 individuals.
#'
#' @format
#' The original data set is based on the work of Deb and Trivedi (1997) analyze data on 4406 individuals, 
#' aged 66 and over, who are covered by Medicare, a public insurance program. Originally obtained from the US National Medical
#' Expenditure Survey (NMES) for 1987/88, the data are available from the data archive of
#' the Journal of Applied Econometrics at http://www.econ.queensu.ca/jae/1997-v12.3/deb-trivedi/.
#' In AZIAD package we work with the number of physicians office visits for the patients.Based on
#' the analysis of kstest.A and kstest.B and lrt.A the data belongs to zero-inflated beta negative binomial
#' or beta negative binomial hurdle model.
#'
#' @source \itemize{\item http://www.jstatsoft.org/v27/i08/paper}
#'
#' @references  \itemize{\item Zeileis, A. and Kleiber, C. and Jackma, S. (2008). "Regression Models for Count Data in R". JSS 27, 8, 1–25.}
#' @export
#' @examples
#' ofp
#' set.seed(1008)
#' \donttest{d1=kstest.A(ofp,nsim=200,bootstrap=TRUE,dist="geometric")}
#' \donttest{d2=kstest.A(ofp,nsim=200,bootstrap=TRUE,dist="zibnb")}
#' \donttest{lrt.A(d1,d2)}       #0
"ofp"

