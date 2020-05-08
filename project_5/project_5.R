P <- t(matrix(c(0,1,0,0,.25,0,.25,.5,0,.5,0,.5,0,.5,.5,0), nrow = 4))

powers <- replicate(8, diag(4), simplify = F)
for (k in 2:(8+1)) {
  powers[[k]] <- powers[[k-1]]%*%P
}

pstat <- solve(rbind(t(diag(4)-P)[-4,],rep(1,4)),c(0,0,0,1))
round(pstat,4)
DB <- diag(4)-diag(4)
n <- length(pstat)
for (i in 1:n){
  for (j in 1:n){
      DB[i,j] <- (P[i,j] == pstat[j]/pstat[i]*P[j,i])
  }
}

#-------------Q2-------------------#
set.seed(0)
M.H_con <- function(f,q,rq,init,t){
  sample <- as.vector(init)
  for (i in 2:(t+1)){
    Yt <- rq(sample[i-1])
    u <- runif(1)
    if (u<= min(c(1,(f(Yt)*q(Yt,sample(i-1)))
                  /(f(sample[i-1])*q(sample[i-1],Yt))))) {
      sample[i] <- Yt
    }else{
      sample[i] <- sample[i-1]
    }
  }
  sample
}

f <- function(x){exp(-2*abs(x)^5)}
k <- 1/integrate(f,-Inf,Inf)$value
sample <- M.H_con(f, function(x,y) 1, 
                  function(x) rnorm(1,mean = x, sd = .5),0,10^4)
estimate <- mean(sample[-(1:100)]^2)

f2 <- function(x) x^2*k*f(x)
numestimate <- integrate(f2,-Inf,Inf)
estimate
numestimate


sample <- lapply(c(-2,-1,1,2), 
                 function(s) M.H_con(f, function(x,y) 1, 
                                     function(x) rnorm(1,mean = x, sd = .01),
                                     s, 10^4))

plot(0:10000, sample[[1]], type = "l", ylim = c(-2,2),xlab = "t",ylab = "Xt")
lines(0:10000, sample[[2]],col = 2)
lines(0:10000, sample[[3]],col = 3)
lines(0:10000, sample[[4]],col = 4)
legend("bottomright",legend = c("X0 = -2", "X0 = -1","X0 = 1","X0 = 2"),
       col = c(1,2,3,4), lty = c(1,1,1,1))




sample2 <- lapply(c(.01,.5,1,5), 
                  function(s) M.H_con(f, 
                                      function(x,y) 1, 
                                      function(x) rnorm(1,mean = x, sd = s),
                                      0, 10^4)[-(1:250)])
library(coda)
ESS <- 1:4
for (i in 1:4){
  ESS[i] <- effectiveSize(mcmc(sample2[[i]]))
}
ESS

#-------------Q3-------------------#
set.seed(0)
gibbs_sampler <- function(init,t,functions){
  x <- init
  n <- length(x)
  Xt <- matrix(rep(0,n*(t+1)),ncol = n)
  for (i in 0:t+1){
    Xt[i,] <- x
    for (j in 1:n){
      marginal <- functions[[j]]
      x[j] <- marginal(x[-j])
    }
  }
  Xt
}

f1 <- function(x) rnorm(1,mean = 1/sqrt(2)*.8*x,sd = sqrt((1-.8^2)))
f2 <- function(x) rnorm(1,mean = sqrt(2)*.8*x,sd = sqrt((1-.8^2)*2))
sample <- gibbs_sampler(c(0,0),10^4,c(f1,f2))[-(1:50),]

dim(sample[(0<=sample[,1] & sample[,1]<=1 
            & 0<=sample[,2] & sample[,2]<=2),])[1]/dim(sample)[1]


#-------------Q4-------------------#
data(iris)
summary(iris)
X<- iris$Petal.Width
Y <- iris$Sepal.Width

bootstrap <- function(x,n, fun, alpha = .05, 
                      two.tailed = T, studentized = F, Efron = F){
  
  bs <- replicate(n,x[sample(1:length(x),length(x),replace = T)])
  stat <- apply(bs,2,fun)
  
  if (studentized){
    sigma = apply(bs,2,sd)
  }else{
    sigma = 1
  }
  if (Efron){
    tx <- 0
  }else{
    tx <- fun(x)
  }
  stat <- (stat - tx)/sigma
  if (two.tailed){
    CI <- tx - sd(x)*as.vector(quantile(stat,c(1-alpha/2,alpha/2)))
  }else{
    CI <- tx - sd(x)*as.numeric(quantile(stat,alpha))
  }
  return(list("Sample" = bs, "Statistic" = stat, "Confidence_Interval" = CI))
}


set.seed(0)
bootstraps <- bootstrap(X,10^4,function(x)mean(x),studentized = T)
bootstraps$Confidence_Interval

set.seed(0)
bootstraps <- bootstrap(X,10^4,function(x)mean(x^2),studentized = T)
bootstraps$Confidence_Interval




R<- cor(X,Y, method = "spearman")
bY <- replicate(1e4,cor(X,sample(Y,length(Y), replace = T),
                        method = "spearman"))
hist(bY)
pv <- length(bY[abs(bY)>=abs(R)])/1e4


