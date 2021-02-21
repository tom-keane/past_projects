library(kernlab)
library(ggplot2)

set.seed(0)
data_q3 <- read.csv("C:\\Users\\Thomas\\Dropbox\\University\\Imperial\\Semester 2\\
                    Machine Learning\\Coursework\\Coursework 2\\dataQ3.csv")

plot(data_q3, type = "l")
hist(data_q3[,2], breaks = 50)
summary(data_q3[,2])

x <- as.matrix(data_q3$Months.since.Jan.1960)
y <- as.matrix(data_q3$temp)

x_train <- as.matrix(data_q3$Months.since.Jan.1960)[-(502:550)]
y_train <- as.matrix(data_q3$temp)[-(502:550)]

x_valid <- as.matrix(data_q3$Months.since.Jan.1960)[502:526]
y_valid <- as.matrix(data_q3$temp)[502:526]

x_test <- as.matrix(data_q3$Months.since.Jan.1960)[526:550]
y_test <- as.matrix(data_q3$temp)[526:550]

gaussianprocess <- function(x, y, xnew, kern_fun, meanx, sigma_n, sigma_f, l , ...){
  n <- length(y)
  f <- function(x,y) kern_fun(x, y, sigma_f, l, ...)
  
  kern1 <- kernelMatrix(kernel = f, x = x)
  kern2 <- kernelMatrix(kernel = f, x = xnew)
  kern3 <- kernelMatrix(kernel = f, x = xnew, y = x)
  temp <- solve(kern1 + sigma_n * diag(n))
  
  posteriorexp <- kern3 %*% temp %*%  (y - meanx)
  posteriorvar <- kern2 - kern3 %*% temp %*% t(kern3)
  list("Expectation" = posteriorexp, "Covariance" = posteriorvar)
}


logmarginal <-  function(param, x, y, kern_fun){
  n <- length(y)
  f <- function(x, y) kern_fun(x, y, param[2], param[3])
  sigma_n <- param[1]
  kern <- kernelMatrix(kernel = f, x = x)
  1/2 * t(y) %*% solve(kern + sigma_n * diag(n)) %*% y + 1/2 * 
    log(det(kern + sigma_n * diag(n)))
}



gausskernel <- function(x, y, sig, l){
  sig * exp((- t((x - y)) %*% (x - y))/(l ^ 2))
}
class(gausskernel) <- "kernel"
gaussparameters <- optim(c(1, 1, 1), logmarginal, x = x_valid, y = y_valid, 
                         kern_fun = gausskernel)$par

gauss_train <- gaussianprocess(c(x_train,x_valid), c(y_train,y_valid), 
                               c(x_train,x_valid), gausskernel, 0, 
                               gaussparameters[1], gaussparameters[2], 
                               gaussparameters[3])

gauss_test <- gaussianprocess(c(x_train,x_valid), c(y_train,y_valid), 
                              x_test, gausskernel, 0, 
                              gaussparameters[1], gaussparameters[2], 
                              gaussparameters[3])

gauss_test_score <- logmarginal(gaussparameters, x_test, 
                                y_test, gausskernel)


maternkernel <- function(x, y, sig, l){
  sig * exp(- sqrt(3) * sqrt(sum((x - y)^2))/l) *
    (1 + sqrt(3) * sqrt(sum((x - y)^2))/l)
}
class(maternkernel) <- "kernel"

maternparameters <- optim(c(1, 1, 1), logmarginal, x = x_valid, 
                          y = y_valid, kern_fun = maternkernel)$par

matern_train <- gaussianprocess(c(x_train,x_valid), 
                                c(y_train,y_valid), c(x_train,x_valid),
                                maternkernel, 0, maternparameters[1], 
                                maternparameters[2], maternparameters[3])

matern_test <- gaussianprocess(c(x_train,x_valid), 
                               c(y_train,y_valid), x_test, maternkernel,
                               0, maternparameters[1], 
                               maternparameters[2], maternparameters[3])

matern_test_score <- logmarginal(maternparameters, 
                                 x_test, y_test, maternkernel)


periodickernel <- function(x, y, sig, l, k){
  x <- c(cos(k*x),sin(k*x))
  y <- c(cos(k*y),sin(k*y))
  sig * exp((- t((x - y)) %*% (x - y))/(l ^ 2))
}
class(periodickernel) <- "kernel"

logmarginalp <-  function(param, x, y, kern_fun){
  n <- length(y)
  f <- function(x, y) kern_fun(x, y, param[2], param[3], param[4])
  sigma_n <- param[1]
  kern <- kernelMatrix(kernel = f, x = x)
  1/2 * t(y) %*% solve(kern + sigma_n * diag(n)) %*% y + 1/2 * 
    log(det(kern + sigma_n * diag(n)))
}

periodicparameters <- optim(c(1, 1, 1, 1), logmarginalp, x = x_valid, 
                            y = y_valid, kern_fun = periodickernel)$par

periodic_train <- gaussianprocess(c(x_train,x_valid), c(y_train,y_valid), 
                                  c(x_train,x_valid), periodickernel, 0, 
                                  periodicparameters[1], periodicparameters[2],
                                  periodicparameters[3], periodicparameters[4])

periodic_test <- gaussianprocess(c(x_train,x_valid), c(y_train,y_valid), x_test, 
                                 periodickernel, 0, periodicparameters[1], 
                                 periodicparameters[2], periodicparameters[3],
                                 periodicparameters[4])

periodic_test_score <- logmarginalp(periodicparameters, x_test, y_test, 
                                    periodickernel)

round(gaussparameters, 4)
round(maternparameters, 4)
round(periodicparameters, 4)

round(gauss_test_score, 4)
round(matern_test_score, 4)
round(periodic_test_score, 4)


gauss <- gaussianprocess(x, y, x, gausskernel, 0, 
                         gaussparameters[1], gaussparameters[2], 
                         gaussparameters[3])

ggplot() + geom_point(aes(x = x, y = y)) + 
    geom_ribbon(aes(x = x ,ymin = gauss$Expectation - 
                      1.96*sqrt(diag(gauss$Covariance)),
                    ymax = gauss$Expectation + 
                      1.96*sqrt(diag(gauss$Covariance)), 
                    fill = "95% CI"), alpha = .5) +
    geom_line(aes(x = x , y = gauss$Expectation, colour = "Expectation")) +
    scale_color_manual(name = NULL,values = c('Expectation' = 'black')) +
    scale_fill_manual(name = NULL,values = c('95% CI' = 'lightblue'))+
    labs(x="Month since January 1960", y="Temperature", col = "")+
    theme_light()

    
March_prediction <- gaussianprocess(x, y, (2011-1960)*12+3, gausskernel, 0, 
                                    gaussparameters[1], gaussparameters[2], 
                                    gaussparameters[3])

round(March_prediction$Expectation,4)
round(sqrt(March_prediction$Covariance),4)
