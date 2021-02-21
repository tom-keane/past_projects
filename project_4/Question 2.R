#---------------Q2-----------------
library(randomForest)
library(caret)
library(doParallel)
library(keras)
library(ggplot2)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)


data_q2 <- read.csv("C:\\Users\\Thomas\\Dropbox\\University\\Imperial\\
                    Semester 2\\Machine Learning\\Coursework\\
                    Coursework 2\\dataQ2.csv")

ggplot() + geom_histogram(aes(data_q2[,1]), binwidth = 1, 
                          colour="black", fill="white") + 
  labs(x="Year") + theme_light()

N <- nrow(data_q2)
trainsize <- round(N*.6)
valsize <- round(N*.2)
testsize <- N - valsize - trainsize

set.seed(0)

indices <- sample(1:N, size=N, replace=FALSE)
indtrain <- indices[1:trainsize]
indvalid <- indices[(1+trainsize):(valsize+trainsize)]
indtest <- indices[(N-testsize+1):N]

trainset <- data_q2[indtrain, ]
validset <- data_q2[indvalid, ]
testset <- data_q2[indtest, ]

x_train <- data_q2[-indtest , -1]
y_train <- data_q2[-indtest , 1]

p <- ncol(data_q2)-1


#----------------------Random Forest-----------------------------------------
Grid <- expand.grid(.mtry = c(p/4, p/3, p/2))
tr <- trainControl(method = 'cv', number = 1, index = list(Fold1 = (indtrain) ),
                   indexOut = list(Fold1 = (indvalid)))
tuning.rf <- caret::train(data_q2[ , -1],data_q2[ , 1], method="rf", 
                          trControl = tr, tuneGrid = Grid)
tuning.rf$results['RMSE']

tree_performance <- matrix(rep(0,9), nrow = 3)
best_mtry <- p/3


i <- 0
ntrees <- c(250, 500, 1000)
for (n in ntrees){
  i <- i + 1
  j <- 0
  for (size in c(25, 50, 100)){
    j <- j + 1
    Grid <- expand.grid(.mtry = best_mtry)
    tuning.rf <- train(data_q2[ , -1],data_q2[ , 1], method="rf", trControl = tr, 
                       tuneGrid = Grid, nodesize = size, ntree = n)
    tree_performance[i,j] <- as.numeric(tuning.rf$results['RMSE'])
  }
}


best_ntrees <- ntrees[which.min(apply(tree_performance, 2, min))]

size_performance <- matrix(rep(0,5), nrow = 1)
sizes <- c(5, 10, 15, 20, 25)
j <- 0
for (size in sizes){
  j <- j + 1
  tuning.rf <- train(data_q2[ , -1],data_q2[ , 1], method="rf", trControl = tr, 
                     tuneGrid = Grid, nodesize = size, ntree = best_ntrees)
  size_performance[j] <- as.numeric(tuning.rf$results['RMSE'])
}

best_size <- sizes[which.min(size_performance)]

final_rf <- randomForest(x_train, y_train, ntree = best_ntrees, 
                         nodesize = best_size,
                         mtry = best_mtry, importance = T)

#----------------------Neural Network---------------------------


x_norm <- scale(x_train)
y_minmax <- (y_train - min(y_train))/(max(y_train)-min(y_train))


build_model <- function(act_funct, learnrate, hidden, 
                        epochs = 150, validation = .25) {
  if (length(hidden) == 1){
    model <- keras_model_sequential() %>%
      layer_dense(units = hidden[1], activation = act_funct, 
                  input_shape = 90) %>%
      layer_dense(units = 1)
  } else {
    model <- keras_model_sequential() %>%
      layer_dense(units = hidden[1], activation = act_funct, input_shape = 90) %>%
      layer_dense(units = hidden[2], activation = act_funct) %>%
      layer_dense(units = 1)
  }
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_sgd(lr = learnrate),
    metrics = list("mean_absolute_error")
  )
  
  history <- model %>% fit(
    x_norm, y_minmax,
    epochs = epochs,
    validation_split = validation
  )
  list(model,history)
}

model_relu <- build_model("relu", .001, c(100))
model_sigmoid <- build_model("sigmoid", .0005, c(100))
model_tanh <- build_model("tanh", .003, c(100))

model_relu[[2]]$metrics$val_loss[150]
model_sigmoid[[2]]$metrics$val_loss[150]
model_tanh$metrics[[2]]$val_loss[150]


model_relu_1layer50 <- build_model("relu", .001, c(50), 250)
model_relu_1layer100 <- build_model("relu", .001, c(100), 250)
model_relu_2layer5050 <- build_model("relu", .001, c(50, 50), 250)
model_relu_2layer10050 <- build_model("relu", .001, c(100, 50), 250)
model_relu_2layer100100 <- build_model("relu", .001, c(100, 100), 250)

model_relu_1layer50[[2]]$metrics$val_loss[250]
model_relu_1layer100[[2]]$metrics$val_loss[250]
model_relu_2layer5050[[2]]$metrics$val_loss[250]
model_relu_2layer10050[[2]]$metrics$val_loss[250]
model_relu_2layer100100[[2]]$metrics$val_loss[250]


finalNN <- build_model("relu", .001, c(100), 250, 0)
predictions <- as.numeric(predict_on_batch(finalNN[[1]], 
                                           scale(data_q2[indtest, -1])))

predictions_unscaled <- predictions * (max(y_train)-min(y_train)) + 
  min(y_train)

NN_mse <- sum((predictions_unscaled - data_q2[indtest, 1])^2)/2504

rf_predictions <- predict(final_rf, data_q2[indtest, -1])
rf_mse <- sum((rf_predictions - data_q2[indtest, 1])^2)/2504

NN_mse
rf_mse

variableimportance <- importance(final_rf)
variableimportance <- data.frame(variable = row.names(variableimportance), 
                                 Increase = variableimportance[,1])

ggplot(variableimportance[1:20,], aes(x = reorder(variable, -Increase), 
                                      y = Increase)) + 
  geom_bar(stat = "identity",  fill = "lightgrey", colour = "black") +
  labs(x = "Variable", y = "Avg. Increase in OOB MSE") + theme_light()


