#--------------- Part 1 Q1-----------------
library(ggplot2)
data_q1 <- read.csv("C:\\Users\\Thomas\\Dropbox\\University\\Imperial\\
                    Semester 2\\Machine Learning\\Coursework\\Coursework 2\\
                    dataQ1.csv")

set.seed(0)

x <- data_q1[,-1]
y <- data_q1[,1]
x_norm <- scale(x)
summary(y)

covariancematrix <- cov(x_norm)
eigens <- eigen(covariancematrix)
values <- eigens$values
vectors <- eigens$vectors

variance_exp <- values / sum(values)
cumsum(variance_exp)

no_dim_90 <- which.max(cumsum(variance_exp)>= .9)
no_dim_95 <- which.max(cumsum(variance_exp)>= .95)
no_dim_99 <- which.max(cumsum(variance_exp)>= .99)

trans_x_2d <- x_norm%*%vectors[,1:2]

ggplot() + geom_point(aes(trans_x_2d[,1], trans_x_2d[,2], col = y)) + 
  labs(col='Diagnosis', x = "First PC", y = "Second PC") +
  scale_color_hue(labels = c("Benign", "Malignant"))

#--------------- Part 2 Q1-----------------
library(e1071)
library(caret)
library(cluster)

manhattandissim <- dist(x_norm, method = "manhattan")
eucliddissim <- dist(x_norm, method = "euclidean")

man.average <- hclust(manhattandissim, method="average")
man.complete <- hclust(manhattandissim, method = "complete")

euclid.average <- hclust(eucliddissim, method="average")
euclid.complete <- hclust(eucliddissim, method = "complete")


plot(man.average, xlab = "")
plot(man.complete, xlab = "")

plot(euclid.average, xlab = "")
plot(euclid.complete, xlab = "")



h_clusters <- cutree(man.complete, 2)
h_clusters <- as.factor(h_clusters)
levels(h_clusters) <- c("M", "B")
h_clusters <- relevel(h_clusters, "B")

confusionMatrix(h_clusters, y)


sumofsquares <- matrix(rep(0,10),ncol = 1)

for (k in 1:10){
  kmeansobject <- kmeans(x_norm, k, nstart = 50)
  sumofsquares[k] <- kmeansobject$tot.withinss
}

plot(1:10,sumofsquares,"b", main = "Number of clusters vs Sum of Squares", 
     ylab= 'Total within Sum of Squares',xlab = "Number of Clusters")


d <- dist(x_norm, method="euclidean")^2
avg_silhouette <- rep(0,4)

for (k in 2:5){
  kmeansobject <- kmeans(x_norm, k, nstart = 50)
  sil <- silhouette(kmeansobject$cluster,d)
  avg_silhouette[k-1] <- mean(sil[,3])
}
avg_silhouette

kmeansobject <- kmeans(x_norm, 2, nstart = 50)
kmeans_cluster <- as.factor(kmeansobject$cluster)
levels(kmeans_cluster) <- c("M", "B")
kmeans_cluster <- relevel(kmeans_cluster, "B")
confusionMatrix(kmeans_cluster, y)

ggplot() + geom_point(aes(trans_x_2d[,1], trans_x_2d[,2], col = kmeans_cluster)) + 
  labs(col='', x = "First PC", y = "Second PC") +
  theme(legend.position = "none")

ggplot() + geom_point(aes(trans_x_2d[,1], trans_x_2d[,2], col = h_clusters)) + 
  labs(col='Diagnosis', x = "First PC", y = "Second PC") +
  scale_color_hue(labels = c("Benign", "Malignant"))

