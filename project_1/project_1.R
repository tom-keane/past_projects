library(reticulate)
library(ggplot2)
library(ggrepel)

use_python("/usr/bin/python3")
gensim <- import("gensim")

twitter <- gensim$downloader$load('glove-twitter-200')
wiki <- gensim$downloader$load('glove-wiki-gigaword-300')
google <- gensim$downloader$load('word2vec-google-news-300')


get_vector <- function(model, word){model$get_vector(word)}
vget_vector <- Vectorize(get_vector, "word")

inner_product <- function(a,b){sum(a*b) / ( sqrt(sum(a * a)) * sqrt(sum(b * b)) )}

create_subspace <- function(model, words_1, words_2){
  vectors_1 <- vget_vector(model, words_1)
  vectors_2 <- vget_vector(model, words_2)
  mean_vector_1 <- apply(vectors_1, 1, mean)
  mean_vector_2 <- apply(vectors_2, 1, mean)
  difference <- mean_vector_1 - mean_vector_2
  list("Subspace" = difference, "Vectors_1" = vectors_1, "Vectors_2" = vectors_2)
}

permutation_test <- function(N, current_bias, model, attr_vectors_1, attr_vectors_2, tar_subspace){
  results <- rep(F,N)
  l1 <- ncol(attr_vectors_1)
  l2 <- ncol(attr_vectors_2)
  attr_set <- matrix(c(attr_vectors_1, attr_vectors_2), ncol = l1+l2)
  for (i in 1:N){
    perm <- attr_set[,sample(1:(l1+l2))]
    mean_vector_1 <- apply(perm[,1:l1], 1, mean)
    mean_vector_2 <- apply(perm[,(l1+1):(l1+l2)], 1, mean)
    perm_difference <- mean_vector_1 - mean_vector_2
    result <- inner_product(perm_difference, tar_subspace)
    results[i] <- (result >= current_bias)
  }
  pval <- mean(results)
  CI <- pval + sd(results)/sqrt(length(results))*qnorm(c(0.025,0.975))
  list("P_value" = pval, "CI" = CI)
}

bias <- function(model, attr_words_1, attr_words_2, tar_words_1, tar_words_2){
  attribute_subspace <- create_subspace(model, attr_words_1, attr_words_2)
  target_subspace <- create_subspace(model, tar_words_1, tar_words_2)
  bias <- inner_product(attribute_subspace$Subspace, target_subspace$Subspace)
  p_val <- permutation_test(10000, bias, model, attribute_subspace$Vectors_1, attribute_subspace$Vectors_2, target_subspace$Subspace)
  list("Bias" = bias, 
       "P_value" = p_val$P_value,
       "95%_CI" = p_val$CI,
       "Attribute_subspace" = attribute_subspace, 
       "Target_subspace" = target_subspace)
}

projection <- function(a, b){
  a <- a/sqrt(sum(a * a))
  bproj <- sum(a*b)
  bproj
}
vprojection <- function(a,bmatrix){
  projections <- rep(0, ncol(bmatrix))
  for (i in 1:ncol(bmatrix)){
    projections[i] <- projection(a,bmatrix[ ,i])
  }
  projections
}

white_word <- c("white", "caucasian", "european", "light")
black_word <- c("black", "african", "dark", "negro")


NY_black_names <- c('Noah','Elijah','Aiden','Jeremiah','Jayden','Ethan','Josiah','Joshua',
                    'Amir','Mason','Tyler','Liam','Christian','Michael','Isaiah','Jacob',
                    'Carter','Ayden','Justin','David','Christopher','Chase','Daniel',
                    'Malachi','Logan','Madison','Ava','Aaliyah','Chloe','London','Taylor',
                    'Kayla','Olivia','Nevaeh','Serenity','Skylar','Fatoumata','Abigail',
                    'Savannah','Gabrielle','Brielle','Arianna','Nyla','Faith',
                    'Khloe','Ariel','Isabella','Mia','Mariam', 'Leah')

#Name removed due to wiki - 'Makayla'

NY_white_names <- c('David','Joseph','Moshe','Jacob','Michael','Benjamin','James','Daniel',
                    'Alexander','Jack','Samuel','John','Adam','Matthew','Henry','Chaim',
                    'Abraham','Nicholas','Ryan','William','Ethan','Liam','Noah','Charles',
                    'Thomas','Esther','Leah','Sarah','Olivia','Chaya','Rachel','Emma','Ava',
                    'Miriam','Sophia','Emily','Ella','Chana','Mia','Isabella','Charlotte',
                    'Sofia','Maya','Rivka','Sara','Alexandra','Abigail','Elizabeth','Anna',
                    'Victoria')

for (name in NY_black_names) {if (name %in% NY_white_names) {print(name)}}

b_idx <- rep(T,50)
w_idx <- rep(T,50)

for (i in 1:50) {
  if (NY_black_names[i] %in% NY_white_names) {
    w_idx[which(NY_white_names == NY_black_names[i])] <- F
    b_idx[i] <- F
  }
}
NY_black_names <- NY_black_names[b_idx]
NY_white_names <- NY_white_names[w_idx]


white_names <- c('Adam','Chip','Harry','Josh','Roger','Alan','Frank','Ian','Justin','Ryan',
                 'Andrew','Fred','Jack','Matthew', 'Stephen','Brad','Greg','Jed','Paul',
                 'Todd','Brandon','Hank','Jonathan','Peter','Wilbur','Amanda','Courtney',
                 'Heather','Melanie','Sara','Amber','Crystal','Katie','Meredith','Shannon',
                 'Betsy','Donna','Kristin','Nancy','Stephanie','Ellen','Lauren','Peggy',
                 'Colleen','Emily','Megan','Rachel','Wendy')

black_names <- c('Alonzo','Jamel','Lerone','Theo','Alphonse','Jerome','Leroy','Torrance','Darnell',
                 'Lamar','Lionel','Rashaun','Tyree','Deion','Lamont','Malik','Terrence','Tyrone',
                 'Lavon','Marcellus','Nichelle','Shereen','Ebony','Latisha','Shaniqua', 'Jasmine',
                 'Latonya','Tanisha','Tia','Lakisha','Latoya','Yolanda', 'Malika','Tawanda','Yvette')


#Intelligence, Education, Employment
positive_word_list <- c("intelligent", "bright", "smart", "clever", 
                        "educated", "knowledgeable", "literate", "trained",
                        "skilled", "creative", "rich", "diligent")

negative_word_list <- c("unintelligent", "dim", "stupid","dense",
                        "uneducated", "ignorant", "illiterate", "untrained",
                        "unskilled", "uncreative", "poor", "lazy")

set.seed(0)

wiki_results_old <- bias(wiki, tolower(white_names), tolower(black_names), positive_word_list, negative_word_list)
twitter_results_old <- bias(twitter, tolower(white_names), tolower(black_names), positive_word_list, negative_word_list)
google_results_old <- bias(google, white_names, black_names, positive_word_list, negative_word_list)

wiki_results_explicit <- bias(wiki, white_word, black_word, positive_word_list, negative_word_list)
twitter_results_explicit <- bias(twitter, white_word, black_word, positive_word_list, negative_word_list)
google_results_explicit <- bias(google, white_word, black_word, positive_word_list, negative_word_list)

wiki_results_implicit <- bias(wiki, tolower(NY_white_names), tolower(NY_black_names), positive_word_list, negative_word_list)
twitter_results_implicit <- bias(twitter, tolower(NY_white_names), tolower(NY_black_names), positive_word_list, negative_word_list)
google_results_implicit <- bias(google, NY_white_names, NY_black_names, positive_word_list, negative_word_list)



#-----------------------Plotting----------------------------
#-----------------------Plot 1------------------------------
wiki1 <- c(vprojection(wiki_results_implicit$Attribute_subspace$Subspace, vget_vector(wiki, positive_word_list)),
           vprojection(wiki_results_implicit$Attribute_subspace$Subspace, vget_vector(wiki, negative_word_list)))


wiki2 <- c(vprojection(wiki_results_explicit$Attribute_subspace$Subspace, vget_vector(wiki, positive_word_list)),
           vprojection(wiki_results_explicit$Attribute_subspace$Subspace, vget_vector(wiki, negative_word_list)))

wiki3 <- c(vprojection(wiki_results_old$Attribute_subspace$Subspace, vget_vector(wiki, positive_word_list)),
           vprojection(wiki_results_old$Attribute_subspace$Subspace, vget_vector(wiki, negative_word_list)))

colours <- c(rep("pos",12),rep("neg",12))
labels <- c(positive_word_list, negative_word_list)



df <- data.frame()


p <- ggplot(df) + xlim(-2.1, 2.1) + ylim(-1.5, 1.5)
p <- p + geom_segment(aes(x = rep(-2.1,3),xend = rep(2.1,3),y = c(-1,0,1), yend = c(-1,0,1)), color='grey')
p <- p + geom_segment(aes(x = 0, xend = 0, y = -1.5, yend = 1.5), color = "grey", linetype=2)
p <- p + geom_point(aes(x = wiki1, y=rep(0,24), col = colours))
p <- p + geom_text_repel(aes(x = wiki1, y=rep(0,24), label = labels, 
                            col = colours), direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(-.45,.45), seed = 5)

p <- p + geom_point(aes(x = wiki2, y=rep(1,24), col = colours))
p <- p + geom_text_repel(aes(x = wiki2, y=rep(1,24), label = labels, 
                             col = colours), direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(.55,1.55))

p <- p + geom_point(aes(x = wiki3, y=rep(-1,24), col = colours))
p <- p + geom_text_repel(aes(x = wiki3, y=rep(-1,24), label = labels, 
                             col = colours),direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(-.55,-1.55))

p <- p + annotate("text", x = c(-2.1,2.1), y = c(-1.5,-1.5), label = c("Black","White"), size = 6)
p <- p + annotate("text", x = rep(-2.1,3), y = c(-.95,.05,1.05), 
                  label = c("Stereotypical","Implicit","Explicit"),
                  color = "grey")

p <- p  + theme_void() + theme(legend.position = "none")
p <- p + scale_color_manual(values = c("pos" = "#00bfc4", "neg" = "#f8766d"))
p

#-----------------------Plot 2------------------------------
wiki1 <- c(vprojection(wiki_results_implicit$Attribute_subspace$Subspace, vget_vector(wiki, positive_word_list)),
           vprojection(wiki_results_implicit$Attribute_subspace$Subspace, vget_vector(wiki, negative_word_list)))

google1 <- c(vprojection(google_results_implicit$Attribute_subspace$Subspace, vget_vector(google, positive_word_list)),
           vprojection(google_results_implicit$Attribute_subspace$Subspace, vget_vector(google, negative_word_list)))

twitter1 <- c(vprojection(twitter_results_implicit$Attribute_subspace$Subspace, vget_vector(twitter, positive_word_list)),
           vprojection(twitter_results_implicit$Attribute_subspace$Subspace, vget_vector(twitter, negative_word_list)))

colours <- c(rep("pos",12),rep("neg",12))
labels <- c(positive_word_list, negative_word_list)


df <- data.frame()

for (seed in 40:50){
p <- ggplot(df) + xlim(-2.1, 2.1) + ylim(-1.5, 1.7)
p <- p + geom_segment(aes(x = rep(-2.1,3),xend = rep(2.1,3),y = c(-1,0,1), yend = c(-1,0,1)), color='grey')
p <- p + geom_segment(aes(x = 0, xend = 0, y = -1.5, yend = 1.5), color = "grey", linetype=2)
p <- p + geom_point(aes(x = wiki1, y=rep(0,24), col = colours))
p <- p + geom_text_repel(aes(x = wiki1, y=rep(0,24), label = labels, 
                             col = colours), direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(-.45,.45))

p <- p + geom_point(aes(x = google1, y=rep(1,24), col = colours))
p <- p + geom_text_repel(aes(x = google1, y=rep(1,24), label = labels, 
                             col = colours), direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(.55,1.7), seed = seed)

p <- p + geom_point(aes(x = twitter1, y=rep(-1,24), col = colours))
p <- p + geom_text_repel(aes(x = twitter1, y=rep(-1,24), label = labels, 
                             col = colours),direction = "y", size = 3.5, segment.alpha = 0,
                         ylim  = c(-.55,-1.55))

p <- p + annotate("text", x = c(-2.1,2.1), y = c(-1.5,-1.5), label = c("Black","White"), size = 6)
p <- p + annotate("text", x = rep(-2.1,3), y = c(-.95,.05,1.05), 
                  label = c("Twitter","Wiki","Google"),
                  color = "grey")

p <- p  + theme_void() + theme(legend.position = "none")
p <- p + scale_color_manual(values = c("pos" = "#00bfc4", "neg" = "#f8766d"))
print(p)}

