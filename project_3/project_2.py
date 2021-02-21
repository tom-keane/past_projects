import pandas as pd
import reg
import numpy as np
from scipy.stats import spearmanr
from scipy.io import mmread
from scipy.sparse import hstack
import NaiveBayes as nb

# ----------------------------Question 1-----------------------------
my_data = pd.read_csv("01788365.csv")

my_data.describe().round(4)
print("Pearson Correlation Coefficients with y:")
for i in range(5):
    print(my_data.columns[i], round(np.corrcoef(my_data.iloc[:, i], my_data["y"])[0, 1], 4))

print("Spearman Correlation Coefficients with y:")
for i in range(5):
    print(my_data.columns[i], round(spearmanr(my_data.iloc[:, i], my_data["y"])[0], 4))

np.random.seed(0)
train, test = reg.train_test_split(my_data, .8)

linear_model = reg.Regression(train)
linear_model.fit("y ~ .")
print(linear_model.beta_hat)
print("MSE:", reg.mean_square_error(linear_model.x, linear_model.y, linear_model.coefs))
test_x, test_y, _, _,  = reg.Regression.design_matrix(test, "y ~ .", intercept=True)
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, linear_model.coefs))

linear_ridge = reg.Regression(train)
linear_ridge.ridge(10, "y~.", k=5, tuner="Optimize")
print(linear_ridge.beta_hat)
print("Ridge Parameter:", linear_ridge.ridge_param)
print("MSE:", reg.mean_square_error(linear_ridge.x, linear_ridge.y, linear_ridge.coefs))
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, linear_ridge.coefs))


linear_ridge = reg.Regression(train)
linear_ridge.ridge([i for i in np.arange(0, 10000, 50)], "y~.", k=5, tuner="Grid Search")
print("\n",linear_ridge.beta_hat)
print("Ridge Parameter:", linear_ridge.ridge_param)
print("MSE:", reg.mean_square_error(linear_ridge.x, linear_ridge.y, linear_ridge.coefs))
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, linear_ridge.coefs))


quad_model = reg.Regression(train)
quad_model.fit("y ~ (.)**2")
print(quad_model.beta_hat)
print("MSE:", reg.mean_square_error(quad_model.x, quad_model.y, quad_model.coefs))
test_x, test_y, _, _,  = reg.Regression.design_matrix(test, "y ~ (.)**2", intercept=True)
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, quad_model.coefs))

quad_ridge = reg.Regression(train)
quad_ridge.ridge(10, "y~(.)**2", k=5, tuner="Optimize")
print(quad_ridge.beta_hat)
print("Ridge Parameter:", quad_ridge.ridge_param)
print("MSE:", reg.mean_square_error(quad_ridge.x, quad_ridge.y, quad_ridge.coefs))
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, quad_ridge.coefs))


quad_ridge = reg.Regression(train)
quad_ridge.ridge([i for i in np.arange(0, 1000, 5)], "y~(.)**2", k=5, tuner="Grid Search")
print(quad_ridge.beta_hat)
print("Ridge Parameter:", quad_ridge.ridge_param)
print("MSE:", reg.mean_square_error(quad_ridge.x, quad_ridge.y, quad_ridge.coefs))
print("MSE on unseen:", reg.mean_square_error(test_x, test_y, quad_ridge.coefs))


final_model = reg.Regression(my_data)
final_model.ridge(10, "y~(.)**2", k=5, tuner="Optimize")
print(final_model.beta_hat)
print("Ridge Parameter:", final_model.ridge_param)
print("MSE:", reg.mean_square_error(final_model.x, final_model.y, final_model.coefs))

points_to_be_predicted = pd.DataFrame([[-0.95, 1.76, -2.04, 0.82, -0.85, 0], [-1.27, 2.58, -1.08, 2.96, -1.83, 0],
                                       [-0.17, 1.30, 0.78, -2.79, -0.28, 0], [-3.48, 0.99, 0.78, 5.18, 1.45, 0]],
                                      columns=["x1", "x2", "x3", "x4", "x5", "y"])

x_predict, _, _, _ = reg.Regression.design_matrix(points_to_be_predicted, "y~(.)**2", intercept=True)
print("Predictions", final_model.predict(x_predict))

# ----------------------------Question 2-----------------------------

x = mmread("features.mtx").tocsr()
x_bern = x[:, :]
x_bern[x > 1] = 1
n, m = x_bern.shape
y = pd.read_csv("messages.csv")["category"]
y = np.array(y.astype('category').cat.codes).reshape(n, 1)

total_words = np.sum(x)
freq_of_word = np.sum(x, axis=0)/total_words
prop_of_messages = np.sum(x_bern, axis=0)/np.sum(x_bern)

most_freq_word = np.max(np.sum(x, axis=0))
print("Most Frequent word occurred",most_freq_word,"times across all messages.")
vocab_with_one_occurrence = len(np.where(np.sum(x, axis=0) <= 1)[0])
print("Number of words in vocab that occur once or less in the data set:", vocab_with_one_occurrence)
vocab_in_20_messages = len(np.where(np.sum(x_bern, axis=0) >= 20)[0])
print("Number of words in vocab that appear in 20 or more messages:", vocab_in_20_messages)


total_spam = np.sum(y)
total_ham = n - total_spam

data_bern = hstack((x_bern, y)).toarray()
data_mult = hstack([x, y]).toarray()

np.random.seed(0)
train_bern, test_bern = reg.train_test_split(data_bern, .8)
np.random.seed(0)
train_mult, test_mult = reg.train_test_split(data_mult, .8)

NB_bern = nb.NaiveBayes(train_bern[:, 0:-1], train_bern[:, -1], "Bernoulli")
NB_bern.train()
NB_bern.score(test_bern[:, 0:-1], test_bern[:, -1])
print(NB_bern.confusion)
print(NB_bern.performance)

NB_mult = nb.NaiveBayes(train_mult[:, 0:-1], train_mult[:, -1], "Multinomial")
NB_mult.train()
NB_mult.score(test_mult[:, 0:-1], test_mult[:, -1])
print(NB_mult.confusion)
print(NB_mult.performance)
