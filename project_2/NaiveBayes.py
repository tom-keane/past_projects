import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, x, y, family):
        self.x = x
        if type(y) == pd.DataFrame:
            self.y = y.astype('category').cat.codes
        elif type(y) == np.ndarray or type(y) == np.mat:
            self.y = y
        else:
            raise TypeError("y must be passed as a pandas Data frame or an encoded Numpy array.")
        self.family = family
        self.no_classes = None
        self.theta = None
        self.priors = None
        self.performance = None

    def train(self):
        self.no_classes = np.max(self.y) + 1
        n, m = self.x.shape
        self.priors = np.zeros((self.no_classes, 1))
        self.theta = np.zeros((self.no_classes, m))
        if self.family == "Bernoulli":
            assert np.max(self.x) == 1
            for i in range(self.no_classes):
                points_in_class = np.where(self.y == i)[0]
                self.priors[i, ] = len(points_in_class)/n
                self.theta[i, ] = (np.sum(self.x[points_in_class, :], axis=0)+1)/(len(points_in_class)+m)
        elif self.family == "Multinomial":
            for i in range(self.no_classes):
                points_in_class = np.where(self.y == i)[0]
                self.priors[i, :] = len(points_in_class)/n
                self.theta[i, :] = (np.sum(self.x[points_in_class, :], axis=0) + 1)/(np.sum(self.x[points_in_class, :]) + m)
        else:
            raise ValueError("Invalid family choice, please choose Bernoulli or Multinomial.")
        return self

    def predict(self, x_new):
        no_of_points = x_new.shape[0]
        classification = np.zeros(no_of_points)
        if self.family == "Bernoulli":
            for i in range(no_of_points):
                prediction_point = np.asmatrix(x_new[i, :])
                theta_0 = 1 - self.theta[:, :]
                theta_1 = self.theta[:, :]
                log_theta_0 = np.log(theta_0) * np.transpose(1 - prediction_point)
                log_theta_1 = np.log(theta_1) * np.transpose(prediction_point)
                likelihood = log_theta_0 + log_theta_1
                posterior = likelihood + np.log(self.priors)
                classification[i] = np.argmax(posterior)
        elif self.family == "Multinomial":
            for i in range(no_of_points):
                prediction_point = np.asmatrix(x_new[i, :])
                posterior = np.log(self.theta)*np.transpose(prediction_point) + np.log(self.priors)
                classification[i] = int(np.argmax(posterior))
        else:
            raise ValueError("Invalid family choice, please choose Bernoulli or Multinomial.")
        return classification

    @staticmethod
    def confusion_matrix(y_actual, y_predict):
        actual = pd.Series(y_actual, name='Actual')
        predicted = pd.Series(y_predict, name='Predicted')
        df_confusion = pd.crosstab(predicted, actual)
        return df_confusion

    def score(self, x_new, y_actual):
        y_predict = self.predict(x_new)
        df_confusion = NaiveBayes.confusion_matrix(y_actual, y_predict)
        confusion = np.mat(df_confusion)
        metrics = {"Accuracy": np.sum(np.diag(confusion))/np.sum(confusion),
                   "Sensitivity": confusion[0, 0] / np.sum(confusion, axis=0)[0, 0],
                   "Specificity": confusion[1, 1] / np.sum(confusion, axis=0)[0, 1],
                   "Precision": float(confusion[0, 0] / np.sum(confusion, axis=1)[0]),
                   "F1 score": None}
        metrics["F1 score"] = float(2 * (metrics["Precision"] * metrics["Sensitivity"]) / (metrics["Precision"] + metrics["Sensitivity"]))
        self.performance = metrics
        self.confusion = df_confusion
        return self
