import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Regression:
    def __init__(self, data):
        """"
        :param data: A pandas dataframe object containing the base features for the regression
        """
        self.data = data
        self.beta_hat = None
        self.coefs = None
        self.x = None
        self.y = None
        self.ridge_param = None
        self.grid_search_results = None

    @staticmethod
    def eq_parser(data, formula):
        """ Parses regression formula into the relevant indexes to construct the correct design matrix.
        :param data: A pandas dataframe object containing the base features for the regression
        :param formula: A string object using the following convention:
                      ~ is equivalent to the equality sign in the regression equation
                      : indicates an interaction term
                      **2 indicates a quadratic term
        :return: x are the base elements of the design matrix.
                 y is the dependent variable
                 higher_order are the column indexes of any interactions or quadratics
                 coef_labels is a list of coefficient names in order
        """
        dependent = formula.split("~")[0].strip()
        independent = formula.split("~")[1].strip()

        if independent[0] == "(" and independent[-4:] == ")**2":
            independent = independent[1:-4]
            interaction_all_pairs = True
        else:
            interaction_all_pairs = False

        independent = [s.strip() for s in independent.split('+')]
        if independent[0] == '.':
            independent = [s for s in data.columns if s != dependent]

        design_col_names = [s for s in independent if s in data.columns]
        m = len(design_col_names)
        names_to_index = {design_col_names[i]: i for i in range(0, m)}

        if interaction_all_pairs:
            interaction_name = [design_col_names[i] + ":" + design_col_names[j] for i in range(0, m-1) for j in
                                range(i+1, m)]
            quadratic_name = [design_col_names[i] + "**2" for i in range(0, m)]
        else:
            interaction_name = [s for s in independent if ':' in s]
            quadratic_name = [s for s in independent if "**2" in s]

        interaction_idx_pairs = [[names_to_index[s] for s in x.split(":")] for x in interaction_name]
        quadratic_idx = [[names_to_index[s.replace("**2", "")], names_to_index[s.replace("**2", "")]] for s in
                         quadratic_name]

        x = np.mat(data[design_col_names])
        y = np.mat(data[dependent])
        y = np.transpose(y)

        higher_order = quadratic_idx + interaction_idx_pairs
        coef_labels = design_col_names + quadratic_name + interaction_name
        return x, y, higher_order, coef_labels

    @staticmethod
    def design_matrix(data, formula, intercept):
        """
        Takes input of data and formula and outputs the final design matrix
        :param data: A pandas dataframe object containing the base features for the regression
        :param formula: A string object using the following convention:
                      ~ is equivalent to the equality sign in the regression equation
                      : indicates an interaction term
                      **2 indicates a quadratic term
        :param intercept: A Boolean indicating whether an intercept term should be included
        :return: x are the base elements of the design matrix.
                 y is the dependent variable
                 higher_order are the column indexes of any interactions or quadratics
                 coef_labels is a list of coefficient names in order
        """
        x, y, higher_order, coef_labels = Regression.eq_parser(data, formula)
        if type(intercept) != bool:
            raise TypeError("Intercept must be a boolean")

        for i in higher_order:
            temp = np.multiply(x[:, i[0]], x[:, i[1]])
            x = np.concatenate((x, temp), axis=1)

        n = np.shape(x)[0]
        if intercept:
            x = np.concatenate((np.ones((n, 1)), x), axis=1)
        return x, y, higher_order, coef_labels

    def fit(self, formula, intercept=True):
        """
        A regression function which takes the data and regression formula as input and outputs the regression coefficients
        :param formula: A string object using the following convention:
                      ~ is equivalent to the equality sign in the regression equation
                      : indicates an interaction term
                      **2 indicates a quadratic term
        :param intercept: A Boolean indicating whether an intercept term should be included, default is True.
        :return: coefs: a numpy matrix of coefficients for mathematical reasons
                 beta_hat: a pandas dataframe of regression coefficients
        """
        if type(self.data) != pd.DataFrame:
            raise TypeError("Data must be passed as a pandas DataFrame object.")

        self.x, self.y, higher_order, beta_hat_names = Regression.design_matrix(self.data, formula, intercept)
        if intercept:
            beta_hat_names = ["Intercept"] + beta_hat_names
        self.coefs = np.linalg.inv(np.transpose(self.x) * self.x) * np.transpose(self.x) * self.y
        self.beta_hat = pd.DataFrame(self.coefs, columns=["Coefficients"], index=beta_hat_names)
        return self

    @staticmethod
    def parameter_grid_search(x, y, m, ridge_param, sample_idx, intercept):
        mod_identity = np.identity(m)
        if intercept:
            mod_identity[0, 0] = 0
        parameter_scores = []
        k = len(sample_idx)
        for param in ridge_param:
            model_score = 0
            for i in range(0, k):
                training_idx = sample_idx[:]
                del training_idx[i]
                training_idx = np.concatenate(training_idx).ravel()
                testing_idx = sample_idx[i]
                x_train_set = x[training_idx, :]
                y_train_set = y[training_idx, :]
                x_test_set = x[testing_idx, :]
                y_test_set = y[testing_idx, :]
                model_coefficients = np.linalg.inv(np.transpose(x_train_set) * x_train_set + param
                                                   * mod_identity) * np.transpose(x_train_set) * y_train_set
                model_score += mean_square_error(x_test_set, y_test_set, model_coefficients)
            parameter_scores.append(model_score / k)

        best_ridge_param = ridge_param[parameter_scores.index(min(parameter_scores))]
        return best_ridge_param, parameter_scores

    @staticmethod
    def cost_function(ridge_param, x, y, m, sample_idx, intercept):
        mod_identity = np.identity(m)
        if intercept:
            mod_identity[0, 0] = 0
        k = len(sample_idx)
        model_score = 0
        for i in range(0, k):
            training_idx = sample_idx[:]
            del training_idx[i]
            training_idx = np.concatenate(training_idx).ravel()
            testing_idx = sample_idx[i]
            x_test_set = x[testing_idx, :]
            y_test_set = y[testing_idx, :]
            x_train_set = x[training_idx, :]
            y_train_set = y[training_idx, :]
            model_coefficients = np.linalg.inv(np.transpose(x_train_set) * x_train_set + ridge_param * mod_identity) * np.transpose(x_train_set) * y_train_set
            model_score += mean_square_error(x_test_set, y_test_set, model_coefficients)
        parameter_score = (model_score / k)
        return parameter_score

    def ridge(self, ridge_param, formula, intercept=True, tuner=None, k=0, random_seed = 0):
        """
        A regression function which takes the data and regression formula as input and outputs the regression coefficients
        :param random_seed:
        :param ridge_param: regularisation parameter for ridge regression, defaulted to 0.
        :param formula: A string object using the following convention:
                      ~ is equivalent to the equality sign in the regression equation
                      : indicates an interaction term
                      **2 indicates a quadratic term
        :param intercept: A Boolean indicating whether an intercept term should be included, default is True.
        :param tuner: String indicating tuner choice. Defaults to None.
                      Can be set to "Optimize" or "Grid Search"
        :param k: number of folds to be used by k-fold cross validation if parameter tuning is required.
        :return: beta_hat: a pandas dataframe of regression coefficients
        """
        np.random.seed(random_seed)
        if type(self.data) != pd.DataFrame:
            raise TypeError("Data must be passed as a pandas DataFrame object.")

        self.x, self.y, higher_order, beta_hat_names = Regression.design_matrix(self.data, formula, intercept)

        if intercept:
            beta_hat_names = ["Intercept"] + beta_hat_names

        n, m = np.shape(self.x)

        if tuner == "Grid Search":
            if type(ridge_param) == list:
                sample_idx = k_foldCV(self.y, k)
                self.ridge_param, self.grid_search_results = Regression.parameter_grid_search(self.x, self.y, m, ridge_param, sample_idx, intercept)
            else:
                raise TypeError("ridge_param must be a sequence of possible parameters passed as a list for"
                                " Grid Search")
        elif tuner == "Optimize":
            sample_idx = k_foldCV(self.y, k)
            optim = minimize(fun=Regression.cost_function, x0=np.array(ridge_param),
                             args=(self.x, self.y, m, sample_idx, intercept), bounds=[(0, None)])
            self.ridge_param = float(optim.x)
        elif tuner is None:
            self.ridge_param = ridge_param
        else:
            raise ValueError("Invalid tuner option; please select 'Grid Search' or 'Optimize'.")

        modified_identity = np.identity(m)
        if intercept:
            modified_identity[0, 0] = 0
        w = (np.transpose(self.x) * self.x) + (self.ridge_param * modified_identity)
        inv = np.linalg.inv(w)
        self.coefs = inv * (np.transpose(self.x) * self.y)
        self.beta_hat = pd.DataFrame(self.coefs, columns=["Coefficients"], index=beta_hat_names)

        return self

    def predict(self, new_x):
        if type(new_x) != np.matrix:
            raise TypeError("X values must be a numpy matrix.")
        prediction = new_x * self.coefs
        return prediction


def mean_square_error(x, y, coefficients):
    n = np.shape(y)[0]
    mse = float(1 / n * (np.transpose(y - x * coefficients) * (y - x * coefficients)))
    return mse


def r_square(x, y, coefficients):
    residuals = y - x * coefficients
    ss_reg = np.square(residuals).sum()
    ss_tot = np.square(y - y.mean()).sum()
    r_sq = 1 - ss_reg / ss_tot
    return r_sq


def k_foldCV(y, k):
    n = np.shape(y)[0]
    bucket_size = int(np.floor(n / k))
    init_sample_idx = np.random.choice(n, n, replace=False)
    left_overs = init_sample_idx[(k * bucket_size):n]
    assert len(left_overs) <= k
    cv_samples_idx = [init_sample_idx[i*bucket_size : (i + 1)*bucket_size] for i in range(0, k)]
    for i in range(0, len(left_overs)):
        cv_samples_idx[i] = np.append(cv_samples_idx[i], left_overs[i])
    return cv_samples_idx


def train_test_split(data, train_prop):
    if type(data) == pd.DataFrame:
        n = len(data)
        train_size = round(n*train_prop)
        idx = np.arange(n)
        np.random.shuffle(idx)
        assert n >= idx.shape[0]
        train = data.iloc[idx[0:train_size], :]
        test = data.iloc[idx[train_size:], :]
    elif type(data) == np.mat or type(data) == np.ndarray:
        n = data.shape[0]
        train_size = round(n*train_prop)
        idx = np.arange(n)
        np.random.shuffle(idx)
        assert n >= idx.shape[0]
        train = data[idx[0:train_size], :]
        test = data[idx[train_size:], :]
    else:
        raise TypeError("data must be passed as a pandas Data frame or Numpy array.")
    return train, test
