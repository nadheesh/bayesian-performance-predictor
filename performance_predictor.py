"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

seed = 42
np.random.seed(seed)


class BayesianRBFRegression:

    def fit(self, X, y):
        """
        train model
        :param X:
        :param y:
        :return:
        """

        # bayesian rbf kernel using gaussian processes
        with pm.Model() as self.model:
            # hyper-prior to tune
            ls = pm.Beta("ls", alpha=1, beta=1)

            cov_func = pm.gp.cov.ExpQuad(X.shape[1], ls=ls)
            # cov_func = pm.gp.cov.Polynomial(X.shape[1], 0.1, 3, 0.1)

            # Specify the GP.
            self.gp = pm.gp.Marginal(cov_func=cov_func)

            # Place a GP prior over the function f.
            sigma = pm.HalfNormal("sigma", 0.1)
            y_ = self.gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

            # inference
            # self.map_trace = pm.fit(50000).sample(1000)
            self.map_trace = [pm.find_MAP()]
            # map1 = pm.sample(2000, tune=1000)

    def predict(self, X, with_error=False):
        """
        predict using the train model
        :param X:
        :return:
        """
        if not hasattr(self, 'model'):
            raise AttributeError("train the model first")

        # fcond = gp.conditional('fcond', eval_X)
        # pred_samples = pm.sample_ppc([map1], vars=[fcond], samples=5000)
        # y_predb, error = pred_samples['fcond'].mean(axis=0), pred_samples['fcond'].std(axis=0)
        y_pred, error = self.gp.predict(X, point=self.map_trace, diag=True)

        if with_error:
            return y_pred, error
        return y_pred


def mean_absolute_percentage_error(y_true, y_pred):
    """
    compute mean absolute percentage error
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def category_to_int(df, columns):
    """
    covert categories to integer codes
    :param df: pandas data-frame
    :param columns: columns to process. can be a string or a list of strings
    :return:
    """
    for col in columns:
        df[col] = df[col].astype('category')

    df[columns] = df[columns].apply(lambda x: x.cat.codes)

    return df


def read_csv(path, scaler, feature_idx, label_idx, shuffle=True):
    """
    read csv given the path

    :param path:
    :param scaler:
    :param feature_idx:
    :param label_idx:
    :param shuffle:
    :return:
    """
    df = pd.read_csv(path)
    df = category_to_int(df, ["Name"])

    X = df.iloc[:, feature_idx]
    y = df.iloc[:, label_idx]

    X = scaler.fit_transform(X, y)
    y = y.values.flatten()

    # if shuffle:
    #     permuted_idx = np.random.permutation(np.arange(0, X.shape[0]))
    #     X = X[permuted_idx]
    #     y = y[permuted_idx]

    return X, y


def eval_SVR_baseline(X, y, eval_X, eval_y):
    """
    evaluate svr model.
    :param X:
    :param y:
    :param eval_X:
    :param eval_y:
    :return: predict_y, mse, mape
    """
    # mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    # mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    # grid search params
    svr_params = {'kernel': ['linear', 'poly', 'rbf'],
                  'C': [0.1, 1, 100, 1000],
                  'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                  'degree': [2, 3, 4]
                  }

    # initial model
    _SVR = SVR(coef0=0.1, tol=0.001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

    # grid search - initiation
    # TODO change to random grid search - can be more efficient
    svr = GridSearchCV(_SVR, param_grid=svr_params,
                       cv=5,
                       scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1)

    # execute grid search - select best svr model
    best_svr = svr.fit(X, y).best_estimator_

    pred_y = best_svr.predict(eval_X)

    return pred_y, np.sqrt(mean_squared_error(eval_y, pred_y)), mean_absolute_percentage_error(eval_y, pred_y)


def eval_linear_baseline(X, y, eval_X, eval_y):
    """
    evaluate linear regression model
    :param X:
    :param y:
    :param eval_X:
    :param eval_y:
    :return: predict_y, mse, mape
    """
    lr = LinearRegression()
    lr.fit(X, y)
    pred_y = lr.predict(eval_X)

    return pred_y, np.sqrt(mean_squared_error(eval_y, pred_y)), mean_absolute_percentage_error(eval_y, pred_y)


def eval_bayesian_rbf(X, y, eval_X, eval_y):
    """
    evaluate linear regression model
    :param X:
    :param y:
    :param eval_X:
    :param eval_y:
    :return: predict_y, error, mse, mape
    """
    lr = BayesianRBFRegression()
    lr.fit(X, y)
    pred_y, error = lr.predict(eval_X, True)

    return pred_y, error, np.sqrt(mean_squared_error(eval_y, pred_y)), mean_absolute_percentage_error(eval_y, pred_y)


# PARAMETERS

# label_index = 23  # latency col
# feature_indices = [0, 1, 2, 3, 4]

label_index = 29 # throughput col
feature_indices = [0, 1, 2, 3]

if __name__ == '__main__':
    # TODO change the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    _X, _y = read_csv("data/train.csv", scaler, feature_indices, label_index)
    eval_X, eval_y = read_csv("data/test.csv", scaler, feature_indices, label_index)

    pred_bayes, error, mse_bayes, mape_bayes = eval_bayesian_rbf(np.copy(_X), np.copy(_y), np.copy(eval_X), np.copy(eval_y))
    pred_svr, mse_svr, mape_svr = eval_SVR_baseline(_X, _y, eval_X, eval_y)
    pred_lr, mse_lr, mape_lr = eval_linear_baseline(_X, _y, eval_X, eval_y)

    print("SVR MSE : %f and MAPE :%f" % (mse_svr, mape_svr))
    print("Bayesian MSE : %f and MAPE :%f" % (mse_bayes, mape_bayes))
    print("LinearRegression MSE : %f and MAPE :%f" % (mse_lr, mape_lr))

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(eval_y, eval_y, ls="--", color='black', label="true", alpha=0.7, lw=1)
    ax.errorbar(eval_y, pred_bayes, yerr=error * 1000, fmt='o', label='bayesian', c='y', alpha=0.5, marker="o")
    ax.scatter(eval_y, pred_lr, c='g', label="linear regression", alpha=1, marker="*", lw=3)
    ax.scatter(eval_y, pred_svr, c='r', label="svr", alpha=1, marker="+")
    ax.set_ylabel("prediction")
    ax.set_xlabel("true y")
    ax.legend(loc=0)

    plt.show()
