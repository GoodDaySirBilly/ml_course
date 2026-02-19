# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import openml as oml

# Hide convergence warning for now
import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Download NO2 data. Takes a while the first time.
no2 = oml.datasets.get_dataset(547)
X, y, _, _ = no2.get_data(target=no2.default_target_attribute); 
attribute_names = list(X)

df = pd.DataFrame(X, columns=attribute_names).join(pd.DataFrame(list(y),columns=['target']))
df = df.sort_values(['day','hour_of_day']).drop('day',axis=1)

# df.plot(use_index=False,figsize=(20,5),cmap=cm.get_cmap('brg'));

X = X.drop('day',axis=1)

without_wind_dir = df.drop('wind_direction',axis=1)

# without_wind_dir.plot(use_index=False,figsize=(20,5),cmap=cm.get_cmap('brg'));

# X = X.drop('hour_of_day',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.model_selection import cross_validate

def evaluate_learners(models, X, y):
    
    train_scores, test_scores = [], []

    for model in models: 

        cv_results = cross_validate(model, X, y, scoring="r2", return_train_score=True)

        train_scores.append(cv_results["train_score"])
        test_scores.append(cv_results["test_score"])

    return train_scores, test_scores


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor()
]

train_scores, test_scores = evaluate_learners(models, X, y)

def show_benchmarks(train_scores, test_scores):
    fig, ax = plt.subplots(figsize=(10,6))
    width=0.45


    ax.barh(np.arange(len(train_scores)), np.mean(test_scores, axis=1), width,
            yerr= np.std(test_scores, axis=1), color='green', label='test R^2')

    ax.barh(np.arange(len(train_scores))-width, np.mean(train_scores, axis=1), width,
            yerr= np.std(train_scores, axis=1), color='red', label='train R^2')

    for i, te, tr in zip(np.arange(len(train_scores)), test_scores, train_scores):

        ax.text(0, i, "{:.3f} +- {:.3f}".format(np.mean(te),np.std(te)), color=('white' if np.mean(te)>0.1 else 'black'), va='center')
        ax.text(0, i-width, "{:.3f} +- {:.3f}".format(np.mean(tr),np.std(tr)), color=('white' if np.mean(tr)>0.1 else 'black'), va='center')

    labels = [c.__class__.__name__ if not hasattr(c, 'steps') else c.steps[0][0] + "_" + c.steps[1][0] for c in models]
    ax.set(yticks=np.arange(len(train_scores))-width/2, yticklabels=labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

show_benchmarks(train_scores, test_scores)

##################################################

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def grid_params(model, params):


    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=params, 
        cv=5,                 
        scoring='r2',   
        n_jobs=-1,            
        verbose=1,
        return_train_score=True           
    )

    grid_search.fit(X, y)

    print(f"\nModel name: {model.__class__.__name__}\n")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (R^2 score): {grid_search.best_score_}")

    best_model = grid_search.best_estimator_

    return best_model, grid_search.cv_results_

best_ridge, ridge_results = grid_params(Ridge(), {'alpha': np.logspace(-12, 12, num=25)})
best_lasso, lasso_results = grid_params(Lasso(), {'alpha': np.logspace(-12, 12, num=25)})
best_elastic_net, elastic_results = grid_params(
    ElasticNet(), {
        'alpha': np.logspace(-12, 12, num=25),
        'l1_ratio': np.linspace(0.1, 1., num=10)
    }
)
best_knn, knn_results = grid_params(KNeighborsRegressor(), {'n_neighbors': np.linspace(1, 50, dtype=int)})


models = [
    LinearRegression(),
    best_ridge, best_lasso, best_elastic_net, best_knn
]

train_scores, test_scores = evaluate_learners(models, X, y)

show_benchmarks(train_scores, test_scores)

def hyperparam_r2_plot(x_train, y_train, x_test, y_test, xlabel, ylabel, title):

    plt.figure()
    plt.plot(x_train, y_train, label="train")
    plt.plot(x_test, y_test, label="test")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


alpha_ticks = np.linspace(-12, 12, num=25)
l1_ticks = np.linspace(0.1, 1., num=10)
n_neighbors = np.linspace(1, 50, dtype=int)
print(ridge_results.keys())
hyperparam_r2_plot(alpha_ticks, ridge_results["mean_test_score"], alpha_ticks, ridge_results["mean_train_score"], "alpha, 10^", "Mean R^2", "Ridge")
hyperparam_r2_plot(alpha_ticks, lasso_results["mean_test_score"], alpha_ticks, lasso_results["mean_train_score"], "alpha, 10^", "Mean R^2", "Lasso")
hyperparam_r2_plot(n_neighbors, knn_results["mean_test_score"], n_neighbors, knn_results["mean_train_score"], "n_neighbors", "Mean R^2", "kNN")


import seaborn as sns

def heatmap(values, xlabel, ylabel, xticklabels, yticklabels):

    plt.figure()
    ax = sns.heatmap(values, annot=True, cmap='YlGnBu', fmt=".2f")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

heatmap(elastic_results["mean_test_score"].reshape(10, 25), "alpha, 10^", "l1_ratio", alpha_ticks, l1_ticks)


######################################

feature_names = list(X.columns)

good_alpha = 0.001
large_alpha = 1.0

models_good = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=good_alpha),
    "Lasso": Lasso(alpha=good_alpha),
    "ElasticNet": ElasticNet(alpha=good_alpha, l1_ratio=0.5),
}

models_large = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=large_alpha),
    "Lasso": Lasso(alpha=large_alpha),
    "ElasticNet": ElasticNet(alpha=large_alpha, l1_ratio=0.5),
}


def extract_coefs(model):
    return model.coef_


def plot_coefficients(models_dict, title):
    plt.figure(figsize=(11, 6))
    x_axis = np.arange(len(feature_names))
    for model_name, model in models_dict.items():
        model.fit(X, y)
        coefs = extract_coefs(model)
        plt.scatter(x_axis, coefs, label=model_name, s=60, alpha=0.8)
    plt.xticks(x_axis, feature_names, rotation=45, ha="right")
    plt.ylabel("Coefficient value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


plot_coefficients(models_good, "Exercise 3.1: Coefficients (good alpha)")
plot_coefficients(models_large, "Exercise 3.2: Coefficients (large alpha)")

plt.show()