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

X = X.drop('hour_of_day',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.model_selection import cross_validate

def evaluate_learners(models, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _, features_num = X_train.shape
    models_num = len(models)

    train_scores, test_scores = np.zeros((models_num, features_num)), np.zeros((models_num, features_num))

    for model_index, model in enumerate(models): 

        model_array = [clone(model) for _ in range(features_num)]

        for feature_index in range(features_num):

            X_train_feature = X_train.to_numpy()[:, feature_index]
            X_test_feature = X_test.to_numpy()[:, feature_index]

            X_train_feature = X_train_feature[:, np.newaxis]
            X_test_feature = X_test_feature[:, np.newaxis]
            
            model_array[feature_index].fit(X_train_feature, y_train)

            score_train = model_array[feature_index].score(X_train_feature, y_train)
            score_test = model_array[feature_index].score(X_test_feature, y_test)

            train_scores[model_index, feature_index] = score_train
            test_scores[model_index, feature_index]= score_test



    return train_scores, test_scores


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet()
]

train_scores, test_scores = evaluate_learners(models, X, y)
print(train_scores.shape)

# Plot a bar chart of the train and test scores of all the classifiers, including the variance as error bars
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
#ax.set(yticks=np.arange(len(train_scores))-width/2, yticklabels=labels)
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2)


plt.show()