# developed by Nazim A.Belabbaci aka NazimBL
# Summer 2022

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from numpy import loadtxt

dataset = loadtxt('data_final.csv', delimiter=",")
# split data into X and y
X = dataset[:,1:304264]
Y = dataset[:,0]

## Hyper Parameter Space
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

classifier = XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=25,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
timer(start_time) # timing ends here for "start_time" variable

print('bestscore:')
print(random_search.best_score_)
print(random_search.best_estimator_)
print(random_search.best_params_)

