# developed by Nazim A.Belabbaci aka NazimBL
# Summer 2022

import numpy as np
import xgboost

dataset = np.loadtxt('data_final.csv', delimiter=",")
# split data into X and y
X = dataset[:,1:304264]
Y = dataset[:,0]
test_size = 0.3

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size = 0.7)

# CV model
model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3,
              enable_categorical=False, gamma=0.2, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=15,
              min_child_weight=5, missing=None, monotone_constraints='()',
              n_estimators=100, n_jobs=36, num_parallel_tree=1,
              predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=5)
print(score)