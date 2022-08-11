# developed by Nazim A.Belabbaci aka NazimBL
# Summer 2022

import numpy as np
from sklearn import metrics
from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = loadtxt('data_final.csv', delimiter=",")
# split data into X and y
X = dataset[:,1:304264]
Y = dataset[:,0]

# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3,
              enable_categorical=False, gamma=0.2, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=15,
              min_child_weight=5, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=36, num_parallel_tree=1,
              predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)

classifier.fit(X_train,y_train,verbose=True ,
               early_stopping_rounds=200,eval_metric='auc',
               eval_set=[(X_test,y_test)])


##plot top 20 most imortant features
plot_importance(classifier,max_num_features=20,importance_type='gain')
plt.savefig('xgb_feature_importance.png')

##plot importance trees
for i in range(0,100):
    plot_tree(classifier,num_trees=i)
    plt.savefig('xgb_tree_'+str(i)+'.png')

#ROC, AUC
y_preds = classifier.predict_proba(X_test)
# take the second column because the classifier outputs scores for the 0 class as well
preds = y_preds[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)
auc_score = metrics.auc(fpr, tpr)

# clear current figure
plt.clf()
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

# it's helpful to add a diagonal to indicate where chance
# scores lie (i.e. just flipping a coin)
plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.savefig('cfdna_xgb_ROC.png')