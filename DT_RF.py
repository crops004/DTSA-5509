from math import exp
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from EDA import df

'''
Create data sets for decision tree regressor
'''

y = df['home_run'].values
X = df.drop('home_run', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

'''
Creating a DecisionTreeRegressor and seeing how it performs
'''

dt = DecisionTreeRegressor().fit(X_train, y_train,)
y_pred_DTR = dt.predict(X_test)
MSEdt = mean_squared_error(y_test, y_pred_DTR)

Models.append('Decision Tree Regressor')
Mean_Squared_Errors.append(MSEdt)

'''
MSE of 45.194 with standard DecisionTreeRegressor. Running GridSearchCV to find best parameters for DTR
'''

rf = DecisionTreeRegressor()
parameters = {'max_depth':[3,5,7,10], 'min_samples_leaf':[1,2,4,6,8,10]}
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_
clf.best_score_

'''
Best estimator is DTR with max_depth=7 and min_samples_leaf=8
'''
rf = best_estimator
y_pred = rf.predict(X_test)
MSEbestDTR = mean_squared_error(y_test, y_pred)

Models.append('Best Decision Tree Regressor')
Mean_Squared_Errors.append(MSEbestDTR)

'''
MSE = 32.736 when using best_estimator
'''


'''
Pruning with ccp_alpha
'''

dt = DecisionTreeRegressor().fit(X_train, y_train)

path = dt.cost_complexity_pruning_path(X_train,y_train) #post pruning
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = [] # VECTOR CONTAINING CLASSIFIERS FOR DIFFERENT ALPHAS

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(ccp_alpha = ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

train_scores = []
test_scores = []

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

max_test_score_index = test_scores.index(max(test_scores))
best_alpha = ccp_alphas[max_test_score_index]
best_test = test_scores[max_test_score_index]
best_train = train_scores[max_test_score_index]

print('alpha', best_alpha)
print('test', best_test)
print('train', best_train)

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

clf = DecisionTreeRegressor(ccp_alpha = best_alpha, max_depth=7, min_samples_leaf=8).fit(X_train, y_train)
y_pred = clf.predict(X_test)
MSEprunedDTR = mean_squared_error(y_test, y_pred)

Models.append('Pruned Decision Tree Regressor')
Mean_Squared_Errors.append(MSEprunedDTR)

'''
MSE = 31.87 after pruning with optimal parameters
'''





'''
Beginning with RandomForestRegressor, which randomly samples the data and features. 
'''

regr = RandomForestRegressor(max_depth=7, min_samples_leaf=8).fit(X_train, y_train)
y_pred = regr.predict(X_test)
MSErf = mean_squared_error(y_test, y_pred)

Models.append('Random Forest Regressor')
Mean_Squared_Errors.append(MSErf)

'''
MSE = 22.553
'''

'''
Using GridSearchCV to tests for optimal parameters
'''

rf = RandomForestRegressor()
parameters = {'max_depth':[3,5,7,10], 'min_samples_leaf':[1,2,4,6,8,10], 'n_estimators':[10,50,100,200],
              'max_samples':[None, 1, 300, 600]}
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_

'''
Best estimator shows that the optimal parameters for RandomForestRegressor are max_depth=10, min_samples_leaf=2, 
max_samples=600, n_estimators=200
'''

regr = RandomForestRegressor(max_depth=10, min_samples_leaf=2, max_samples=600, n_estimators=200).fit(X_train, y_train)
y_pred_regr = regr.predict(X_test)
MSEbestrf = mean_squared_error(y_test, y_pred_regr)

Models.append('Best Random Forest Regressor')
Mean_Squared_Errors.append(MSEbestrf)

'''
After running RandomForestRegressor with optimal paramters, MSE down to 21.681
'''


'''
Boosting the ensemble with AdaBoost. Starting with standard AdaBoost then GridSearch for optimal parameters
'''

regr_ada = AdaBoostRegressor().fit(X_train, y_train)
y_pred = regr_ada.predict(X_test)
MSEada = mean_squared_error(y_test, y_pred)

Models.append('AdaBoost')
Mean_Squared_Errors.append(MSEada)

'''
Initial adaBoost has MSE of 24.72. GridSearchCV for best params
'''

rf = AdaBoostRegressor()
parameters = {'n_estimators':[50,100], 'learning_rate':[1,5,10], 'loss':['linear','square','exponential']}
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_

regr_ada = AdaBoostRegressor(learning_rate=1, n_estimators=100).fit(X_train, y_train)
y_pred = regr_ada.predict(X_test)
MSEbestada = mean_squared_error(y_test, y_pred)

Models.append('Best AdaBoost')
Mean_Squared_Errors.append(MSEbestada)

'''
MSE = 24.474 with optimal parameters
'''

'''
AdaBoost with optimnal RandomForest parameters
'''

regr_ada_rf = AdaBoostRegressor(estimator=RandomForestRegressor(max_depth=10, min_samples_leaf=2, max_samples=600,
                                                                n_estimators=200), n_estimators=100,
                                                                loss='square').fit(X_train, y_train)
y_pred = regr_ada_rf.predict(X_test)
MSErfada = mean_squared_error(y_test, y_pred)

Models.append('RF AdaBoost')
Mean_Squared_Errors.append(MSErfada)

'''
MSE = 19.068 with optimal rf parameters
'''

features_importance = []
features = list(df.columns)
features.pop(0)


for index, item in enumerate(features):
    features_importance.append([features[index], regr_ada_rf.feature_importances_[index]])

features_importance.sort(key = lambda x: x[1], reverse=True)

'''
Most important features to AdaBoost were 'barrel_batted_rate', 'hit', and 'ab'. Everything else much lower
'''

x = features
y = regr_ada_rf.feature_importances_
sorted_data = sorted(zip(x,y), key=lambda x: x[1], reverse=True)
x_values, y_values = zip(*sorted_data)

fig, ax = plt.subplots(figsize=(9.6,7.2))
ax.set_xlabel("Feature")
ax.set_ylabel("Feature Importancce")
ax.set_title("Feature Importance")
ax.bar(x_values, y_values)
ax.set_xticklabels(x_values, rotation=40, ha='right')
plt.subplots_adjust(bottom=0.27)
plt.show()

'''
Compare all different regressions
'''
x = Models
y = Mean_Squared_Errors
sorted_data = sorted(zip(x,y), key=lambda x: x[1])
x_values, y_values = zip(*sorted_data)

fig, ax = plt.subplots(figsize=(9.6,7.2))
ax.set_xlabel("Models")
ax.set_ylabel("MSE")
ax.set_title("MSE for Different Regressions")
ax.bar(x_values, y_values)
ax.set_xticklabels(x_values, rotation=40, ha='right')
plt.subplots_adjust(bottom=0.27)
plt.show()

'''
Discussion and conclusion - Based on the handful of regression performed on my data, a RandomForestRegressor with 
AdaBoost was the most effective model. This was not surprising, as RF generally out performs linear regression and 
decision trees. It was interesting that multi linear regression performed better than decision tree regression, this
may have been related to that fact that while I had a high number of features, only a small number were atctually 
important to the regression. My initial dataset did not work for a regression because there was a combination of four 
variables that perfectly calculated my target, so I had an R2 of 1.0 and none of the other features were important. But
I was able to easily adjust for this by removing those features from the dataset. To improve on this analysis, you 
could add many more features. With the addition of Statcast to MLB, there are now hundreds of advanced statistics that 
are automatically tracked for every game, and I sampled just a handful that I thought would be significant. That being
said, after achieving not very good results with any of my regressions (high MSE with all of them), I do think that 
modeling home runs from other statistics may not be a wortwhile venture.
'''