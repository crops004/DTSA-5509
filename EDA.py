'''
The topic for this project will be predicting the number of home runs a player will hit in a season of
MLB. The dataset I am using for this project has been collected from
https://baseballsavant.mlb.com/leaderboard/statcast, and includes a number of
traditional baseball states (batting average, at bats, hits) as well as many newer advanced stats (average exit
velocity, hard hit percentage, launch angle). The data is readily available, and easily downloaded to a .csv file.
For my data, I am usuing statistics gathered during the 2018 through 2023 seasons (omitting shortened 2020 season, 2023
statistics through 06/23/2023). This will be a regression problem, and I plan to use
random forest to boost my regression. The goal of the project would be to predict the number of home runs a player
would hit given his other statistics.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pylab
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
Importing data and inspecting data.  
'''
df = pd.read_csv('stats.csv')
df.info()
df.describe()

'''
Inspection shows that I have 22 columns (features), the majority of which are int 
data types or float data types. Two columns are object data type, these are the players names. I have 1897 
rows (samples) in most columns. Also reveals four columns that can be removed. player_id is a reference column used
by baseballsavant.com, and year, first_name, and last_name won't be needed.
'''

df = df.drop('year', axis=1)
df = df.drop('player_id', axis=1)
df = df.drop(' first_name', axis=1)
df = df.drop('last_name', axis=1)

'''
Checking for null values. 
'''

null_counts = pd.isnull(df).sum()
df = df.dropna()

'''
Checking revealed a very small amount. Since my dataset is large enough, I will
just drop these rows from the dataset. 
'''

'''
Visualizing and describing data. Histogram of each feature created
Moving home runs (my target) to the first column
'''

features = list(df.columns)
neworder = ['home_run', 'player_age', 'ab', 'hit', 'single', 'double', 'triple',
            'batting_avg', 'exit_velocity_avg', 'launch_angle_avg', 'sweet_spot_percent', 'barrel_batted_rate',
            'solidcontact_percent', 'flareburner_percent', 'poorlyunder_percent', 'poorlytopped_percent',
            'hard_hit_percent', 'avg_best_speed']
df=df[neworder]

for item in features:
    plt.hist(df[item])
    plt.xlabel(item)
    plt.ylabel('Count')
    plt.show()

'''
home_run - what I am trying to find
player_age - Age of each player in the given year. Will be interesting to see if correlated with HR
ab - At bats in the season. Should be somewhat correlated with HR since you need to have at bats to hit HR
    Very large percentage of the data appears to be at the very low (<300) end of thd data.
    Will be droping samples with very low 'ab' so they don't distort data.
hit - number of hits in the year
single - number of singles in the year
double - number of doubles in the year
triple - number of triples in the year
batting_avg - number of hits / number of at bats
exit_velocity_avg - average speed of the baseball off of the players bat
launch_angle_avg - angle that the ball leaves the bat from. I expect this to be highly correlated with HR
sweet_spot_percent - percent of batted balls with a launch angle between 8 and 32 degrees. Should be highly 
    correlated with HR
barrel_batted_rate - percentage of batted balls with a launch angle and exit velocity combination have led to 
    a minimum .500 batting average and 1.500 slugging percentage since Statcast was implemented 
    Major League wide in 2015.
solidcontact_percent - percentage of batted balls deemed to have solid contact
flareburner_percent - percentage of batted balls that are burners (hard ground balls) or flares (short pop flies)
poorlyunder_percent - percentage of batted balls that are weak pop ups
poorlytopped_percent - percentage of batted balls that are weak ground balls (impossible to be a HR)
hard_hit_percent - percentage of batted balls with exit velocity over 95 mph
avg_best_speed - average of 50% of players hardest hit balls
'''

'''
Removing rows of players with less than 260 at bats in a season from the training data. This is a bit arbitrary,
but after 130 at bats you are no longer considered a rookie in MLB, and I needed a cutoff to remove players 
with very few at bats whose data could skew the results, so I doubled that number. 
After dropping these samples, the dataset now contains 1210 samples
'''

df = df.drop(df[df['ab'] < 260].index)

'''
Making correlation matrix
'''

corr = df.corr()

plt.figure(figsize=(12, 12))
sns.heatmap(corr, cmap="viridis", annot=True, cbar=False, fmt='.2f')
plt.show()

'''
Best predictor of home runs based on the correlation matrix is 'barrel_batted_rate', with the next bests being 
'avg_best_speed', 'hit', and 'avg_exit_velocity'. 'barrel_batted_rate' is highly correlated with 'avg_best_speed' and
'avg_exit_velocity', so those are not surprising. 'ab' is not correlated strongly with 'barrel_batted_rate'
so that is interesting.
'''




'''
Splitting data to run a simple linear regression model. 
'''

X_train, X_test = train_test_split(df, test_size=0.20, random_state=0)
model = smf.ols(formula='home_run ~ barrel_batted_rate', data=df)
res = model.fit()
print(res.summary())

'''
Adj R squared of only 0.501 on the best guess predictor
of 'barrel_batted_rate'
'''

'''
Plot of 'barrel_batted_rate' vs 'home_run' with linear regression line
'''

sns.lmplot(x='barrel_batted_rate',y='home_run', data=df)
plt.show()

'''
Calculating MSE for model. Creating lists to store future results for later analysis
'''
Models = []
Mean_Squared_Errors = []

y_pred = res.predict(X_test)
y_true = X_test['home_run'].values
MSElinear = mean_squared_error(y_true, y_pred)

Models.append('Linear Regression')
Mean_Squared_Errors.append(MSElinear)

'''
MSE of 53.13
'''

'''
Testing all other features as predictor. 'barrel_batted_rate' was the best predictor
'''

headers = list(df.columns)
headers = headers[1:]
best_predictor = ''
best_r_squared = 0

for item in headers:
    test = item
    model = smf.ols(formula='home_run ~ ' + test, data=df)
    res = model.fit()
    r_squared = res.rsquared

    if r_squared > best_r_squared:
        best_predictor = item
        best_r_squared = r_squared

print(best_predictor, best_r_squared)

'''
Testing polynomial regression up to N=10. 
'''
# return updated best_degree and best_r_squared
best_degree = 1

formulaString = 'home_run ~ barrel_batted_rate'
for n in range(2, 11):
    formulaString = formulaString + ' + np.power(barrel_batted_rate,' + str(n) + ')'
    model = smf.ols(formula=formulaString, data=df)
    res = model.fit()
    r_squared = res.rsquared

    if r_squared > best_r_squared:
        best_degree = n
        best_r_squared = r_squared

print(best_degree, best_r_squared)

'''
Best R squared was at degree 10. Adj R2 of 0.507
'''

'''
MSE of polynomial regression
'''

model = smf.ols('home_run ~ barrel_batted_rate + np.power(barrel_batted_rate,2) +'
                'np.power(barrel_batted_rate,3) + np.power(barrel_batted_rate,4) +'
                'np.power(barrel_batted_rate,5) + np.power(barrel_batted_rate,6) +'
                'np.power(barrel_batted_rate,7) + np.power(barrel_batted_rate,8) +'
                'np.power(barrel_batted_rate,9) + np.power(barrel_batted_rate,10)', data=df).fit()

y_pred = model.predict(X_test)
y_true = X_test['home_run'].values
MSEpoly = mean_squared_error(y_true, y_pred)

Models.append('Polynomial Regression')
Mean_Squared_Errors.append(MSEpoly)

'''
MSE of 53.568
'''


'''
Multi-linear regression.
'''

header_string = ''
for index, x in enumerate(headers):
    if index == 0:
        header_string += x
    else:
        header_string += ' + ' + x

model = smf.ols('home_run ~ ' + header_string, data=df).fit()
print(model.summary())

'''
Found a problem right away. Hits - (singles + doubles + triples) = home runs, so
everything else has terrible P values. Dropping those 4 features and re running. 
'''

headers.remove('hit')
headers.remove('single')
headers.remove('double')
headers.remove('triple')

header_string = ''
for index, x in enumerate(headers):
    if index == 0:
        header_string += x
    else:
        header_string += ' + ' + x

model = smf.ols('home_run ~ ' + header_string, data=df).fit()
print(model.summary())

'''
After removing hit, single, double, & triple, got a model with an adjusted R quared of 0.812, and 8 significant
features based off p values < 0.05
Multi-model with interactions. Removed insignificant interactions (p > 0.05) prior to running
'''

sig_features = ['ab', 'batting_avg', 'launch_angle_avg', 'sweet_spot_percent', 'barrel_batted_rate',
                'solidcontact_percent', 'flareburner_percent', 'hard_hit_percent']

model_multi = smf.ols('home_run ~ (ab * batting_avg) + (ab * launch_angle_avg) + (ab * sweet_spot_percent) + '
                      '(ab * barrel_batted_rate) + (ab * solidcontact_percent) + (ab * flareburner_percent) + '
                      '(ab * hard_hit_percent) + (batting_avg * launch_angle_avg) + (batting_avg * sweet_spot_percent)'
                      '+ (batting_avg * barrel_batted_rate) + (batting_avg * solidcontact_percent) + '
                      '(batting_avg * flareburner_percent) + (batting_avg * hard_hit_percent) + '
                      '(launch_angle_avg * sweet_spot_percent) + (launch_angle_avg * barrel_batted_rate) + '
                      '(launch_angle_avg * solidcontact_percent) + (launch_angle_avg * flareburner_percent) + '
                      '(launch_angle_avg * hard_hit_percent) + (sweet_spot_percent * barrel_batted_rate) + '
                      '(sweet_spot_percent * solidcontact_percent) + (sweet_spot_percent * flareburner_percent) + '
                      '(sweet_spot_percent * hard_hit_percent) + (barrel_batted_rate * solidcontact_percent) + '
                      '(barrel_batted_rate * flareburner_percent) + (barrel_batted_rate * hard_hit_percent)'
                      '+ (solidcontact_percent * flareburner_percent) + (solidcontact_percent * hard_hit_percent)'
                      '+ (flareburner_percent * hard_hit_percent)', data=df).fit()

print(model_multi.summary())

model_multi = smf.ols('home_run ~ (ab * batting_avg) + (ab * launch_angle_avg) + (ab * sweet_spot_percent) + '
                      '(ab * barrel_batted_rate) + (ab * flareburner_percent) + (batting_avg * launch_angle_avg)'
                      '+ (barrel_batted_rate * hard_hit_percent)',
                      data=df).fit()

print(model_multi.summary())

'''
With interactions achieved adj R2 of 0.854. Removing insignificant interactions and rerunning yields same results
'''

'''
Decided to add 'hit' back in to the model, but leaving hit type (single, double, triple) out to see how
it affects the model
'''

headers.append('hit')

header_string = ''
for index, x in enumerate(headers):
    if index == 0:
        header_string += x
    else:
        header_string += ' + ' + x

model = smf.ols('home_run ~ ' + header_string, data=df).fit()
print(model.summary())

'''
After adding hit back in, adj R2 the same at 0.812. Still have 8
significant features, with hit replacing batting average
'''

sig_features = ['ab', 'hit', 'launch_angle_avg', 'sweet_spot_percent', 'barrel_batted_rate', 'solidcontact_percent',
                'flareburner_percent', 'hard_hit_percent']

model_multi = smf.ols('home_run ~ (ab * hit) + (ab * launch_angle_avg) + (ab * sweet_spot_percent) + '
                      '(ab * barrel_batted_rate) + (ab * solidcontact_percent) + (ab * flareburner_percent) + '
                      '(ab * hard_hit_percent) + (hit * launch_angle_avg) + (hit * sweet_spot_percent) + '
                      '(hit * barrel_batted_rate) + (hit * solidcontact_percent) + (hit * flareburner_percent) + '
                      '(hit * hard_hit_percent) + (launch_angle_avg * sweet_spot_percent) + '
                      '(launch_angle_avg * barrel_batted_rate) + (launch_angle_avg * solidcontact_percent) + '
                      '(launch_angle_avg * flareburner_percent) + (launch_angle_avg * hard_hit_percent)'
                      '+ (sweet_spot_percent * barrel_batted_rate) + (sweet_spot_percent * solidcontact_percent) + '
                      '(sweet_spot_percent * flareburner_percent) + (sweet_spot_percent * hard_hit_percent)'
                      '+ (barrel_batted_rate * solidcontact_percent) + (barrel_batted_rate * flareburner_percent) + '
                      '(barrel_batted_rate * hard_hit_percent) + (solidcontact_percent * flareburner_percent) + '
                      '(solidcontact_percent * hard_hit_percent) + (flareburner_percent * hard_hit_percent)'
                      , data=df).fit()

print(model_multi.summary())

model_multi = smf.ols('home_run ~ (hit * launch_angle_avg) + (barrel_batted_rate * hard_hit_percent)',
                      data=df).fit()

print(model_multi.summary())

'''
After running with model interactions, achieved and adj R2 of 0.854, same as when batting_avg was in place of hits.
Interestingly, there were only two significant interactions in this version of the model, vs 7 in the first version
'''



'''
Forward selection up to k=7. 
'''

best = ['',0]
for p in headers:
    model  = smf.ols(formula='home_run~'+p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr1 = smf.ols(formula='home_run ~ barrel_batted_rate', data=X_train).fit()

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr1.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr2 = smf.ols(formula=train_hr1.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr2.model.formula)
train_hr2.rsquared_adj

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr2.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr3 = smf.ols(formula=train_hr2.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr3.model.formula)
train_hr3.rsquared_adj

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr3.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr4 = smf.ols(formula=train_hr3.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr4.model.formula)
train_hr4.rsquared_adj

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr4.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr5 = smf.ols(formula=train_hr4.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr5.model.formula)
train_hr5.rsquared_adj

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr5.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr6 = smf.ols(formula=train_hr5.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr6.model.formula)
train_hr6.rsquared_adj

best = ['',0]
for p in headers:
    model  = smf.ols(formula=train_hr6.model.formula+ '+' + p, data=X_train).fit()
    print(p, model.rsquared)
    if model.rsquared>best[1]:
        best = [p, model.rsquared]
print('best:',best)

train_hr7 = smf.ols(formula=train_hr6.model.formula+ '+' + best[0],data=X_train).fit()
print(train_hr7.model.formula)
train_hr7.rsquared_adj

k = [1, 2, 3, 4, 5, 6, 7]
test_hr1 = smf.ols(formula=train_hr1.model.formula, data=X_test).fit()
test_hr2 = smf.ols(formula=train_hr2.model.formula, data=X_test).fit()
test_hr3 = smf.ols(formula=train_hr3.model.formula, data=X_test).fit()
test_hr4 = smf.ols(formula=train_hr4.model.formula, data=X_test).fit()
test_hr5 = smf.ols(formula=train_hr5.model.formula, data=X_test).fit()
test_hr6 = smf.ols(formula=train_hr6.model.formula, data=X_test).fit()
test_hr7 = smf.ols(formula=train_hr7.model.formula, data=X_test).fit()

adjr2_train = [train_hr1.rsquared_adj, train_hr2.rsquared_adj, train_hr3.rsquared_adj, train_hr4.rsquared_adj,
               train_hr5.rsquared_adj, train_hr6.rsquared_adj, train_hr7.rsquared_adj]

adjr2_test = [test_hr1.rsquared_adj, test_hr2.rsquared_adj, test_hr3.rsquared_adj, test_hr4.rsquared_adj,
               test_hr5.rsquared_adj, test_hr6.rsquared_adj, test_hr7.rsquared_adj]

plt.plot(k, adjr2_train, c='red', label='Train')
plt.scatter(k, adjr2_test, c='green', label='Test')
plt.xlabel("# of Factors")
plt.ylabel("Adj Rsquared")
plt.legend()
plt.show()

'''
Adjusted R2 sharply increases for testing and training data from k=1 to k=2, but then starts to level
off quickly. k=5 is likely the best solution, as afterwards there was no meaningful increase in adj R2.
'''

'''
Predicting home runs on the test data, and calculating MSE to compare future models
'''
y_pred = test_hr5.predict(X_test)
y_true = X_test['home_run'].values
MSEmultilinear = mean_squared_error(y_true, y_pred)

Models.append('Multi Linear Regression')
Mean_Squared_Errors.append(MSEmultilinear)

'''
MSE values of 21.367 for test data
'''