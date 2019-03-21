# -*- coding: utf-8 -*-

# %% import of packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

# %% import data
x_train=pd.read_json("NEW TRAINING DATA - August 2017 to September 2018.json", lines=False, orient='columns')
x_test=pd.read_json("NEW TEST DATA - October 2018 to March 2019.json", lines=False, orient='columns')

# %% Transform data
x_train['result']=x_train['result'].replace(['np',5],0)
x_train['age']=x_train['age'].fillna((x_train['age'].mean()))
x_train['humidity']=x_train['humidity'].fillna(method='bfill')
x_train['jockeyClaim']=x_train['jockeyClaim'].interpolate()
x_train['rating']=x_train['rating'].fillna((x_train['rating'].mean()))
x_train['temperature']=x_train['temperature'].fillna(method='bfill')
x_train=x_train[pd.notnull(x_train['sex'])]
x_train['trackCondition']=x_train['trackCondition'].fillna('GOOD')
x_train['trackRating']=x_train['trackRating'].fillna((x_train['trackRating'].mean()))
x_train['weather']=x_train['weather'].fillna(method='bfill')
x_train['windspeed']=x_train['windspeed'].fillna((x_train['windspeed'].mean()))

# %% Encode data
x_train.drop(['horseName','jockeyName','trainerName','careerWin', 'deadPlace', 'deadWin','fastPlace', 'fastWin', 'heavyPlace', 'heavyWin','horseLastMonth', 
'horseTrend','jockeyClaim','slowPlace','trackCondition','venueDistancePlace','venueDistanceWin','bettingOdds'],axis=1,inplace=True)



lb = LabelEncoder()
x_train["class"] = lb.fit_transform(x_train["class"])
x_train["sex"] = lb.fit_transform(x_train["sex"])
#x_train["trackCondition"] = lb.fit_transform(x_train["trackCondition"])
x_train["venue"] = lb.fit_transform(x_train["venue"])
x_train["weather"] = lb.fit_transform(x_train["weather"])

cols = x_train.columns[x_train.dtypes.eq('object')]
x_train[cols] = x_train[cols].apply(pd.to_numeric, errors='coerce')
y_train=x_train['result']
x_train.drop(['result','date'],axis=1,inplace=True)

# acount for imbalances
os=SMOTE(random_state=15)
x_train_os, y_train_os=os.fit_sample(x_train, y_train)

scaler = MinMaxScaler(feature_range=(0, 1))
x_train_os = scaler.fit_transform(x_train_os)

# %% Calibrate Model
"""To retrain the model on new training data kindly remove "#" from all below lines of code"""
model= lgb.LGBMClassifier(learning_rate=0.1,num_leaves=21,n_estimators=500,min_child_samples=10,min_data_in_leaf=15,boosting_type='dart',verbosity=3,random_state=15)
model.fit(x_train_os,y_train_os)

"""saving model To save new trained model kindly remove "#" from below line of code"""
model.booster_.save_model('horse_race_prediction.txt')

# %% Transform test data in the same fashion as train data
x_test['result']=x_test['result'].replace(['np',5],0)
x_test['age']=x_test['age'].fillna((x_test['age'].mean()))
x_test['humidity']=x_test['humidity'].fillna(method='bfill')
x_test['jockeyClaim']=x_test['jockeyClaim'].interpolate()
x_test['rating']=x_test['rating'].fillna((x_test['rating'].mean()))
x_test['temperature']=x_test['temperature'].fillna(method='bfill')
x_test=x_test[pd.notnull(x_test['sex'])]
x_test['trackCondition']=x_test['trackCondition'].fillna('GOOD')
x_test['trackRating']=x_test['trackRating'].fillna((x_test['trackRating'].mean()))
x_test['weather']=x_test['weather'].fillna(method='bfill')
x_test['windspeed']=x_test['windspeed'].fillna((x_test['windspeed'].mean()))

x_test.drop(['horseName','jockeyName','trainerName','careerWin', 'deadPlace', 'deadWin','fastPlace', 'fastWin', 'heavyPlace', 'heavyWin','horseLastMonth', 
'horseTrend','jockeyClaim','slowPlace','trackCondition','venueDistancePlace','venueDistanceWin','bettingOdds'],axis=1,inplace=True)

x_test["class"] = lb.fit_transform(x_test["class"])
x_test["sex"] = lb.fit_transform(x_test["sex"])
#x_test["trackCondition"] = lb.fit_transform(x_test["trackCondition"])
x_test["venue"] = lb.fit_transform(x_test["venue"])
x_test["weather"] = lb.fit_transform(x_test["weather"])

cols = x_test.columns[x_test.dtypes.eq('object')]
x_test[cols] = x_test[cols].apply(pd.to_numeric, errors='coerce')
y_test=x_test['result']
x_test_RI=x_test['runnerID']
x_test.drop(['result','date'],axis=1,inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
x_test = scaler.fit_transform(x_test)

# %% Evaluate Model
"""loading saved model"""
clf = lgb.Booster(model_file='horse_race_prediction.txt')


train_pred=clf.predict(x_train_os)
# why 5 probabilities?
train_preds_os=np.argmax(train_pred,axis=1)

test_pred=clf.predict(x_test)
test_preds_os=np.argmax(test_pred, axis=1)

# y_test - actual rank, train_preds_os - predicted rank
train_accuracy_os = accuracy_score(y_train_os,train_preds_os)
test_accuracy_os = accuracy_score(y_test, test_preds_os)
print('Train accuracy :', train_accuracy_os)
print('Test accuracy :', test_accuracy_os)
train_precision_os = precision_score(y_train_os,train_preds_os,average=None)
test_precision_os = precision_score(y_test, test_preds_os, average=None)
print('Train precision for class 1 :', train_precision_os[1])
print('Test precision for class 1 :', test_precision_os[1])
print("")
print("Confusion Matrix_train_os")
print('-------------------------------')
print(pd.crosstab(y_train_os,train_preds_os,rownames=['True'], colnames=['Predicted'], margins=True))
print("Confusion Matrix_test_os")
print('-------------------------------')
print(pd.crosstab(y_test,test_preds_os,rownames=['True'], colnames=['Predicted'], margins=True))
print("Classification Report_train_os")
print('-------------------------------')
print(classification_report(y_train_os, train_preds_os))
print("Classification Report_test_os")
print('-------------------------------')
print(classification_report(y_test, test_preds_os))



predictions_prob = clf.predict(x_test)
predictions_df = pd.DataFrame({'id':x_test_RI,
                               'True Position' : y_test, 'Predicted Position' : test_preds_os, 
                               'Probablilty of Out of any Position (Likelihood)' : predictions_prob[:,0]*100, 
                               'Probablilty of First Position (Likelihood)' : predictions_prob[:,1]*100,
                              'Probablilty of Second Position  (Likelihood)' : predictions_prob[:,2]*100, 
                               'Probablilty of Third Position  (Likelihood)' : predictions_prob[:,3]*100,
                              'Probablilty of Fourth Position  (Likelihood)' : predictions_prob[:,4]*100})

#Relevant METRIC 
print('Probability to predict the first place right:',
      len(predictions_df[(predictions_df['True Position']==1) & (predictions_df['Predicted Position']==1)])/len(predictions_df[predictions_df['True Position']==1]))

#saving results in csv format
predictions_df.to_csv("result.csv", index=False)

#saving results in json format
result=predictions_df.to_json(orient='records')
with open('results.json', 'w') as f:
    f.write(result)
