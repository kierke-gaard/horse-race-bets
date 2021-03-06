# -*- coding: utf-8 -*-

# %% 
# IMPORT PACKAGES
from functools import partial, reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import datetime as dt

pd.set_option('display.max_rows', 100)
pd.reset_option('display.max_columns', None)

# CONFIGURATION
n_estimators = 40
univariate_statistics = False
load_model_and_not_fit = False
start_time = dt.datetime.now().replace(microsecond=0)

# IMPORT DATA
test_raw = pd.read_json(r"data/11am Test Data - October 2018 to April 2019.json",
                          lines=False, orient='columns')
train_raw = pd.read_json(r"data/11am Training Data - August 2017 to September 2018.json",
                         lines=False, orient='columns')
cols = set(train_raw.columns.values)

# %% Univariate Statistics
def attribute_characteristics_in_test_set(df_train: pd.DataFrame,  df_test: pd.DataFrame):
      def occurence_prob(df1, df2, attribute):   
            return sum(df2[attribute].isin(df1[attribute]))/len(df2)
      print('Probability that attribute characteristics in the training set occur also in the test set.')
      categorical_cols = df_train.dtypes[df_train.dtypes == 'object'].index
      return pd.Series(map(partial(occurence_prob, df_train, df_test), categorical_cols),
                        categorical_cols)

def histograms_of_frequencies(df: pd.DataFrame):
      categorical_cols = df.dtypes[df.dtypes == 'object'].index
      for c in categorical_cols:
            df[c].value_counts().plot(title=c);
            plt.pause(1)

if univariate_statistics:
      histograms_of_frequencies(train_raw)
      attribute_characteristics_in_test_set(train_raw, test_raw)


# %% Transform data
cols_exclude = \
            [#'horseName', 
            #  'jockeyName', 
            #   'trainerName',
            #   'careerWin',
              'runnerID',
            #   'deadPlace',
            #   'deadWin',
            #   'fastPlace',
            #   'fastWin',
            #   'heavyPlace',
            #   'heavyWin',
            #   'horseLastMonth', 
            #   'horseTrend',
            #   'jockeyClaim',
            #   'slowPlace',
            #   'trackCondition',
            #   'venueDistancePlace',
            #   'venueDistanceWin',
            #   'bettingOdds',
              'result',
              'date' # include season, month, weekday
              ]

def preprocess(x, cols_exclude):
      # fill na
      x['age']=x['age'].fillna((x['age'].mean()))
      x['humidity']=x['humidity'].fillna(method='bfill') 
      x['jockeyClaim']=x['jockeyClaim'].interpolate() 
      x['rating']=x['rating'].fillna((x['rating'].mean())) 
      x['temperature']=x['temperature'].fillna(method='bfill') 
      x=x[pd.notnull(x['sex'])]
      x['trackCondition']=x['trackCondition'].fillna('GOOD')
      x['trackRating']=x['trackRating'].fillna((x['trackRating'].mean()))
      x['weather']=x['weather'].fillna(method='bfill')
      x['windspeed']=x['windspeed'].fillna((x['windspeed'].mean()))
      x['bettingOds']=x['bettingOdds'].replace(0.,np.nan)

      # encode categories with numbers
      x['result']=x['result'].replace(['np',5,4,3,2],0)
      lb = LabelEncoder()
      x["class"] = lb.fit_transform(x["class"])
      x["sex"] = lb.fit_transform(x["sex"])
      #x_train["trackCondition"] = lb.fit_transform(x_train["trackCondition"])
      x["venue"] = lb.fit_transform(x["venue"])
      x["weather"] = lb.fit_transform(x["weather"])
      cols = x.columns[x.dtypes.eq('object')]
      x[cols] = x[cols].apply(pd.to_numeric, errors='coerce')
      
      # feature selection
      y = x['result']
      x.drop(cols_exclude, axis=1, inplace=True)

      return x, y

x_train, y_train = preprocess(train_raw, cols_exclude)
x_test, y_test = preprocess(test_raw, cols_exclude)

# %% Calibrate Model
if load_model_and_not_fit:
      model = lgb.Booster(model_file='horse_race_prediction.txt')
else:
      model = lgb.LGBMClassifier(learning_rate=0.1,
                              num_leaves=21,
                              n_estimators=n_estimators,
                              min_child_samples=10,
                              min_data_in_leaf=15,
                              boosting_type='gbdt',
                              verbosity=3,
                              random_state=15)
      model.fit(x_train, y_train)
      model.booster_.save_model('horse_race_prediction.txt')

# %% Evaluate Model
def evaluate(model, x, y_true):
      y_pred = model.predict(x)
      s = 'Accuracy: ' + str(accuracy_score(y_true, y_pred)) \
            + '\nPrecision for first place: ' + str(precision_score(y_true, y_pred, average=None)[1]) \
            + '\n\nConfusion Matrix\n-------------------------------\n' \
            +  pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).to_string() \
            + '\n\nClassification Report\n-------------------------------\n' \
            + classification_report(y_true, y_pred)
      return s

report = '\n--- OUT OF SAMPLE EVALUATION ------------------------\n' \
       + evaluate(model, x_test, y_test) \
       + '--- IN SAMPLE EVALUATION ------------------------------\n' \
       + evaluate(model, x_train, y_train) \

# %% Print and Save Evaluation Report
running_time = 'Running Time: ' + str(dt.datetime.now().replace(microsecond=0) - start_time) 
print(running_time + '\n' + report)
with open('results.txt', 'w') as f:
      f.write('Date: ' + str(dt.datetime.now()) + '\n')
      f.write(running_time + '\n')
      f.write('Experimental Setup:\n')
      f.write('Excluded Cols' + str(cols_exclude) + '\n\n')
      f.write('Model' + model.__doc__ + '\n')
      f.write('Model parameters' + str(model.get_params()) +'\n\n')
      f.write(report)
