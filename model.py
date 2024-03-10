import pandas as pd
import os
from visualize import get_landmarks
import csv
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
#import fastai 
from fastai.tabular.all import *

"""
def populate_dataset(image_folder,df):
  for filename in os.listdir(image_folder):
    name, ext = os.path.splitext(filename)
    if 'Space' in name:
        name = 'Space'
    elif 'Del' in name:
        name = 'Del'
    else:
        name = name[0]
    landmarks = get_landmarks(image_folder+'/'+filename)
    if len(landmarks) < 21:
        continue 
    data = pd.DataFrame({'letter':[name], '1.X':[landmarks[0].x], '2.X':[landmarks[1].x], '3.X':[landmarks[2].x], '4.X':[landmarks[3].x],
    '5.X':[landmarks[4].x], '6.X':[landmarks[5].x], '7.X':[landmarks[6].x], '8.X':[landmarks[7].x], '9.X':[landmarks[8].x], '10.X':[landmarks[9].x],
    '11.X':[landmarks[10].x], '12.X':[landmarks[11].x], '13.X':[landmarks[12].x],'14.X':[landmarks[13].x], '15.X':[landmarks[14].x],'16.X':[landmarks[15].x],
    '17.X':[landmarks[16].x], '18.X':[landmarks[17].x], '19.X':[landmarks[18].x],'20.X':[landmarks[19].x], '21.X':[landmarks[20].x],
    '1.Y':[landmarks[0].y], '2.Y':[landmarks[1].y], '3.Y':[landmarks[2].y], '4.Y':[landmarks[3].y],
    '5.Y':[landmarks[4].y], '6.Y':[landmarks[5].y], '7.Y':[landmarks[6].y], '8.Y':[landmarks[7].y], '9.Y':[landmarks[8].y], '10.Y':[landmarks[9].y],
    '11.Y':[landmarks[10].y], '12.Y':[landmarks[11].y], '13.Y':[landmarks[12].y],'14.Y':[landmarks[13].y], '15.Y':[landmarks[14].y],'16.Y':[landmarks[15].y],
    '17.Y':[landmarks[16].y], '18.Y':[landmarks[17].y], '19.Y':[landmarks[18].y],'20.Y':[landmarks[19].y], '21.Y':[landmarks[20].y],
    '1.Z':[landmarks[0].z], '2.Z':[landmarks[1].z], '3.Z':[landmarks[2].z], '4.Z':[landmarks[3].z],
    '5.Z':[landmarks[4].z], '6.Z':[landmarks[5].z], '7.Z':[landmarks[6].z], '8.Z':[landmarks[7].z], '9.Z':[landmarks[8].z], '10.Z':[landmarks[9].z],
    '11.Z':[landmarks[10].z], '12.Z':[landmarks[11].z], '13.Z':[landmarks[12].z],'14.Z':[landmarks[13].z], '15.Z':[landmarks[14].z],'16.Z':[landmarks[15].z],
    '17.Z':[landmarks[16].z], '18.Z':[landmarks[17].z], '19.Z':[landmarks[18].z],'20.Z':[landmarks[19].z], '21.Z':[landmarks[20].z]})

    df = pd.concat([data,df.loc[:]]).reset_index(drop=True)
  return df

df = pd.read_csv('data/train.csv')
df1 = populate_dataset('data/train_images/A',df)
df2 = populate_dataset('data/train_images/B',df)
df3 = populate_dataset('data/train_images/C',df)
df4 = populate_dataset('data/train_images/D',df)
df5 = populate_dataset('data/train_images/E',df)
df6 = populate_dataset('data/train_images/F',df)
df7 = populate_dataset('data/train_images/G',df)
df8 = populate_dataset('data/train_images/H',df)
df9 = populate_dataset('data/train_images/I',df)
df10 = populate_dataset('data/train_images/J',df)
df11 = populate_dataset('data/train_images/K',df)
df12 = populate_dataset('data/train_images/L',df)
df13 = populate_dataset('data/train_images/M',df)
df14 = populate_dataset('data/train_images/N',df)
df15 = populate_dataset('data/train_images/O',df)
df16 = populate_dataset('data/train_images/P',df)
df17 = populate_dataset('data/train_images/Q',df)
df18 = populate_dataset('data/train_images/R',df)
df19 = populate_dataset('data/train_images/S',df)
df20 = populate_dataset('data/train_images/T',df)
df21 = populate_dataset('data/train_images/U',df)
df22 = populate_dataset('data/train_images/V',df)
df23 = populate_dataset('data/train_images/W',df)
df24 = populate_dataset('data/train_images/X',df)
df25 = populate_dataset('data/train_images/Y',df)
df26 = populate_dataset('data/train_images/Z',df)
df27 = populate_dataset('data/train_images/Space',df)
df28 = populate_dataset('data/train_images/Del',df)


dataset = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28])
dataset.to_csv('dataset.csv',index=False)

"""

df = pd.read_csv('dataset.csv')
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(df, procs=[Normalize],
                   cat_names = [],
                   cont_names = ['1.X', '2.X', '3.X','4.X','5.X','6.X','7.X','8.X','9.X','10.X','11.X','12.X','13.X','14.X','15.X','16.X','17.X','18.X','19.X','20.X','21.X',
                                    '1.Y', '2.Y', '3.Y','4.Y','5.Y','6.Y','7.Y','8.Y','9.Y','10.Y','11.Y','12.Y','13.Y','14.Y','15.Y','16.Y','17.Y','18.Y','19.Y','20.Y','21.Y',
                                    '1.Z', '2.Z', '3.Z','4.Z','5.Z','6.Z','7.Z','8.Z','9.Z','10.Z','11.Z','12.Z','13.Z','14.Z','15.Z','16.Z','17.Z','18.Z','19.Z','20.Z','21.Z'
                                ],
                   y_names='letter',
                   splits=splits)

dls = to.dataloaders(bs = 64,valid_pct = 0.7)

X_train, y_train = dls.train.xs, dls.train.ys.values.ravel()
X_test, y_test = dls.valid.xs, dls.valid.ys.values.ravel()

#print(X_train.head())
#print(y_train[:5])
#print(df.head())

#dataset = pd.read_csv('dataset.csv')
#X = dataset.drop('letter',axis = 1)   
#y = dataset['letter']

#label_map = {'A': 0, 'B': 1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'H': 7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V':21, 'W':22, 'X':23, 'Y':24, 'Z': 25, 's':26, 'd':27}
#y = dataset['letter'].map(label_map)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.7, stratify= y, shuffle= True) 


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))
print('  Params: ')

for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


params = trial.params
model = XGBClassifier(**params)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

model.save_model('model.json')
