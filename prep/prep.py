import argparse
from azureml.core import Run
from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
 
run=Run.get_context()

parser=argparse.ArgumentParser('prep')

parser.add_argument('--train',type=str,help='train')
parser.add_argument('--test',type=str,help='test')
parser.add_argument('--scaler',type=str,help='test')

args=parser.parse_args()

# filename="pima-indians-diabetes.csv"
# names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Class']
# Data=read_csv(filename,names=names)

dataframe=run.input_datasets["raw_data"].to_pandas_dataframe()
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

# print(X[0])

scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(X)

test_size=0.33
seed=7

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size, random_state=seed)

train=np.column_stack((X_train,Y_train))
test=np.column_stack((X_test,Y_test))

os.makedirs(args.train,exist_ok=True)
os.makedirs(args.test,exist_ok=True)

np.savetxt(args.train+'train.txt',train,fmt='%f')
np.savetxt(args.test+'test.txt',train,fmt='%f')

# model=LogisticRegression(max_iter=100000)
# model.fit(X_train,Y_train)
 
if not os.path.isdir(args.scaler):
    os.mkdir(args.scaler)
# result=model.score(X_test,Y_test)

joblib.dump(scaler,args.model+'/scaler.joblib')
# x_sample=np.array([7,187,68,39,304,37.7,0.254,41]).reshape(1,-1)
# x_pred=scaler.transform(x_sample)
# pred=model.predict(x_sample)
