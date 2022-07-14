
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

from diabetes.prep import X_train
 
parser=argparse.ArgumentParser('train')

parser.add_argument('--train',type=str,help='train')
parser.add_argument('--test',type=str,help='test')
parser.add_argument('--scaler',type=str,help='test')

args=parser.parse_args()

train=np.loadtxt(args.train+'train.txt',dtype=float)
test=np.loadtxt(args.test+'test.txt',dtype=float)

X_train=train[:,0:8]
Y_train=train[:,8]
X_test=test[:,0:8]
X_test=test[:,8]

model=LogisticRegression(max_iter=100000)
model.fit(X_train,Y_train)
 
if not os.path.isdir(args.model):
    os.mkdir(args.model)

joblib.dump(model,args.model+'/model.joblib')

