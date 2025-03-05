
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
38 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import  RandomForestClassifier 
from sklearn.svm import SVC 
import joblib 
data=pd.read_csv("diabetes.csv")   # Reading Dataset 
corr=data.corr(method='pearson')   # Checking Correlation 
diabetes_positive_count = len(data.loc[data['Outcome'] == 1]) 
diabetes_negative_count = len(data.loc[data['Outcome'] == 0]) 
print("total positve count:{0} and total negative 
count:{1}".format(diabetes_positive_count,diabetes_negative_count)) 
cmap=sns.diverging_palette(220,10,as_cmap=True) 
sns.heatmap(corr,cmap=cmap,vmax=.3,square=True,linewidths=6,cbar_kws={"shrink":.5}) 
colormap=plt.cm.viridis 
plt.figure(figsize=(12,12)) 
plt.title('Pearson Correlation of Features', y=1.05, size=15) 
sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 
linecolor='white',annot=True) 
x=data.iloc[:,0:-1] 
y=data.iloc[:,-1] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
sc = StandardScaler() 
x_train=sc.fit_transform(x_train) 
x_test=sc.fit_transform(x_test) 
log = LogisticRegression() 
39 
log.fit(x_train, y_train)   
y_pred=log.predict(x_test) 
print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred)) 
# Saving Diabetes predct Train Model 
filename = 'Diabetes-pred_model.sav' 
joblib.dump(log, filename) 
# load the model from disk 
loaded_model = joblib.load(filename) 
# Use the loaded model to make predictions 
loaded_model.predict(x_test) 