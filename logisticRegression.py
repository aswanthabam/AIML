#Load the dataset
#import pandas
import pandas as pd
# load dataset
pima = pd.read_csv("/content/Pima.csv") # open

pima.head()
print(pima.shape) # no of rows and columns
pima.describe() # describe

#Selecting Feature
#split dataset in features and target variable
target = ['Outcome']
features = list(set(list(pima.columns))-set(target))  
X = pima[features].values # Features
y = pima[target].values # Target variable
pima[features] = pima[features]/pima[features].max()
pima.describe()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40, random_state=40)
print(X_train.shape) # Splitted Training data
print(X_test.shape) # Splitted Test Data

# Model Development and Prediction
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix

from sklearn.metrics import classification_report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred,target_names=target_names))

#Area Under the Curve
import matplotlib.pyplot as plt
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#AUC score for the case is 0.87.
#AUC score 1 represents a perfect classifier, and 0.5 represents a worthless classifier
