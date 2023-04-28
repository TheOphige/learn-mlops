# imports
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix

# # load in  test data
print('loading in  test data...')
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetes']
test_set = pd.read_csv("TestSet.csv", header=None, names=col_names)
test_set.head()

X_test= test_set.drop('diabetes', axis=1).values
y_test= test_set['diabetes'].copy().values


# handle missing data and scale
print('Handling missing data and scaling...')
pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

X_test_prep = pipeline.fit_transform(X_test)

# load model
print('loading model...')
log_reg = joblib.load('final_model.pkl')

# predict test data
print('Predicting test data...')
y_pred = log_reg.predict(X_test_prep)
accuracy= accuracy_score(y_test, y_pred)
c_matrix = confusion_matrix(y_test, y_pred)

print('The validation accuracy is', accuracy*100)
print(c_matrix)

# print the first 30 true and predicted responses
print('Printing the first 30 true and predicted responses...')
print('True:', y_test[0:30])
print('Pred:', y_pred[0:30])