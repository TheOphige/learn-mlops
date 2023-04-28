# import modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix

# import data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetes']
df = pd.read_csv("pima-indians-diabetes.data", header=None, names=col_names)

# ## Train, val, Test split
from sklearn.model_selection import train_test_split
X= df.drop('diabetes', axis=1).values
y= df['diabetes'].copy().values
X_train_1, X_test, y_train_1, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_1, y_train_1, test_size=0.2, random_state=42)

# ### saving test set for later use
x_df= pd.DataFrame(X_test) 
y_df= pd.DataFrame(y_test)

test_set = pd.concat([x_df, y_df], axis=1)
test_set.to_csv('TestSet.csv', header= False)

# Data preparation
# ## Handling Missing Values and scaling
print('Handling Missing Values and scaling...')

pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=0,strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

X_train_prep = pipeline.fit_transform(X_train)
X_val_prep = pipeline.fit_transform(X_val)


# ## create models
print('creating models....')
models = [('log_reg', LogisticRegression())]
results= []
names= []
evale= []
cme= []
for name, model in models:
    model.fit(X_train_prep, y_train)
    joblib.dump(model, name +'.pkl')
    accuracy= model.score(X_train_prep, y_train)*100
    val_accuracy= model.score(X_val_prep, y_val)*100
    cv= cross_val_score(model, X_train_prep, y_train,
                            scoring='accuracy', cv=10)     
    y_pred = model.predict(X_val_prep)
    pre= precision_score(y_val, y_pred)
    recall= recall_score(y_val, y_pred)
    y_scores = model.predict_proba(X_val_prep)[:, 1]
    auc= roc_auc_score(y_val, y_scores)    
    results.append([accuracy, pre, recall, auc, cv])
    names.append(name)
    eval= "%s:\t%f\t%f\t%f\t%f\t%f\t%f (%f)" % (name, accuracy, val_accuracy, pre*100, recall*100, auc*100, cv.mean()*100, cv.std()*100)
    evale.append(eval)
    y_pred = model.predict(X_val_prep)
    c_matrix = confusion_matrix(y_val, y_pred)
    cme.append([name, c_matrix])

print("""\t\t training\t validating
NAME\t\t ACCURACY\tACCURACY\tPRECISION\tRECALL\t\t   AUC\t\t CV_MEAN (CV_STD)""") 
print('='*117)
for eval in evale:
    print(eval)
    print('='*117)

for cm in cme:
    print(cm)
    print('='*75)


# # validation
print('validating...')
log_reg = joblib.load('log_reg.pkl')
y_pred = log_reg.predict(X_val_prep)
accuracy= accuracy_score(y_val, y_pred)
c_matrix = confusion_matrix(y_val, y_pred)
print('The validation accuracy is', accuracy*100)
print(c_matrix)

# print the first 30 true and predicted responses
print('Printing the first 30 true and predicted responses...')
print('True:', y_val[0:30])
print('Pred:', y_pred[0:30])


# # Optimization

# ## Hyperparameter tuning for log_reg
from sklearn.model_selection import GridSearchCV
param_grid = [{'penalty': ['l2'],
                     'C': np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(log_reg, param_grid, cv=5,
                       scoring='precision_macro' )
grid_search.fit(X_train_prep, y_train)

print(' Here is the score, parameters of the best model')
print('score:', grid_search.best_score_)
print('Parameters:', grid_search.best_params_)
print('Best model:', grid_search.best_estimator_)

y_pred = grid_search.best_estimator_.predict(X_val_prep)
print('Classification Report:')
print(classification_report(y_val, y_pred))
y_scores = grid_search.best_estimator_.predict_proba(X_val_prep)[:, 1]
print('ROC AUC score:',roc_auc_score(y_val,y_scores))


# # save model
print('Saving the final model...')
final_model= grid_search.best_estimator_
joblib.dump(final_model, 'final_model.pkl')
print('Final model', final_model, 'has been saved. Done :)')
