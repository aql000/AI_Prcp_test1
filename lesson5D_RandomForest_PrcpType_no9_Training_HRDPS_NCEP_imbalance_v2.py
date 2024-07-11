#! /usr/bin/env python3
# https://www.datacamp.com/tutorial/random-forests-classifier-python
#Import scikit-learn dataset library

#Load dataset
import pandas as pd
import os, glob
import joblib
import numpy as np
import pickle
import os, glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
import imblearn

data = []
filename = '/home/aql000/proj_site6/ML_precipitation_type/20221001_20230331_HRDPS_NCEP_all_no9.txt'
with open(os.path.join(os.getcwd(), filename),'r') as f:
    #      next(f)
          first_line = f.readline()
          for line in f:
            data.append(line.strip().split(','))  
f.close()        
df = pd.DataFrame(data)

    
# Import train_test_split function
index1=first_line.strip().split(',')
df.columns = index1
y = df.iloc[:,166]  #prcp type catogary
x = df.iloc[:,2:166]      #Met Features
#x = df.iloc[:, list(range(25, 40)) + [157, 158, 159, 160, 161, 162, 163, 164, 165, 167]]


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0,shuffle=False) # 70% training and 30% test
# Create a Random Forest Classifier instance
random.seed(1234)
model = RandomForestClassifier(n_estimators=300)


# Train the model using the training sets y_pred=blzd_rf_model.predict(X_test)
model.fit(X_train,y_train)


# Do a prediction using the model with the test data
y_pred=model.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy, all predictors: ",metrics.accuracy_score(y_test, y_pred))


# Save the model_RDPS, "prcp_rf_model_RDPS_v1")
# Split dataset into training set and test set
joblib.dump(model, "prcp_rf_model_HRDPS_NCEP_no9_6mth")

# Feature importance plot
feature_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
# Save feature importances to a text file
feature_importances.to_csv('feature_importances.txt', header=True, index=True)

'''
plt.figure(figsize=(10, 8))  # You can adjust the size of the figure
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.tight_layout()  # Adjust layout to make room for elements
plt.savefig('feature_importance.png')  # Save the figure to a file
plt.show()
''' and None


#Another
# Plotting
# Get the predictor names	    
index1=first_line.split(",")
index=index1[2:166]

# Get the feature (predictor) importance values for plotting
feature_imp = pd.Series(model.feature_importances_,index).sort_values(ascending=False)

# Select the top 30 most important features
top_features = feature_importances.head(30)  # Select only the top 30 features

# Save the top features to a CSV file
top_features.to_csv('top_30_features.csv', header=True, index=True)

# Print a confirmation
print("Top features saved to 'top_30_features.csv'.")

'''
# Plotting the top 30 feature importances
plt.figure(figsize=(10, 8))  # Set figure size
sns.barplot(x=top_features.values, y=top_features.index)  # Create bar plot
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Top 30 Important Features")
plt.tight_layout()  # Adjust the layout to make room for plot elements
plt.savefig('top_30_feature_importance.png')  # Save the figure to a file
plt.show()

''' and None

#build the new model with only top 30 features 
y = df.iloc[:,166]                         # prcp type category
x = df[feature_imp.index[0:30]].copy()   # selected Met Features

                                     
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42) # 70% training and 30% test

X_train.to_csv('X_train_30_features.csv', header=True, index=True)



RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# Create a Random Forest Classifier
random.seed(1234)
model_reduced = RandomForestClassifier(n_estimators=300, max_depth=3)


# Train the model using the training sets y_pred=blzd_rf_model.predict(X_test)
model_reduced.fit(X_train,y_train)


# Prediction on test set
y_pred=model_reduced.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy, reduced predictor set: ",metrics.accuracy_score(y_test, y_pred))

joblib.dump(model_reduced, "model_reduced30_HRDPS_NCEP_no9_6mth")
exit( )

# use reduced model
with open ('objs.pkl', 'rb')  as f:
    feature_imp = pickle.load(f)
prcp_rf_model_reduced = joblib.load("model_reduced30_HRDPS_NCEP_no9_6mth")
x = df[feature_imp.index[0:30]].copy() 
y_pred_reduced=prcp_rf_model_reduced.predict(x)
print("Accuracy RducedMdl:",metrics.accuracy_score(y, y_pred_reduced))

y_array=y.array
yyy=np.stack((y_array,y_pred_reduced),axis=1)
#numpy.savetxt("prcp_type_obs_pred.csv", yyy, delimiter=",")
pd.DataFrame(yyy).to_csv("prcp_type_RDPS_forecast_20221001_20230331_reduced.csv")
print("Well done!") 
#"""
