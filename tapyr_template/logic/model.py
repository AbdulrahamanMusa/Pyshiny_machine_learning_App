import sqlite3
import pandas as pd
import re
import numpy as np
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,f1_score


models = {
  
'Logistic Regression': LogisticRegression(),
'Support Vector Machine': SVC(kernel='rbf', gamma='auto'),
'Decision Tree': DecisionTreeClassifier(),
'Random Forest': RandomForestClassifier(),
'Gradient Boosting': GradientBoostingClassifier()
}

Stat_model=["OLS","WLS","GLS","GLSAR","RecursiveLS"]


# Create SQLite database connection
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Select data from the table
query = "SELECT * FROM patient"
df = pd.read_sql_query(query, conn)

# Remove NaN values and update the original DataFrame (in-place)
df.dropna(inplace=True)

# Close the connection (optional - good practice)
conn.close()
# Now df should not contain NaN values

df=df[["PatientID", "Gender", "Age", "AgeCat", "Occupation", "Ethnicity", "Education", "Height", "Weight", "BMI", 
                    "BP_sytolic", "BP_Distolic", "BP_category", 
                    "Pusle_rate", "Temperature", "Diet_compliance", "Smoking_status", 
                    "Alcohol_consumption", "Physical_activity", 
                    "Consultation_fee"]]

# print(df)

# Assigned code number to represent body mass index 
def bmi_category(bmi):
  if bmi <= 18.5:
    return 0
  elif 18.5 < bmi <= 24.9:
    return 0
  elif 25.0 <= bmi < 30.0: 
    return 1
  else:
    return 1
# Add a new column 'BMI_code' with categories
df = df.assign(BMI_code=df['BMI'].apply(bmi_category))


# # Assigned code number to represent Blood pressure
def bp_category(bp):
  if bp == 'Normal':
    return 0
  elif bp == 'Elevated':
   return 0    
  elif bp == 'Stage 1 hypertension':  # Double equals for comparison
    return 1
  else:
    return 2  # Assuming 'Stage 2 hypertension' maps to 2

# Add a new column 'BP_code' with categories
df = df.assign(BP_code=df['BP_category'].apply(bp_category))

# Data Mapping/Preprocessing
gender_mapping = {"Male": 0, "Female": 1}
agecat_mapping = {"5-14yrs": 0, "15-24yrs": 1, "25-34yrs": 2, "35-44yrs": 3, "45-54yrs": 4, "55yers and above": 5}
occupation_mapping = {"Unemployed": 0, "Business": 1, "Farmer": 2, "Student": 3, "Civil servant": 4, "Tailor": 5, "Carpenter": 6, "Engineer": 7, "Others": 8}
education_mapping = {"Non-formal": 0, "Primary": 1, "Secondary": 1, "Tertiary": 3}
bp_mapping = {'Normal': 0, 'Elevated': 0, 'Stage 1 hypertension': 1, 'Stage 2 hypertension': 1}
diet_compliance_mapping = {"No": 0, "Yes": 1}
smoking_status_mapping = {"No": 0, "Yes": 1}
alcohol_consumption_mapping = {"No": 0, "Yes": 1}
physical_activity_mapping = {"No": 0, "Yes": 1}
df.loc[:,'BMI_code'] = df['BMI'].apply(bmi_category)

# Asigning code numbers to categorical varibales 
df.loc[:,"PatientID"] = df["PatientID"]
df.loc[:,"Gender_code"] = df["Gender"].map(gender_mapping)
df.loc[:,"AgeCat_code"] = df["AgeCat"].map(agecat_mapping)
df.loc[:,"Occupation_code"] = df["Occupation"].map(occupation_mapping)
df.loc[:,"Education_code"] = df["Education"].map(education_mapping)
df.loc[:,'BMI_code'] = df['BMI'].apply(bmi_category)
df.loc[:,"BP_code"] = df["BP_category"].map(bp_mapping)  # Assuming BP_category stores class labels
df.loc[:,"Diet_compliance_code"] = df["Diet_compliance"].map(diet_compliance_mapping)
df.loc[:,"Smoking_status_code"] = df["Smoking_status"].map(smoking_status_mapping)
df.loc[:,"Alcohol_consumption_code"] = df["Alcohol_consumption"].map(alcohol_consumption_mapping)
df.loc[:,"Physical_activity_code"] = df["Physical_activity"].map(physical_activity_mapping)

# Displayed the coded data 
feature_df=df[["PatientID","Gender_code", "AgeCat_code", "Occupation_code", 
          "Education_code", "Diet_compliance_code", "Smoking_status_code",
          "Alcohol_consumption_code", "Physical_activity_code"]]
feature_df=feature_df.dropna()
# print(feature_df)

# Preprocess the data
# Exclude 'Unnamed: 0' column
X = feature_df.drop(['PatientID'], axis=1)
X=np.asarray(X)
# X[0:5]

# Class for Dibetic 
y_DM = df['BMI_code'] = df['BMI_code'].astype('int')
y_DM=np.asarray(df['BMI_code'])
# y_DM[0:5]

# Train and test split for Dibetic
X_train_DM, X_test_DM, y_train_DM, y_test_DM = train_test_split(X, y_DM, test_size=0.30, random_state=21)
print ('Train set:', X_train_DM.shape,  y_train_DM.shape)
print ('Test set:', X_test_DM.shape,  y_test_DM.shape)

for name, model in models.items():
  model.fit(X_train_DM, y_train_DM)  # Train each model in the dictionary

# clf =SVC(kernel='rbf')
# print(clf.fit(X_train, y_train))

# yhat = clf.predict(X_test)
# yhat [0:5]


#==============================================
# Class for  Hypertension
y_BP = df['BP_code'] = df['BP_code'].astype('int')
y_BP= np.asarray(df['BP_code'])
# y_BP[0:5]

# Train and test split for hypertension
X_train_BP, X_test_BP, y_train_BP, y_test_BP = train_test_split(X, y_BP, test_size=0.30, random_state=21)
print ('Train set:', X_train_BP.shape,  y_train_BP.shape)
print ('Test set:', X_test_BP.shape,  y_test_BP.shape)

for name, model in models.items():
  model.fit(X_train_BP, y_train_BP)  # Train each model in the dictionary

# clf =SVC(kernel='rbf')
# print(clf.fit(X_train, y_train))

# yhat = clf.predict(X_test)
# yhat [0:5]
