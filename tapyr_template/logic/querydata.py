from pathlib import Path
import pandas as pd
import sqlite3
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
# Select data from the table
query = "SELECT * FROM patient"
df = pd.read_sql_query(query, conn)


# # Load the dataset from the CSV file
# df = pd.read_csv(Path(__file__).parent / "data.csv", na_values="NA")

total_patients = df.shape[0]  # Total number of patients
total_males = df[df["Gender"] == "Male"].shape[0]  # Total number of male patients
total_females = df[df["Gender"] == "Female"].shape[0]  # Total number of female patients
# df_bp =df[df["BP_category"].shape[0]]
# Define the list of patient symptoms
patient_symptoms = ["Fever", "Cough", "Headache"]  # list of symptoms


# cursor.execute("SELECT COUNT(*) as Children_U5 FROM patient where Age < 5")
# Children_U5 = cursor.fetchone()[0]

cursor.execute("SELECT id, BP_category FROM patient")
Bp = cursor.fetchall()
df_bp = pd.DataFrame(Bp, columns=['id', 'BP_category'])

        
cursor.execute("SELECT id, Consultation_fee, Visit_date FROM patient")
payment = cursor.fetchall()
# Define the query using f-strings for multi-line representation
query = f"""
SELECT SUM(Consultation_fee) AS total,
       Round(AVG(Consultation_fee),2) AS Average,
       MIN(Consultation_fee) AS min,
       MAX(Consultation_fee) AS max
FROM patient
"""
statistics = pd.read_sql_query(query, conn)

#----------------ENCODINGS
query = "SELECT * FROM patient"
data = pd.read_sql_query(query, conn)
# Remove NaN values and update the original DataFrame (in-place)
data.dropna(inplace=True)

Explordata=data[["PatientID", "Gender", "Age", "AgeCat", "Occupation", "Ethnicity", "Education", "Height", "Weight", "BMI", 
                    "BP_sytolic", "BP_Distolic", "BP_category", 
                    "Pusle_rate", "Temperature", "Diet_compliance", "Smoking_status", 
                    "Alcohol_consumption", "Physical_activity", 
                    "Consultation_fee"]]

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
data = data.assign(BMI_code=data['BMI'].apply(bmi_category))

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
data = data.assign(BP_code=data['BP_category'].apply(bp_category))


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
data.loc[:,'BMI_code'] = data['BMI'].apply(bmi_category)


# Asigning code numbers to categorical varibales 
data.loc[:,"PatientID"] = data["PatientID"]
data.loc[:,"Gender_code"] = data["Gender"].map(gender_mapping)
data.loc[:,"AgeCat_code"] = data["AgeCat"].map(agecat_mapping)
data.loc[:,"Occupation_code"] = data["Occupation"].map(occupation_mapping)
data.loc[:,"Education_code"] = data["Education"].map(education_mapping)
data.loc[:,'BMI_code'] = data['BMI'].apply(bmi_category)
data.loc[:,"BP_code"] = data["BP_category"].map(bp_mapping)  # Assuming BP_category stores class labels
data.loc[:,"Diet_compliance_code"] = data["Diet_compliance"].map(diet_compliance_mapping)
data.loc[:,"Smoking_status_code"] = data["Smoking_status"].map(smoking_status_mapping)
data.loc[:,"Alcohol_consumption_code"] = data["Alcohol_consumption"].map(alcohol_consumption_mapping)
data.loc[:,"Physical_activity_code"] = data["Physical_activity"].map(physical_activity_mapping)

# Displayed the coded data 
codededata=data[["PatientID", "Gender_code", "AgeCat_code", "Occupation_code", 
          "Education_code", "BMI", "BMI_code", "Temperature","Age", "BP_code",
          "Diet_compliance_code", "Smoking_status_code",
          "Alcohol_consumption_code", "Physical_activity_code"]]
codededata=codededata.dropna()

choices_column=["Gender_code","AgeCat_code", "Occupation_code", 
          "Education_code", "BMI", "BMI_code", "Temperature","Age", "BP_code",
          "Diet_compliance_code", "Smoking_status_code",
          "Alcohol_consumption_code", "Physical_activity_code","Pusle_rate","Consultation_fee"]
