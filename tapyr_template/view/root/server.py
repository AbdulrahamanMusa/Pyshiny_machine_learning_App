from shiny import Inputs, Outputs, Session, ui, reactive, render
import sqlite3
import datetime
import pandas as pd
from itables.shiny import DT
from itables import show
from openpyxl import Workbook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plotnine import aes, geom_point, ggplot
import plotly.express as px
from shinywidgets import render_widget
from tapyr_template.logic.about import model_infor_pop
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

from tapyr_template.logic.querydata import total_patients,total_females,total_males, patient_symptoms,df



from tapyr_template.logic.model import models,X_train_DM, X_test_DM, y_train_DM, y_test_DM, feature_df
from tapyr_template.logic.model import X_train_BP, X_test_BP, y_train_BP, y_test_BP
from tapyr_template.logic.utils import contact_us, categorize_blood_pressure
from tapyr_template.logic.querydata import Explordata,codededata,df
# Create SQLite database connection
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

def server(input: Inputs, output: Outputs, session: Session):
    model_infor_pop()
    @reactive.Effect
    @reactive.event(input.submit)
    def _():
        m = ui.modal(
            "Data submitted successfully!",
            title="Notification",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    @reactive.Effect
    @reactive.event(input.submit)
    def submit():
        if input.submit():
            PatientID = input.PatientID()
            FirstName=input.FirstName()
            LastName=input.LastName() 
            Gender=input.Gender()
            Age=input.Age()
            AgeCat=input.AgeCat()
            Occupation=input.Occupation()
            Ethnicity=input.Ethnicity()
            Education=input.Education()
            Height=input.Height() 
            Weight=input.Weight()
            BMI=input.BMI() 
            BP_sytolic=input.BP_sytolic()
            BP_Distolic=input.BP_Distolic()
            BP_category=input.BP_category()
            Pusle_rate=input.Pusle_rate()
            Temperature=input.Temperature()
            Diet_compliance=input.Diet_compliance()
            Smoking_status=input.Smoking_status()
            Alcohol_consumption=input.Alcohol_consumption()
            Physical_activity=input.Physical_activity()
            Symptoms = str(input.Symptoms())
            Complain=input.Complain()
            Consultation_fee=input.Consultation_fee()
            VisitType=input.VisitType() 
            Visit_date=input.Visit_date() 
    
            # Insert data into the database
            cursor.execute('''INSERT INTO patient (
                    PatientID, FirstName, LastName, Gender, Age, 
                    AgeCat, Occupation, Ethnicity, Education, Height, Weight, BMI, 
                    BP_sytolic, BP_Distolic, BP_category, 
                    Pusle_rate, Temperature, Diet_compliance, Smoking_status, 
                    Alcohol_consumption, Physical_activity, 
                    Symptoms, Complain, Consultation_fee, VisitType, 
                    Visit_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (PatientID, FirstName, LastName, Gender, Age, AgeCat, Occupation, Ethnicity, Education, Height, Weight, BMI, 
                    BP_sytolic, BP_Distolic, BP_category, Pusle_rate, 
                    Temperature, Diet_compliance,
                    Smoking_status, 
                    Alcohol_consumption, Physical_activity, Symptoms,
                    Complain, Consultation_fee, VisitType, Visit_date))         
            conn.commit()
            # conn.close()
    #----------------------Delete Record from Database-----------
    @reactive.Effect
    @reactive.event(input.delete)
    def _():
        m = ui.modal(
            "Record Deleted successfully!",
            title="Notification",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    @reactive.Effect
    @reactive.event(input.delete)
    def delete():
        if input.delete():
            PatientID = input.PatientID()
            cursor.execute("DELETE FROM patient WHERE PatientID=?", (PatientID,))
            conn.commit()
    #this part deal with appointment action button when user click on
    @reactive.Effect
    @reactive.event(input.App)
    def _():
        m = ui.modal(
            """Please Book the Appointment with Pateint for the next visit:""",
            contact_us(),
            title="Appoinment",
            easy_close=True,
            footer=None,
        ) 
        ui.modal_show(m)
    
    @render.ui
    @reactive.event(input.msg)
    def data_table():
        if input.msg():
            # Copy the link to the Google Spreadsheet CSV file
            df = "Replace googlesheetlink  that read your datasheet from googlesheet"
            # Read the CSV file into a Pandas Dataframe
            df = pd.read_csv(df)
            df=df[['Name','Message:', 'Email:', 'Address:', 'Phone number:']]
            df = df.sort_values(by='Message:', ascending=True)
            # return render.DataGrid(df.tail(), filters=True)
            return ui.HTML(DT(df,layout={"top": "searchBuilder"},keys=True,classes="display nowrap compact", 
                                        buttons=["pageLength","copyHtml5", "csvHtml5", "excelHtml5",'print']))
    
    # This section allow user to see the app demo when click on the info button
    @reactive.Effect
    @reactive.event(input.inf)
    def _():
        m = ui.modal(
            """This is a Machine learning/Artificial intelligence EHR Software I built using Shiny for Python Framework to help physicians/Doctors predict each patient who is at risk of having Hypertension or Diabetic using various machine learning models in a resource-limited setting
        Please navigate through the app to explore more""",
        ui.hr(),
        ui.HTML(
        'Replace this with your Youtube channel',
        ),
        ui.hr(),
        ui.tags.a(ui.strong("Welcome to A-Musa Data-Solution@2023"),
        href=("https://am-datasolution.com/about.html")),
            
            title="Notification",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    
    #-----------------------------BMI-Calculation-Section----------------
    @render.code
    def bmi_output():
        bmi = calculate_bmi()
        if bmi is not None:
            return f"Your BMI is: {bmi}"
        else:
            return "Please enter valid height and weight values."
    @reactive.Calc
    def calculate_bmi():
        Height = input.Height()
        Weight = input.Weight()
        if Height is not None and Weight is not None:
            Height = float(Height) / 100
            bmi = Weight / (Height ** 2)
            return round(bmi, 2)
        else:
            return None, None
    #-----------------------------Blood-Pressure-Section by calling the function from above----------------
    @render.code
    def bp_output():
        bp_category = categorize_blood_pressure(input.BP_sytolic(), input.BP_Distolic())
        return f"Your blood pressure is: {bp_category}"  
    
    #-----------------------------Data-Upload-Section----------------                       
    @render.ui
    @reactive.event(input.Load_DB_Button)
    def showtable():
        if input.file1() is None:
            return "Please upload an Excel file"
        files: list[FileInfo] = input.file1()
        for file in files:
            xls = pd.read_excel(file["datapath"], sheet_name=None)
            for sheet_name, df in xls.items():
                if 'PatientID' in df.columns:
                    df = df[~df['PatientID'].duplicated()]
                df.to_sql(sheet_name, conn, if_exists='append', index=False)
        query = 'SELECT * FROM Outpatient'
        df2 = pd.read_sql_query(query, conn)
        # return render.DataTable(df2.head(7))
        return ui.HTML(DT(df2,layout={"top": "searchBuilder"}, 
                        keys=True,classes="display nowrap compact", 
                                            buttons=["pageLength","copyHtml5", "csvHtml5", "excelHtml5",'print']))
        
# This section will allow user to retrive data from the database
    @render.ui
    @reactive.event(input.Refresh)
    def retriveData():
            if input.Refresh():
                # Retrieve patien data from the database
                query = ("""SELECT PatientID,FirstName, LastName,Gender, AgeCat,Occupation,Education,Consultation_fee
                FROM patient""")
                df = pd.read_sql_query(query, conn)
                return ui.HTML(DT(df,layout={"top": "searchBuilder"}, 
                                keys=True,classes="display nowrap compact", 
                                                    buttons=["pageLength","copyHtml5", "csvHtml5", "excelHtml5",'print']))
                
    #----------------------Delete Record from the Table
    @reactive.Effect
    @reactive.event(input.delR)
    def _():
        m = ui.modal(
            "Record Deleted successfully!",
            title="Notification",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    @reactive.Effect
    @reactive.event(input.delR)
    def delete():
        if input.delR():
            PatientID = input.PID()
            cursor.execute("DELETE FROM patient WHERE PatientID=?", (PatientID,))
            conn.commit() 
#---------------------This for Machine Learning Section
    @render.ui
    def datacode():
        # selected_ids = 
        df = Explordata
        return ui.HTML(DT(df,layout={"top": "searchBuilder"},keys=True,classes="display nowrap compact", 
                                    buttons=["pageLength","copyHtml5", "csvHtml5", "excelHtml5",'print']))
        
#------------------This is for predic section
    @reactive.Calc
    def get_patient():
        patient_id = input.patient_id()
        return feature_df[feature_df['PatientID'] == str(patient_id)]
    
    @render.data_frame
    def tablepredic():
        return get_patient()
    
    @reactive.Calc
    def selected_patient():
        # Get the selected patient ID
        selected_patient_id = input.patient_id()
        # Find the corresponding record in feature_df
        record = feature_df.loc[feature_df['PatientID'] == selected_patient_id].drop('PatientID', axis=1).values
        predictios= {}
        for name, model in models.items():
            predictios[name] ="Non-Diabetic" if model.predict(record)[0]==0 else "Diabetic"
        
        return predictios
    
    @output
    @render.text
    def result():
        if input.subDM():
            predictios = selected_patient()
            output_text = ""
            for model, prediction in predictios.items():
                output_text += f"This case is {prediction} according to {model}\n"
            return output_text
        else:
            return ""

    @output
    @render.text
    def recommendation():
        if input.subDM():
            predictions = selected_patient()
            if predictions['Logistic Regression'] == 1 or predictions['Support Vector Machine'] == 1:
                return "Recommendation: Additional tests needed."
            else:
                return "Recommendation: No additional tests needed."
        else:
            return ""
# #-------------------------Model Accuracy
    @output
    @render.text
    @reactive.Calc
    @reactive.event(input.predict)
    def accuarcy():
        if input.model():
            model = models[input.model()]
            model.fit(X_train_DM, y_train_DM)
            y_pred = model.predict(X_test_DM)
            accuracy = accuracy_score(y_test_DM, y_pred)
            report = classification_report(y_test_DM, y_pred)
            return f"Accuracy: {accuracy}\nClassification Report:\n{report}"

    import itertools
    @output
    @render.plot
    @reactive.Calc
    @reactive.event(input.predict)
    def plot_confusion_matrix():
        if input.model():
            model = models[input.model()]
            model.fit(X_train_DM, y_train_DM)
            y_pred = model.predict(X_test_DM)
            cm = confusion_matrix(y_test_DM, y_pred, labels=[0, 1])
            np.set_printoptions(precision=2)

            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(['Non-Diabetic(0)', 'Diabetic(1)']))
            plt.xticks(tick_marks, ['Non-Diabetic(0)', 'Diabetic(1)'], rotation=45)
            plt.yticks(tick_marks, ['Non-Diabetic(0)', 'Diabetic(1)'])

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

#----------------HYpertensionSegment
   
    @reactive.Calc
    def selected_patient_Bp():
        # Get the selected patient ID
        selected_patient_id = input.patient_id()
        # Find the corresponding record in feature_df
        record = feature_df.loc[feature_df['PatientID'] == selected_patient_id].drop('PatientID', axis=1).values
        predictios= {}
        for name, model in models.items():
            predictios[name] ="Non-Hypertensive" if model.predict(record)[0]==0 else "Hypertensive"
        
        return predictios
   
   
    @output
    @render.text
    def result_Bp():
        if input.subDM():
            predictios = selected_patient_Bp()
            output_text = ""
            for model, prediction in predictios.items():
                output_text += f"This case is {prediction} according to {model}\n"
            return output_text
        else:
            return ""

    @output
    @render.text
    def recommendation_Bp():
        if input.subDM():
            predictions = selected_patient()
            if predictions['Logistic Regression'] == 1 or predictions['Support Vector Machine'] == 1:
                return "Recommendation: Additional tests needed."
            else:
                return "Recommendation: No additional tests needed."
        else:
            return ""
# #-------------------------Model Accuracy
    @output
    @render.text
    @reactive.Calc
    @reactive.event(input.predict)
    def accuarcy_Bp():
        if input.model():
            model = models[input.model()]
            model.fit(X_train_BP, y_train_BP)
            y_pred = model.predict(X_test_BP)
            accuracy = accuracy_score(y_test_BP, y_pred)
            report = classification_report(y_test_DM, y_pred)
            return f"Accuracy: {accuracy}\nClassification Report:\n{report}"

    import itertools
    @output
    @render.plot
    @reactive.Calc
    @reactive.event(input.predict)
    def plot_confusion_matrix_Bp():
        if input.model():
            model = models[input.model()]
            model.fit(X_train_BP, y_train_BP)
            y_pred = model.predict(X_test_BP)
            cm = confusion_matrix(y_test_BP, y_pred, labels=[0, 1])
            np.set_printoptions(precision=2)

            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(['Non-Hypertensive(0)', 'Hypertensive(1)']))
            plt.xticks(tick_marks, ['Hypertensive(0)', 'Hypertensive(1)'], rotation=45)
            plt.yticks(tick_marks, ['Hypertensive(0)', 'Hypertensive(1)'])

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')



#--------------This Data Exploration Section also
    keep_rows = reactive.value([True] * len(codededata))
    @reactive.calc
    def data_with_keep():
        df = codededata.copy()
        df["keep"] = keep_rows()
        return df
    @render.code()
    def statmodel():
        var_y, var_x, = input.vary(),input.varx()
        modelT=input.modelT()
        df = data_with_keep()
        df_keep = df[df["keep"]]
        mod = getattr(sm, modelT)(df_keep[var_y], df_keep[var_x])
        res = mod.fit()
        return res.summary()

#-------------ggplotSection
    @render.plot
    def gg_plot():
        var_y, var_x, = input.vary(),input.varx()
        df = data_with_keep()
        df_keep = df
        p = ggplot(df_keep, aes(var_x, var_y)) + geom_point() 
        return p
   
#------------------Dashboard Section--------------
    
    @reactive.Calc
    def filtered_df() -> pd.DataFrame:
        """Returns a Pandas data frame that includes only the desired rows based on selected filters"""

        # Filter the data based on the selected visitType
        if input.visitType() == "All":
            filtered_data = df.copy()  # No specific filter, return all data
        elif input.visitType() == "New":
            filtered_data = df[df['VisitType'] == "New"].copy()  # Filter by New visits
        else:
            filtered_data = df[df['VisitType'] == "Follow-up"].copy()  # Filter by Follow-up visits

        # Filter the data based on selected ethnicity
        if input.ethnicity() != "All":
            filtered_data = filtered_data[filtered_data['Ethnicity'] == input.ethnicity()].copy()

        # Filter the data based on selected symptoms
        if "All" not in input.symptoms():
            filtered_data = filtered_data[filtered_data['Symptoms'].apply(lambda x: any(symptom in x for symptom in input.symptoms()))].copy()

        # # Apply the date range filter if visit_date is selected
        # if input.visit_date():
        #     start_date, end_date = pd.to_datetime(input.visit_date(), format='%Y-%m-%d')
        #     filtered_data = filtered_data[(filtered_data['Visit_date'].apply(pd.Timestamp) >= start_date) & 
        #                                 (filtered_data['Visit_date'].apply(pd.Timestamp) <= end_date)].copy()
        # else:
        #     # If visit_date filter is not selected, return all records
        #     filtered_data = df.copy()

        return filtered_data

    @output
    @render.ui
    def totalpatients():
        return filtered_df().shape[0]

    @output
    @render.ui
    def totalmales():
        return filtered_df()[filtered_df()["Gender"] == "Male"].shape[0]

    @output
    @render.ui
    def totalfemales():
        return filtered_df()[filtered_df()["Gender"] == "Female"].shape[0]
    
    @render.data_frame
    def freq_and_pct():
        result = pd.DataFrame()
        filtered_data = filtered_df()  # Get the filtered DataFrame
        if not filtered_data.empty:
            freq = filtered_data['BP_category'].value_counts()
            pct = freq / freq.sum() * 100
            result = pd.DataFrame({'Blood-Pressure': freq.index, 'Frequency': freq.values, 'Percentage': pct.round(1).values})
        return render.DataTable(result)
    
    @output
    @render_widget 
    def height_weight_chart():
        """Generates a scatter plot with filters applied"""
        filtered_data = filtered_df()

        fig = px.scatter(filtered_data, x='Weight', y='Height', size=filtered_data['Weight'].apply(lambda x: len(str(x))), template='simple_white')
        return fig

    
    @render.data_frame
    def bod_mass_index():
        result_df = pd.DataFrame()
        filtered_data = filtered_df()  # Get the filtered DataFrame
        if not filtered_data.empty:
            filtered_data["BMI_range"] = pd.cut(filtered_data["BMI"], bins=[0, 18.5, 24.9, 29.9, filtered_data["BMI"].max()], labels=["Underweight", "Healthy weight", "Overweight", "Obese"])
            result_df = filtered_data.groupby("BMI_range").size().reset_index(name='count')
            result_df['percentage'] = result_df.groupby('BMI_range')['count'].transform(lambda x: round((x / x.sum()) * 100, 1))
            return render.DataTable(result_df)
     
    
    @render_widget
    def age_chart():
        """Generates a bar chart with filtered data based on selected age categories"""
        
        filtered_data = filtered_df()
        result = filtered_data.groupby('AgeCat').size().reset_index(name='count')
        fig = px.bar(result, x="AgeCat", y="count", template='simple_white')
        fig.update_layout(xaxis_title="Age Category", yaxis_title="Count")
        return fig        
   
    
    ui.markdown("Consultation Fee")
    @render.data_frame
    def freqe():
        """Displays a DataTable with statistics based on filtered data"""
        
        filtered_data = filtered_df()
        filtered_data['Consultation_fee'] = pd.to_numeric(filtered_data['Consultation_fee'], errors='coerce')
        # Perform statistics operations on the filtered data
        total = filtered_data['Consultation_fee'].sum()
        average = filtered_data['Consultation_fee'].mean().round(2)
        minimum = filtered_data['Consultation_fee'].min()
        maximum = filtered_data['Consultation_fee'].max()
        
        # Create a DataFrame with the aggregated results
        statistics = pd.DataFrame({
            'total': [total],
            'Average': [average],
            'min': [minimum],
            'max': [maximum]
        }) 
        return render.DataTable(statistics)
    
    
    
    
    @render_widget
    def payment_fee():
        """Generates a line chart of Consultation Fee Trend Over Time with filtered data"""
        
        filtered_data = filtered_df()
        dfpy = pd.DataFrame(filtered_data, columns=["id", "Consultation_fee", "Visit_date"])
        # Ensure Visit_date is in a datetime format for plotting
        dfpy["Visit_date"] = pd.to_datetime(dfpy["Visit_date"])

        # Create the line chart using Plotly Express
        fig = px.line(
            dfpy,
            x="Visit_date",  # Set Visit_date as the x-axis
            y="Consultation_fee",  # Set Consultation_fee as the y-axis
            title="Consultation Fee Trend Over Time",
        )
        
        return fig