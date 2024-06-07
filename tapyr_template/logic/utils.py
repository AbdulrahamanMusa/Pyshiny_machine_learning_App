from loguru import logger
from shiny import ui
from tapyr_template.logic.querydata import codededata
from tapyr_template.logic.model import models

@logger.catch(reraise=True)
def dataupload():
    return ui.markdown(
    """
    This section allow user to upload excel data file to SQLite database and displayed
    10 recond from the uploaded dataset """),ui.input_file("file1", "Choose CSV File", accept=[".xlsx"], multiple=False),ui.input_checkbox("header", "Header", True), ui.input_action_button("Load_DB_Button",
    "Load Data",
    style = "bordered",
    width = "90%"),
    
occupations = ["Business", "Farmer", "Student","Civil servant",  "Tailor", "Carpenter", "Engineer","Others", "Unemployed","Under-Care"]

patient_symptoms = [
    "Pain",
    "Fever",
    "Difficulty breathing",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Bleeding",
    "Injuries",
    "Infections",
    "Fatigue",
    "Anxiety",
    "Headache",
    "Loss of appetite",
    "Coughing",
    "Chills",
    "Sore throat",
    "Nasal congestion",
    "Stiff neck",
    "Other"
]

@logger.catch(reraise=True)
def sidebar_ui():
    return ui.div(
        ui.div(
        ui.input_switch("record", "Show-UI", False),
        style =
        "width: 90%;" 'border-radius: 10px;'
            "background-color: lightblue;"),
    ui.hr(),
    ui.div(
        ui.input_switch("msg", "Message", False),
        style =
        "width: 90%;" 'border-radius: 10px;'
            "background-color: white;"
            ),
    ui.hr(),
    ui.input_action_button("App", "Book", style = "bordered", width = "90%"),   
    ui.hr(),
    ui.input_action_button("inf", "info", style = "bordered", width = "90%"),    
),


@logger.catch(reraise=True)
def ui_register():
    return ui.div(
        ui.h1("Patient Database Management"),

        ui.div(
        ui.input_text("PatientID", "PatientID"),
        ui.input_text("FirstName", "Frist Name"),
        ui.input_text("LastName", "Last Name"),
        ui.input_select("Gender", "Gender", choices=["Male", "Female"]),
        ui.input_numeric("Age", "Age", value=0),
        ui.input_selectize("AgeCat", "Age-category", choices=["U-5yrs","5-14yrs", "15-24yrs", "25-34yrs", "35-44yrs", "45-54yrs", "55yers and above"]),
        ui.input_selectize("Occupation", "Occupation", choices=[""]+occupations),
        ui.input_select("Ethnicity", "Ethnicity", choices=["Hausa-Fulani","Yoruba","Igbo","Ijaw","Kanuri","Ibibio","Tiv","Others"]),
        ui.input_selectize("Education", "Education level", choices=["Under-Care","Primary","Secondary", "Tertiary", "Non-formal"]),
        
        style=("float: left;" 'margin-left: 25px;'),
        ),
        ui.div(
        ui.input_numeric("Height", "Height (in cm)",0, min=0, max=1000),
        ui.input_numeric("Weight", "Weight (in kg)", 0, min=0, max=1000),
        ui.output_text_verbatim("bmi_output"),
        ui.input_numeric("BMI", "Body Mass Index", value=0),
        ui.input_numeric("BP_sytolic", "BP_sytolic", value=0),
        ui.input_numeric("BP_Distolic", "BP_Distolic", value=0),
        ui.output_text_verbatim("bp_output"), 
        ui.input_select("BP_category", "BP_category", choices=["Normal", "Elevated", "Stage 1 hypertension","Stage 2 hypertension"]),    
        ui.input_numeric("Pusle_rate", "Pusle_rate", value=0),
        ui.input_numeric("Temperature", "Temperature", value=0),
        style=("float: left;" 'margin-left: 15px;'), 
        ),
        ui.div(
        ui.input_select("Diet_compliance", "Diet compliance", choices=["Yes","No", "N/A"]),
        ui.input_select("Smoking_status", "Smoking status", choices=["Yes","No", "N/A"]),
        ui.input_select("Alcohol_consumption", "Alcohol consumption", choices=["Yes","No", "N/A"]),
        ui.input_select("Physical_activity", "Physical activity", choices=["Yes","No", "N/A"]),
        ui.input_selectize("Symptoms", "Symptoms", choices=[""]+patient_symptoms , multiple=True),  
        ui.input_text_area("Complain", "Complain Presented"),
        ui.input_numeric("Consultation_fee", "Consultation_fee", value=0),
        ui.input_select("VisitType", "Type of Visit", choices=["New", "Follow-up"]),
        ui.input_date("Visit_date", "Visit Date"),
        ui.hr(),
        ui.div(
        ui.input_action_button("submit", "Save Record",class_="btn btn-primary"),
        ui.input_action_button("delete", "Delete Record", class_="btn btn-danger"),
        style=('display: inline-block;'"display:flex;")
        ), 
        style=("float: right;" 'margin-left: 15px;'),
        ),         
        style=(
            'border-radius: 10px;'
            "background-color: pink;"
            "float: left;"
            # 'position: fixed;'
            # "overflow: auto;"
            'margin-top: 1px;'
            'margin-left: 15px;'
            "flex-direction: row;"
            
            ),
    ),#end of UI Form

@logger.catch(reraise=True)
def contact_us():
    return ui.div(
            ui.HTML(
                'This is my googlesheet link for book appointment'
                ),
            style=(
            'padding: 4px 10px;'
            'display: inline-block;'
            'border-radius: 10px;'
            "background-color:pink;"
            'margin-top: 2px;'
            ""),
            ),
@logger.catch(reraise=True) 
def categorize_blood_pressure(systolic, diastolic):
    if systolic < 120 or diastolic < 80:
        return "Normal"
    elif 120 == systolic <= 129 or diastolic <= 80:
        return "Elevated"
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return "Stage1 hypertension"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage2 hypertension"
    else:
        return "Invalid input"   
      
# Get the systolic and diastolic blood pressure from the user
systolic = ui.input_numeric("BP_sytolic", "BP_sytolic", value=0)
diastolic = ui.input_numeric("BP_Distolic", "BP_Distolic", value=0)

@logger.catch(reraise=True)
def retrivedata():
    return ui.div(
        ui.input_text("PID", "Type PatientID to Delete Record"), 
        ui.input_action_button("Refresh", "Show Record",class_="btn btn-success"),
        ui.input_action_button("delR", "Delete Record",class_="btn btn-danger"),
        style=('display: inline-block;')
        ),
    
    
# Machine_learning Section utils
@logger.catch(reraise=True)
def predic_sidebar_ui():
    return ui.div(
        ui.input_select("patient_id", "Seach PatientID", choices=codededata["PatientID"].tolist()),
                            ui.hr(),
                            ui.input_action_button("subDM", "Predict Diabetic"),
                            ui.hr(),
                            ui.input_select('model', 'Select model', choices=list(models.keys())),
                            ui.hr(),
                            ui.input_action_button("predict", "model-Report"),
                            ui.h6("Model CLassification Report "),
    )

