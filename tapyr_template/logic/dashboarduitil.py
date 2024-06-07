from shiny import ui
import faicons as fa
from shinywidgets import output_widget
# from tapyr_template.logic.querydata import total_patients,total_females,total_males, Children_U5
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


def filtter():
    return ui.div(
    ui.accordion(
    ui.accordion_panel("Filters",
    ui.input_select("visitType", "Type of Visit", choices=["All", "New", "Follow-up"], width="auto"),
    ui.input_date_range("visit_date", "Visit Date"),
    ui.input_select("ethnicity", "Ethnicity", choices=["All", "Hausa-Fulani", "Yoruba", "Igbo", "Ijaw", "Kanuri", "Ibibio", "Tiv", "Others"], width="auto"),
    ui.input_selectize("symptoms", "Symptoms", choices=["All"] + patient_symptoms,selected=("All"), multiple=True, width="auto"),
            class_="d-flex align-items-center gap-1", 
    ),
    open=False
  ) 
)
   
    
# Add main content
ICONS = {
    "users-rectangle": fa.icon_svg("users-rectangle"),
    "wallet": fa.icon_svg("wallet"),
    "people-arrows": fa.icon_svg("people-arrows"),
    "children": fa.icon_svg("children"),
    "person-cane": fa.icon_svg("person-cane"),
    "person-pregnant": fa.icon_svg("person-pregnant"),
    }


def Valueboxes():
    return ui.div(
        ui.row(
    ui.layout_column_wrap(      
    ui.value_box("Total patients", ui.output_ui("totalpatients"),showcase=ICONS["users-rectangle"],theme="bg-gradient-orange"),
    ui.value_box("Total males", ui.output_ui("totalmales"),theme="bg-gradient-blue-purple",showcase=ICONS["person-cane"]),
    ui.value_box("Total females", ui.output_ui("totalfemales"),showcase=ICONS["person-pregnant"],theme="bg-gradient-indigo-yellow"),
    # ui.value_box("Total Patients",(Children_U5),theme="gradient-blue-indigo"),        
             fill=False
            ),
        )        

    )
def dashboard_body():
    return ui.div(
    ui.layout_columns(
    ui.card(
    ui.markdown("Blood Pressure Category"),
      ui.output_data_frame("freq_and_pct")),
      ui.card(
      ui.markdown("Classification of BMI"),
      ui.output_data_frame("bod_mass_index")),
      col_widths=[6, 6],row_heights=(2, 3),height="250px",),
      ui.layout_columns(
        ui.card(
        ui.markdown("Age distribution of patients"),
          output_widget("age_chart")),
        ui.card(
          output_widget("payment_fee")
        ),
        col_widths=[4, 8,]),
    ui.layout_columns(
    ui.card( ui.markdown("Generated Income "),
              ui.output_data_frame("freqe")),
    ui.card(
    ui.markdown("Weight and Height Chart"),
      output_widget("height_weight_chart")),
      col_widths=[4, 8,])
       
    )