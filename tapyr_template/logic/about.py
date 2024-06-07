from shiny import ui

def AppInfo():
    return ui.div(
    ui.accordion(
    ui.accordion_panel("Author",
        ui.tags.img( src="Replace-my-picture", height="420px",
           style="float: right;"
            'border-radius: 15px;'),
            ui.div(
            ui.h5(
                ui.markdown(
                    """
                Hello everyone I am Abdulrahaman A Musa Passionate Data Scientist in the field of Artificial Intelligent (AI) Machine learning 
                and predictive modelling in the Academic, 
                Health and Humanitarian sectors
               

              ## Our areas of expertise include:
              -   Building R and Python Shiny Dashboard
              -   Business Analytics Dashboard (Power BI)
              -   Training/Coaching on Data Analysis using the following
                  -   R and Python program
                  -   SPSS and Epi-Info software
              -   Building of Survey Platform
              -   Scripting and developing data collection tools
              -   Setting/Implementing of EHR in the private Health sector
              -   Monitoring and Evaluation of program/survey

                - For more information visit:https://am-datasolution.com/.
                - If you want to hire me for Full-Stack Data Science App contact me @ abdulrahaman@am-datasolution.com.
                                            """,)
    ),
    
    style='margin-right: 10px;''margin-top: 70px;' 'margin-bottom: 10px;'
            "background-color: white"                      
    ), 
  ),
    open=False  
  ),        
   
 )
    
def WorkInfo():
    return ui.div(
    ui.accordion(
    ui.accordion_panel("How does it Work?",
              ui.layout_columns(
              ui.div(
              ui.tags.img( src="static/img/aboutEHR.png", height="440px", style="float: left;" 'border-radius: 15px;'),
                   ui.HTML(
                    "replace this with your yutube channel"
                    ),
                )
                ),
            ui.div(
            ui.h5(
                ui.markdown(
                    """                
                ## Model Used
                1. Logistic Regression, 
                2. Support Vector Machine, 
                3. Decision Tree, 
                4. Random Forest, and 
                5. Gradient Boosting.
                
                The app used a Synthetic data.

                ## Usage
                
                1. Select patientID on the left sidebar panel under predicion Tab.
                2. Click on the "Predict" button to see the predicted case.
                3. Select a model from the dropdown list.
                4. The app will display the predicted result, accuracy, classification report, and a confusion matrix plot.

                ## Dataset

                The dataset used for training and testing the models is split into X (input features) and y (target variable) using a 70-30 train-test split.

                ## Acknowledgements
                - Shiny for Python by Posit.
                - Appsilon 
                                            """,)
    ),
    
    style='margin-right: 10px;''margin-top: 70px;' 'margin-bottom: 10px;'
            "background-color: white"                      
    ), 
  ),
    open=False 
  ),        
    
 )
    
    
def TemsInfo():
    return ui.div(
    ui.accordion(
    ui.accordion_panel("Terms & Conditions",
            ui.div(
            ui.h5(
                ui.markdown(
                """
                Terms and Conditions
**Welcome to the EHR Software for Predicting Hypertension and Diabetes Risk in Patients.**

By accessing and using this Machine Learning/Artificial Intelligence Electronic Health Record (EHR) Software, you agree to abide by the following Terms and Conditions:

**Predictive Models:** This software utilizes various machine learning models to predict the risk of Hypertension and Diabetes in patients.

**Physician Assistance:** The software is designed to assist physicians in identifying patients at risk in resource-limited settings.

**Accuracy Disclaimer:** While the predictive models aim to provide valuable insights, they may not be 100% accurate. It is essential for healthcare professionals to use their clinical judgment.

**Confidentiality:** Patient data input into the software is confidential and should be handled in accordance with applicable privacy regulations.

**Security:** Users are responsible for safeguarding their login credentials and ensuring the security of patient information stored in the software.

**Medical Advice:** The software's predictions are not a substitute for professional medical advice. Healthcare decisions should be based on comprehensive patient assessments.

**Data Integrity:** Users are encouraged to input accurate and up-to-date patient information for reliable predictions.

By using this EHR Software, you acknowledge and agree to these Terms and Conditions. Please contact the software administrator for any questions or
                                            """,)
    ),
    
    style='margin-right: 10px;''margin-top: 70px;' 'margin-bottom: 10px;'
            "background-color: white"                      
    ), 
  ),
    open=False 
  ),        
    
 )
    
def model_infor_pop():
      mod = ui.modal(
            ui.card(
            ui.row(
                ui.layout_columns(
                   ui.h4(
                        ui.layout_columns(
                        ui.card(
                         ui.img(src="images/Amusa.png", style="width: 120px; display: left; margin: 0 auto;",),
                         ui.h6("This is a Machine learning/Artificial intelligence EHR Software I built using shiny for Python help physican to predict each patient who is at risk of having Hypertension or Diabetic using various machine learning models in a resource-limited setting"),
                        )),
                         ui.hr(),
                         ui.h6(ui.markdown("""Hello everyone I am Abdulraham A Musa Passionate Data/ML Scientist in the field of Artificial Intelligent (AI) Machine learning and predictive modelling in the Academic, Health and Humanitarian sectors
                        For more information visit:https://am-datasolution.com/.
                    
                     """),
                         ui.hr(),
                         ui.a("github-link", href="https://github.com/AbdulrahamanMusa",class_="docs-link",)),
                         
                    ),
                ),
            ),
        ),
        title=ui.span("Executive Summary:", style="color: #486590; font-size: 2rem;"),
        easy_close=True,
        footer=None,
    )
      return ui.modal_show(mod) 
    