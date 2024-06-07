from shiny import ui
from tapyr_template.logic.utils import dataupload, ui_register, sidebar_ui, retrivedata, predic_sidebar_ui
from tapyr_template.logic.querydata import choices_column
from tapyr_template.logic.model import Stat_model
from tapyr_template.logic.dashboarduitil import filtter,Valueboxes,dashboard_body
from tapyr_template.logic.about import AppInfo,WorkInfo,TemsInfo

def get_dashboard_ui() -> ui.Tag:
    return ui.page_navbar(   
                                    
    ui.nav_spacer(),
    ui.nav_panel(ui.input_action_button("HomePage","Home Page",class_="btn btn-info"),
    ui.navset_card_pill(   
    ui.nav_panel("Regsiter", 
        ui.layout_sidebar(  
        ui.sidebar(
        sidebar_ui(), # I called this function from my utils file
        open ='open', bg="pink",width="13.9%", 
        ),
        ui.panel_conditional("input.record",        
        ui_register(),# I haved Called From form the utility
        ),
        
        ui.panel_conditional("input.msg",
        ui.output_ui("data_table"),
      ), 
    )   
    ),
    
    ui.nav_panel("Upload-Data",
        ui.layout_sidebar(
        ui.sidebar(
        dataupload(),
        open ='open', bg="pink",width="15.9%", 
        ),
        ui.output_ui("showtable")
       ),            
     ),
    ui.nav_panel("Retrive-Data",
    retrivedata(),
    ui.hr(),
    ui.output_ui("retriveData"),
      ), #close ui.nav_panel for Retrive Data
     ), #close ui.navset_card_pill under HomePage section         
    ), # close ui.nav_panel for HomePage section
     
   ui.nav_panel(ui.input_action_button("MachineLearning","Machine-Learning",class_="btn btn-info"),
          ui.navset_card_pill( 
          ui.nav_panel("Data-Explor",
          ui.output_ui("datacode"), 
          ui.card(
          ui.div(
          ui.input_select("vary", "Variables-Y", choices=[""]+choices_column, width="auto"),
          ui.input_select("varx", "Variables-X", choices=[""]+choices_column, width="auto"),
          ui.input_select("modelT", "Chose the Stat-Model-Type", choices=[""]+Stat_model, width="auto"),
          style=('display: inline-block;'"display:flex;")
          ),
          ui.output_text_verbatim("statmodel"),
          ui.hr(),
          ui.output_plot("gg_plot")
          )      
        ),
          ui.nav_panel("Prediction-Tab",
        ui.layout_sidebar(  
        ui.sidebar(
        predic_sidebar_ui(),# This function was from the utils
         
        ),     
        ui.output_data_frame("tablepredic"),
          ui.output_text_verbatim("result"),
          ui.output_text_verbatim("recommendation"),
           ui.output_text_verbatim("result_Bp"),
          ui.output_text_verbatim("recommendation_Bp"),
          ui.layout_columns(
          ui.output_text_verbatim("accuarcy"),
          ui.output_plot("plot_confusion_matrix"),
          ),
          ui.hr(),
          ui.layout_columns(
          ui.output_text_verbatim("accuarcy_Bp"),
          ui.output_plot("plot_confusion_matrix_Bp")
          ),
    )
                       
    ),
      ),           
   ), # Closing of Machine-learning page
   
   ui.nav_panel(ui.input_action_button("Dashboard","Dashboard",class_="btn btn-info"),
      ui.card(
      filtter(),
      Valueboxes(),
      dashboard_body(),
      full_screen=True,
      style=(
        'padding: 4px 10px;'
        "background-color: pink;"
        'display: inline-block;'
        'border-radius: 10px;'
        "background-color:pink;"
        'margin-top: 2px;'
      ))
   ),
   
    # this part 
   ui.nav_panel(ui.input_action_button("About","About",class_="btn btn-info"),
           AppInfo(),
           ui.hr(),
           WorkInfo(),
           ui.hr(),
           TemsInfo(),  
   ),
   title="Machine-Learning",
  #  fillable_mobile=False, 
   bg= "navy",
  #  collapsible = True,
   
   inverse= True,
   footer= ui.strong(ui.h4(ui.HTML("""<marquee> A-Musa Data-Solution @ 2024 </marquee>""")))
  #  position='fixed-top',
   

)

    
    
