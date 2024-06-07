import pytest
from shiny import ui

from tapyr_template.logic.utils import dataupload
from tests.helpers.logging_helpers import log_contain_message



def dataupload():
    return ui.markdown(
    """
    This section allow user to upload excel data file to SQLite database and displayed
    10 recond from the uploaded dataset """),ui.input_file("file1", "Choose CSV File", accept=[".xlsx"], multiple=False),ui.input_checkbox("header", "Header", True), ui.input_action_button("Load_DB_Button",
    "Load Data",
    style = "bordered",
    width = "90%"),
