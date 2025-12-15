# Establish a pint unit registry for consistent units
import pint
units = pint.UnitRegistry()

from importlib import resources
path = resources.files("htbam_db_api") / "units" / "RFU.txt"
units.load_definitions(path)
