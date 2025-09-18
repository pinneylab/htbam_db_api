from htbam_db_api.exceptions import HtbamDBException
from htbam_db_api.csv_processing import CSV_DATA_LABELS, parse_concentration
from htbam_db_api.csv_processing import process_dataframe_kinetics, process_dataframe_binding
from htbam_db_api.data import Data3D

from pathlib import Path

import pandas as pd
import numpy as np

def verify_file_exists(file_path: str) -> None:
        '''
        Verifies that a file exists at the given path.
        Returns an informative Error message if not.

            Parameters:
                    file_path (str): Path to the file

            Returns:
                    None
        '''

        # exists?
        if Path(file_path).is_file():
            return True

        # if not, check if the parent file even exists
        parent_file_exists = False
        parent_file_contents = []

        parent_file = Path(file_path).parent

        while not parent_file_exists and parent_file != Path('/'):
            if Path(parent_file).exists():
                parent_file_exists = True
                parent_file_contents = [str(f) for f in Path(parent_file).iterdir() if f.is_file()]
            else:
                parent_file = parent_file.parent
        
        if not parent_file_exists:
            raise HtbamDBException(f"File {file_path} does not exist. We cannot find any files matching the path provided")
        else:
            raise HtbamDBException(f"File {file_path} does not exist. We found the parent file {parent_file} but it does not contain the file you requested.\n \
                                   We found the following files in the parent directory:\n" + "\n".join(parent_file_contents))

def load_run_from_csv(csv_path: str, run_type:str, conc_unit_str: str) -> Data3D:
    '''
    Loads a run from a CSV file, and processes it into a dict of numpy arrays.

    Arguments:
        csv_path: The path to the CSV file
        run_type: The type of run (kinetics, standard curve, etc.)
        conc_unit_str: The unit string for the concentration (e.g. 'nM', 'uM', etc.)

    Returns:
        A dict of numpy arrays in the 'kinetics' or 'binding' format.
    '''
    ### Load CSV
    with open(csv_path, 'r') as f:
        df = pd.read_csv(f)

    ### Pre-process CSV
    ### TODO: Unify standard curve and kinetics CSV formats on microscope, so we don't have to juggle here.
    L = CSV_DATA_LABELS # shorthand for labels dict.
    # The standard curve CSVs look different that the usual kinetics. Let's rectify that:
    if L['time'] not in df.columns:
        df[L['time']] = 0
    # First, we convert the raw concentration string to a float:
    if L['raw_concentration'] in df.columns:
        # Kinetics CSV format
        df[L['concentration']] = df[L['raw_concentration']].apply(lambda x: parse_concentration(x, conc_unit_str))
    else:
        # Standard curve CSV format
        df[L['concentration']] = df[L['standardcurve_concentration']]
    # Create unique Chamber_IDs as "x,y"
    df[L['chamber_IDs']] = df[L['chamber_x']].astype(str) + ',' + df[L['chamber_y']].astype(str)
    
    ### Sort
    # Sort the df first by chamber_id (using x and y to keep order correct), then by concentration, then by time
    # Warning: Modifying in-place here.
    df = df.sort_values(by=[L['chamber_x'], L['chamber_y'], L['concentration'], L['time']])

    ### Process data into numpy arrays:
    # N.F. I think we can have different functions to do this. Ideally we should keep data in the generally flexible "kinetics" format.
    # This format should work for kinetics (multiple concentrations and timepoints), inhibition, and standard curves (single timepoint).
    data_processing_functions = {
        'kinetics': process_dataframe_kinetics,
        'binding':  process_dataframe_binding
    }

    # Pass our dataframe into the function which will process into a dict of numpy arrays.
    run_data = data_processing_functions[run_type](df)

    return run_data