from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from typing import List, Tuple
import numpy as np
import json
from pathlib import Path
import re
from copy import deepcopy

# These are the human-readable labels for our data. Some are from the CSV, while others (like chamber_IDs) are constructed.
# This is what we will be returning when we parse each CSV. 
# The dimensions are noted.

# Metadata notes:
# Dep var types
# Something like "from"?
# Something like "masked with"? 
# Fit model

# Maybe turn all the structs into (n_conc, n_chamb, n_time, n) where n could be 1 or more.

DATA_TYPE_STRUCTURE = {
    # For assay data that contains 3D data, like raw RFU or product amounts. (concentrations, timepoints, and chambers)
    "3D_data": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        'dep_vars':
            [   "dep_var",    # (n_concentrations, n_time_points, n_chambers)
            ],
        "meta": [
            "dep_var_type" # (e.g. "luminance", "product", etc)
        ],
        },
    # For data that varies by concentration and chamber # (We've lost the time dimension).
    "2D_data": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        "dep_vars"  : [
            "dep_var",   # (n_conc, n_chamb)
        ],
        "meta": {
            "dep_var_type" # (e.g. "luminance", "product", etc)
        }
        },
    # For masking out certain concentrations or entire chambers.
    "2D_data_mask": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        "dep_vars"  : [
            "mask",   # (n_conc, n_chamb)
        ],
        "meta": {
            }
        },
    # For 2D data with several fit parameters per chamber (e.g. standard curve slope/intercept, or enzyme kinetics fit parameters).
    "2D_data_fits": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        "dep_vars"  : [
            "dep_var"   # (n_conc, n_chamb, n_fit_params)
        ],
        "meta": [
            "parameters", # (n_params)
            "model" # mm_model, etc
        ]
        },
    # E.g. for enzyme concentration data, which is just a 1D array of concentrations for each chamber. We've lost both time and concentration.
    "1D_data": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        "dep_vars"  : {
            "dep_vars",            # (n_chamb)
        },
        "meta": [
            "dep_var_type" # (e.g. "luminance", "product", etc)
        ]
    },
    # E.g. for final model fitting, where we have several params per chamber (Vmax, Km, etc). 
    "1D_data_fits": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        "dep_vars"  : {
            "dep_vars",            # (n_chamb, n_params
        },
        "meta": {
            "model",     # e.g. mm_model
            "parameters" # (n_params)
        }
    }
}

### Current as of 6/4/25, but I want to change to a 
# DATA_TYPE_STRUCTURE = {
#     # For RFU data over time, with multiple concentrations.
#     "RFU_data": 
#         {'indep_vars':
#             [   'concentration', # (n_concentrations)
#                 'chamber_IDs',   # (n_chambers)
#                 'sample_IDs',    # (n_chambers)
#                 "button_quant_sum",  # (n_chambers)
#                 "time",          # (n_concentrations, n_time_points)
#             ],
#         'dep_vars':
#             [   "luminance",    # (n_concentrations, n_time_points, n_chambers)
#             ]
#         },
#     # For standard curve slope/intercept (several per chamber)
#     "linear_fit_data": 
#         {'indep_vars':
#             [   'concentration', # (n_concentrations)
#                 'chamber_IDs',   # (n_chambers)
#                 'sample_IDs',    # (n_chambers)
#                 "button_quant_sum",  # (n_chambers)
#                 "time",          # (n_concentrations, n_time_points)
#             ],
#         "dep_vars"  : [
#             "slope",   # (n_conc, n_chamb)
#             "intercept", # (n_conc, n_chamb)
#             "r_squared", # (n_conc, n_chamb)
#         ],
#         "meta": {
#             "fit": "luminance_vs_time",
#             "model": "LinearRegression",
#         }
#         },
#     "linear_fit_data_mask": 
#         {'indep_vars':
#             [   'concentration', # (n_concentrations)
#                 'chamber_IDs',   # (n_chambers)
#                 'sample_IDs',    # (n_chambers)
#                 "button_quant_sum",  # (n_chambers)
#                 "time",          # (n_concentrations, n_time_points)
#             ],
#         "dep_vars"  : [
#             "mask",   # (n_conc, n_chamb)
#         ],
#         "meta": {
#             }
#         },
#     "concentration_data": 
#         {'indep_vars':
#             [   'concentration', # (n_concentrations)
#                 'chamber_IDs',   # (n_chambers)
#                 'sample_IDs',    # (n_chambers)
#                 "button_quant_sum",  # (n_chambers)
#                 "time",          # (n_concentrations, n_time_points)
#             ],
#         "dep_vars"  : {
#             "concentration",            # (n_chamb)
#         },
#         "meta": {
#         }
#     }
# }

# These are the CSV columns that correspond with our human-readable labels.
CSV_DATA_LABELS = {
    'concentration': 'concentration',   # we will construct this from raw_concentration
    'raw_concentration': 'series_index',
    'chamber_IDs': 'chamber_IDs',                 # we construct this from chamber_x, chamber_y
    'sample_IDs': 'id',
    'chamber_x': 'x',
    'chamber_y': 'y',
    'time': 'time_s',
    'luminance': 'sum_chamber',
    'button_quant_sum': 'summed_button_BGsub_Button_Quant',
    'standardcurve_concentration': 'concentration_uM', # This is because on the microscope, the concentration is named differently for standard experiments. We should change it to be uniform.
}

class AbstractHtbamDBAPI(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def get_standard_data(self, standard_name: str) -> Tuple[List[float], np.ndarray]:
    #     raise NotImplementedError

    # @abstractmethod
    # def get_run_assay_data(self, run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     raise NotImplementedError
    
    # @abstractmethod
    # def create_analysis(self, run_name: str):
    #     raise NotImplementedError

class LocalHtbamDBAPI(AbstractHtbamDBAPI):
   

    def __init__(self, standard_curve_data_path: str, standard_name: str, standard_substrate: str, standard_units: str,
                  kinetic_data_path: str, kinetic_name: str, kinetic_substrate: str, kinetic_units: str):
        super().__init__()

        # Verify that the files exist
        self._verify_file_exists(standard_curve_data_path)
        self._verify_file_exists(kinetic_data_path)
        
        # The data is in format 'kinetics' for both standard curve and kinetics.
        standard_data = self.load_run_from_csv(standard_curve_data_path, 'kinetics', standard_units)
        kinetics_data = self.load_run_from_csv(kinetic_data_path, 'kinetics', kinetic_units)
        
        self._init_json_dict()

        # Populate with metadata, which was stored in the kinetics dataframe
        self.set_metadata('chamber_IDs', kinetics_data['indep_vars']['chamber_IDs'])
        self.set_metadata('sample_IDs', kinetics_data['indep_vars']['sample_IDs'])

        self.add_run(standard_name, standard_data)
        self.add_run(kinetic_name, kinetics_data)

        # Use the kinetic data to create a separate button_quant run, for later processing:
        self.add_run('button_quant', self.process_button_quant_from_kinetics(kinetics_data))

        return
    
    def _verify_file_exists(self, file_path: str) -> None:
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

    def _init_json_dict(self) -> None:
        '''
        Populates an initial dictionary with chamber specific metadata.

        Parameters:
                None

        Returns:
                None
        ''' 
        self._json_dict = dict()
        self._json_dict["metadata"] = dict() # Will contain chamber_IDs, sample_IDs as 1D numpy arrays of shape (n_chambers, )
        self._json_dict["runs"] = dict()

    def parse_concentration(self, conc_str: str, unit_name: str) -> float:
            '''
            Currently, we're storing substrate concentration as a string in the kinetics data.
            This will be changed in the future to store as a float + unit as a string. For now,
            we will parse jankily.

            Arguments:
                conc_str: The concentration string to parse
                unit_name: The unit name to remove from the string

            Returns:
                The concentration as a float
            '''
            #first, remove the unit and everything following
            conc = conc_str.split(unit_name)[0]
            #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
            conc = float(conc.replace("_", "."))
            return conc
    
    def load_run_from_csv(self, csv_path: str, run_type:str, conc_unit_str: str) -> dict:
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
            df[L['concentration']] = df[L['raw_concentration']].apply(lambda x: self.parse_concentration(x, 'nM'))
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
            'kinetics': self.process_dataframe_kinetics,
            'binding':  self.process_dataframe_binding
        }

        # Pass our dataframe into the function which will process into a dict of numpy arrays.
        data_dict = data_processing_functions[run_type](df)

        return data_dict
    
    def process_dataframe_kinetics(self, df):
        '''
        Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'RFU_data' format.
        
        Arguments:
            df: DataFrame of experiment data

        Returns:
            {
                "RFU_data": 
                    {
                    'data_type': 'RFU_data',
                    'indep_vars':
                        {   'concentration':    concentrations, # (n_concentrations)
                            'chamber_IDs':      chamber_ids, # (n_chambers)
                            'sample_IDs':       sample_ids, # (n_chambers)
                            "button_quant_sum": button_quant, # (n_chambers)
                            "time":             time_array # (n_concentrations, n_time_points)
                        },
                    'dep_vars':
                        {   "luminance",        RFU_array # (n_concentrations, n_time_points, n_chambers)
                        }
                    }
            }
        '''
        L = CSV_DATA_LABELS # shorthand for labels dict.

        # Chamber_IDs(length n_chambers)
        chamber_ids = df[L["chamber_IDs"]].unique()      # n_chambers

        # Get sample IDs (length n_chambers)
        chamber_to_sample_map = df.set_index(L["chamber_IDs"])[L['sample_IDs']].to_dict() # Create a mapping of Chamber_IDs to Sample_IDs
        sample_ids = np.array([chamber_to_sample_map[chamber] for chamber in chamber_ids]) # Map the Sample_IDs to the unique Chamber_IDs

        # Get button quant (length n_chambers)
        if L['button_quant_sum'] in df.columns:
            chamber_to_button_quant_map = df.set_index(L["chamber_IDs"])[L['button_quant_sum']].to_dict() # Create a mapping of Chamber_IDs to Button_Quant
            button_quant = np.array([chamber_to_button_quant_map[chamber] for chamber in chamber_ids]) # Map the Button_Quant to the unique Chamber_IDs
        else: 
            button_quant = np.nan

        # Chamber_IDs(length n_concentrations)
        concentrations = df[L['concentration']].unique()  # n_concentrations

        # Time array (n_concentrations, n_timepoints). The values are time in seconds.
        time_array = np.array( list(
                                    df[df[L["chamber_IDs"]] == df[L["chamber_IDs"]][0]] # Get just the first chamber
                                    .groupby(L['concentration'])[L['time']]                                  # Group by the concentration column, and get just the time values.
                                    .apply(list)) )                                                              # And convert the times for each concentration to a list.
                                                                                                                # Then we convert to a list of lists, and then to a numpy array.

        # RFU array (n_concentrations, n_timepoints, n_chambers)
        RFU_list_by_conc = []
        for conc_index, concentration in enumerate(concentrations):
            time_values = time_array[conc_index]
            #print(time_values)
            RFU_list_by_time = []
            for time_index, time in enumerate(time_values):
                # Get the RFU values for this concentration and time
                rfu_values = df[(df[L['concentration']] == concentration) & (df[L['time']] == time)][L['luminance']].to_list()
                RFU_list_by_time.append(rfu_values)
            # Append the RFU values for this concentration to the list
            RFU_list_by_conc.append(RFU_list_by_time)
        # Convert the list to a numpy array
        RFU_array = np.array(RFU_list_by_conc)

        data_dict = {
            'data_type': 'RFU_data',
            'indep_vars': {
                'concentration':    concentrations,     # (n_concentrations)
                'chamber_IDs':      chamber_ids,        # (n_chambers)
                'sample_IDs':       sample_ids,         # (n_chambers)
                'button_quant_sum': button_quant,       # (n_chambers)
                'time':             time_array          # (n_concentrations, n_time_points)
            },
            'dep_vars': {
                'luminance':        RFU_array           # (n_concentrations, n_time_points, n_chambers)
            }
        }

        return data_dict

    def process_dataframe_binding(self, df):
        '''
        Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'binding' format.
        TODO: Not implemented.
        '''
        pass

    def process_button_quant_from_kinetics(self, kinetics_data: dict) -> dict:
        '''
        Processes the kinetics data to create a button quant run.
        
        Arguments:
            kinetics_data: The kinetics data in the 'RFU_data' format.

        Returns:
            A dict of numpy arrays in the 'RFU_data' format for button quant.
            We manually set there to be 0 timepoints and 0 substrate concenrtations, so we effectively
            have a 1D array of shape (n_chambers) for the button_quant data.
        '''
        sample_ids = kinetics_data['indep_vars']['sample_IDs']
        chamber_ids = kinetics_data['indep_vars']['chamber_IDs']
        button_quant_sum = kinetics_data['indep_vars']['button_quant_sum']

        button_quant_data = {'indep_vars':
                                {   'concentration': np.array([]), # (n_concentrations)
                                    'chamber_IDs':   chamber_ids,  # (n_chambers)
                                    'sample_IDs':    sample_ids, # (n_chambers)
                                    "button_quant_sum": button_quant_sum, # (n_chambers)
                                    "time":   np.array([]),          # (n_concentrations, n_time_points)
                                },
                            'dep_vars':
                                {   "luminance":    button_quant_sum, # (n_concentrations, n_time_points, # n_chambers)
                                },
                            'data_type': 'RFU_data'
                            }
        
        return button_quant_data
        
    def __repr__(self) -> str:
        '''
        Returns a string representation of the object.

        Returns:
                str: A string representation of the object
        '''
        def recursive_string(d: dict, indent: int, width=5) -> str:
            s = "\t"*indent + '{\n'
            for i, (key, value) in enumerate(d.items()):
                if i == width:
                    s += "\t"*indent +"...\n"
                    break
                s += "\t"*indent + f"{key}: "
                if isinstance(value, dict):
                    s += "\n" + recursive_string(value, indent+1)
                else:
                    data_string = ""
                    data_string += str(type(value)) + " "
                    if isinstance(value, np.ndarray):
                        data_string += str(value.shape) + " "
                    value_string = str(value)
                    value_string = value_string.replace("\n", " ").replace("\t", " ")
                    if len(value_string) > 30:
                        data_string += value_string[:30] + "..."
                    else:
                        data_string += value_string
                    s += f"{data_string}\n"
            s += "\t"*indent + '}\n'
            return s
        
        return recursive_string(self._json_dict, 0)

    ### GETTERS & SETTERS
    def add_run(self, run_name: str, run_data: dict) -> None:
        '''
        Adds a run to the database.

                Parameters:
                        run_name (str): Name of the run
                        run_data (dict): Data for the run

                Returns:
                        None
        '''
        # Check if the format matches one in DATA_TYPE_STRUCTURE:
        if run_data['data_type'] not in DATA_TYPE_STRUCTURE.keys():
            raise HtbamDBException(f"Run data type {run_data['data_type']} not supported. Supported types: {DATA_TYPE_STRUCTURE.keys()}")
        
        # Add to the database:
        self._json_dict['runs'][run_name] = run_data
        return
    
    def get_run(self, run_name: str) -> dict:
        '''
        Gets a run from the database.

                Parameters:
                        run_name (str): Name of the run

                Returns:
                        dict: Data for the run
        '''
        if run_name not in self._json_dict['runs'].keys():
            raise HtbamDBException(f"Run {run_name} not found in database.")
        
        return self._json_dict['runs'][run_name]
    
    def get_metadata(self, name: str) -> dict:
        '''
        Gets metadata from the database.

                Parameters:
                        name (str): Name of the metadata

                Returns:
                        dict: Metadata
        '''
        if name not in self._json_dict['metadata'].keys():
            raise HtbamDBException(f"Metadata {name} not found in database.")
        
        return self._json_dict['metadata'][name]
    
    def get_run_names(self):
        '''
        Gets the names of all runs in the database.
            
        Parameters:
            None
        Returns:
            list: List of run names
        '''
        return [key for key in self._json_dict['runs'].keys()]

    def set_metadata(self, name: str, value: str) -> None:
        '''
        Sets metadata in the database.

                Parameters:
                        name (str): Name of the metadata
                        value (str): Value of the metadata

                Returns:
                        None
        '''
        self._json_dict['metadata'][name] = value

         
    # def export_json(self):
    #     '''This writes the database to file, as a dict -> json'''
    #     with open('db.json', 'w') as fp:
    #         json.dump(self._json_dict, fp, indent=4)

class HtbamDBException(Exception):
    pass