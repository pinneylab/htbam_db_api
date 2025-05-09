from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from typing import List, Tuple
import numpy as np
import json
from pathlib import Path
import re
from copy import deepcopy

# DATA_TYPE_STRUCTURE = {
#     "standard": 
#         {"conc_label": "concentration_uM",
#          "luminance_label": "sum_chamber",
#          "time_label": "time_s"},
#     "kinetic": 
#         {"conc_label": "series_index",
#          "luminance_label": "sum_chamber",
#          "time_label": "time_s"},
#     "binding":
#         {"conc_label": None}
# }

# These are the human-readable labels for our data. Some are from the CSV, while others (like chamber_IDs) are constructed.
# This is what we will be returning when we parse each CSV. 
# The dimensions are noted.
DATA_TYPE_STRUCTURE = {
    "kinetic": 
        {'indep_vars':
            [   'concentration', # (n_concentrations)
                'chamber_IDs',   # (n_chambers)
                'sample_IDs',    # (n_chambers)
                "button_quant_sum",  # (n_chambers)
                "time",          # (n_concentrations, n_time_points)
            ],
        'dep_vars':
            [   "luminance",    # (n_concentrations, n_time_points, n_chambers)
            ]
        }
}

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

    @abstractmethod
    def get_run_assay_data(self, run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    # @abstractmethod
    # def create_analysis(self, run_name: str):
    #     raise NotImplementedError

# def _squeeze_df(df: pd.DataFrame, grouping_index: str, squeeze_targets: List[str]):
#     '''
#     Squeezes a dataframe along a given column to de-tidy the target data into lists.

#             Parameters:
#                     grouping_index (str): Aggregation column name
#                     squeeze_targets ([str]): List of columns to reduce values to lists

#             Returns:
#                     sqeeuzed_df (pd.DataFrame): DF with columns == [grouping_index, *squeeze_targets]
#     '''

#     squeeze_func = lambda x : pd.Series([x[grouping_index].values[0]] + [x[col].tolist() for col in squeeze_targets], 
#                                         index=[grouping_index] + squeeze_targets)
#     return df.groupby(grouping_index).apply(squeeze_func)

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

        #standard_df = pd.read_csv(standard_curve_data_path)
        #standard_df['indices'] = standard_df.x.astype('str') + ',' + standard_df.y.astype('str')

        #self._standard_src_data = dict()
        #self._standard_src_data["standard_0"] = {"name": standard_name, "substrate": standard_substrate,
        #                                         "conc_unit": standard_units, "data": standard_df}
    
        #kinetic_df = pd.read_csv(kinetic_data_path)
        #kinetic_df['indices'] = kinetic_df.x.astype('str') + ',' + kinetic_df.y.astype('str')
        #self._kinetic_src_data = dict()
        #self._kinetic_src_data["kinetic_0"] = {"name": kinetic_name, "substrate": kinetic_substrate,
        #                                         "conc_unit": kinetic_units, "data": kinetic_df}
        
        self._init_json_dict()

        # Populate with metadata, which was stored in the kinetics dataframe
        self._json_dict['metadata']['chamber_IDs'] = kinetics_data['indep_vars']['chamber_IDs'] # (n_concentrations, )
        self._json_dict['metadata']['sample_IDs'] = kinetics_data['indep_vars']['sample_IDs'] # (n_concentrations, )

        self._json_dict['runs'][standard_name] = standard_data
        self._json_dict['runs'][kinetic_name] = kinetics_data

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
            '''
            #first, remove the unit and everything following
            conc = conc_str.split(unit_name)[0]
            #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
            conc = float(conc.replace("_", "."))
            return conc
    
    def load_run_from_csv(self, csv_path: str, run_type:str, conc_unit_str: str) -> dict:

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
        Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'kinetics' format.
        
        Arguments:
            df: DataFrame of experiment data

        Returns:
            {
                "kinetic": 
                    {'indep_vars':
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

    def _load_button_quant_data(self, run_name: str) -> None:
        '''
        Populates an json_dict with button quant data with the following schema:
        {button_quant: {
         
            1,1: {
                sum_chamber: [...],
                std_chamber: [...]
            },
            ...
            }}

                Parameters:
                        None

                Returns:
                        None
        ''' 
        try:   
            unique_buttons = self._kinetic_src_data[run_name]["data"][["summed_button_Button_Quant","summed_button_BGsub_Button_Quant",
            "std_button_Button_Quant", "indices"]].drop_duplicates(subset=["indices"]).set_index("indices")
        except KeyError:
            raise HtbamDBException("ButtonQuant columns not found in kinetic data.")
        self._json_dict["button_quant"] = unique_buttons.to_dict("index")  

    def __repr__(self) -> str:
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
                    s += f"{value}\n"
            s += "\t"*indent + '}\n'
            return s
        
        return recursive_string(self._json_dict, 0)
    
    # NF. Is this every used? looks like the funcitons inside don't exist.
    # def add_run_data(self, run_type: str, data_path: str, run_name: str, run_substrate: str, run_units: str):
       
    #     if run_type == "standard":
    #         stadard_run_num = len([key for key in self._json_dict['runs'].keys() if 'standard' in key])
    #         standard_df = pd.read_csv(data_path)
    #         standard_df['indices'] = standard_df.x.astype('str') + ',' + standard_df.y.astype('str')

    #         self._standard_src_data = dict()
    #         self._standard_src_data[f"standard_{stadard_run_num}"] = {"name": run_name, "substrate": run_substrate,
    #                                                 "conc_unit": run_units, "data": standard_df}
    #         self._load_std_data(f"standard_{stadard_run_num}")
    #         print(f"Added run data to database.\n Run ID: standard_{stadard_run_num}\n Run name: {run_name}\n Run type: {run_type}\n Run substrate: {run_substrate}\n Run units: {run_units}")
    #     elif run_type == "kinetic":
    #         kinetic_run_num = len([key for key in self._json_dict['runs'].keys() if 'kinetic' in key])
    #         kinetic_df = pd.read_csv(data_path)
    #         kinetic_df['indices'] = kinetic_df.x.astype('str') + ',' + kinetic_df.y.astype('str')
    #         self._kinetic_src_data = dict()
    #         self._kinetic_src_data[f"kinetic_{kinetic_run_num}"] = {"name": run_name, "substrate": run_substrate,
    #                                                 "conc_unit": run_units, "data": kinetic_df}
    #         self._load_kinetic_data(f"kinetic_{kinetic_run_num}")
    #         self._load_button_quant_data(f"kinetic_{kinetic_run_num}")
    #         print(f"Added run data to database.\n Run ID: kinetic_{kinetic_run_num} Run name: {run_name}\n Run type: {run_type}\n Run substrate: {run_substrate}\n Run units: {run_units}")
    #     else:
    #         raise HtbamDBException(f"Run type {run_type} not supported. Supported types: ['standard', 'kinetic']")
        
    #     return

    def get_run_names(self):
        return [key for key in self._json_dict['runs'].keys()]
    
    def get_run_assay_data(self, run_name):
        '''This function takes as input an HTBAM Database object.
        For each kinetics run, we have 
        It returns 3 numpy arrays:
        chamber_ids: an array of the chamber ids (in the format '1,1' ... '32,56')
            shape: (n_chambers,)
        luminance_data: an array of the luminance data for each chamber
            shape: (n_time_points, n_chambers, n_assays)
        conc_data: an array of the concentration data for each chamber.
            shape: (n_assays,)
        time_data: an array of the time data for each time point.
            shape: (n_time_points, n_assays)
        '''
    
        chamber_idxs = np.array(list(self._json_dict['chamber_metadata'].keys()))
        luminance_data = None
        time_data = None
        conc_data = np.array([])

        #Each assay may have recorded a different # of time points.
        #First, we'll just check what the max # of time points is:
        max_time_points = 0
        for assay in self._json_dict["runs"][run_name]['assays'].keys():
            current_assay_time_points = len(np.array(self._json_dict["runs"][run_name]['assays'][assay]['time_s']))
            if current_assay_time_points > max_time_points:
                max_time_points = current_assay_time_points

        for assay in self._json_dict["runs"][run_name]['assays'].keys():
            
            #to make things easier later, we'll be sorting the datapoints by time value.
            #Get time data:
            #collect from DB
            current_time_array = np.array(self._json_dict["runs"][run_name]['assays'][assay]['time_s'])
            current_time_array = current_time_array.astype(float) #so we can pad with NaNs
            #pad the array with NaNs if there are fewer time points than the max
            current_time_array = np.pad(current_time_array, (0, max_time_points - len(current_time_array)), 'constant', constant_values=np.nan)
            #sort, and capture sorting idxs:
            sorting_idxs = np.argsort(current_time_array)
            current_time_array = current_time_array[sorting_idxs]
            current_time_array = np.expand_dims(current_time_array, axis=1)
            #add to our dataset
            if time_data is None:
                time_data = current_time_array
            else:
                time_data = np.concatenate([time_data, current_time_array], axis=1)

            #Get luminance data:
            current_luminance_array = None
            for chamber_idx in chamber_idxs:
                #collect from DB
                current_chamber_array = np.array(self._json_dict["runs"][run_name]['assays'][assay]['chambers'][chamber_idx]['sum_chamber'])
                #set type to float:
                current_chamber_array = current_chamber_array.astype(float)
                #pad the array with NaNs if there are fewer time points than the max
                current_chamber_array = np.pad(current_chamber_array, (0, max_time_points - len(current_chamber_array)), 'constant', constant_values=np.nan)
                #sort by time:
                current_chamber_array = current_chamber_array[sorting_idxs]
                #add a dimension at the end:
                current_chamber_array = np.expand_dims(current_chamber_array, axis=1)

                if current_luminance_array is None:
                    current_luminance_array = current_chamber_array
                else:
                    current_luminance_array = np.concatenate([current_luminance_array, current_chamber_array], axis=1)
            #add a dimension at the end:
            current_luminance_array = np.expand_dims(current_luminance_array, axis=2)
            #add to our dataset
            if luminance_data is None:
                luminance_data = current_luminance_array
            else:
                luminance_data = np.concatenate([luminance_data, current_luminance_array], axis=2)
            
            #Get concentration data:
            #collect from DB
            current_conc = self._json_dict["runs"][run_name]['assays'][assay]['conc']
            conc_data = np.append(conc_data, current_conc)

        #sort once more, by conc_data:
        sorting_idxs = np.argsort(conc_data)
        conc_data = conc_data[sorting_idxs]

        #sort luminance data by conc_data:
        luminance_data = luminance_data[:,:,sorting_idxs]
        
        return chamber_idxs, luminance_data, conc_data, time_data
    
    def get_entity_data(self, entity_name: str):
        chamber_idxs = np.array(list(self._json_dict['chamber_metadata'].keys()))
        luminance_data = None
        time_data = None
        conc_data = np.array([])

        #Each assay may have recorded a different # of time points.
        #First, we'll just check what the max # of time points is:
        time_dict = self._get_entity_independent_data(entity_name, 'time_s')
        max_time_points = max([len(v) for v in time_dict.values()])
        
        # for assay in self._json_dict["runs"][entity_name]['assays'].keys():
        #     current_assay_time_points = len(np.array(self._json_dict["runs"][entity_name]['assays'][assay]['time_s']))
        #     if current_assay_time_points > max_time_points:
        #         max_time_points = current_assay_time_points

        for assay in self._json_dict["runs"][entity_name]['assays'].keys():
            
            #to make things easier later, we'll be sorting the datapoints by time value.
            #Get time data:
            #collect from DB
            current_time_array = np.array(self._json_dict["runs"][entity_name]['assays'][assay]['time_s'])
            current_time_array = current_time_array.astype(float) #so we can pad with NaNs
            #pad the array with NaNs if there are fewer time points than the max
            current_time_array = np.pad(current_time_array, (0, max_time_points - len(current_time_array)), 'constant', constant_values=np.nan)
            #sort, and capture sorting idxs:
            sorting_idxs = np.argsort(current_time_array)
            current_time_array = current_time_array[sorting_idxs]
            current_time_array = np.expand_dims(current_time_array, axis=1)
            #add to our dataset
            if time_data is None:
                time_data = current_time_array
            else:
                time_data = np.concatenate([time_data, current_time_array], axis=1)

            #Get luminance data:
            current_luminance_array = None
            for chamber_idx in chamber_idxs:
                #collect from DB
                current_chamber_array = np.array(self._json_dict["runs"][entity_name]['assays'][assay]['chambers'][chamber_idx]['sum_chamber'])
                #set type to float:
                current_chamber_array = current_chamber_array.astype(float)
                #pad the array with NaNs if there are fewer time points than the max
                current_chamber_array = np.pad(current_chamber_array, (0, max_time_points - len(current_chamber_array)), 'constant', constant_values=np.nan)
                #sort by time:
                current_chamber_array = current_chamber_array[sorting_idxs]
                #add a dimension at the end:
                current_chamber_array = np.expand_dims(current_chamber_array, axis=1)

                if current_luminance_array is None:
                    current_luminance_array = current_chamber_array
                else:
                    current_luminance_array = np.concatenate([current_luminance_array, current_chamber_array], axis=1)
            #add a dimension at the end:
            current_luminance_array = np.expand_dims(current_luminance_array, axis=2)
            #add to our dataset
            if luminance_data is None:
                luminance_data = current_luminance_array
            else:
                luminance_data = np.concatenate([luminance_data, current_luminance_array], axis=2)
            
            #Get concentration data:
            #collect from DB
            current_conc = self._json_dict["runs"][entity_name]['assays'][assay]['conc']
            conc_data = np.append(conc_data, current_conc)

        #sort once more, by conc_data:
        sorting_idxs = np.argsort(conc_data)
        conc_data = conc_data[sorting_idxs]

        #sort luminance data by conc_data:
        luminance_data = luminance_data[:,:,sorting_idxs]
        
        return chamber_idxs, luminance_data, conc_data, time_data
    
    def _get_entity_independent_data(self, entity_name: str, ind_variable_name):
        assays = self._json_dict["runs"][entity_name]['assays'].items()
        return {k: v[ind_variable_name] for k, v in assays}
    
    def _init_analysis(self, run_name):
        if 'analyses' not in self._json_dict['runs'][run_name].keys():
            self._json_dict['runs'][run_name]['analyses'] = {}

    def add_analysis(self, run_name, analysis_type, chamber_idx, analysis_data):
    
        supported_analysis_types = ['linear_regression', 'ic50_raw', 'mm_raw', 'bgsub_linear_regression']
        self._init_analysis(run_name)

        if analysis_type not in supported_analysis_types:
            raise HtbamDBException(f"Analysis type {analysis_type} not supported. Supported types: {supported_analysis_types}")
        
        if analysis_type not in self._json_dict['runs'][run_name]['analyses'].keys():
            self._json_dict['runs'][run_name]['analyses'][analysis_type] = {'chambers': {}} #initialize the dictionary

        self._json_dict['runs'][run_name]['analyses'][analysis_type]['chambers'][chamber_idx] = analysis_data

    def add_filtered_assay(self, run_name: str, assay_type: str, assay_data: dict):
        supported_assay_types = ['filtered_initial_rates']
        if assay_type not in supported_assay_types:
            raise HtbamDBException(f"Analysis type {assay_type} not supported. Supported types: {supported_assay_types}")
        
        self._init_analysis(run_name)

        #initialize the dictionary
        self._json_dict['runs'][run_name]['analyses'][assay_type] = assay_data

    def add_sample_analysis(self, run_name: str, analysis_type: str, sample_name: str, sample_data: dict):
        supported_assay_types = ['ic50_filtered', 'mm_filtered']
        if analysis_type not in supported_assay_types:
            raise HtbamDBException(f"Analysis type {analysis_type} not supported. Supported types: {supported_assay_types}")
        if analysis_type not in self._json_dict['runs'][run_name]['analyses'].keys():
            self._json_dict['runs'][run_name]['analyses'][analysis_type] = {'samples': {}}

        self._json_dict['runs'][run_name]['analyses'][analysis_type]['samples'][sample_name] = sample_data

    def get_sample_analysis_dict(self, run_name: str, analysis_type: str):
        return self._json_dict['runs'][run_name]['analyses'][analysis_type]['samples']
         
    def get_filters(self, run_name, assay_type):
        return self._json_dict['runs'][run_name]['analyses'][assay_type]["filters"]
    
    def get_analysis(self, run_name, analysis_type, param):
       
        chamber_idxs = self._json_dict['runs'][run_name]['analyses'][analysis_type]['chambers']
        query_result = {chamber_idx: 
                        self._json_dict['runs'][run_name]['analyses'][analysis_type]['chambers'][chamber_idx][param] 
                        for chamber_idx in chamber_idxs}
        return query_result

    def get_chamber_name_dict(self):
        return {chamber_idx: subdict['id'] for chamber_idx, subdict in self._json_dict['chamber_metadata'].items()}
    
    def get_chamber_name_to_id_dict(self):
        chamber_name_to_idx = {}
        for chamber_idx, subdict in self._json_dict['chamber_metadata'].items():
            name = subdict['id']
            if name not in chamber_name_to_idx.keys():
                chamber_name_to_idx[name] = [chamber_idx]
            else:
                chamber_name_to_idx[name].append(chamber_idx)
        return chamber_name_to_idx
    
    def get_button_quant_data(self, chamber_idx, button_quant_type='summed_button_BGsub_Button_Quant'):
        return self._json_dict['button_quant'][chamber_idx][button_quant_type]
    
    def get_concentration_units(self, run_name):
        return self._json_dict['runs'][run_name]['conc_unit']
    
    def combine_runs(self, run_names: List[str]):

        # check that all runs are of the same type
        run_types = [run_name.split('_')[0] for run_name in run_names]
        if len(set(run_types)) > 1:
            raise HtbamDBException("Runs must be of the same type to be combined.")
        
        # check that all runs are of type 'kinetic'
        if run_types[0] != 'kinetic':
            raise HtbamDBException("Runs must be of type 'kinetic' to be combined.")
        
        # check that all runs are in the database
        for run_name in run_names:
            if run_name not in self._json_dict['runs'].keys():
                raise HtbamDBException(f"Run {run_name} not found in database.")
            
        new_run_num = len([key for key in self._json_dict['runs'].keys() if 'kinetic' in key])
        
        new_run_dict = {}
        for run_name in run_names:
            if "linear_regression" not in new_run_dict.keys():
                new_run_dict["linear_regression"] = deepcopy(self._json_dict['runs'][run_name]['analyses']['linear_regression'])
            else:
                for chamber_idx in self._json_dict['runs'][run_name]['analyses']['linear_regression']['chambers'].keys():
                    for key, value in new_run_dict["linear_regression"]["chambers"][chamber_idx].items():
                        
                        if key == 'mask':
                            continue
                        new_run_dict["linear_regression"]["chambers"][chamber_idx][key] = np.concatenate([value, self._json_dict['runs'][run_name]['analyses']['linear_regression']['chambers'][chamber_idx][key]])
                    #print(self._json_dict['runs'][run_name]['analyses']['linear_regression']['chambers'][chamber_idx])
                    #new_run_dict["linear_regression"]["chambers"][chamber_idx] = np.concatenate(new_run_dict["linear_regression"]["chambers"][chamber_idx], self._json_dict['runs'][run_name]['analyses']['linear_regression']['chambers'][chamber_idx])
            if "filtered_initial_rates" not in new_run_dict.keys():
                new_run_dict["filtered_initial_rates"] = deepcopy(self._json_dict['runs'][run_name]['analyses']['filtered_initial_rates'])
            elif len(new_run_dict["filtered_initial_rates"]["filters"]) != len(self._json_dict['runs'][run_name]['analyses']['filtered_initial_rates']['filters']):
                raise HtbamDBException("Runs must have the same number of filters to be combined.")
            else:
                for i in range(len(new_run_dict["filtered_initial_rates"]["filters"])):
                    new_run_dict["filtered_initial_rates"]["filters"][i] = np.concatenate([new_run_dict["filtered_initial_rates"]["filters"][i], 
                                                                                           self._json_dict['runs'][run_name]['analyses']['filtered_initial_rates']['filters'][i]],
                                                                                           axis=1)
        self._json_dict['runs'][f'kinetic_{new_run_num}']= {"analyses": new_run_dict}
        return f'kinetic_{new_run_num}'

    def remove_run(self, run_name: str):
        if run_name not in self._json_dict['runs'].keys():
            raise HtbamDBException(f"Run {run_name} not found in database.")
        del self._json_dict['runs'][run_name]   

    def remove_analysis(self, run_name: str, analysis_type: str):
        if analysis_type not in self._json_dict['runs'][run_name]['analyses'].keys():
            raise HtbamDBException(f"Analysis {analysis_type} not found in database.")
        del self._json_dict['runs'][run_name]['analyses'][analysis_type]  
         
    def export_json(self):
        '''This writes the database to file, as a dict -> json'''
        with open('db.json', 'w') as fp:
            json.dump(self._json_dict, fp, indent=4)

class HtbamDBException(Exception):
    pass