from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from typing import List, Tuple
import numpy as np
import json
from pathlib import Path
import pint

CURRENT_VERSION = "0.0.1"

class AbstractHtbamDBAPI(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def get_standard_data(self, standard_name: str) -> Tuple[List[float], np.ndarray]:
    #     raise NotImplementedError

def _squeeze_df(df: pd.DataFrame, grouping_index: str, squeeze_targets: List[str]):
    '''
    Squeezes a dataframe along a given column to de-tidy the target data into lists.

            Parameters:
                    grouping_index (str): Aggregation column name
                    squeeze_targets ([str]): List of columns to reduce values to lists

            Returns:
                    sqeeuzed_df (pd.DataFrame): DF with columns == [grouping_index, *squeeze_targets]
    '''

    squeeze_func = lambda x : pd.Series([x[grouping_index].values[0]] + [x[col].tolist() for col in squeeze_targets], 
                                        index=[grouping_index] + squeeze_targets)
    return df.groupby(grouping_index).apply(squeeze_func)

class HTBAM_Experiment(AbstractHtbamDBAPI):

    def __init__(self, file:str, new:bool=False, units_registry=None):
        super().__init__()
        self._experiment_file = Path(file)

        if units_registry is None:
            units_registry = pint.UnitRegistry()
        self.ureg = units_registry
        #make sure these are set up
        self.ureg.setup_matplotlib(True)
        self.ureg.define('RFU = [luminosity]')

        #does it have the correct extension?
        if self._experiment_file.suffix != ".HTBAM":
            raise ValueError(f"File {self._experiment_file} does not have the correct extension. Must be .HTBAM")

        if not new:
            data = self._get_dict_from_file()
        else:
            data = self._init_dict()
            Path(self._experiment_file).touch()
            self._write_file(data)

        #is it the correct version?
        if data["file_version"] != CURRENT_VERSION:
            print(f"Warning: File {self._experiment_file} was created with a different version. You're currently using {CURRENT_VERSION}.")

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
        
        data = self.get('') #get the whole thing
        return recursive_string(data, 0)
    
    def get(self, path):
        '''
        Returns the data from a given "path" in the database.
        Input: 
            path (str): the "path" to the data to be returned.
                ex: "runs/standard_0/assays/0/chambers/1,1/sum_chamber"
        Output:
            data: the data at the given path.
        '''
        #is path a str or Path object?
        if type(path) == str:
            path = Path(path)

        #if root, return the whole thing
        if path == "":
            return self._get_dict_from_file()
        
        #split the path object:
        path = path.parts
        data = self._get_dict_from_file()
        path_traversed = ""
        for p in path:
            # TODO: get wildcard working, like db.get('runs/standard_0/assays/*/chambers/1,1/sum_chamber')
            # if p == "*":
            #     #if we're at a wildcard, find the matching values in ALL children:
            #     if type(data) == dict:
            #         children = list(data.keys())
            #     else:
            #         raise ValueError(f"Wildcard found at {path_traversed}, which is a {type(data)}. Wildcards can only be used on dicts.")
            #     data_list = [self.get(path_traversed + child + "/") for child in children]
            #     for d in data_list:
            #         print(d)
            #     data = np.concatenate(data_list)
            #     return data
            
            #handle errors
            if p not in data:
                error_string = f"Path {path_traversed+p} not found in database. \n"
                if type(data) == dict:
                    error_string += f"Made it to {path_traversed}, which has keys {data.keys()}"
                else:
                    error_string += f"Made it to {path_traversed}, which is a {type(data)}"
                raise ValueError(error_string)
            data = data[p]
            path_traversed += p + "/"

        #if this is a quantity, convert to a pint.Quantity
        if type(data) == dict:
            if 'is_quantity' in data.keys() and data['is_quantity']:
                data = self._dict_to_quantity(data)

        # #if this is a dict, convert all quantities to pint.Quantities
        # if type(data) == dict:
        #     data = self._make_quantities_from_serialized_dict(data)
        
        return data
    
    ##########################################################################################
    ########################### READING / WRITING to/from FILE  ##############################
    ##########################################################################################

    def _init_dict(self) -> dict:
        '''
        Populates an initial dictionary with chamber specific metadata.
                Parameters:
                        None

                Returns:
                        None
        '''        
        return {
            "file_version": CURRENT_VERSION,
            "chamber_metadata": {},
            "button_quant": {},
            "runs": {},
        }

    def _get_dict_from_file(self) -> dict:
        '''
        Returns the dictionary stored in the .HTBAM file.
            Parameters: None
            Returns: json_dict (dict): Dictionary stored in the .HTBAM file.
        '''
        with open(self._experiment_file, 'r') as fp:
            json_dict = json.load(fp)
        return json_dict

    def _write_file(self, data):
        '''This writes the database to file, as a dict -> json.
        This will overwrite the existing file.'''
        with open(self._experiment_file, 'w') as fp:
            json.dump(data, fp, indent=4)

    def _update_file(self, path, new_data):
        '''This appends data to a given path in the database'''
        #is path a str or Path object?
        if type(path) == str:
            path = Path(path)

        #is new_data a quantity? If so, convert to dict
        if type(new_data) == pint.Quantity:
            new_data = self._quantity_to_dict(new_data)

        path = path.parts
        full_data = self._get_dict_from_file()
        current_data = full_data #this will be updated as we traverse the path
        path_traversed = ""

        #N.B.: If anything is passed by value instead of by reference, this will break.
        for p in path[:-1]:
            #handle errors
            if p not in current_data:
                error_string = f"Path {path_traversed+p} not found in database. \n"
                if type(current_data) == dict:
                    error_string += f"Made it to {path_traversed}, which has keys {current_data.keys()}"
                else:
                    error_string += f"Made it to {path_traversed}, which is a {type(current_data)}"
                raise ValueError(error_string)
            
            #continue down the path
            current_data = current_data[p]
            path_traversed += p + "/"

        #update the data
        if path[-1] in current_data: 
            if current_data[path[-1]] != {}: #We can't overwrite data, but we'll allow overwriting blank placeholder dicts.
                raise ValueError(f"Path {path_traversed+path[-1]} already exists in database. Overwriting data is forbidden.")
        current_data[path[-1]] = new_data
        
        #write to file
        #now, we've iterated down our full dict and changed some part of it. We'll pass back the full dict.
        self._write_file(full_data)

    ##########################################################################################
    ######################### SERIALIZE / DESERIALIZE QUANTITY DATA ##########################
    ##########################################################################################
    def _dict_to_quantity(self, data: dict) -> pint.Quantity:
        '''
        Converts a dictionary with keys 'values', 'unit', and 'is_quantity' to a pint.Quantity.
        This allows us to receive our data from a json file.
        '''
        if not data['is_quantity']:
            raise ValueError("Data is not a quantity.")
        
        try:
            return np.array(data['values']) * self.ureg(data['unit'])
        except:
            raise ValueError(f"Could not convert data to quantity. Data: {data}")
        
    def _quantity_to_dict(self, quantity: pint.Quantity) -> dict:
        '''
        Converts a pint.Quantity to a dictionary with keys 'values', 'unit', and 'is_quantity'.
        The reason we need this is so we can store our data in a JSON. json.dump() can't handle np.arrays or pint.Quantities.
        '''
        return {
            'values': quantity.magnitude.tolist(),
            'unit': str(quantity.units),
            'is_quantity': True
        }
    
    def _make_serializable_dict(self, data: dict) -> dict:
        '''
        Converts all pint.Quantities in a dictionary to dictionaries with keys 'values', 'unit', and 'is_quantity'.
        This allows us to store our data in a JSON. json.dump() can't handle np.arrays or pint.Quantities.
        '''
        for key, value in data.items():
            if isinstance(value, pint.Quantity):
                data[key] = self._quantity_to_dict(value)
            elif isinstance(value, dict):
                data[key] = self._make_serializable_dict(value)
        return data
    
    def _make_quantities_from_serialized_dict(self, data: dict) -> dict:
        '''
        Converts all dictionaries with keys 'values', 'unit', and 'is_quantity' to pint.Quantities.
        This allows us to receive our data from a json file.
        '''
        for key, value in data.items():
            if isinstance(value, dict):
                if 'is_quantity' in value.keys() and value['is_quantity']:
                    data[key] = self._dict_to_quantity(value)
                else:
                    data[key] = self._make_quantities_from_serialized_dict(value)
        return data

    ##########################################################################################
    ######################### LOADING EXPERIMENT DATA FROM CSVs  #############################
    ##########################################################################################
    #(make _load_chamber_metadata !)
    def _load_chamber_metadata(self,standard_data_df) -> None:
        '''
        Populates an json_dict with kinetic data with the following schema:
        {button_quant: {
         
            1,1: {
                sum_chamber: {'values': [...],
                            'unit': 'RFU',
                            'is_quantity': True},
                std_chamber: {'values': [...],
                            'unit': 'RFU',
                            'is_quantity': True},
            },
            ...
            }}

        Parameters:
            standard_data_df (pd.DataFrame): Dataframe from standard curve

        Returns:
            None
        '''    
        unique_chambers = standard_data_df[['id','x_center_chamber', 'y_center_chamber', 'radius_chamber', 
            'xslice', 'yslice', 'indices']].drop_duplicates(subset=["indices"]).set_index("indices")
        self._update_file(Path("chamber_metadata"), unique_chambers.to_dict("index") )

    def load_standard_data_from_file(self, standard_curve_data_path: str, standard_name: str, standard_type: str, standard_units: str) -> None:
        '''
        Populates an dict with standard curve data with the following schema, and saves to the .HTBAM file:
        {standard_run_#: {
            name: str,
            type: str,
            assays: {
                1: {
                    conc: float,
                    time:
                        {'values': [...],
                        'unit': 's',
                        'is_quantity': True},
                    chambers: {
                        1,1: {
                            sum_chamber: {'values': [...],
                                          'units': 'RFU',
                                          'is_quantity': True},
                            std_chamber: {'values': [...],
                                          'unit': 'RFU'}
                                          'is_quantity': True},
                        },
                        ...
                        }}}}

        Parameters:
            standard_curve_data_path (str): Path to standard curve data
            standard_name (str): Name of standard curve
            standard_type (str): Type of standard curve
            standard_units (str): Units of standard curve

        Returns:
                None
        '''
        #First, check if our standard_units is a valid unit of concentration:
        if self.ureg(standard_units).dimensionality != self.ureg.molar.dimensionality:
            raise ValueError(f"Units {standard_units} are not a valid unit of concentration.\nIs your capitalization correct?")
        
        standard_data_df = pd.read_csv(standard_curve_data_path)
        standard_data_df['indices'] = standard_data_df.x.astype('str') + ',' + standard_data_df.y.astype('str')
        i = 0    
        std_assay_dict = {}
        #TODO: this bad convention of column names will be phased out soon.
        for prod_conc, subset in standard_data_df.groupby("concentration_uM"):
            squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=['sum_chamber', 'std_chamber'])
            squeezed["time_s"] = pd.Series([[0]]*len(squeezed), index=squeezed.index) #this tomfoolery is used to create a list with a single value, 0, for the standard curve assays.
            
            #turn prod_conc into a pint.Quantity:
            prod_conc = np.array(prod_conc) * self.ureg(standard_units)
            
            #turn it into a pint.Quantity:
            time_quantity = squeezed.iloc[0]["time_s"] * self.ureg("s")

            #now, we need to properly format the data for each chamber:
            chambers_dict_unformatted = squeezed.drop(columns=["time_s", "indices"]).to_dict("index")
            chambers_dict = {}
            for chamber_coord, chamber_data in chambers_dict_unformatted.items():
                chambers_dict[chamber_coord]  = {}
                for key, value in chamber_data.items():
                    #convert to pint.Quantity:
                    chambers_dict[chamber_coord][key] = value * self.ureg("RFU")

            #make dict for each assay
            std_assay_dict[i] = {
                "conc": prod_conc, #serialize using our custom _quantity_to_dict function
                "time": time_quantity, 
                "chambers": chambers_dict}
            
            i += 1

        std_run_num = len([key for key in self.get(Path('runs')) if "standard_" in key])
        standard_data_dict = {
            "name": standard_name,
            "type": standard_type,
            "assays": std_assay_dict
            }
        
        #append to file
        standard_data_dict = self._make_serializable_dict(standard_data_dict) #convert all quantities to dicts so we can save to json
        self._update_file(Path("runs") / f"standard_{std_run_num}", standard_data_dict)
        
        #update chamber metadata
        self._load_chamber_metadata(standard_data_df)
       
    def load_kinetics_data_from_file(self, kinetic_data_path: str, kinetic_name: str, kinetic_type: str, kinetic_units: str) -> None:
        '''
        Populates an dict with kinetic data with the following schema, and saves to the .HTBAM file:
        {kinetics_run_#: {
            name: str,
            type: str,
            assays: {
                1: {
                    conc: float,
                    time: {'values': [...],
                            'unit': 's',
                            'is_quantity': True},,
                    chambers: {
                        1,1: {
                            sum_chamber: {'values': [...],
                                          'unit': 'RFU',
                                          'is_quantity': True},
                            std_chamber: {'values': [...],
                                          'unit': 'RFU'}
                                          'is_quantity': True},
                        },
                        ...
                        }}}}

        Parameters:
            kinetic_data_path (str): Path to kinetic data
            kinetic_name (str): Name of kinetic data
            kinetic_type (str): Type of kinetic data
            kinetic_units (str): Units of kinetic data

        Returns:
                None
        '''    

        kinetic_data_df = pd.read_csv(kinetic_data_path)
        kinetic_data_df['indices'] = kinetic_data_df.x.astype('str') + ',' + kinetic_data_df.y.astype('str')

        def parse_concentration(conc_str: str):
            '''
            Currently, we're storing substrate concentration as a string in the kinetics data.
            This will be changed in the future to store as a float + unit as a string. For now,
            we will parse jankily.
            '''
            print('Warning: parsing concentration from string')
            #first, remove the unit and everything following
            conc = conc_str.split(kinetic_units)[0]
            #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
            conc = float(conc.replace("_", "."))
            return conc
        
        i = 0    
        kin_dict = {}
        for sub_conc, subset in kinetic_data_df.groupby("series_index"):
            squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=["time_s",'sum_chamber', 'std_chamber'])
            
            #turn sub_conc into a pint.Quantity:
            sub_conc = np.array(parse_concentration(sub_conc)) * self.ureg(kinetic_units)

            #turn time into a pint.Quantity:
            time_quantity = np.array(squeezed.iloc[0]["time_s"]) * self.ureg("s")

            #now, we need to properly format the data for each chamber:
            chambers_dict_unformatted = squeezed.drop(columns=["time_s", "indices"]).to_dict("index")
            chambers_dict = {}
            for chamber_coord, chamber_data in chambers_dict_unformatted.items():
                chambers_dict[chamber_coord]  = {}
                for key, value in chamber_data.items():
                    #convert to pint.Quantity:
                    chambers_dict[chamber_coord][key] = value * self.ureg("RFU")

            #make dict for each assay
            kin_dict[i] = {
                "conc": sub_conc, 
                "time": time_quantity,
                "chambers": chambers_dict}
            i += 1

        kinetics_run_num = len([key for key in self.get(Path('runs')) if "kinetics_" in key])
        kinetics_data_dict = {
            "name": kinetic_name,
            "type": kinetic_type,
            "assays": kin_dict
        }
        
        #append to file
        kinetics_data_dict = self._make_serializable_dict(kinetics_data_dict) #convert all quantities to dicts so we can save to json
        self._update_file(Path("runs") / f"kinetics_{kinetics_run_num}", kinetics_data_dict)

    def load_button_quant_data_from_file(self, kinetic_data_path: str) -> None:
        '''
        Populates an json_dict with kinetic data with the following schema:
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
        kinetic_data_df = pd.read_csv(kinetic_data_path)
        kinetic_data_df['indices'] = kinetic_data_df.x.astype('str') + ',' + kinetic_data_df.y.astype('str')
        try:
            unique_buttons = kinetic_data_df[["summed_button_Button_Quant","summed_button_BGsub_Button_Quant",
            "std_button_Button_Quant", "indices"]].drop_duplicates(subset=["indices"]).set_index("indices")
        except KeyError:
            raise HtbamDBException("ButtonQuant columns not found in kinetic data.")
            
        button_quant_dict = unique_buttons.to_dict("index")
        #convert to pint.Quantity:
        for chamber_coord, chamber_dict in button_quant_dict.items():
            for key, value in chamber_dict.items():
                button_quant_dict[chamber_coord][key] = np.array([value]) * self.ureg("RFU")


        button_quant_dict = self._make_serializable_dict(button_quant_dict) #convert all quantities to dicts so we can save to json
        self._update_file("button_quant", button_quant_dict)


    ##########################################################################################
    ############################## UTILITIES: ANALYSIS  ######################################
    ##########################################################################################
    def get_chamber_coords(self):
        '''
        Returns the chamber ids for a given run.
        Input:
            None
        Output:
            chamber_ids: an array of the chamber ids (in the format '1,1' ... '32,56')
                shape: (n_chambers,)
        '''
        chamber_coords = np.array(list(self.get('chamber_metadata').keys()))
        return chamber_coords

    def get_chamber_names(self):
        '''
        Returns the chamber names for a given run.
        Input:
            None
        Output:
            chamber_names: an array of the chamber names (in the format 'ecADK'...)
                shape: (n_chambers,)
        '''
        metadata_dict = self.get('chamber_metadata')
        chamber_coords = self.get_chamber_coords() #this way chamber_coords and chamber_names are always the same order.
        chamber_names = np.array([metadata_dict[chamber_coord]['id'] for chamber_coord in chamber_coords])
        return chamber_names
    
    def get_run_data(self, run_name):
        '''
        Returns the data from a given run as numpy arrays.
        Input: 
            run_name (str): the name of the run to be converted to numpy arrays.
        Output:
            chamber_ids: an array of the chamber ids (in the format '1,1' ... '32,56')
                shape: (n_chambers,)
            luminance_data: an array of the luminance data for each chamber
                shape: (n_time_points, n_chambers, n_assays)
            conc_data: an array of the concentration data for each chamber.
                shape: (n_assays,)
            time_data: an array of the time data for each time point.
                shape: (n_time_points, n_assays)
        '''
        #get data from this run as a dict:
        run_data = self.get(Path("runs") / run_name)
        #convert all serialized quantities to pint.Quantities:
        run_data = self._make_quantities_from_serialized_dict(run_data)

        #get chamber_coords from file:
        chamber_coords = np.array(list(self.get('chamber_metadata').keys()))
        luminance_data = None
        time_data = None
        conc_data = np.array([])

        #Each assay may have recorded a different # of time points.
        #First, we'll just check what the max # of time points is:
        max_time_points = 0
        for assay in run_data['assays'].keys():
            current_assay_time_points = len(run_data['assays'][assay]['time'])
            if current_assay_time_points > max_time_points:
                max_time_points = current_assay_time_points

        for assay in run_data['assays'].keys():
            #to make things easier later, we'll be sorting the datapoints by time value.
            #Get time data:
            current_time_array = run_data['assays'][assay]['time']
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
            for chamber_idx in chamber_coords:
                #collect from DB
                current_chamber_array = run_data['assays'][assay]['chambers'][chamber_idx]['sum_chamber']
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
            current_conc = run_data['assays'][assay]['conc']
            conc_data = np.append(conc_data, current_conc)

        #sort once more, by conc_data:
        sorting_idxs = np.argsort(conc_data)
        conc_data = conc_data[sorting_idxs]

        #sort luminance data by conc_data:
        luminance_data = luminance_data[:,:,sorting_idxs]
        
        return chamber_coords, luminance_data, conc_data, time_data
    
    def save_new_analysis(self, run_name, analysis_name, chamber_dict):
        '''
        Creates a new analysis in the database. 
        Input:
            analysis_name (str): the name of the analysis to be created.
            run_name (str): the name of the run to be analyzed.
            analysis_type (str): the type of analysis to be performed.
            analysis_params (dict): a dictionary of parameters for the analysis.
        Output:
            None
        '''

        analysis_path = Path("runs") / run_name / 'analyses' / analysis_name

        #use get(path) to check if the analysis already exists:
        if 'analyses' not in self.get(Path("runs") / run_name).keys():
            self._update_file(Path("runs") / run_name / "analyses", {})

        #our analysis must have one entry for each chamber. Let's verify this:
        chamber_coords = self.get_chamber_coords()
        for chamber_coord in chamber_coords:
            if chamber_coord not in chamber_dict.keys():
                raise ValueError(f"Chamber {chamber_coord} is missing from the analysis data. \n \
                                    Analysis must have one entry for each chamber.")

        analysis_dict = {
            "chambers": chamber_dict,
        }

        #write to file
        analysis_dict = self._make_serializable_dict(analysis_dict)
        self._update_file(analysis_path, analysis_dict)
    

    

    def export_json(self):
        '''This writes the database to file, as a dict -> json'''
        with open('db.json', 'w') as fp:
            json.dump(self._json_dict, fp, indent=4)

class HtbamDBException(Exception):
    pass
