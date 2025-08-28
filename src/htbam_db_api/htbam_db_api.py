from abc import ABC, abstractmethod, abstractproperty
from htbam_db_api.data import Data2D, Data3D, Data4D, Meta
import pandas as pd
from typing import List, Tuple
import numpy as np
import json
from pathlib import Path
import re
from copy import deepcopy

from htbam_db_api.exceptions import HtbamDBException
from htbam_db_api.io import verify_file_exists, load_run_from_csv

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
        verify_file_exists(standard_curve_data_path)
        verify_file_exists(kinetic_data_path)
        
        # The data is in format 'kinetics' for both standard curve and kinetics.
        standard_data = load_run_from_csv(standard_curve_data_path, 'kinetics', standard_units)
        kinetics_data = load_run_from_csv(kinetic_data_path, 'kinetics', kinetic_units)
        
        self._init_json_dict()

        # Populate with metadata, which was stored in the kinetics dataframe
        # TODO: I think this is unused currently.
        #self.set_metadata('chamber_IDs', kinetics_data['indep_vars']['chamber_IDs'])
        #self.set_metadata('sample_IDs', kinetics_data['indep_vars']['sample_IDs'])

        self.add_run(standard_name, standard_data)
        self.add_run(kinetic_name, kinetics_data)

        # Use the kinetic data to create a separate button_quant run, for later processing:
        self.add_run('button_quant', self.process_button_quant_from_kinetics(kinetics_data))

        return
    

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
        button_quant_sum = kinetics_data.indep_vars.button_quant_sum  # (n_chambers,)
        # Add a new dim, so it's (n_chambers, 1)
        button_quant_sum = button_quant_sum[..., np.newaxis]  # (n_chambers, 1)

        # copy over indep vars
        indep_vars = kinetics_data.indep_vars
        meta = Meta()

        button_quant_data = Data2D(
            indep_vars=indep_vars,
            meta=meta,
            dep_var=np.array(button_quant_sum),  # (n_chambers, 1)
            dep_var_type=["luminance"],  # e.g. "luminance", "product", etc
        )
        
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
    def add_run(self, run_name: str, run_data) -> None:
        '''
        Adds a run to the database.

                Parameters:
                        run_name (str): Name of the run
                        run_data (Data4D, Data3D, etc): Data for the run

                Returns:
                        None
        '''
        # Check if the format matches one of the allowed dataclasses:
        if not isinstance(run_data, (Data2D, Data3D, Data4D)):
            raise HtbamDBException(f"Run data must be of type Data2D, Data3D, or Data4D. Got {type(run_data)}")
        
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
        # TODO: unused
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
        # TODO: This should also return the run names of the runs in db_conn
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
        # TODO: unused
        self._json_dict['metadata'][name] = value

         
    # def export_json(self):
    #     '''This writes the database to file, as a dict -> json'''
    #     with open('db.json', 'w') as fp:
    #         json.dump(self._json_dict, fp, indent=4)
