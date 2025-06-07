import numpy as np

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

def process_dataframe_kinetics(df):
    '''
    Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'RFU_data' format.
    
    Arguments:
        df: DataFrame of experiment data

    Returns:
        data_3d: Data3D object, with metadata, indep_vars, and dep_vars.
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
        button_quant = np.nan * np.ones(len(chamber_ids))  # If no button quant, fill with NaNs

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
    # Expand by 1 axis so it's (n_concentrations, n_timepoints, n_chambers, 1)
    RFU_array = RFU_array[..., np.newaxis]

    ### Create the data object:
    from htbam_db_api.data import Data4D, IndepVars, Meta
    # Independent variables:
    indep_vars = IndepVars(concentrations, chamber_ids, sample_ids, button_quant, time_array)
    # Meta data:
    meta = Meta()  # Currently empty, but can be extended in the future.
    # 3D Data object:
    data_4d = Data4D(indep_vars=indep_vars, 
                     meta=meta,
                     dep_var=RFU_array, 
                     dep_var_type=['luminance'])

    return data_4d

def process_dataframe_binding(self, df):
    '''
    Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'binding' format.
    TODO: Not implemented.
    '''
    pass

def parse_concentration(conc_str: str, unit_name: str) -> float:
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
