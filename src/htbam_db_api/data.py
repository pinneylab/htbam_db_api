from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy

@dataclass
class IndepVars:
    concentration: np.ndarray     # (n_conc,)
    chamber_IDs: np.ndarray       # (n_chamb,)
    sample_IDs: np.ndarray        # (n_chamb,)
    button_quant_sum: np.ndarray  # (n_chamb,)
    time: np.ndarray              # (n_conc, n_time)

    def __post_init__(self):
        for var in self.concentration, self.chamber_IDs, self.sample_IDs, self.button_quant_sum:
            if var.ndim != 1:
                raise ValueError(f"Expected 1D array, got {var.shape} for {var}")
        if self.time.ndim != 2:
            raise ValueError(f"time must be 2D, got {self.time.shape}")

@dataclass
class Meta:
    # TODO: flesh this out
    masks: dict = field(default_factory=dict)
    model_fit: dict = field(default_factory=dict)

@dataclass
class Data4D:
    indep_vars: IndepVars
    meta: Meta = field(default_factory=Meta)

    dep_var: np.ndarray           # (n_conc, n_time, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    

    def __post_init__(self):
        # make a full copy so original IndepVars isn’t shared
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()  # validate copied indep vars
        if self.dep_var.ndim != 4:
            raise ValueError(f"dep_var must be 4D, got {self.dep_var.shape}")

@dataclass
class Data3D:
    indep_vars: IndepVars
    meta: Meta = field(default_factory=Meta)

    dep_var: np.ndarray           # (n_conc, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    
    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 3:
            raise ValueError(f"dep_var must be 2D, got {self.dep_var.shape}")

@dataclass
class Data2D:
    indep_vars: IndepVars
    meta: Meta = field(default_factory=Meta)

    dep_var: np.ndarray           # (n_chambers, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    
    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 2:
            raise ValueError(f"dep_var must be 2D, got {self.dep_var.shape}")
