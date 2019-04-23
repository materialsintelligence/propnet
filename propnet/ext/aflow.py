from matminer.data_retrieval.retrieve_AFLOW import AFLOWDataRetrieval


class AFLOWRetrieve(AFLOWDataRetrieval):
    # Mapping is keyed by AFLOW keyword, valued by propnet property
    mapping = {
        "auid": "external_identifier_aflow",
        "Egap": "band_gap",
        "ael_bulk_modulus_reuss": "bulk_modulus",
        "ael_bulk_modulus_voigt": "bulk_modulus",
        "ael_elastic_anistropy": "elastic_anisotropy",
        "ael_poisson_ratio": "poisson_ratio",
        "ael_shear_modulus_reuss": "shear_modulus",
        "ael_shear_modulus_voigt": "shear_modulus",
        "agl_acoustic_debye": "debye_temperature",
        "agl_debye": "debye_temperature",
        "agl_gruneisen": "gruneisen_parameter",
        # Listed per unit cell, need new symbol and model for conversion
        # "agl_heat_capacity_Cp_300K": "cell_heat_capacity_constant_pressure",
        # "agl_heat_capacity_Cv_300K": "cell_heat_capacity_constant_volume",
        "agl_thermal_conductivity_300K": "thermal_conductivity",
        "agl_thermal_expansion_300K": "thermal_expansion_coefficient",
        # This returns the volumes of all the atoms' volumes
        # "bader_atomic_volumes": ""
        "structure": "structure",
        "compound": "formula",
        "energy_atom": "energy_per_atom",
        "enthalpy_formation_atom": "formation_energy_per_atom"
    }
    unit_map = {
        "Egap": "eV",
        "ael_bulk_modulus_reuss": "gigapascal",
        "ael_bulk_modulus_voigt": "gigapascal",
        "ael_elastic_anistropy": "dimensionless",
        "ael_poisson_ratio": "dimensionless",
        "ael_shear_modulus_reuss": "gigapascal",
        "ael_shear_modulus_voigt": "gigapascal",
        "agl_acoustic_debye": "kelvin",
        "agl_debye": "kelvin",
        "agl_gruneisen": "dimensionless",
        # "agl_heat_capacity_Cp_300K": "boltzmann_constant",
        # "agl_heat_capacity_Cv_300K": "boltzmann_constant",
        "agl_thermal_conductivity_300K": "watt/meter/kelvin",
        "agl_thermal_expansion_300K": "1/kelvin",
        "energy_atom": "eV/atom",
        "enthalpy_formation_atom": "eV/atom"
    }
    
    