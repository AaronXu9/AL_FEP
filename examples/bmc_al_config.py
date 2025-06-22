"""
Configuration file for BMC Active Learning with GNINA Oracle
"""

# Experiment configuration
EXPERIMENT_CONFIG = {
    "name": "BMC_AL_GNINA",
    "description": "Active Learning on BMC FEP validation set using GNINA oracle",
    
    # Data files
    "data": {
        "sdf_file": "data/targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf",
        "protein_file": "data/BMC_FEP_protein_model_6ZB1.pdb",
        "output_dir": "results/bmc_al_gnina"
    },
    
    # Active learning parameters
    "active_learning": {
        "strategy": "uncertainty_sampling",
        "batch_size": 10,
        "max_rounds": 20,
        "initial_size": 50,
        "random_seed": 42
    },
    
    # GNINA docking configuration
    "gnina": {
        "engine": "gnina",
        "scoring_function": "default",
        "cnn_scoring": True,
        "exhaustiveness": 8,
        "num_poses": 1,
        "search_space": {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0
        }
    },
    
    # Molecule selection criteria
    "selection": {
        "diversity_weight": 0.3,
        "uncertainty_weight": 0.7,
        "filter_druglike": True,
        "max_mw": 500,
        "min_mw": 150
    },
    
    # Output configuration
    "output": {
        "save_individual_rounds": True,
        "save_sdf_files": True,
        "save_plots": True,
        "save_statistics": True
    }
}

# Molecular descriptor configuration
DESCRIPTOR_CONFIG = {
    "basic": ["mw", "logp", "hbd", "hba", "tpsa"],
    "extended": ["rotatable_bonds", "aromatic_rings", "heavy_atoms", "formal_charge"],
    "fingerprints": ["morgan", "rdkit", "maccs"]
}

# Stopping criteria
STOPPING_CRITERIA = {
    "max_rounds": 20,
    "convergence_threshold": 0.01,
    "min_improvement": 0.005,
    "patience": 3
}
