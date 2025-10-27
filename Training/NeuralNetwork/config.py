# config.py
MODEL_CONFIGS = {
    # ──────────────────────────────────────────────────────────────────────────
    # Three “quick” White‐side models
    # ──────────────────────────────────────────────────────────────────────────

    "white_quick_1": {
        "data_path": "CSVFiles/White.csv",
        "samples": 500000,
        "save_path": "Chess_Full_Training_v1",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 42,
    },

    "white_quick_2": {
        "data_path": "CSVFiles/White.csv",
        "samples": 500000,
        "save_path": "Chess_white_quick_2",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 43,
    },

    "white_quick_3": {
        "data_path": "CSVFiles/White.csv",
        "samples": 500000,
        "save_path": "Chess_white_quick_3",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 44,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Three “quick” Black‐side models
    # ──────────────────────────────────────────────────────────────────────────

    "black_quick_1": {
        "data_path": "CSVFiles/Black.csv",
        "samples": 500000,
        "save_path": "Chess_black_quick_1",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 42,
    },

    "black_quick_2": {
        "data_path": "CSVFiles/Black.csv",
        "samples": 500000,
        "save_path": "Chess_black_quick_2",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 43,
    },

    "black_quick_3": {
        "data_path": "CSVFiles/Black.csv",
        "samples": 500000,
        "save_path": "Chess_black_quick_3",
        "threshold": 20,
        "batch_size": 2048,
        "residual_blocks": 4,
        "filters": 128,
        "dense_units": 256,
        "dropout_rate": 0.2,
        "learning_rate_phase1": 0.0005,
        "learning_rate_phase2": 5e-05,
        "epochs_phase1": 40,
        "epochs_phase2": 5,
        "epochs_curriculum": 3,
        "epochs_final": 3,
        "test_size": 0.1,
        "patience": 4,
        "show_progress": True,
        "plot_history": True,
        "curriculum_bins": 4,
        "seed": 44,
    },

    "combined_quick_enqueue": {
        # Paths to the three *trained* white models
        "white_models": [
            "Chess_White_Quick_1/final_model.h5",
            "Chess_White_Quick_2/final_model.h5",
            "Chess_White_Quick_3/final_model.h5"
        ],

        # Paths to the three *trained* black models
        "black_models": [
            "Chess_Black_Quick_1/final_model.h5",
            "Chess_Black_Quick_2/final_model.h5",
            "Chess_Black_Quick_3/final_model.h5"
        ],

        # Data for ensemble training
        "white_data_path": "CSVFiles/White.csv",
        "white_samples": 1000000,
        "black_data_path": "CSVFiles/Black.csv",
        "black_samples": 1000000,

        "save_path": "Chess_Combined_Quick",

        # Training hyperparameters
        "batch_size": 2048,
        "learning_rate": 0.0005,
        "epochs": 5, 
        "freeze_submodels": True, 
        "finetune_epochs": 2
    }
}