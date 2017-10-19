"""
Configuration module
"""

training_options = {
    "num_class": 2,
    "image_shape": (160, 576),
    "epochs": 30,
    "batch_size": 20,
    "keep_prob": 0.5,
    "learning_rate": 0.001,
    "is_restore": False,
    "restore_model_path": "./runs/2017-10-18-07:55:49.851706"
}

data_paths = {
    "data_dir": "./data",
    "runs_dir": "./runs",
    "model_dir": "./model"
}