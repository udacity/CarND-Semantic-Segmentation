"""
Configuration module
"""

training_options = {
    "num_class": 2,
    "image_shape": (160, 576),
    "epochs": 300,
    "batch_size": 1,
    "keep_prob": 0.5,
    "learning_rate": 0.001
}

data_paths = {
    "data_dir": "./data",
    "runs_dir": "./runs",
    "model_dir": "./model"
}