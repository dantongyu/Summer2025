[
    {
        "------Directorys------": "------Directorys------",
        "save_dir": "models_ckpt",
        "ios_dataset" : true,
        "train_test_split": 0.8,
        "validation": true,
        "max_dataset_size": 100000,
        "dataset_paths": ["mapping_data"],
        "num_workers_preprocessing": 16,
        "num_workers": 6,
        "---Augmentation Params---": "---Augmentation Params---",
        "resize" : 224,
        "brightness" : 0.1,
        "contrast" : 0.1,
        "hue" : 0.05,
        "saturation" : 0.05,
        "steering_factor": 155,
        "max_augmentation": 155,
        "bcurve_scale": 15,
        "num_aug": 2,
        "seq_stride_min": 5,
        "seq_stride_max": 7,
        "----Model Features----": "----Model Features----",
        "model": "gru",
        "model_spec": "h2x500"
        "sequence_length" : 10,
        "permute": true,
        "edge_filter": true,
        "crop_filter": false,
        "old_model": false,
        "turning_weight": 5,
        "epochs": 100,
        "lr": 0.009196,
        "bs": 256,
        "eps": 7.271e-8,
        "b1": 0.5288,
        "b2": 0.5614,
        "output_features": 10,
        "save_interval": 100,
        "device" : "mps",
        "----Model Conversion----": "----Model Conversion----",
        "input_model": "ckpt_800.pth",
        "output_model": "ckpt_800.mlpackage",
        "conversion": "coreml",
        "----Model Training----": "----Model Training----",
        "task": "train",
        "visualize": false,
        "sweep_id": null,
        "sweep_configuration": 
        {
            "method" : "random",
            "name" : "sweep",
            "metric": 
            {
                "goal": "maximize",
                "name":"val_angle"
            },
            "parameters":
            {
                "lr": {"max":0.01,"min":1e-6},
                "b1": {"max":0.9, "min":0.1},
                "b2": {"max":0.9, "min":0.1},
                "eps":{"max":1e-7, "min":1e-9}
            }
        },
        "------End------": "------End------"
    }
]
