# Training Pipeline using PyTorch

## Dependencies

Explicit dependencies:

- ipykernel
- ipywidgets
- matplotlib
- numpy
- opencv-python
- pytorch
- scipy
- torchdata (optional for tfrecord support)
- torchvision
- tqdm
- ultralytics(optional for using yolo)
- wandb

## Dataset

Before running the script, make sure the dataset is with correct structure. We assume the dataset is in the `mydataset` folder.

There are three different types of dataset structure (for historical reason): `openbot`, `openbot_v2`, and `ios`. The `openbot` and `openbot_v2` is for openbot (Android) dataset, and `ios` is for iOS dataset. Use `ios_dataset` switch in config file to switch between openbot and ios.

Below shows the dataset structure for each type of dataset.

1. `openbot` dataset

    ```
    mydataset
    └── tfrecords
        ├── train.tfrec
        └── test.tfrec
    ```

2. `openbot-v2` dataset

    ```
    mydataset
    ├── train
    │   ├── <dataset_folder_1>
    │   │   ├── images
    │   │   │   ├── xxxx_preview.jpeg
    │   │   │   └── [* more images *]
    │   │   └── sensor_data
    │   │       ├── ctrlLog.txt
    │   │       ├── indicatorLog.txt
    │   │       ├── inferenceTime.txt
    │   │       └── rgbFrames.txt
    │   └── [* more dataset_folder *]
    └── test
        └── [* same structure as train *]
    ```

3. `ios` dataset

    ```
    mydataset
    ├── train
    │   ├── <dataset_folder_1>
    │   │   ├── depth
    │   │   │   ├── xxxxxxxx.jpeg
    │   │   │   └── [* more depth images *]
    │   │   ├── images
    │   │   │   ├── xxxxxxxx.jpeg
    │   │   │   └── [* more images *]
    │   │   ├── control
    │   │   └── motion
    │   └── [* more dataset_folder *]
    └── test
        └── [* same structure as train *]
    ```

You can use

    tree --filelimit 10 mydataset

to verify the data structure.

For `openbot-v2` dataset, use the following command to prepare the dataset

    python prepare_dataset.py mydataset

## Training

Run `train.py` with config file:

    CONFIG=<config>.json python train.py

Refer to `config.json.ex` for an example of the config file.

You can use `DEVICE=<device>` environment variable to specify the device to use, e.g.,

    DEVICE=cuda:0 CONFIG=config-resnet.json python train.py

## Known Issues

- Training script using multiprocessing to process data, which is incompatible with GNU OpenMP. The way to work around this is to set `torch.set_num_threads(1)` for the worker process.

- NumPy prefer intel openmp while PyTorch prefer GNU OpenMP.  To ensure GNU OpenMP is used, always import PyTorch before NumPy.
