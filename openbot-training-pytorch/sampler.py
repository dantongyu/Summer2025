'''
Name: sampler.py
Description: Sampler for reading data from the dataset and setting up data for training
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import pathlib

import mp_sharedmap
import numpy as np
import torch
import os
from device import device
from klogs import kLogger
from openbot import list_dirs, load_labels, load_labels_map
from scipy.stats import truncnorm
from torch.utils.data import Dataset, random_split
from torchvision import io, transforms
from typing import List
import pandas as pd

TAG = "SAMPLER"
log = kLogger(TAG)

import config
from config import CONFIG

def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)

from scipy.special import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals


def process_data(sample):
    '''
    Processes a sample and return it in the correct formats
    Args:
        sample (tuple): tuple of (steering, throttle, image)
    Returns:
        tuple: tuple of (steering, throttle, image)
    '''
    steer, throttle, image = sample
    if CONFIG["device"] != "mps":
        if isinstance(image, str):
            image = io.read_image(image, mode=io.ImageReadMode.RGB)
        else:
            data = torch.tensor(bytearray(image), dtype=torch.uint8)
            image = io.decode_image(data, mode=io.ImageReadMode.RGB)
    return steer, throttle, image


class ImageSampler(Dataset):
    '''
    ImageSampler class - a class for sampling images from the dataset

    Args:
        dataset_path (str): path to dataset
        use_cuda (bool): whether to use cuda for image processing
            If Ture there will be ~400M * num_of_processes additional GPU memory usage.
            It's recommended to set the number of processes to round 4 if use_cuda is True.
            Use False if there is not enough GPU memory, or the CPU is not the bottleneck.

    Methods:
        prepare_datasets(tfrecords)
        load_sample(dataset_paths)
        load_sample_tfrecord(dataset_path)
        load_sample_openbot(dataset_path)
        process(img, steering, throttle)
    '''
    def __init__(self, dataset_path, max_size=100000, testing=False):
        self.max_size = max_size
        self.testing = testing
        self.device = torch.device(CONFIG["device"])

        self.transforms = []
        self.transforms.append(transforms.Resize((CONFIG["resize"], CONFIG["resize"]), antialias=True))

        jitter_config = ["brightness", "contrast", "hue", "saturation"]
        if not testing and any(CONFIG[x] != 0 for x in jitter_config):
            self.transforms.append(
                transforms.ColorJitter(**{x: CONFIG[x] for x in jitter_config})
            )

        self.transform = transforms.Compose(self.transforms)

        self.df = pd.DataFrame()
        self.steering_factor = CONFIG["steering_factor"]  # when steering is 1, we move 300 pixel (assumption)
        max_aug = CONFIG["max_augmentation"]
        self.output_features = CONFIG["output_features"] 
        self.max_steering_aug = max_aug / self.steering_factor
        self.scale = self.max_steering_aug / 2
        self.prepare_datasets(dataset_path)

    def test_config(self):
        print(CONFIG)

    def load_sample_tfrecord(self, dataset_path) -> List:
        '''
        Loads a sample from a tfrecord dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        from torchdata.datapipes.iter import FileOpener
        return [
            (sample["steer"].item(), sample["throttle"].item(), sample["image"][0]) 
            for sample in FileOpener(dataset_path, mode="b").load_from_tfrecord()
        ]

    def load_sample_openbot(self, dataset_path):
        '''
        Loads a sample from an openbot dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        samples = []
        count = 0
        for image_path, ctrl_cmd in load_labels(dataset_path, list_dirs(dataset_path)).items():
            if pathlib.Path(image_path).exists():
                samples.append((
                    float(ctrl_cmd[1]) / 255.0,  # steer
                    float(ctrl_cmd[0]) / 255.0,  # throttle
                    image_path,  # image
                ))
            else:
                log.debug(f"File not found: {image_path}")
                # XXX: do not report every missing file before we fix the frame matching problem
                count += 1

        if count > 0:
            log.error(f"Found {count} missing images")

        return samples

    def load_sample_ios(self, dataset_paths):
        '''
        Loads a sample from an iOS dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        samples = []
        count = 0
        for image_path, ctrl_cmd in load_labels(dataset_paths, list_dirs(dataset_paths), ios=True).items():
            if pathlib.Path(image_path).exists():
                samples.append((
                    (float(ctrl_cmd[0]) * 2 / 255.0) - 1,  # steer
                    (float(ctrl_cmd[1]) * 2 / 255.0) - 1,  # throttle
                    image_path,  # image
                ))
            else:
                log.debug(f"File not found: {image_path}")
                # XXX: do not report every missing file before we fix the frame matching problem
                count += 1

        if count > 0:
            log.error(f"Found {count} missing images")

        return samples

    def load_sample(self, dataset_paths):
        '''
        Loads a sample from a generic dataset
        Args:
            dataset_paths (str): path to dataset
        Returns:
            list: list of samples
        '''
        # XXX: Some compatibility with old tfrecord datasets
        # This is not that efficient, so eventually we will  
        # put everything into a datapipe.
        samples = []
        for dataset_path in dataset_paths:
            if CONFIG["ios_dataset"]: 
                samples.extend(self.load_sample_ios(dataset_path))
            elif not pathlib.Path(dataset_path).is_dir():
                samples.extend(self.load_sample_tfrecord([str(dataset_path)]))
            else:
                samples.extend(self.load_sample_openbot(dataset_path))
        return samples

    def map(self, fn, dataset):
        if CONFIG["device"] == "mps":#CONFIG["ios_dataset"]:
            steering = []
            throttle = []
            imgs = []
            for tup in dataset:
                str, thr, img = fn(tup)
                steering.append(str)
                throttle.append(thr)
                imgs.append(img)
            self.size = len(steering)
            self.df.insert(0, "steering", steering)
            self.df.insert(1, "throttle", throttle)
            self.df.insert(2, "image", imgs)
            return steering,throttle,imgs
        else:
            return mp_sharedmap.map(
                fn, dataset, num_workers=CONFIG["num_workers_preprocessing"])

    def bin_index(self, steering):
        #take the range from -1 to 1 and divide it into nBins many ranges
        #return the index of the bin that the steering angle falls into
        return int((steering + 1) * (self.nBins - 1) / 2)
        

    def resample(self, dataset):
        self.nBins = 100
        self.bins = np.zeros(self.nBins)
        binDict = {}
        for tup in dataset:
            index = self.bin_index(tup[0])
            self.bins[index] += 1
            binDict[tup[2]] = index
            
        #now equalize the number of bins by adding more samples to the bins with less samples
        #find the bin with the most samples
        maxBin = np.argmax(self.bins)
        maxBinCount = self.bins[maxBin]
        #now find the bins with less samples and add more samples to them
        samplesToAddPerBinDict = np.zeros(self.nBins, dtype=int)
        for i in range(self.nBins):
            if i != maxBin and self.bins[i] != 0:
                #add more samples to this bin
                #find the number of samples to add
                numSamplesToAdd = maxBinCount - self.bins[i]
                #find the number of samples in this bin
                numSamples = self.bins[i]
                #find the total number to add to each sample
                numSamplesToAddPerSample = int(numSamplesToAdd / numSamples)
                #now add the samples to bin dictionary
                samplesToAddPerBinDict[i] = numSamplesToAddPerSample
        
        #now create a new dataset with the new samples
        newDataset_ind = []
        for idx, tup in enumerate(dataset):
            if samplesToAddPerBinDict[binDict[tup[2]]] != 0:
                newDataset_ind.append(samplesToAddPerBinDict[binDict[tup[2]]])
            else:
                newDataset_ind.append(1)

        return newDataset_ind

    def prepare_datasets(self, dataset_paths):
        """adds the datasets found in directories: dirs to each of their corresponding member variables

        Parameters:
        -----------
        dirs : string/list of strings
            Directories where dataset is stored"""
        dataset = self.load_sample(dataset_paths)
        if len(dataset) > self.max_size:
            dataset_ind = np.sort(np.random.choice(range(len(dataset)), self.max_size, replace=False))
            dataset = [dataset[i] for i in dataset_ind]

        self.dataset_weight = self.resample(dataset)
        self.size = len(dataset)
        self.steering, self.throttle, self.imgs = self.map(process_data, dataset)

    def process_image(self, img, aug_pixel, endShape=(224, 224)):
        '''
        Processes an image and returns it in the correct format
        Args:
            img (np.array): image to be processed
            aug_pixel (int): number of pixels to augment
            rangeY (tuple): range of pixels in Y direction to crop
            rangeX (tuple): range of pixels in X direction to crop
            endShape (tuple): shape of the output image
        Returns:
            np.array: processed image
        Note:
            change the value of rangeY to(58,282) for center crop and (136,360) for bottom crop
        '''
        if CONFIG["ios_dataset"]:
            rangeY=(75,405)
            rangeX=(155,485)
        else:
            rangeX = (155, 485)
            rangeY = (0, 330)

        new_rangeX = (rangeX[0] + aug_pixel, rangeX[1] + aug_pixel)
        img = img[:, rangeY[0]:rangeY[1], new_rangeX[0]:new_rangeX[1]]
        return self.transform(img)

    def get_aug(self, loc):
        max_steering_aug = self.max_steering_aug
        scale = self.scale

        l = max(-1, loc - max_steering_aug)
        r = min(1, loc + max_steering_aug)
        a, b = (l - loc) / scale, (r - loc) / scale
        r = truncnorm(a, b, loc, scale).rvs().astype('float32')
        diff = loc - r
        aug = (diff * self.steering_factor).astype('int32')
        return r, aug

    def process(self, img, steering, throttle):
        '''
        Processes an image and returns it in the correct format, applies translation
        Args:
            img (np.array): image to be processed
            steering (float): steering angle
            throttle (float): throttle angle
        Returns:
            np.array: processed image
            float: steering angle
            float: throttle
        '''
        # Translate the image randomly
        # XXX: should we do image augmentation for testing?
        # XXX: some simple experiments show results shows that
        # XXX: without augmentation the loss and angle error
        # XXX: are basically the same, so it's probably not ok
        # XXX: not to augment the images for testing.
        if self.testing:
            aug = 0
        else:
            r, aug = self.get_aug(steering)
            steering = r  # The pixel to angle conversion is approximate

        if CONFIG["device"] == "mps":
            img = io.read_image(img, mode=io.ImageReadMode.RGB)

        #convert steering and throttle to tensor
        output = torch.tensor([steering, throttle])
        img = self.process_image(img, aug)
        input = img / 256.0

        return input, output

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = self.imgs[index]
        steering = self.steering[index]
        throttle = self.throttle[index]
        return self.process(img, steering.item(), throttle.item())


def load_full_dataset(testing_only=False):
    '''
    Loads a full dataset from a list of paths
    Args:
        dataset_paths (list): list of paths to datasets
        train_test_split (float): percentage of data to be used for training
        use_cuda (bool): whether to use cuda for image processing
            See ImageSampler for more details
    Returns:
        tuple: trainset, testset
    '''
    train_datasets = []
    test_datasets = []
    valid_datasets = []
    for dataset_path in CONFIG["dataset_paths"]:
        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.is_dir() or CONFIG["ios_dataset"]:
            for data_category in os.listdir(dataset_path):
                if data_category == "train":
                    for d in os.listdir(dataset_path / "train"):
                        train_datasets.append(dataset_path / "train" / d)
                elif data_category == "test":
                    for d in os.listdir(dataset_path / "test"):
                        test_datasets.append(dataset_path / "test" / d)
                elif data_category == "valid":
                    for d in os.listdir(dataset_path / "valid"):
                        valid_datasets.append(dataset_path / "valid" / d)
        else:
            if (dataset_path / "train").is_dir():
                train_datasets.append(dataset_path / "train")
            if (dataset_path / "tfrecords" / "train.tfrec").exists():
                train_datasets.append(dataset_path / "tfrecords" / "train.tfrec")
            if (dataset_path / "tfrecords" / "test.tfrec").exists():
                test_datasets.append(dataset_path / "tfrecords" / "test.tfrec")
            if (dataset_path / "test").is_dir():
                test_datasets.append(dataset_path / "test")

    # Create samplers now
    max_size = CONFIG["max_dataset_size"]
    max_size_test = max_size // 10
    if not testing_only:
        trainset = ImageSampler(train_datasets, max_size=max_size)
        if test_datasets:
            testset = ImageSampler(
                test_datasets, max_size=max_size_test, testing=True)
        else:
            train_size = int(len(trainset) * CONFIG["train_test_split"])
            trainset, testset = random_split(
                trainset, [train_size, len(trainset) - train_size])
    # testing_only == True below
    elif test_datasets:
        testset = ImageSampler(
            test_datasets, max_size=max_size_test, testing=True)
    elif valid_datasets:
        testset = ImageSampler(
            valid_datasets, max_size=max_size_test, testing=True)
    else:
        log.warning(
            "No test/validation dataset found, "
            "using training dataset for testing")
        testset = ImageSampler(
            train_datasets, max_size=max_size_test, testing=True)

    # create valid if it exists
    if not testing_only:
        if valid_datasets and CONFIG["validation"]:
            valid_datasets = ImageSampler(
                valid_datasets, max_size=max_size_test)
            log.info(
                f"Training: {len(trainset)}, Testing {len(testset)}, "
                f"Validation {len(valid_datasets)}")
            return trainset, testset, valid_datasets
        else:
            log.info(f"Training: {len(trainset)}, Testing {len(testset)}")
            return trainset, testset
    else:
        log.info(f"Testing: {len(testset)}")
        return testset
