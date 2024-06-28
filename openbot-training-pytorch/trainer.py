'''
Name: trainer.py
Description: Trainer to facilitate training the model, logging, and saving
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import collections
import os
import pathlib
import pickle
import shutil
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import wandb
from device import device
from klogs import kLogger
from models.base import ModelBase
from tqdm.auto import tqdm

TAG = "TRAINER"
log = kLogger(TAG)

import config
from config import CONFIG


def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)


tqdm = partial(tqdm, dynamic_ncols=True)


def to(D, device):
    if isinstance(D, collections.abc.Sequence):
        D = tuple(d.to(device) for d in D)
    elif isinstance(D, collections.abc.Mapping):
        D = {k: v.to(device) for k, v in D.items()}
    elif isinstance(D, torch.Tensor):
        D = D.to(device)
    else:
        raise TypeError(f"Unknown type for dataset: {type(D)}")
    
    return D


class Trainer:
    '''
    Trainer class - a class for training the model, logging, and saving
    Args:
        save_dir (str): path to save directory
        model (torch.nn.Module): model to train
        optim (torch.optim.Optimizer): optimizer to use
        turning_weight (int): weight to use for turning
        epochs (int): number of epochs to train for
        lr (float): learning rate
        bs (int): batch size
    Methods:
        load(fname)
        save(fname)
        train(sampler_train, sampler_test)
    '''
    def __init__(self, model: ModelBase, optim: torch.optim.Optimizer):
        self.model = model
        self.optim = optim
        self.turning_weight = CONFIG["turning_weight"]
        self.epochs = CONFIG["epochs"]
        self.bs=CONFIG["bs"]
        self.lr=CONFIG["lr"]
        self.of=CONFIG["output_features"]

        self.save_dir = pathlib.Path(CONFIG["save_dir"]).joinpath(f"{model.NAME}_{CONFIG['model_spec']}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.save_dir.joinpath("ckpt")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metrics = {}
        self.i = 0

    def update_params(self, epochs=None, lr=None, bs=None, output_features=None):
        '''
        Updates parameters
        Args:
            epochs (int): number of epochs to train for
            lr (float): learning rate
            bs (int): batch size
        Returns:
            None
        '''
        if epochs:
            self.epochs = epochs
        if lr:
            self.lr = lr
        if bs:
            self.bs = bs
        if output_features:
            self.of = output_features

    def load(self):
        # Allow for loading from old format
        if self.save_dir.joinpath("trainer_log.npz").exists():
            return self._load_old()

        trainer = self.save_dir.joinpath("trainer_log.pkl")
        save_model = self.save_dir.joinpath("last.pth")
        if trainer.exists and save_model.exists():
            log.info(f"Loading trainer from {self.save_dir}")

            state = torch.load(save_model)
            self.model.load_state_dict(state["state"])
            self.optim.load_state_dict(state["optim"])

            with open(trainer, "rb") as f:
                self.i = pickle.load(f)
                self.best_metrics = pickle.load(f)

    def save(self):
        torch.save({
            "state": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, self.save_dir.joinpath(f"last.pth"))

        with open(self.save_dir.joinpath("trainer_log.pkl"), "wb") as f:
            pickle.dump(self.i, f)
            pickle.dump(self.best_metrics, f)

    def _load_old(self) -> None:
        '''
        Loads a trainer from a file
        Args:
            fname (str): path to file
        Returns:
            None
        '''
        trainer = self.save_dir.joinpath("trainer_log.npz")
        save_model = self.save_dir.joinpath("last.pth")

        if trainer.exists() and save_model.exists():
            log.info(f"Loading trainer from {self.save_dir}")

            state = torch.load(save_model)
            self.model.load_state_dict(state["state"])
            self.optim.load_state_dict(state["optim"])

            data = np.load(trainer)
            self.i = data["i"].item()
            self.best_metrics == {
                "loss": data["best_loss"],
                "angle_err": 1 - data["best_angle_metric"],
                "direction_err": 1 - data["best_direction_metric"],
            }

    def train(self, sampler_train, sampler_test, sampler_valid=None):
        '''
        Trains the model
        Args:
            sampler_train (torch.utils.data.DataLoader): training data
            sampler_test (torch.utils.data.DataLoader): testing data
        Returns:
            None
        '''
        epochs = self.epochs
        batches_train = len(sampler_train)
        batches_test = len(sampler_test)
        batches_valid = len(sampler_valid) if sampler_valid else 0

        epochs_bar = tqdm(total=epochs, initial=self.i)
        epochs_bar.set_description("Epochs")
        batch_bar = tqdm(total=batches_train)

        epochs_bar.refresh()
        while self.i < epochs:
            # Training
            batch_bar.set_description("Training")
            batch_bar.reset(batches_train)

            for _, D in enumerate(sampler_train):
                # Tensor Processing
                D = to(D, device)

                # Using the model for inference
                self.optim.zero_grad()
                loss, pred = self.model.forward_loss(D)

                # Backpropogation
                loss.backward()
                self.optim.step()

                # Some extra metrics to grade performance by
                loss = loss.item()
                metrics = self.model.metrics(pred, D)
                metrics["loss"] = loss

                batch_bar.set_postfix(ordered_dict=OrderedDict(
                    **{k: f"{v: .2g}" for k, v in metrics.items()}
                ))
                batch_bar.update()

                metrics = {f"{k}/train": v for k, v in metrics.items()}
                metrics["epoch"] = self.i
                wandb.log(metrics)

            # Validation/Testing
            batch_bar.set_description("Test")
            batch_bar.reset(batches_test)
            test_metrics = []
            for _, D in enumerate(sampler_test):
                # Tensor Processing
                D = to(D, device)

                # Test on Validation Set
                with torch.no_grad():
                    loss, pred = self.model.forward_loss(D)

                loss = loss.item()
                metrics = self.model.metrics(pred, D)
                metrics["loss"] = loss
                test_metrics.append(metrics)

                batch_bar.set_postfix(ordered_dict=OrderedDict(
                    **{k: f"{v: .2g}" for k, v in metrics.items()}
                ))
                batch_bar.update()

            test_metrics = {
                k: np.nanmean([i[k] for i in test_metrics])
                for k in test_metrics[0].keys()
            }
            current_metrics = test_metrics.copy()

            test_metrics = {f"{k}/test": v for k, v in test_metrics.items()}
            test_metrics["epoch"] = self.i
            wandb.log(test_metrics)

            if sampler_valid:
                batch_bar.set_description("Validation")
                batch_bar.reset(batches_valid)
                val_metrics = []
                for _, D in enumerate(sampler_valid):
                    # Tensor Processing
                    D = to(D, device)

                    # Test on Validation Set
                    with torch.no_grad():
                        loss, pred = self.model.forward_loss(D)

                    loss = loss.item()
                    metrics = self.model.metrics(pred, D)
                    metrics["loss"] = loss
                    val_metrics.append(metrics)

                    batch_bar.set_postfix(ordered_dict=OrderedDict(
                        **{k: f"{v: .2g}" for k, v in metrics.items()}
                    ))
                    batch_bar.update()

                val_metrics = {
                    k: np.nanmean([i[k] for i in val_metrics])
                    for k in val_metrics[0].keys()
                }
                current_metrics = val_metrics.copy()

                val_metrics = {f"{k}/val": v for k, v in val_metrics.items()}
                val_metrics["epoch"] = self.i
                wandb.log(val_metrics)

            for k, v in current_metrics.items():
                if k not in self.best_metrics or v < self.best_metrics[k]:
                    self.best_metrics[k] = v
                    torch.save({
                        "state": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }, os.path.join(self.save_dir, f"best_{k}.pth"))

            self.i += 1
            # save every `save_interval' epochs
            if (self.i - 1) % CONFIG["save_interval"] == 0:
                self.save()
                shutil.move(
                    self.save_dir.joinpath("last.pth"),
                    self.ckpt_dir.joinpath(f"ckpt_{self.i-1}.pth")
                )

            batch_bar.refresh()
            epochs_bar.update()
