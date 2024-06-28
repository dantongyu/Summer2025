
import pathlib
from collections import OrderedDict

import numpy as np
import torch
from config import CONFIG
from dataloader import FastDataLoader
from device import device
from metrics import angle_metric, direction_metric
from metrics import loss_path as loss_fn
from models import get_model
from sampler import load_full_dataset
from tqdm.auto import tqdm
from klogs import kLogger

TAG = "TEST"
log = kLogger(TAG)


model = get_model(CONFIG["model"])().to(device)
model_save_dir = pathlib.Path(CONFIG["save_dir"]).joinpath(
    f"{CONFIG['model']}_{CONFIG['model_spec']}").joinpath("last.pth")
model.load_state_dict(torch.load(model_save_dir)["state"])
log.info(f"Loaded model from {model_save_dir}")
model.eval()

testset = load_full_dataset(testing_only=True)
sampler_test = FastDataLoader(
    testset, batch_size=CONFIG["bs"], shuffle=True,
    num_workers=CONFIG["num_workers"], pin_memory=True)

batches_test = len(sampler_test)
test_avg = torch.zeros(batches_test, 3)  # loss, angle, direction_sum
direction_num = 0
batch_bar = tqdm(total=batches_test, desc="Test")
for i, (input, expected) in enumerate(sampler_test):
    Y = expected.to(device)
    X = input.to(device)

    with torch.no_grad():
        Y_pred = model(X)

    # Test on Validation Set
    tst_loss = loss_fn(Y, Y_pred)
    ang_metric = angle_metric(Y_pred, Y)
    dir_metric, num = direction_metric(Y_pred, Y)
    test_avg[i] = torch.stack([tst_loss, ang_metric, dir_metric])
    direction_num += num

    batch_bar.set_postfix(ordered_dict=OrderedDict(
        Loss=f"{tst_loss.item(): .2g}",
        Angle=f"{ang_metric.item(): .2g}",
        Direction=f"{dir_metric.item() / num if num > 0 else np.nan: .2g}",
    ))
    batch_bar.update()

validation_avg = test_avg.sum(dim=0)
validation_avg[:2] /= batches_test
validation_avg[2] /= direction_num

print(f"Test Loss: {validation_avg[0]: .2g}")
print(f"Test Angle: {validation_avg[1]: .2g}")
print(f"Test Direction: {validation_avg[2]: .2g}")
