"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import yaml
import wandb
import csv
import h5py


class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.cfg_file = os.path.join(self.logdir, "cfg.yaml")
        self.acc_file = os.path.join(self.logdir, "acc.csv")
        self.loss_file = os.path.join(self.logdir, "loss.csv")
        self.ws_file = os.path.join(self.logdir, "ws.h5")
        self.acc_keys = None
        self.loss_keys = None
        self.logging_ws = False
        self.run = wandb.init(project="autoarborist", name="rolt")

    def log_cfg(self, cfg):
        print("===> Saving cfg parameters to: ", self.cfg_file)
        with open(self.cfg_file, "w") as f:
            yaml.dump(cfg, f)
        wandb.log(cfg)

    def log_model(self, model_path, epoch):
        self.run.log_model(path=model_path)

    def log_acc(self, accs):
        if self.acc_keys is None:
            self.acc_keys = [k for k in accs.keys()]
            with open(self.acc_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writeheader()
                writer.writerow(accs)
        else:
            with open(self.acc_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writerow(accs)
        wandb.log(accs)

    def log_loss(self, losses):
        # valid_losses = {k: v for k, v in losses.items() if v is not None}
        valid_losses = losses
        if self.loss_keys is None:
            self.loss_keys = [k for k in valid_losses.keys()]
            with open(self.loss_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writeheader()
                writer.writerow(valid_losses)
        else:
            with open(self.loss_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writerow(valid_losses)
        wandb.log(valid_losses)

    def log_ws(self, e, ws):
        mode = "a" if self.logging_ws else "w"
        self.logging_ws = True

        key = "Epoch{:02d}".format(e)
        # wandb.log(
        #     {
        #         "epoch": e,
        #     }
        # )
        # wandb.Artifact(ws, type="weights", description=f"weights_{key}")
        # wandb.log_model(self.ws_file)
        filepath = os.path.join(self.logdir, f"epoch{e}.h5")
        filepath = self.ws_file
        with h5py.File(filepath, mode) as f:
            g = f.create_group(key)
            for k, v in ws.items():
                g.create_dataset(k, data=v)
