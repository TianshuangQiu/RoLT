import os
import json
import pdb

root_dir = "/home/ethantqiu/tree/data/los_angeles"
train_dir = f"{root_dir}/train"
test_dir = f"{root_dir}/test"
val_dir = f"{root_dir}/test"

with open("config/tree/config.json", "r") as r:
    config = json.load(r)

train_output = []
test_output = []
val_output = []
for k, v in config.items():
    curr_train_dir = f"{train_dir}/{k}"
    curr_test_dir = f"{test_dir}/{k}"
    curr_val_dir = f"{test_dir}/{k}"

    for img in os.listdir(curr_train_dir):
        train_output.append(f"{curr_train_dir}/{img} {v} \n")

    for img in os.listdir(curr_test_dir):
        test_output.append(f"{curr_test_dir}/{img} {v} \n")

    for img in os.listdir(curr_val_dir):
        val_output.append(f"{curr_val_dir}/{img} {v} \n")

with open("config/tree/train.txt", "w") as w:
    w.writelines(train_output)

with open("config/tree/test.txt", "w") as w:
    w.writelines(test_output)

with open("config/tree/val.txt", "w") as w:
    w.writelines(val_output)
