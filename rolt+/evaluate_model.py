from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from EfficientNet import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import wandb
import torchmetrics
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch WebVision Training")
# parser.add_argument("name", help="name of wandb experiment")
parser.add_argument("--batch_size", default=32, type=int, help="train batchsize")
parser.add_argument(
    "--lr", "--learning_rate", default=0.01, type=float, help="initial learning rate"
)
parser.add_argument("--alpha", default=0.5, type=float, help="parameter for Beta")
parser.add_argument(
    "--lambda_u", default=0, type=float, help="weight for unsupervised loss"
)
parser.add_argument(
    "--p_threshold", default=0.5, type=float, help="clean probability threshold"
)
parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
parser.add_argument("--num_epochs", default=80, type=int)
parser.add_argument("--id", default="", type=str)
parser.add_argument("--seed", default=123)
parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--num_class", default=100, type=int)
parser.add_argument(
    "--data_path", default="./dataset/", type=str, help="path to dataset"
)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="autoarborist",
#     # track hyperparameters and run metadata
#     config=args,
#     name="ROLT_" + args.name,
# )
recall = torchmetrics.Recall(
    task="multiclass", average="macro", num_classes=args.num_class
).to("cuda")


def test(epoch, net1, net2, test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # if batch_idx > 3:
            #     break
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            acc_meter.add(outputs, targets)
            recall.update(predicted, targets)
    accs = acc_meter.value()
    return accs


def eval_train_baseline(model, all_loss):
    model.eval()
    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            recall.update(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
            sys.stdout.write("\r")
            sys.stdout.write(
                "| Evaluating loss Iter[%3d/%3d]\t" % (batch_idx, num_iter)
            )
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def eval_train(model, all_loss=None):
    model.eval()
    total_features = torch.empty((0, feat_size)).cuda()
    total_labels = torch.empty(0).long().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(tqdm(eval_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats = model(inputs, return_features=True)
            total_features = torch.cat((total_features, feats))
            total_labels = torch.cat((total_labels, targets))

    # pdb.set_trace()
    cfeats = get_knncentroids(feats=total_features, labels=total_labels)
    ncm_classifier.update(cfeats)
    ncm_logits = ncm_classifier(total_features, None)[0]

    refined_ncm_logits = ncm_logits
    refine_times = 1
    for _ in range(refine_times):
        mask = get_gmm_mask(refined_ncm_logits, total_labels)
        refined_cfeats = get_knncentroids(
            feats=total_features, labels=total_labels, mask=mask
        )
        ncm_classifier.update(refined_cfeats)
        refined_ncm_logits = ncm_classifier(total_features, None)[0]

    prob = get_gmm_prob(refined_ncm_logits, total_labels)
    return prob, None


def get_gmm_mask(ncm_logits, total_labels):
    mask = torch.zeros_like(total_labels).bool()

    for i in range(args.num_class):
        this_cls_idxs = total_labels == i
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_preds = gmm.predict(this_cls_logits)
        inner_cluster = gmm.means_.argmax()

        this_cls_mask = mask[this_cls_idxs]
        this_cls_mask[gmm_preds == inner_cluster] = True

        if (gmm_preds != inner_cluster).all():
            this_cls_mask |= True  # not to exclude any instance

        mask[this_cls_idxs] = this_cls_mask
    return mask


def get_gmm_prob(ncm_logits, total_labels):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = total_labels == i
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()


def get_knncentroids(feats=None, labels=None, mask=None):
    feats = feats.cpu().numpy()
    labels = labels.cpu().numpy()

    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        # pdb.set_trace()
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype("bool")
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()

        centroids = []
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_ == i) & mask_], axis=0))
        return np.stack(centroids)

    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels, mask)

    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {
        "mean": featmean,
        "uncs": un_centers,
        "l2ncs": l2n_centers,
        "cl2ncs": cl2n_centers,
    }


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = EfficientNet(num_classes=args.num_class)
    # model = InceptionResNetV2(num_classes=args.num_class)
    # wandb.log(
    #     {
    #         "model parameters": sum(
    #             p.numel() for p in model.parameters() if p.requires_grad
    #         )
    #     }
    # )
    model = model.cuda()
    return model


stats_log = open("./checkpoint/%s" % (args.id) + "_stats.txt", "w")
test_log = open("./checkpoint/%s" % (args.id) + "_acc.txt", "w")


warm_up = 1

loader = dataloader.treevision_dataloader(
    batch_size=args.batch_size,
    num_workers=5,
    root_dir=args.data_path,
    log=stats_log,
    num_class=args.num_class,
)
eval_loader = loader.run("eval_train")
feat_size = 1280
# feat_size = 1536
ncm_classifier = KNNClassifier(feat_size, args.num_class)

print("| Building net")
net1 = create_model()
net2 = create_model()
net1.load_state_dict(
    torch.load("/home/ethantqiu/RoLT/checkpoint/net1_efficientnet_s_1.pth")
)
net2.load_state_dict(
    torch.load("/home/ethantqiu/RoLT/checkpoint/net1_efficientnet_s_2.pth")
)

cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# wandb.watch(net1, log_graph=True)
# wandb.watch(net2, log_graph=True)


CE = nn.CrossEntropyLoss(reduction="none")
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
web_valloader = loader.run("test")
web_acc = test(0, net1, net2, web_valloader)
# imagenet_acc = test(epoch, net1, net2, imagenet_valloader)
imagenet_acc = [0, 0]
print(recall.compute())

test_log.flush()
recall.reset()
print("\n==== net 1 evaluate training data loss ====")
prob1, all_loss[0] = eval_train(net1, all_loss[0])
print("\n==== net 2 evaluate training data loss ====")
prob2, all_loss[1] = eval_train(net2, all_loss[1])
print(recall.compute())
