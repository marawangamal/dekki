import os
import os.path as osp
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anki_dataset import ANKIDataset
from utils import AverageMeter, get_experiment_name, save_yaml, apply_padding
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torchmetrics as tm

import hydra
from omegaconf import DictConfig, OmegaConf

from focal_loss import FocalLoss

import transformers


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True,
                 loss="ce", weight=None, samples_per_class=None, gamma=0, **kwargs):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.weight = weight
        self.samples_per_class = samples_per_class
        self.loss = {
            "ce": torch.nn.CrossEntropyLoss(
                # weight=weight,
                reduction="none"
            ),
            "focal": FocalLoss(
                gamma=gamma,
                # alpha=weight
            )
        }[loss]

    def forward(self, x, y=None, mask=None, compute_loss=False):
        """

        Args:
            x: Input features [B, T, D]
            y: Ground truth labels [B, T, 1]
            mask: Mask for valid labels [B, T]
            compute_loss: Override to compute loss

        Returns:
            (tuple) If training, returns loss (float), probs [B, T, C]
            (torch.tensor) If testing, returns prediction [B, T, C]
        """
        h, _ = self.lstm(x)

        if self.training or compute_loss:
            assert y is not None, "Labels must be provided during training"
            yhat = self.linear(h)  # [B, T, C]
            yhat_masked = yhat[mask == 1]
            y_masked = y[mask == 1]
            loss = self.loss(yhat_masked, y_masked)  # [B, T]

            if self.samples_per_class is not None:

                current_sample_prob = self.samples_per_class / self.samples_per_class.sum()
                adjusted_sample_prob = current_sample_prob.prod() / current_sample_prob
                adjusted_sample_prob = adjusted_sample_prob / adjusted_sample_prob.sum()
                # zero_prob = 1.0 - adjusted_sample_prob[y_masked].float()
                new_sample_prob = adjusted_sample_prob[y_masked].float()
                random_tensor = torch.rand(new_sample_prob.size(), device=loss.device)
                mask = random_tensor < new_sample_prob  # 0.5 > 1 => False
                loss = loss * mask.float()

                # for each index in list samples_per_class, compute product of the rest of the indices
                # zero_prob = 1.0 / self.weight[y_masked].float()
                # Normalize
                # normalizing_factor = (1 / self.weight).sum()
                # zero_prob = torch.div(zero_prob, normalizing_factor.sum())
                # random_tensor = torch.rand(zero_prob.size(), device=loss.device)
                # mask = random_tensor > zero_prob  # 0.5 > 1 => False
                # loss = loss * mask.float()

            elif self.weight is not None:
                weights = self.weight[y_masked].float()
                loss = loss * weights

            loss = loss.mean()

            return loss, torch.softmax(yhat, dim=-1)

        else:
            yhat = self.linear(h)  # [B, T, C]
            return torch.softmax(yhat, dim=-1)


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        # B, T, C = x.shape ==> B, T, 1
        output = self.linear(x)
        return output


def load_data(args, batch_size):
    # Load data
    # timestamp, card id, time since last review, time spent viewing, note length, ease
    train_ds = ANKIDataset(args['data']['input'], split="train", shuffle=False)
    test_ds = ANKIDataset(args['data']['input'], split="test")
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)
    return test_dl, train_dl, train_ds


def collate_fn(batch):

    source = [x["source"] for x in batch]  # [B, T1, C], [
    target = [x["target"] for x in batch]  # [B, T1]
    mask = [x["mask"] for x in batch]

    source, target, mask = apply_padding(source, target, mask)

    return {"source": source, "target": target, "mask": mask}


def save_ckpt(args, best_eval_metrics, epoch, eval_metrics, model, optimizer, scheduler, output_path, train_metrics):
    # Save model checkpoint
    if best_eval_metrics is None or eval_metrics['f1'] > best_eval_metrics['f1']:
        best_eval_metrics = eval_metrics

        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'model_state_dict': model_state_dict,
            'torchrandom_state': torch.get_rng_state(),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'best_eval_metrics': best_eval_metrics,
            'config': args
        }, osp.join(output_path, "model_best.pth"))

        # Save the latest model
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'model_state_dict': model_state_dict,
            'torchrandom_state': torch.get_rng_state(),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'best_eval_metrics': best_eval_metrics,
            'config': args
        }, osp.join(output_path, "model_latest.pth"))

    elif epoch % 5 == 0 or epoch == args["train"]["epochs"] - 1:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        # Save the latest model
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'model_state_dict': model_state_dict,
            'torchrandom_state': torch.get_rng_state(),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'best_eval_metrics': best_eval_metrics,
            'config': args
        }, osp.join(output_path, "model_latest.pth"))

    return best_eval_metrics


def accuracy(pred, target, mask=None):
    """ Compute accuracy.

    Args:
        pred (torch.Tensor): Integer predictions [B, *]
        target (torch.Tensor): Integer targets [B, *]
        mask (torch.Tensor): Mask for valid labels [B, *]. Optional.

    Returns:
        Accuracy (float)
    """

    if mask is not None:
        pred = pred[mask == 1]
        target = target[mask == 1]
        return (pred == target).float().mean()
    else:
        return (pred == target).float().mean()


def train_epoch(train_dl, model, optimizer, device):
    model.train()
    loss_avg_meter = AverageMeter()
    acc_avg_meter = AverageMeter()

    for i_batch, batch in enumerate(train_dl):

        source, target, mask = batch["source"], batch["target"], batch["mask"]
        source, target, mask = source.to(device), target.to(device), mask.to(device)

        # Forward pass
        loss, probs = model(source, target, mask)
        loss_avg_meter.update(loss.item())

        # Accuracy
        pred_masked = probs.argmax(dim=-1)[mask == 1]  # [N',]
        target_masked = target[mask == 1]  # [N',]
        acc_avg_meter.update(accuracy(pred_masked, target_masked).item())

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i_batch % 1000 == 0:
            print("\tBatch: {}/{} | Loss: {:.4f}".format(i_batch, len(train_dl), loss.item()))

    return {"loss": loss_avg_meter.value, "acc": acc_avg_meter.value}


def eval_epoch(test_dl, model, device, num_classes=4):
    model.eval()

    f1_fn = tm.classification.MulticlassF1Score(num_classes=num_classes)

    with torch.no_grad():
        loss_avg_meter = AverageMeter()
        acc_avg_meter = AverageMeter()
        y_true = []
        y_pred = []
        for i_batch, batch in enumerate(test_dl):
            source, target, mask = batch["source"], batch["target"], batch["mask"]
            source, target, mask = source.to(device), target.to(device), mask.to(device)
            loss, probs = model(source, target, mask, compute_loss=True)  # [N, C]

            loss_avg_meter.update(loss.item())

            # Accuracy
            pred_masked = probs.argmax(dim=-1)[mask == 1]  # [N',]
            target_masked = target[mask == 1]  # [N',]
            acc_avg_meter.update(accuracy(pred_masked, target_masked).item())

            # For confusion matrix and f1
            y_pred.extend(pred_masked.tolist())
            y_true.extend(target_masked.tolist())

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_fn(torch.tensor(y_pred), torch.tensor(y_true))

        return {
            "loss": loss_avg_meter.value, "acc": acc_avg_meter.value, "cf_matrix": cf_matrix,
            "y_true": y_true, "y_pred": y_pred,
            "diag_acc": np.prod(np.diag(
                (cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
            )).round(2),
            "f1": f1
        }


def train(args, exp_name):
    epochs = args['train']['epochs']
    batch_size = args['train']['batch_size']
    output_path = args['data']['output']
    lr = args['train']['lr']

    # Tensorboard
    writer = SummaryWriter(log_dir=osp.join(output_path, exp_name))
    test_dl, train_dl, train_ds = load_data(args, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = {
        "lstm": LSTM,
        "linear": Linear
    }[args['model']['name'].lower()](
        loss=args['loss']['name'].lower(),
        weight=torch.tensor(train_ds.get_class_weights()).to(device),
        # samples_per_class=torch.tensor(train_ds.get_samples_per_class()).to(device),
        **train_ds.get_data_params(),  # Get input_size, output_size, time_index
        **args['model']['params'],  # Get model params, hidden_size, dropout
        **args['loss']['params']  # Get loss params, gamma
    )

    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)
    logging.info("=> Training on {} GPUs.".format(torch.cuda.device_count()))

    optimizer = {
        "adam": torch.optim.Adam(model.parameters(), lr=lr),
        "sgd": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    }[args['train']['optimizer'].lower()]

    if args['train']['scheduler']['name'].lower() == "none":
        scheduler = None
    else:
        scheduler = {
            "linear": transformers.get_scheduler(
                optimizer=optimizer, num_training_steps=epochs, **args['train']['scheduler']
            )
        }[args['train']['scheduler']['name'].lower()]

    # Load ckpt if exists
    ckpt_path = osp.join(output_path, exp_name, "model_latest.pth")
    if osp.exists(ckpt_path):

        ckpt = torch.load(ckpt_path)
        try:
            model.module.load_state_dict(ckpt['model_state_dict'])
        except:
            model.load_state_dict(ckpt['model_state_dict'])

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        best_eval_metrics = ckpt['best_eval_metrics']
        start_epoch = ckpt['epoch'] + 1
        logging.info("=> Loaded checkpoint from: {}".format(ckpt_path))
        logging.info("=> Resuming from epoch: {}".format(start_epoch))
    else:
        logging.info("=> No checkpoint found at: {}".format(ckpt_path))
        best_eval_metrics = None
        start_epoch = 0

    # train
    for epoch in range(start_epoch, epochs):
        # train_metrics = eval_epoch(train_dl, model, device)

        eval_metrics = eval_epoch(test_dl, model, device)
        train_metrics = train_epoch(train_dl, model, optimizer, device)

        log_str = "Epoch: {}/{} | lr: {:.4f} Train Loss: {:.4f} | " \
                  "Eval Loss: {:.4f} | Eval Acc: {:.4f} | Eval Diag Acc: {:.4f} | f1: {:.4f}".format(
            epoch, epochs, optimizer.param_groups[0]['lr'],
            train_metrics["loss"], eval_metrics["loss"], eval_metrics["acc"], eval_metrics["diag_acc"],
            eval_metrics["f1"]
        )
        logging.info(log_str)

        # scheduler.step(eval_metrics['loss'])
        logging.info("Eval Confusion Matrix: \n{}".format(eval_metrics["cf_matrix"]))
        logging.info("Eval Confusion Totals: \n{}\n".format(eval_metrics["cf_matrix"].sum(axis=1)))
        # logging.info("Train Confusion Matrix: \n{}".format(train_metrics["cf_matrix"]))
        # logging.info("Train Confusion Totals: \n{}\n".format(train_metrics["cf_matrix"].sum(axis=1)))

        # Log confusion matrix to tensorboard
        cf_matrix = eval_metrics["cf_matrix"]
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None]).round(2)
        # round to 2 decimal
        logging.info(df_cm)
        plt.figure(figsize=(12, 7))
        hm = sn.heatmap(df_cm, annot=True).get_figure()

        # Save model checkpoint
        best_eval_metrics = save_ckpt(
            args, best_eval_metrics, epoch, eval_metrics, model, optimizer, scheduler, osp.join(output_path, exp_name),
            train_metrics
        )

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("test/loss", eval_metrics["loss"], epoch)
        writer.add_scalar("test/acc", eval_metrics["acc"], epoch)
        writer.add_figure("confusion_matrix", hm, epoch)
        writer.add_scalar("test/diag_acc", eval_metrics["diag_acc"], epoch)
        writer.add_scalar("test/f1", eval_metrics["f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            scheduler.step()


@hydra.main(version_base=None, config_path="./", config_name="configs")
def main(cfg: DictConfig):
    args = OmegaConf.to_container(cfg, resolve=True)

    exp_name_list = get_experiment_name(
        {"train": args["train"], 'loss': args['loss'], "model": args["model"]}
    )
    exp_name = "_".join(["{}{}".format(k, v) for k, v in exp_name_list.items()])

    # Check if experiment already exists
    if osp.exists(osp.join(args["data"]["output"], exp_name, "model_latest.pth")):
        dct = torch.load(osp.join(args["data"]["output"], exp_name, "model_latest.pth"))
        epochs_completed = dct["epoch"]
        if epochs_completed >= args["train"]["epochs"] - 1:
            print("=> Experiment {} already exists. Skipping...".format(exp_name))
            exit()

    if not osp.exists(osp.join(args["data"]["output"], exp_name)):
        os.makedirs(osp.join(args["data"]["output"], exp_name))

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s',
        datefmt='%H:%M:%S',
        filemode='a'
    )
    logging.getLogger().addHandler(logging.FileHandler(osp.join(args["data"]["output"], exp_name, "logging.txt")))

    logging.info("=> Running experiment: {}".format(osp.join(args["data"]["output"], exp_name)))
    save_yaml(args, osp.join(args["data"]["output"], exp_name, "exp_config.yaml"))
    train(args, exp_name)


if __name__ == "__main__":
    main()
