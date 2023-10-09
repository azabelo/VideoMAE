import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import wandb
from torch import Tensor
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, data_for_knn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        wandb.log({"epoch": epoch, "batch": step, "train_loss": loss_value,
                   "min_lr": min_lr, "max_lr": max_lr,
                   "grad_norm": grad_norm, })

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if data_for_knn is not None:
        log_knn_acc(data_for_knn, model)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def log_knn_acc(data_for_knn, model):

    # lightly knn
    train_videos = torch.empty((0, 768), device='cuda')
    test_videos = torch.empty((0, 768), device='cuda')
    train_labels = torch.empty(0, device='cuda')
    test_labels = torch.empty(0, device='cuda')

    # my implementation of knn
    knn_classifier3 = KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='cosine')
    train_videos_np = np.empty((0, 768))
    test_videos_np = np.empty((0, 768))
    train_labels_np = np.empty(0)
    test_labels_np = np.empty(0)

    with torch.no_grad():
        index = 0
        for batch in data_for_knn:
            print("knn step: ", index)
            index += 1
            videos, labels, _ = batch
            # make an empty tensor of False values with shape [bs, 1568]
            empty_mask = torch.zeros((videos.shape[0], 768), dtype=torch.bool)
            output_features_for_knn = model(videos.cuda(), empty_mask.cuda())
            # output_features_video_for_knn = output_features_video_for_knn.cpu().numpy()
            cls_tok_knn = output_features_for_knn[:, 0, :]
            cls_tok_knn = F.normalize(cls_tok_knn, dim=1)
            cls_tok_knn = cls_tok_knn.cuda()
            if index > 100:
                # move to cuda if not already
                test_labels = test_labels.cuda()
                test_videos = test_videos.cuda()
                labels = labels.cuda()
                cls_tok_knn = cls_tok_knn.cuda()
                test_labels = torch.cat((test_labels, labels), 0)
                test_videos = torch.cat((test_videos, cls_tok_knn), 0)
                # test_videos_np = np.append(test_videos, output_features_video_for_knn.reshape(8, -1), axis=0)
                test_labels_np = np.append(test_labels_np, labels.cpu().numpy(), axis=0)
                test_videos_np = np.append(test_videos_np, cls_tok_knn.cpu().numpy(), axis=0)
            else:
                train_labels = train_labels.cuda()
                train_videos = train_videos.cuda()
                labels = labels.cuda()
                cls_tok_knn = cls_tok_knn.cuda()
                train_labels = torch.cat((train_labels, labels), 0)
                train_videos = torch.cat((train_videos, cls_tok_knn), 0)
                # train_videos_np = np.append(train_videos, output_features_video_for_knn.reshape(8, -1), axis=0)
                train_labels_np = np.append(train_labels_np, labels.cpu().numpy(), axis=0)
                train_videos_np = np.append(train_videos_np, cls_tok_knn.cpu().numpy(), axis=0)

        # custom knn
        # Standardize the feature values
        scaler = StandardScaler()
        train_scaled_np = scaler.fit_transform(train_videos_np)
        test_scaled_np = scaler.transform(test_videos_np)
        knn_classifier3.fit(train_scaled_np, train_labels_np)
        predictions3 = knn_classifier3.predict(test_scaled_np)
        knn_accuracy_custom = accuracy_score(test_labels_np, predictions3)
        print("custom knn accuracy", knn_accuracy_custom)

        # lightly knn
        #switch dimensions of the train_videos
        train_videos = train_videos.transpose(0, 1)
        pred_labels = knn_predict(
            test_videos,
            train_videos,
            train_labels,
            num_classes=51,
        )
        print(pred_labels.shape)
        print(test_labels.shape)
        print(pred_labels[0])
        test_labels = test_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        lightly_knn_accuracy = accuracy_score(test_labels, pred_labels)

        wandb.log({"knn_accuracy_lightly": lightly_knn_accuracy, "knn_accuracy_custom": knn_accuracy_custom})

def knn_predict(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 199,
    knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN predictions


    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_labels = sim_labels.long()
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels[:, 0]
