from .mmformer.utils import criterions
from ..data.porcupine_dataset import MODAL2INDEX, LABEL2INDEX
from ..common import miscutils, logutils

import torch
from torch.utils.data import DataLoader

from typing import Optional, Dict, Union
from tqdm.auto import tqdm


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch : int,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Trains the given model for one epoch based on the optimizer given. The training
    progress is displayed with a custom progress bar. At the end of the each batch the
    mean of the batch loss is displayed within the progress bar.

    Args:
        model: Model to train.
        loader: Data loader.
            The batch data should be a dictionary containing "img" and "label" keys.
        loss_function: Loss function to measure the loss during the training.
        optimizer: Optimizer to optimize the loss.
        epoch: Epoch number. Only used in the progress bar to display the current epoch.
        device: Device to load the model and data into. Defaults to None. If set to None
            will be set to ``cuda`` if it is available, else will be set to ``cpu``.

    Returns:
        A dictionary containing statistics about the model training process.
        Keys and values available in the dictionary are as follows:
            ``Mean Loss``: Mean validation loss value for the whole segmentation.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

    model = model.to(device)
    model = model.train()

    train_loss = miscutils.AverageMeter()

    with logutils.etqdm(loader, epoch=epoch) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["image"].to(device)[:,0,...]
            label = batch_data["label"].to(device)[:,0,...]
            info = batch_data["info"]

            N, H, W, D = label.shape

            # create feature mask
            feature_mask = [False for i in range(4)]
            feature_mask[MODAL2INDEX[info["modality"][0]]] = True
            feature_mask = [feature_mask]

            # transform image into suitable format for empty modalities
            image_tr = torch.zeros((N, 4, H , W, D), dtype=image.dtype)
            image_tr[:,MODAL2INDEX[info["modality"][0]]] = image
            
            # convert label to suitable form
            label_tr = torch.zeros((N, 2, H, W, D), dtype=torch.uint8).to(device)
            label_tr[:,0][label == LABEL2INDEX["background"]] = 1
            label_tr[:,1][label == LABEL2INDEX["tumour"]] = 1

            optimizer.zero_grad()

            model.module.is_training = True
            fuse_pred, _, _ = model(image_tr, feature_mask)

            # fuse_pred.shape = (N, 4, H, W, D)
            fuse_pred_tumour, _ = torch.max(fuse_pred[:,1:], dim=1)
            fuse_pred_background = fuse_pred[:,0]

            fuse_pred = torch.stack([fuse_pred_background, fuse_pred_tumour], dim=1)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, label_tr, num_cls=2)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, label_tr, num_cls=2)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            fuse_loss.backward()
            optimizer.step()

            loss_val = fuse_loss.item()

            train_loss.update(loss_val, image.size(0))

            metrics = {
                "Mean Loss": loss_val,
            }

            pbar.log_metrics(metrics)

    history = {
        "Mean Train Loss": train_loss.avg.item(),
    }

    return history