import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

from bts.data.porcupine_dataset import LABEL2INDEX, MODAL2INDEX
from .mmformer.mmformer import Model
from ..common import miscutils, logutils

from tqdm.auto import tqdm
from functools import partial

import numpy as np

from typing import List, Dict, Optional, Union
from monai.metrics import compute_generalized_dice

patch_size = 128

def val_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    epoch: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Evaluates the given model.

    Args:
        model: Model to evaluate.
        loader: Data loader.
            The batch data should be a dictionary containing "img" and "label" keys.
        loss_function: Loss function to measure the loss during the validation.
        roi_size: The spatial window size for inferences.
        sw_batch_size: The batch size to run window slices.
        overlap: Amount of overlap between scans.
        labels: Label key-values configured with DotConfig.
            labels should have `BRAIN` and `TUMOR` keys.
        epoch: Epoch number. Only used in the progress bar to display the current epoch.
        device: Device to load the model and data into. Defaults to None. If set to None
            will be set to ``cuda`` if it is available, else will be set to ``cpu``.

    Raises:
        AssertionError: If labels does not have either of `BRAIN` and `TUMOR` keys.

    Returns:
        A dictionary containing statistics about the model validation process.
        Keys and values available in the dictionary are as follows:
            ``Mean Brain Acc.``: Mean accuracy value for the brain segmentation
            ``Mean Tumor Acc.``: Mean accuracy value for the tumor segmentation
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    num_class = 2

    val_accuracy = miscutils.AverageMeter()

    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()
    
    with torch.no_grad(), logutils.etqdm(loader, epoch=epoch) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["image"].to(device)[:,0,...]
            label = batch_data["label"].to(device)[:,0,...]
            info = batch_data["info"]

            N, H, W, Z = image.size()

            if Z < patch_size:
                continue

            # create feature mask
            feature_mask = [False for i in range(4)]
            feature_mask[MODAL2INDEX[info["modality"][0]]] = True
            feature_mask = [feature_mask]

            # transform image into suitable format for empty modalities
            image_tr = torch.zeros((N, 4, H , W, Z), dtype=image.dtype)
            image_tr[:,MODAL2INDEX[info["modality"][0]]] = image

            # convert label to suitable form
            label_tr = torch.zeros((N, 2, H, W, Z), dtype=torch.uint8).to(device)
            label_tr[:,0][label == LABEL2INDEX["background"]] = 1
            label_tr[:,1][label == LABEL2INDEX["tumour"]] = 1    

            #########get h_ind, w_ind, z_ind for sliding windows
            h_cnt = int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
            h_idx_list = range(0, h_cnt)
            h_idx_list = [h_idx * int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
            h_idx_list.append(H - patch_size)

            w_cnt = int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
            w_idx_list = range(0, w_cnt)
            w_idx_list = [w_idx * int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
            w_idx_list.append(W - patch_size)

            z_cnt = int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
            z_idx_list = range(0, z_cnt)
            z_idx_list = [z_idx * int(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
            z_idx_list.append(Z - patch_size)

            #####compute calculation times for each pixel in sliding windows
            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            weight = weight1.repeat(1, num_class, 1, 1, 1)

            #####evaluation
            pred = torch.zeros(1, num_class, H, W, Z).float().cuda()
            model.module.is_training=False
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = image_tr[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                        pred_part = model(x_input, feature_mask)

                        fuse_pred_tumour, _ = torch.max(pred_part[:,1:], dim=1)
                        fuse_pred_background = pred_part[:,0]

                        pred_part = torch.stack([fuse_pred_background, fuse_pred_tumour], dim=1)

                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
            pred = pred / weight

            pred = pred[:, :, :H, :W, :Z]
            pred = torch.argmax(pred, dim=1)
            
            # N, H, W, D, C
            pred = torch.nn.functional.one_hot(pred).permute(0, 4, 1, 2, 3)
            dice_score = compute_generalized_dice(
                y_pred=pred,
                y = label_tr
            )

            val_accuracy.update(dice_score.cpu())

            # `GROUND` label is excluded
            metrics = {
                "dice_score" : dice_score.cpu().item()
            }

            pbar.log_metrics(metrics)

    # `GROUND` label is excluded
    history = {
        "avg_dice" : val_accuracy.avg
    }

    return history