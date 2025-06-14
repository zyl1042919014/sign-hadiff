# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch
import torch.nn.functional as F

from constants import TARGET_PAD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Batch(torch_batch, pad_index, diff_model):
    src = torch_batch[0]
    src_mask = (src != pad_index).unsqueeze(1)
    nseqs = src.size(0)
    trg_input = None
    trg = None
    trg_mask = None
    trg_lengths = None
    ntokens = None

    file_paths = torch_batch[2]
    use_cuda = True
    target_pad = TARGET_PAD
    # Just Count
    just_count_in = False
    # Future Prediction
    future_prediction = 0

    origin = torch_batch[1].to(device).clone()

    trg = torch_batch[1][:,:,:-1].to(device)
    trg_lengths = torch_batch[1].shape[1]
    # trg_input is used for teacher forcing, last one is cut off
    # Remove the last frame for target input, as inputs are only up to frame N-1

    new_skel = trg[:, :, :].clone().to(device)
    target_skel = diff_model.get_hidden_space(new_skel)
    # orgin_skel = torch.concat((target_skel, trg[:, :, -1].unsqueeze(2)), dim=-1)
    orgin_skel = target_skel
    trg_input = orgin_skel

    trg_lengths = trg_lengths
    # trg is used for loss computation, shifted by one since BOS
    trg = orgin_skel.clone()

    # Just Count
    if just_count_in:
        # If Just Count, cut off the first frame of trg_input
        trg_input = trg_input[:, :, -1:]

    # Future Prediction
    if future_prediction != 0:
        # Loop through the future prediction, concatenating the frames shifted across once each time
        future_trg = torch.Tensor()
        # Concatenate each frame (Not counter)
        for i in range(0, future_prediction):
            future_trg = torch.cat((future_trg, trg[:, i:-(future_prediction - i), :-1].clone()),
                                   dim=2)
        # Create the final target using the collected future_trg and original trg
        trg = torch.cat((future_trg, trg[:, :-future_prediction, -1:]), dim=2)

        # Cut off the last N frames of the trg_input
        trg_input = trg_input[:, :-future_prediction, :]

    # Target Pad is dynamic, so we exclude the padded areas from the loss computation
    trg_mask = (trg_input != target_pad).unsqueeze(1)
    # This increases the shape of the target mask to be even (16,1,120,120) -
    # adding padding that replicates - so just continues the False's or True's
    pad_amount = trg_input.shape[1] - trg_input.shape[2]
    # Create the target mask the same size as target input
    trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
    ntokens = (origin != pad_index).data.sum().item()

    if torch.cuda.is_available():
        return [src.cuda(), src_mask.cuda(), trg_input.cuda(), trg_mask.cuda(), file_paths, trg.cuda(), origin.cuda()]

    return [src, src_mask, trg_input, trg_mask, file_paths, trg, origin]
