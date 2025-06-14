import numpy as np
import math

import torch
from torch.utils.data import DataLoader
from torchtext.data import Dataset

from CVT.Conv_model import Model
from helpers import calculate_dtw

from CVT.batch import Batch
from data_operate.dataset import make_data_iter, collate_fn
from skel_diffusion import SkelDiffusion


def pre_validate_on_data(model: Model,
                         vocab: list,
                         data: Dataset,
                         batch_size: int,
                         diff_model: SkelDiffusion,
                         max_output_length: int,
                         eval_metric: str,
                         loss_function: torch.nn.Module = None,
                         batch_type: str = "sentence",
                         type="val",
                         BT_model=None):
    # valid_iter = make_data_iter(
    #     dataset=data, batch_size=batch_size, batch_type=batch_type,
    #     shuffle=True, train=False)

    valid_iter = make_data_iter(data, vocab)

    pad_index = 1
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        talent_hypotheses = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []
        valid_length = []
        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        valid_latent = []
        batches = 0
        # for valid_batch in iter(valid_iter):
        dev_loader = DataLoader(valid_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        for i, (src, trg, file_path, trg_length) in enumerate(dev_loader):
            # Extract batch
            batch = [src, trg, file_path]
            # create a Batch object from torchtext batch
            batch = Batch(torch_batch=batch,
                          pad_index=pad_index,
                          diff_model=diff_model)
            targets = batch[5]
            origin = batch[6]
            # targets = batch[5]

            # run as during training with teacher forcing
            if loss_function is not None and batch[2] is not None:
                # Get the loss for this batch
                with torch.no_grad():
                    gloss_output, output = model.forward(src=batch[0], trg=batch[6][:, :, :-1])
                    latent_output = model.AE(trg=batch[2][:, :, :])
                    # output = targets.clone()
                    # latent_output = torch.cat((latent_output[:, :, :latent_output.shape[2] // (10)], latent_output[:, :, -1:]),dim=2)
                batch_loss = loss_function(output, targets[:, :, :])

                valid_loss += batch_loss
                total_ntokens += 1
                total_nseqs += batch_size

            # If future prediction
            # if model.future_prediction != 0:
            #     # Cut to only the first frame prediction + add the counter
            #     train_output = torch.cat(
            #         (train_output[:, :, :train_output.shape[2] // (model.future_prediction)], train_output[:, :, -1:]),
            #         dim=2)
            #     # Cut to only the first frame prediction + add the counter
            #     targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),
            #                         dim=2)

            # For just counter, the inference is the same as GTing
            # if model.just_count_in:
            #     output = train_output

            # Add references, hypotheses and file paths to list
            valid_references.extend(origin)
            valid_latent.extend(targets[:, :, :-1])
            valid_hypotheses.extend(output)
            talent_hypotheses.extend(latent_output)
            file_paths.extend(batch[4])
            valid_length.extend(trg_length)
            # Add the source sentences to list, by using the model source vocab and batch indices
            # valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
            #                      range(len(batch.src))])

            valid_inputs.extend(batch[4])
            # Calculate the full Dynamic Time Warping score - for evaluation
            dtw_score = calculate_dtw(targets[:, :, -1], output, trg_length)
            all_dtw_scores.extend(dtw_score)

            # Can set to only run a few batches
            if batches == math.ceil(20 / batch_size):
                break
            batches += 1

        # Dynamic Time Warping scores
        current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths, talent_hypotheses, valid_length, valid_latent
