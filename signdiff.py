import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import math
import yaml
from typing import Optional
from torch.utils.data import DataLoader
from torchtext.data import Dataset
import glob

from CVT.Conv_model import build_model,Model
from CVT.batch import Batch
from data import load_data
from dtw import dtw
from constants import TARGET_PAD
from plot_videos import plot_video, alter_DTW_timing
from data_operate.dataset import make_data_iter, collate_fn
from skel_diffusion import SkelDiffusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
            self.criterion_L1 = nn.L1Loss()

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask
        # loss = self.criterion(preds_masked, targets_masked) + 0.1 * self.criterion_L1(preds_masked, targets_masked)
        loss = self.criterion(preds_masked, targets_masked)
        # Calculate loss just over the masked predictions
        # body_loss = self.criterion(preds_masked[:, :, :8 * 3], targets_masked[:, :, :8 * 3])
        # wrist_loss = self.criterion(preds_masked[:, :, 8 * 3:9 * 3],
        #                             targets_masked[:, :, 8 * 3:9 * 3]) + self.criterion(
        #     preds_masked[:, :, 29 * 3:30 * 3], targets_masked[:, :, 29 * 3:30 * 3])
        # left_hand_loss = self.criterion(preds_masked[:, :, 8 * 3:29 * 3], targets_masked[:, :, 8 * 3:29 * 3])
        # right_hand_loss = self.criterion(preds_masked[:, :, 29 * 3:], targets_masked[:, :, 29 * 3:])
        # w_b = 1
        # w_w = 1
        # w_l = 1
        # w_r = 1
        # loss = w_b * body_loss + w_w * wrist_loss + w_l * left_hand_loss + w_r * right_hand_loss

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss



def calculate_dtw(references, hypotheses, trg_length):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []
    max_length = 100
    left_padding = 6
    # new_hyp = []
    # new_ref = []
    # Remove the BOS frame from the hypothesis
    hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        length = trg_length[i]
        right_padding = max_length - length + 6

        # # Cut the reference down to the max count value
        # _, ref_max_idx = torch.max(ref[:, -1], 0)
        # if ref_max_idx == 0: ref_max_idx += 1
        # # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[left_padding:-right_padding].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        # _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        # if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[left_padding:-right_padding].cpu().detach().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d / acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores

def get_latest_checkpoint(ckpt_dir, post_fix="_every") -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir, post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint
def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

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



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def produce_validation_video(output_joints, inputs, references, display, model_dir, type, trg_length,
                             steps="",
                             diff_model=None,
                             file_paths=None):

    max_length = 100
    left_padding = 6
    # If not at test
    if type == "val_inf":
        dir_name = model_dir + "/videos/Step_{}/".format(steps)
        if not os.path.exists(model_dir + "/videos/"):
            os.mkdir(model_dir + "/videos/")

    elif type == "talent":
        dir_name = model_dir + "/talent_videos/Step_{}/".format(steps)
        if not os.path.exists(model_dir + "/talent_videos/"):
            os.mkdir(model_dir + "/talent_videos/")

    # If at test time
    elif type == "test":
        dir_name = "output/"+model_dir.split("/")[-1] + "_test_videos/"

    # Create model video folder if not exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # For sequence to display
    for i in display:

        length = trg_length[i]
        right_padding = max_length - length + 6

        seq = output_joints[i][left_padding:-right_padding, :]
        ref_seq = references[i][left_padding:-right_padding, :]
        input = inputs[i]
        # Write gloss label
        gloss_label = input[0]
        if input[1] is not "</s>":
            gloss_label += "_" + input[1]
        if input[2] is not "</s>":
            gloss_label += "_" + input[2]

        seq = seq.unsqueeze(1)
        # out_skel = diff_model.sample_after_mlp(seq[:, :, :-1])
        # seq1 = seq[:, :, -1].unsqueeze(2)
        # seq1_out = torch.cat((out_skel, seq1), dim=-1)
        # seq1_out = seq1_out.squeeze(1)
        out_skel = diff_model.sample_after_mlp(seq)
        seq1_out = out_skel.squeeze(1)
        ref_seq = ref_seq[:,:-1]

        # Alter the dtw timing of the produced sequence, and collect the DTW score
        timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq1_out, ref_seq)

        video_ext = "{}_{}".format("{0:.2f}".format(float(dtw_score)).replace(".", "_"),file_paths[i][5:])

        if file_paths is not None:
            sequence_ID = file_paths[i]
        else:
            sequence_ID = None

        # Plot this sequences video
        if "<" not in video_ext:
            plot_video(joints=timing_hyp_seq,
                       file_path=dir_name,
                       video_name=video_ext,
                       references=ref_seq_count,
                       skip_frames=1,
                       sequence_ID=sequence_ID)


# pylint: disable-msg=logging-too-many-args
def CVT_test(cfg_file, ckpt=None) -> None:
    # Load the config file
    cfg = load_config(cfg_file)
    print('testing')

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]
    diff_path = cfg["model"].get("diffusion_path")
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", True)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", 300)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)


    # Load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # Build model and load parameters into it
    model, disc = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)

    model.load_state_dict(model_checkpoint["model_state"])

    # 初始化diffusion模型
    checkpoint = torch.load(diff_path, map_location=device)
    print("init diffusion model")
    model.diffusion_model.load_state_dict(checkpoint['state_dict'])
    # 冻结mlp
    model.diffusion_model.freeze_mlp()
    # 冻结net
    # model.diffusion_model.freeze_unet()



    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # Set up trainer to produce videos
    # trainer = CVTTrainManager(model=model, disc=disc, config=cfg, test=True, ckpt=ckpt)
    regloss = RegLoss(cfg=cfg,
                        target_pad=TARGET_PAD)
    # Validate for this data set
    score, loss, references, hypotheses, \
    inputs, all_dtw_scores, file_paths, talent_hypotheses, test_length, valid_latent = \
        pre_validate_on_data(
            model=model,
            data=test_data,
            batch_size=batch_size,
            max_output_length=max_output_length,
            eval_metric=eval_metric,
            loss_function=regloss,
            batch_type=batch_type,
            vocab=src_vocab,
            diff_model=model.diffusion_model
        )
    # hypotheses, valid_latent
    # hypotheses_array = np.array([i.cpu().detach().numpy() for i in hypotheses])
    # # np.save('gt_latent.npy', hypotheses_array)
    # valid_latent_array = np.array([i.cpu().detach().numpy() for i in valid_latent])
    # np.save('latent.npy', valid_latent_array)
    # Set which sequences to produce video for
    display = list(range(len(hypotheses)))

    # Produce videos for the produced hypotheses
    produce_validation_video(
        output_joints=hypotheses,
        inputs=inputs,
        references=references,
        model_dir=model_dir,
        display=display,
        type="test",
        file_paths=file_paths,
        trg_length=test_length,
        diff_model=model.diffusion_model
    )

    print("testing done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Talent Transformers")
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    # train(cfg_file=args.config)
