from torch.utils.data import Dataset
import torch
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    token2idx = batch[0][3]
    max_length = 100
    left_pad = 6
    new_trg = []
    new_name = []
    new_trg_length = []
    # 将文本数据化
    texts = [batch[idx][0] for idx in range(len(batch))]
    tokens = [[token2idx[token] for token in text] for text in texts]
    sequences_padded = pad_sequence([torch.tensor(tokens) for tokens in tokens], batch_first=True,
                                    padding_value=token2idx['<pad>'])

    # 将骨骼序列进行补帧
    for i in range(len(batch)):
        trg_length = len(batch[i][1])
        new_trg_length.append(trg_length)
        right_pad = max_length - trg_length + 6
        padded_video = torch.cat(
            (
                batch[i][1][0][None].expand(left_pad, -1),
                torch.stack(batch[i][1]),
                batch[i][1][-1][None].expand(right_pad, -1),
            )
            , dim=0)
        new_trg.append(padded_video)
        new_name.append(batch[i][2])

    new_padded_video = torch.stack(new_trg)
    # 文件名
    # name = [batch[idx][2] for idx in batch]  new_trg_length
    return sequences_padded, new_padded_video, new_name, new_trg_length


class make_data_iter(Dataset):

    def __init__(self, dataset, vocab):
        self.datasets = dataset
        self.vocabs = vocab

    def __getitem__(self, item):
        return self.datasets[item].src, self.datasets[item].trg, self.datasets[item].file_paths, self.vocabs

    def __len__(self):
        return len(self.datasets)


if __name__ == "__main__":
    train_data = DataSet(datamode="dev")
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
