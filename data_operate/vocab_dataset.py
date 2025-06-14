from collections import Counter

from torchtext.data import Dataset


def build_vocab(train_data: Dataset, dev_data: Dataset, test_data: Dataset):
    """
    对数据进行填充，并构建词汇表
    """
    vocab = Counter()
    for text in train_data.examples:
        vocab.update(text.src)
    for text in dev_data.examples:
        vocab.update(text.src)
    for text in test_data.examples:
        vocab.update(text.src)
    # 添加起始和结束标志
    start_symbol = "<start>"
    end_symbol = "<end>"
    pad_symbol = "<pad>"
    vocab.update([start_symbol, end_symbol, pad_symbol])

    # 构建词汇表，将单词映射为数值标识
    word2idx = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}

    return word2idx

