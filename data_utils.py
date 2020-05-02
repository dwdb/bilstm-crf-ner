import pickle
import random
from collections import defaultdict

# TAG_VOCAB = {
#     'O': 0,
#     'B-PER': 1,
#     'I-PER': 2,
#     'B-LOC': 3,
#     'I-LOC': 4,
#     'B-ORG': 5,
#     'I-ORG': 6
# }

TRAIN_DATA_PATH = 'data/train.data'

WORD_PATH = 'data/word_vocab.pkl'
TAG_PATH = 'data/tag_vocab.pkl'


def read_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf8') as fr:
        for sent in fr.read().split('\n\n'):
            if sent:
                words, tags = zip(*(i.split('\t') for i in sent.split('\n')))
                data.append((words, tags))
    return data


def vocab_build(corpus_path, word_vocab_path=None, tag_vocab_path=None, min_count=2):
    data = read_corpus(corpus_path)
    word_vocab = defaultdict(int)
    tag_vocab = defaultdict(int)
    for words, tags in data:
        for word in words:
            if word.isdigit():
                word_vocab['<NUM>'] += 1
            elif '\u0041' <= word <= '\u005a' or '\u0061' <= word <= '\u007a':
                word_vocab['<ENG>'] += 1
            else:
                word_vocab[word] += 1
        for tag in tags:
            tag_vocab.setdefault(tag, len(tag_vocab) + 1)

    word_vocab = sorted(word_vocab.items(), key=lambda x: x[1], reverse=True)
    word_vocab = [c[0] for c in word_vocab if
                  c[1] >= min_count and c[0] not in ('<NUM>', '<ENG>')]
    word_vocab = dict(zip(word_vocab, range(1, len(word_vocab) + 1)))

    word_vocab['<PAD>'] = 0
    word_vocab['<UNK>'] = len(word_vocab)

    tag_vocab['<PAD>'] = 0

    pickle.dump(word_vocab, open(word_vocab_path or WORD_PATH, 'wb'))
    pickle.dump(tag_vocab, open(tag_vocab_path or TAG_PATH, 'wb'))

    return word_vocab


def read_vocab(vocab_path):
    with open(vocab_path, 'rb') as fr:
        vocab = pickle.load(fr)
        id2vocab = {item[1]: item[0] for item in vocab.items()}
    return vocab, id2vocab


def pad_sequences(sequences, pad_mark=0, max_len=None):
    seq_list, seq_len_list = [], []
    if sequences:
        if max_len is None:
            max_len = max(map(lambda x: len(x), sequences))
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def words2id(words, vocab):
    indices = []
    for word in words:
        if word.isdigit():
            word = '<NUM>'
        elif '\u0041' <= word <= '\u005a' or '\u0061' <= word <= '\u007a':
            word = '<ENG>'
        if word not in vocab:
            word = '<UNK>'
        indices.append(vocab[word])
    return indices


def tags2id(tags, vocab):
    indices = [vocab[t] for t in tags]
    return indices


def id2tags(indices, vocab):
    tags = [vocab[i] for i in indices]
    return tags


def batch_yield(data, batch_size, shuffle=False):
    if shuffle:
        random.shuffle(data)

    batch_x, batch_y = [], []
    for words, tags in data:

        if len(batch_x) == batch_size:
            yield batch_x, batch_y
            batch_x, batch_y = [], []

        batch_x.append(words)
        batch_y.append(tags)

    if batch_x:
        yield batch_x, batch_y


if __name__ == '__main__':
    # build vocabulary
    vocab_build('data/train.data')
