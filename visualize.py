from typing import Tuple, List
from transformers import BertModel, BertTokenizer
import torch
import numpy
import sklearn
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import argparse
from random import shuffle


def load_data(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as fp:
        l1 = []
        for sentence in fp:
            l1.append(sentence.strip().lower())
    shuffle(l1)
    return l1


def encode_sentences(s: List[str], batch_size=64, avg_pooling=True) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        outputs = []
        for i in range(0, len(s), batch_size):
            x = tokenizer(s[i:min(i + batch_size, len(s))], return_tensors='pt', padding=True)
            x = x.to(device)
            h = model(**x).last_hidden_state
            if avg_pooling:
                output = torch.avg_pool1d(h.transpose(1, 2), h.shape[1])[:, :, 0]
            else:
                output = h[:, 0]
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
    return outputs


def reduce_dim(X: numpy.ndarray, method='tsne', perplexity=50) -> numpy.ndarray:
    if method == 'tsne':
        f = TSNE(n_components=2, perplexity=perplexity)
    elif method == 'pca':
        f = sklearn.decomposition.PCA(n_components=2)
    else:
        assert 0
    out = f.fit_transform(X)
    return out


def visualize(X: numpy.ndarray, indices: List[Tuple[str, int]], point_size):
    markers = ['o', 'x', '^', 'v', 's']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    i = 0
    curr = 0
    plt.figure()
    for label, length in indices:
        plt.scatter(X[0][curr:curr+length], X[1][curr:curr+length], label=label,
                    c=colors[i % len(colors)], marker=markers[i % len(markers)], s=point_size)
        curr += length
        i += 1
    plt.legend()
    plt.show()


def main(args):
    labels, paths = tuple(zip(*map(lambda s: s.split(':', 1), args.label_and_path)))
    num_examples = args.num
    batch_size = args.batch_size
    avg_pooling = args.average_pooling
    point_size = args.size

    data = list(map(lambda it: it[:min(len(it), num_examples)], map(load_data, paths)))
    tensors = list(map(lambda it: encode_sentences(it, batch_size=batch_size, avg_pooling=avg_pooling).cpu().numpy(), data))
    arr2 = reduce_dim(numpy.concatenate(tensors, axis=0)).T
    visualize(arr2, list(zip(labels, map(len, data))), point_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label_and_path', nargs='+', type=str,
                        help='Labels and paths to data.  Format: LABEL:PATH')
    parser.add_argument('-m', '--method', type=str, choices=['pca', 'tsne'], default='tsne',
                        help='The algorithm to reduce dimension. Available: pca, tsne. Default to tsne.')
    parser.add_argument('-p', '--perplexity', type=float, default=50,
                        help='Perplexity parameter when using T-SNE. Default to 50.')
    parser.add_argument('-n', '--num', type=int, default=10000,
                        help='Maximum numbers of samples of each class. Default to 10000.')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='The batch size when encoding sentences. Default to 64.')
    parser.add_argument('-a', '--average_pooling', action='store_true',
                        help='Use average pooling instead of CLS token for sentence encoding.')
    parser.add_argument('-s', '--size', type=float, default=5,
                        help='The point size of the graph. Default to 5.')
    argv = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model: BertModel = BertModel.from_pretrained("bert-base-uncased").to(device)

    main(argv)
