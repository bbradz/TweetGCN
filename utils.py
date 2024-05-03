import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from preprocess import process_file

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def build_graph(features):
    sim_matrix = cosine_similarity(features)
    adj = np.array(sim_matrix)
    adj_norm = adj / adj.sum(axis=1, keepdims=True)

    adj_norm = torch.FloatTensor(adj_norm)

    return adj_norm

def load_data(path="tweet_topic_single/dataset/split_coling2022_random/test_random.single.json", dataset="tweet_topic_single"):
    print('Loading {} dataset...'.format(dataset))

    output = process_file(path)
    features_str, labels_str = output[0], output[1]

    # Convert features and labels from string to correct types
    labels = encode_onehot(np.array(labels_str, dtype=np.float32))
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    features = model.encode(features_str)

    idx_train = range(2719)
    idx_test = range(2719, 3399)

    features = torch.FloatTensor(features) # Tensor of features
    labels = torch.FloatTensor(labels) # Tensor of labels

    train_features = torch.FloatTensor(features[idx_train])
    train_labels = torch.FloatTensor(labels[idx_train])
    test_features = torch.FloatTensor(features[idx_test])
    test_labels = torch.FloatTensor(labels[idx_test])

    return train_features, train_labels, test_features, test_labels

def load_all(path="tweet_topic_single/dataset/split_coling2022_random/test_random.single.json", dataset="tweet_topic_single"):
    print('Loading {} dataset...'.format(dataset))

    output = process_file(path)
    features_str, labels_str = output[0], output[1]

    # Convert features and labels from string to correct types
    labels = np.array(labels_str, dtype=np.float32)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    features = model.encode(features_str)

    return labels, features