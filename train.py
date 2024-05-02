from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import model

from utils import load_data, build_graph
from model import TextGCN

# Load data
train_features, train_labels, test_features, test_labels = load_data()

train_adj = build_graph(train_features)
test_adj = build_graph(test_features)

# Model and optimizer
model = TextGCN(input_size=train_features.shape[1], num_classes=train_labels.shape[1])
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(preds, actual):

    actual = torch.argmax(actual, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = torch.flatten(preds)
    actual = torch.flatten(actual)

    n = actual.shape[0]
    acc = (preds == actual).sum() / n

    return acc

def train(epoch):
    model.train()
    optimizer.zero_grad()

    preds = model(train_features, train_adj)

    training_loss = loss_fn(preds, train_labels)

    softmax = torch.nn.Softmax()
    preds = softmax(preds)

    training_accuracy = accuracy(preds, train_labels)

    training_loss.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(training_loss.item()),
          'acc_train: {:.4f}'.format(training_accuracy.item()))


def test():
    model.eval()
    preds = model(test_features, test_adj)

    testing_loss = loss_fn(preds, test_labels)
    testing_accuracy = accuracy(preds, test_labels)

    print("Test set results:",
          "loss= {:.4f}".format(testing_loss),
          "accuracy= {:.4f}".format(testing_accuracy))


# Train model
for epoch in range(10):
    train(epoch)
print("Optimization Finished!")

# Testing
test()
