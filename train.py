from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from build_graph.py import load_data
from model.py import TextGCN

# Load data
adj_mtx, train_features, train_labels, test_features, test_labels = load_data()

# Model and optimizer
model = TextGCN(input_size=train_features.shape[1], num_classes=train_labels.shape[1])
optimizer = optim.Adam(model.parameters())

def accuracy():
    pass

def train(epoch):
    model.train()
    optimizer.zero_grad()
    
    preds = model(train_features, adj_mtx)

    training_loss = torch.nn.CrossEntropyLoss(preds, train_labels)
    training_accuracy = accuracy(preds, train_labels)

    training_loss.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(training_loss.item()),
          'acc_train: {:.4f}'.format(training_accuracy.item()))


def test():
    model.eval()
    preds = model(test_features, adj_mtx)
    testing_loss = torch.nn.CrossEntropyLoss(preds, train_labels)
    testing_accuracy = accuracy(preds, train_labels)

    print("Test set results:",
          "loss= {:.4f}".format(testing_loss),
          "accuracy= {:.4f}".format(testing_accuracy))


# Train model
for epoch in range(4):
    train(epoch)
print("Optimization Finished!")

# Testing
test()
