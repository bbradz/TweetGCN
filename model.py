import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
# from torch_scatter import scatter_add
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
# from torch_geometric.nn.inits import glorot, zeros

class TextGCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextGCN, self).__init__()

        self.hidden = 64

        self.conv1 = GraphConvLayer(input_size, self.hidden)
        self.conv2 = GraphConvLayer(self.hidden, num_classes)
        
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, inputs, adj_matrix):
        l1 = self.relu(self.conv1(inputs, adj_matrix))
        l2 = self.dropout(l1)
        l3 = self.conv2(l2, adj_matrix)

        return l3

class GraphConvLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = Parameter(torch.FloatTensor(input_size, output_size))
        self.weights.data.uniform_(-0.1, 0.1)

        self.bias = Parameter(torch.zeros(output_size))

    def forward(self, inputs, pyg_graph):
        supports = torch.mm(inputs, self.weights)
        outputs = torch.spmm(pyg_graph, supports) + self.bias

        return outputs