import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class CoGNN(torch.nn.Module):
    def __init__(self, rating_mat, data_u, data_i, embedding_size_ae, embedding_size_graph, hidden_dim, num_layers=2, nhead=2, dropout=0.0,
                 num_layers_transformer=1, num_layers_mlp=1):
        super(CoGNN, self).__init__()
        N, M = rating_mat.shape
        self.rating_mat = rating_mat
        self.graph_u, self.graph_i = Graph(data_u, embedding_size_graph), Graph(data_i, embedding_size_graph)
        self.ae_u, self.ae_i = Autoencoder(input_size=M, hidden_dim=embedding_size_ae), Autoencoder(input_size=N, hidden_dim=embedding_size_ae)
        self.embedding_size = embedding_size_ae + embedding_size_graph
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers_transformer)

        input_size = 3 * self.embedding_size 
        self.mlp = MLP(input_size=input_size, num_layers=num_layers_mlp)

    def forward(self, x):
        user_ids, item_ids = x[:,0], x[:,1]
        users, items = self.rating_mat[user_ids], self.rating_mat.t()[item_ids]
        ae_users, y_users = self.ae_u(users)
        ae_items, y_items = self.ae_i(items)
        graph_users, graph_items = self.graph_u(user_ids), self.graph_i(item_ids)
        embed_users, embed_items = torch.cat([ae_users, graph_users], 1), torch.cat([ae_items, graph_items], 1)
        out_gmf = embed_users * embed_items

        out = torch.cat([embed_users, embed_items], 1).reshape(2, -1, self.embedding_size)  #batch_first=False, batch_size 放中间
        out = self.transformer(out)

        user, item = out[0], out[1]
        out_trans = torch.cat([user, item], 1)
        out = torch.cat([out_trans, out_gmf], 1)
        out = self.mlp(out)

        return out.view(-1), users, y_users, items, y_items


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dim, noise_level=0.):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc11 = nn.Linear(self.input_size, int(self.input_size/2))
        self.fc12 = nn.Linear(int(self.input_size/2), self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, int(self.input_size/2))
        self.fc22 = nn.Linear(int(self.input_size/2), self.input_size)
        
    def encoder(self, x):
        x = self.fc11(x)
        x = F.relu(x)
        h1 = self.fc12(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        x = self.fc21(x)
        x = F.relu(x)
        h2 = self.fc22(x)
        return h2
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode


class Graph(torch.nn.Module):
    def __init__(self, data, output_size=8):
        super(Graph, self).__init__()
        self.x, self.edge_index, self.device = data.x, data.edge_index, data.x.device
        input_size = data.x.shape[1]
        hidden = int(input_size/2)
        self.sage1 = SAGEConv(input_size, hidden)
        self.sage2 = SAGEConv(hidden, output_size)
        self.embeddings = torch.zeros((self.x.shape[0], output_size), device=self.device)

    def forward(self, x):
        if self.training:
            out = self.sage1(self.x, self.edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            self.embeddings = self.sage2(out, self.edge_index)

        return self.embeddings[x]


class MLP(torch.nn.Module):
    def __init__(self, input_size, num_layers):
        super(MLP, self).__init__()
        self.linears = nn.Sequential()
        for i in range(num_layers - 1):
            output_size = input_size // 2
            self.linears.add_module("linear_" + str(i), nn.Linear(input_size, output_size))
            self.linears.add_module("relu_" + str(i), nn.ReLU())
            input_size = output_size
        self.linears.add_module("linear_final", nn.Linear(input_size, 1))
        
    def forward(self, x):
        out = self.linears(x)
        return out