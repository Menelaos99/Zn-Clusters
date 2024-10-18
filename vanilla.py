import os

import numpy as np
import matplotlib.pyplot as plt

import warnings
from tqdm import tqdm
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull

from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import torch
import gpytorch

import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool,global_add_pool
# from torch_geometric.nn.glob import attention
def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
import gpytorch

from helpers import get_graph
from models import BaseGNN, ExactGPModel

# ignore the warning: RuntimeWarning: divide by zero encountered in true_divide
#   cm = (atomic_nums*atomic_nums.T) / pairwise_distances(pos)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#%%

learning_rate=1e-5
EPOCHS = 2000
SAMPLE_SIZE = 8000
save_pos = True
#NOTE: set also the parameter sample_size in the get_suggestions function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used: {device}')

#%%
######################################
#Data transformation
######################################

# list of graphs as Data objects
data_list = []
root_path = os.getcwd() + '/data_processed/'
filenames = os.listdir(root_path)
filenames.sort()
# debug: 
# filenames = filenames[:6]
for filename in filenames:
    if filename.endswith('.pt'):
        graph = torch.load(os.path.join(root_path, filename)) 
        graph.x = graph.x.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.y = torch.tensor(graph.y).to(device)
        data_list.append(graph)


print(f'device of x, edge_attr, y: {graph.x.device}, {graph.edge_attr.device}, {graph.y.device}')

# list of atoms 
with open('data_processed/atoms.txt', 'r') as f:
    lines = f.readlines()
atoms = [line.strip() for line in lines]

# initial positions of cluster 
cluster = data_list[0].pos[:-2]

# note for GP:
# inputs are changing based on transformation
# targets are fixed
targets = torch.tensor([data.y for data in data_list], dtype=torch.float).to(device)
class vanillagnn(nn.Module):
    def __init__(self,
    nodefeat_num=3, edgefeat_num=1,
    nodeembed_to=6, edgeembed_to=4):
        super().__init__()
        ## Embeddings
        self._node_embedding = nn.Sequential(nn.Linear(nodefeat_num, nodeembed_to),nn.ReLU())
        self._node_embednorm = nn.BatchNorm1d(nodeembed_to) 
        self._edge_embedding = nn.Sequential(nn.Linear(edgefeat_num, edgeembed_to), nn.CELU())
        self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)
        
        # Graph Convolutions
        self._first_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'add'

        )
        self._first_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._second_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'add'

        )
        self._second_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._third_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'add'

        )
        self._third_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._fourth_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'add'

        )
        self._fourth_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._fifth_conv = NNConv(
            
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, 2*nodeembed_to), nn.ReLU(),nn.Linear( 2*nodeembed_to, nodeembed_to**2)
                # nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()

            ),aggr= 'add'

        )
        self._fifth_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)


        ## Pooling and actuall prediction NN
        self._pooling = [global_mean_pool, global_max_pool] # takes batch.x and batch.batch as args
        # shape of one pooling output: [B,F], where B is batch size and F the number of node features.
        # shape of concatenated pooling outputs: [B, len(pooling)*F]
        self._predictor = nn.Sequential(
            nn.Linear(nodeembed_to*len(self._pooling), 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.ReLU(),


        )       
        self._predictor.apply(init_weights)



 
    def forward(self, batch: Batch):
        node_features, edges, edge_features,mask, batch_vector = \
            batch.x.float(), batch.edge_index, batch.edge_attr.float(),batch.mask, batch.batch
        ## embed the features
        node_features = self._node_embednorm(
            self._node_embedding(node_features))
        edge_features = self._edge_embednorm(
            self._edge_embedding(edge_features))

        ## do graphs convolutions
        node_features = self._first_conv(
            node_features, edges, edge_features)
        node_features = self._second_conv(
            node_features, edges, edge_features)
        # node_features = self._third_conv_batchnorm(self._third_conv(
        #     node_features, edges, edge_features))

        # node_features = self._fourth_conv_batchnorm(self._fourth_conv(
        #     node_features, edges, edge_features))
        # node_features = self._fifth_conv_batchnorm(self._fifth_conv(
        #     node_features, edges, edge_features))
        
        ## now, do the pooling
        pooled_graph_nodes = torch.cat([p(node_features[mask], batch_vector[mask]) for p in self._pooling], axis=1) 
        outputs = self._predictor(pooled_graph_nodes)
        return outputs # ready for a loss        


# TODO: adjust batch size
train_loader = GeoLoader(data_list, batch_size=len(data_list), shuffle= False) # batch-size is data size for GP

# likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(5)*0.0001) # gpytorch.likelihoods.GaussianLikelihood() 

# TODO: should set noise to zero. No DFT error
feature_extractor = vanillagnn(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
feature_extractor.to(device)

optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=3,
                                                       min_lr=0.0000001)

# for name, param in model_gp.named_parameters():
#     print("Parameter name:", name)
#     print("Parameter shape:", param.shape)
#     print("Parameter value:", param.data)

# NOTE: to solve the convergence problem:
#       remove noise
#       reduce learning rate by decay
criterion = nn.MSELoss()

def train(epoch):
    # "Loss" for GPs - the marginal log likelihood
    for data in train_loader:
        data = data.to(device)
        # zero gradients from previous iteration
        optimizer.zero_grad()
        # output from model
        output = feature_extractor(data)
        # calc loss and backprop gradients
        loss = criterion(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,))
        loss.backward()
        optimizer.step()
    return loss / len(train_loader.dataset) 

loss_history = []
best = {}

feature_extractor.train()
# mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model_gp)

for epoch in range(1, EPOCHS+1):
    # lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    # scheduler.step(loss)
    feature_extractor.eval()
    # means = model_gp(Batch.from_data_list(data_list)).mean.detach().cpu().clone().numpy()
    means = feature_extractor(Batch.from_data_list(data_list))

    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   means : ' % (
    # epoch, EPOCHS, loss.item(),
    # model_gp.gp.covar_module.base_kernel.lengthscale.item(),
    # ), means)
    print('Iter %d/%d - Loss: %.3f   means : ' % (
    epoch, EPOCHS, loss.item(),
    ), means.to(device).reshape(-1,))

       
    feature_extractor.train()

    loss_history.append({"train": loss})


#%%
######################################
#surrogate model
######################################
# get data from simulation:

# cluster = np.random.rand(5,3) # shape: [N_samples, N_atoms, spatial_dimension]
# y = np.random.rand(5,1)


# # select a kernel
# kernel = GPy.kern.MLP(cluster.shape[1],ARD = True)
# kernel += GPy.kern.MLP(cluster.shape[1],ARD = True) 

# # do GP regression (can also divide in train/test)
# model_gp = GPy.models.GPRegression(cluster,y,kernel) 


# model_gp.optimize()
# %%