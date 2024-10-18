import torch
import gpytorch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool,global_add_pool, EdgeConv, TransformerConv, GMMConv, MetaLayer, nearest, GINConv
# from torch_geometric.nn.glob import attention
def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

class EdgeModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim):
        super(EdgeModel, self).__init__()
        # Define MLP for edge feature updates
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, edge_in_dim),

            nn.ReLU(),
            nn.Linear(edge_in_dim, edge_in_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # Concatenate source and destination node features and edge attributes
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

# Node model for updating node features
class NodeModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeModel, self).__init__()
        # Define MLP for node feature updates
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch): 
        return self.node_mlp(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class BaseGNN(nn.Module):
    def __init__(self, nodefeat_num=3, edgefeat_num=2, nodeembed_to=3, edgeembed_to=2):
        super().__init__()
        ## Embeddings
        self._node_embedding = nn.Sequential(nn.Linear(nodefeat_num, nodeembed_to),nn.ReLU())
        self._node_embednorm = nn.BatchNorm1d(nodeembed_to) 
    
        self._edge_embedding = nn.Sequential(nn.Linear(edgefeat_num, edgeembed_to), nn.ReLU())
        self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)
        
        ## Meta layer for joint node and edge feature updates
        self.op1 = MetaLayer(EdgeModel(nodeembed_to, edgeembed_to), NodeModel(nodeembed_to, nodeembed_to), None)
        self.op2 = MetaLayer(EdgeModel(nodeembed_to, edgeembed_to), NodeModel(nodeembed_to, nodeembed_to), None)
        
        ## GIN architecture 
        self.node_ginlayers = nn.ModuleList() 
        
        self.batch_norms = nn.ModuleList()
        self.Nodeinput = NodeModel(nodeembed_to, nodeembed_to)
        self.Edgeinput = EdgeModel(nodeembed_to, edgeembed_to)

        num_layers = 2
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(nodefeat_num, nodeembed_to, nodeembed_to)
            else:
                mlp = MLP(nodeembed_to, nodeembed_to, nodeembed_to)
    
            
            self.node_ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon            

            self.batch_norms.append(nn.BatchNorm1d(nodeembed_to))
        # Graph Convolutions
        self.NNconv1 = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'mean'

        )
        self.batch_norm1 = nn.BatchNorm1d(nodeembed_to) 
        self.NNconv2 = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'mean'

        )
        self.batch_norm2 = nn.BatchNorm1d(nodeembed_to) 
        self._GMMconv = GMMConv(
            nodeembed_to,
            nodeembed_to,
            dim=edgeembed_to,
            kernel_size=3
        )
        
        self._first_convEdge = EdgeConv(
            nn.Sequential(
                nn.Linear(nodefeat_num*2, nodeembed_to), nn.Tanh()
            ),aggr= 'mean',

        )

        # self._first_conv = TransformerConv(
        #     nn.Sequential(
        #         nn.Linear(nodefeat_num*2, nodeembed_to), nn.Tanh()
        #     ),aggr= 'mean',

        # )
        # self._first_conv = TransformerConv(
        #     in_channels=nodefeat_num,
        #     out_channels=nodeembed_to,
        #     heads=1,  
        #     dropout=0.5  
        # )
        
        self.TransformerConv = TransformerConv(
            in_channels=nodefeat_num,
            out_channels=nodeembed_to,
            heads=1,  
            dropout=0.5  
        )

        self._third_convTransfomer = TransformerConv(
            in_channels=nodeembed_to,
            out_channels=nodeembed_to,
            heads=1,  
            dropout=0.5  
        )

        # self._third_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        ## Pooling and actuall prediction NN
        self._pooling = [global_add_pool, global_mean_pool] # takes batch.x and batch.batch as args
        # shape of one pooling output: [B,F], where B is batch size and F the number of node features.
        # shape of concatenated pooling outputs: [B, len(pooling)*F]

    def forward(self, batch: Batch):
        node_features, edges, edge_features,mask, batch_vector , pos, domain= \
            batch.x.float(), batch.edge_index, batch.edge_attr.float(),batch.mask, batch.batch,batch.pos, batch.domain
        ## embed the features
        # node_features = self._node_embednorm(
        #     self._node_embedding(node_features))
        # edge_features = self._edge_embednorm(
        #     self._edge_embedding(edge_features))

        # do graphs convolutions
#################### LAYERS ####################
        # node_features = self._node_embednorm(self._GMMconv(node_features, edges, edge_features)) ### GMM ###
        
        ##### META Layer ###
        # node_features, edge_features, _ = self.op(node_features, edges, edge_features, None, batch_vector)
        # node_features = self._node_embednorm(node_features)
        # edge_features = self._edge_embednorm(edge_features)
        # node_features, edge_features, _ = self.op(node_features, edges, edge_features, None, batch_vector)
        # node_features = self._node_embednorm(node_features)
        # edge_features = self._edge_embednorm(edge_features)
        #### NNConv ###
        node_features =self.batch_norm1(self.NNconv1(node_features, edges, edge_features))
        node_features =self.batch_norm2(self.NNconv2(node_features, edges, edge_features)) 
        
        ### GIN ###
        # for i, layer in enumerate(self.node_ginlayers):
        #     h = layer(node_features, edges.to(torch.int64))
        #     # h = layer(node_features, edges, edge_features, None, batch_vector)
        #     h = self.batch_norms[i](h)
        #     node_features = F.relu(h)
        
        ### TransformerConv ###
        # node_features = self._node_embednorm(self.TransformerConv(node_features, edges))
        
        ## now, do the pooling
        # pooled_graph_nodes = torch.cat([p(node_features[mask], batch_vector[mask]) for p in self._pooling], axis=1) 
        pooled_graph_nodes = torch.cat([p(node_features, batch_vector) for p in self._pooling], axis=1) 
    
        return pooled_graph_nodes # ready for a loss        


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
    
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()
        
        # self.base_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=6)
        # self.base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        self.base_kernel = gpytorch.kernels.RFFKernel(num_samples = 10) #, ard_num_dims = 6
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=6))  # lengthscale for each dimension. Same shape as len(pooling_layer) # add white noise kernel, good for regression
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=6) # lengthscale for each dimension. Same shape as len(pooling_layer) # add white noise kernel, good for regression
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.covar_module =  gpytorch.kernels.RBFKernel(ard_num_dims=6)
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6)#+ gpytorch.kernels.LinearKernel()
        # self.covar_module = gpytorch.kernels.RFFKernel(num_samples = 4, ard_num_dims = 6) #+ gpytorch.kernels.LinearKernel()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6) #  + gpytorch.kernels.LinearKernel(num_dimensions=6)
        # self.covar_module = gpytorch.kernels.ArcKernel(self.base_kernel,
        #                                                 ard_num_dims=6)#radius_prior=gpytorch.priors.GammaPrior(3,2), angle_prior=gpytorch.priors.GammaPrior(0.5,1),
    def forward(self, x):
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GNNGP(gpytorch.models.GP):

    def __init__(self, feature_extractor, gp, train_x, train_y, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.feature_extractor = feature_extractor #Input for the GNN
        self.gp = gp #Input for the GP  
        self.train_x = train_x # must be a List of PyG Data 
        self.train_y = train_y
        if self.training: 
            train_x_features = self.feature_extractor(Batch.from_data_list(self.train_x)) #feed the .xyz to the GNN, but first should be converted to a torch tensor
            self.gp.set_train_data(inputs=train_x_features.to(device), targets=self.train_y, strict=False) #feed the pooled graphs of the GNN to the GP        # consider strict=True 
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x): 
        # setting training features and labels
        if self.training: 
            train_x_features = self.feature_extractor(x)
            # train_x_features = self.scale_to_bounds(train_x_features)

            self.gp.set_train_data(inputs=train_x_features.to(self.device), targets=self.train_y) # consider strict=True
        if self.training:
            x1 = train_x_features
        else:
            # x1 = self.scale_to_bounds(self.feature_extractor(x)).to(device)
            x1 = self.feature_extractor(x).to(self.device)

        # actual forward
        # x1 = self.feature_extractor(x).to(device)
        x2 = self.gp(x1)
        return x2



