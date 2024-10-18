import torch
import gpytorch

import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool,global_add_pool, EdgeConv, TransformerConv, GMMConv
# from torch_geometric.nn.glob import attention
def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
class BaseGNN(nn.Module):
    def __init__(self,
    nodefeat_num=3, edgefeat_num=2,
    nodeembed_to=3, edgeembed_to=2):
        super().__init__()
        ## Embeddings
        self._node_embedding = nn.Sequential(nn.Linear(nodefeat_num, nodeembed_to),nn.ReLU())
        self._node_embednorm = nn.BatchNorm1d(nodeembed_to) 
        self._edge_embedding = nn.Sequential(nn.Linear(edgefeat_num, edgeembed_to), nn.ReLU())
        self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)
        
        # Graph Convolutions
#############################################################################################################################
        self._first_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()
            ),aggr= 'mean'

        )
        # self._first_conv = GMMConv(
        #     nodeembed_to,
        #     nodeembed_to,
        #     dim=edgeembed_to,
        #     kernel_size=3
        # )
        
        # self._first_conv = EdgeConv(
        #     nn.Sequential(
        #         nn.Linear(nodefeat_num*2, nodeembed_to), nn.Tanh()
        #     ),aggr= 'mean',

        # )

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
        self._first_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)
#############################################################################################################################
        self._second_conv = NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()),
            aggr= 'mean'
        )
        
        # self._second_conv = GMMConv(
        #     nodeembed_to,
        #     nodeembed_to,
        #     dim =edgeembed_to,
        #     kernel_size=3
        # )
        
        # self._second_conv = TransformerConv(
        #     in_channels=nodefeat_num,
        #     out_channels=nodeembed_to,
        #     heads=1,  
        #     dropout=0.5  
        # )

        self._second_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)
#############################################################################################################################
        # self._third_conv = NNConv(
        #     nodeembed_to, # first, pass the initial size of the nodes
        #     nodeembed_to, # and their output-size
        #     nn.Sequential(
        #         nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
        #     ),aggr= 'mean'

        # )

        # self._third_conv = TransformerConv(
        #     in_channels=nodeembed_to,
        #     out_channels=nodeembed_to,
        #     heads=1,  
        #     dropout=0.5  
        # )

        # self._third_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        ## Pooling and actuall prediction NN
        self._pooling = [global_add_pool, global_mean_pool] # takes batch.x and batch.batch as args
        # shape of one pooling output: [B,F], where B is batch size and F the number of node features.
        # shape of concatenated pooling outputs: [B, len(pooling)*F]
        self._predictor = nn.Sequential(
            nn.Linear(6,1),
            # nn.ReLU(),
            # nn.Linear(3, 1),
            nn.ELU()

        )       
        self._predictor.apply(init_weights)



 
    def forward(self, batch: Batch):
        # We will unpack the values that define the graph, i.e. 
        # the edges (connections)
        # edge attributes (value of each connection, e.g. distance etc)
        # node values (values that characterize the nodes of the graph)
        node_features, edges, edge_features, batch_vector = \
            batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch 

        ## embed the features, i.e. pass the already existing features through a neural network to gain more complexity
        ## this is not a convolution, i.e. the projection doesnt take into account the neighbours.
        # node_features = self._node_embednorm(
        #     self._node_embedding(node_features))
        # edge_features = self._edge_embednorm(
        #     self._edge_embedding(edge_features))

        # do graphs convolutions 
#############################################################################################################################
        # node_features = self._first_conv_batchnorm(self._first_conv(node_features, edges, edge_features)) #CMM 
        
        node_features =self._first_conv_batchnorm(self._first_conv(    ### NNConv
            node_features, edges, edge_features))
        
        # node_features =self._first_conv_batchnorm(self._first_conv(
        #     node_features, edges))
#############################################################################################################################
        node_features = self._second_conv_batchnorm(self._second_conv(node_features, edges, edge_features)) #CMM 

        # node_features =self._second_conv_batchnorm(self._second_conv(  #### NNConv
            # node_features, edges, edge_features))

        # node_features = self._second_conv_batchnorm(self._second_conv(
        #       node_features, edges))
#############################################################################################################################
        # node_features =self._third_conv_batchnorm(self._third_conv(
        #     node_features, edges, edge_features))

        # node_features = self._third_conv_batchnorm(self._third_conv(node_features, edges))


        ## now, do the pooling
        pooled_graph_nodes = torch.cat([p(node_features, batch_vector) for p in self._pooling], axis=1) 
        ## now pass the pooled representation of the graph through a Neural Network. (Pooled representation is the shape size 
        # regardless of the initial molecule size)
        output = self._predictor(pooled_graph_nodes)
        return output # ready for a loss        
