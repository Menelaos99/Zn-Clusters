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

import gpytorch
import re
from helpers import get_graph
from models import BaseGNN, ExactGPModel
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# ignore the warning: Run   timeWarning: divide by zero encountered in true_divide
#   cm = (atomic_nums*atomic_nums.T) / pairwise_distances(pos)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#%%

learning_rate=1e-3
EPOCHS = 2000
SAMPLE_SIZE = 10000
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
test_string = 'data_0.pt'
find_first = re.compile('_0.pt')
counter_for_cluster = 0
counter_boring = 0
# filenames = filenames[:6]
for filename in filenames:
    if filename.endswith('.pt'):
        graph = torch.load(os.path.join(root_path, filename)) 
        graph.x = graph.x.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.y = torch.tensor(graph.y).to(device)
        data_list.append(graph)
        print(filename)
        if bool(re.search(find_first, filename)):
            counter_for_cluster = counter_boring
        counter_boring += 0
print(data_list)



# data_list_eval = data_list[-2:]
# data_list = data_list[:-2]
# print(data_list_eval, data_list)
# print(f'device of x, edge_attr, y: {graph.x.device}, {graph.edge_attr.device}, {graph.y.device}')

#Create atom list from xyz files
def create_atom_list():
    return()

# list of atoms 
with open('data_processed/atoms.txt', 'r') as f:
    lines = f.readlines()
atoms = [line.strip() for line in lines]

# initial positions of cluster 
cluster = data_list[0].pos[:-1]

cluster_origin = list(zip(atoms[:-1],cluster))
# note for GP:
# inputs are changing based on transformation
# targets are fixed
targets = torch.tensor([data.y for data in data_list], dtype=torch.float).to(device)
# targets_eval = torch.tensor([data.y for data in data_list_eval], dtype=torch.float).to(device)

class GNNGP(gpytorch.models.GP):
    def __init__(self, feature_extractor, gp, train_x, train_y):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp = gp
        self.train_x = train_x # must be a List of pyg Data 
        self.train_y = train_y
        if self.training: 
            train_x_features = self.feature_extractor(Batch.from_data_list(self.train_x))
            self.gp.set_train_data(inputs=train_x_features.to(device), targets=self.train_y, strict=False) # consider strict=True
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x): 
        # setting training features and labels
        if self.training: 
            train_x_features = self.feature_extractor(x)
            # train_x_features = self.scale_to_bounds(train_x_features)

            self.gp.set_train_data(inputs=train_x_features.to(device), targets=self.train_y) # consider strict=True
        if self.training:
            x1 = train_x_features
        else:
            # x1 = self.scale_to_bounds(self.feature_extractor(x)).to(device)
            x1 = self.feature_extractor(x).to(device)

        # actual forward
        # x1 = self.feature_extractor(x).to(device)
        x2 = self.gp(x1)
        return x2

# TODO: adjust batch size
train_loader = GeoLoader(data_list, batch_size=len(data_list), shuffle= False) # batch-size is data size for GP

# likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(5)*0.0001) # gpytorch.likelihoods.GaussianLikelihood() 
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# TODO: should set noise to zero. No DFT error
feature_extractor = BaseGNN(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
gp = ExactGPModel(train_x=None, train_y=None, likelihood=likelihood)
feature_extractor.to(device)
gp.to(device)
model_gp = GNNGP(feature_extractor=feature_extractor,
                gp=gp, train_x=data_list, train_y=targets)
model_gp.to(device)
optimizer = torch.optim.Adagrad(model_gp.parameters(), lr=learning_rate)
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


def train(epoch):
    # "Loss" for GPs - the marginal log likelihood
    for data in train_loader:
        data = data.to(device)
        # zero gradients from previous iteration
        optimizer.zero_grad()
        # output from model
        output = model_gp(data)
        # calc loss and backprop gradients
        loss = -mll(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,))
        loss.backward()
        optimizer.step()
    return loss / len(train_loader.dataset) 

loss_history = []
best = {}
eval_loss = np.inf
model_gp.train()
likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)
# mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model_gp)

for epoch in range(1, EPOCHS+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    scheduler.step(loss)
    model_gp.eval()
    likelihood.eval()
    # means = model_gp(Batch.from_data_list(data_list)).mean.detach().cpu().clone().numpy()
    means = model_gp(Batch.from_data_list(data_list)).mean.detach().cpu().clone().numpy()
    std_print = model_gp(Batch.from_data_list(data_list)).stddev.detach().cpu().clone().numpy()

    # means_eval = model_gp(Batch.from_data_list(data_list_eval)).mean.detach().cpu().clone().numpy()
    # std_eval = model_gp(Batch.from_data_list(data_list_eval)).stddev.detach().cpu().clone().numpy()

    # if np.mean(abs(means_eval - targets_eval.detach().cpu().clone().numpy())) < eval_loss: #TODO: CHANGE THE VALIDATION SET EVALUATION TO BE THE -mll
    #     eval_loss = np.mean(abs(means_eval - targets_eval.detach().cpu().clone().numpy()))
    #     torch.save(model_gp.state_dict(), 'model_weights.pth')
    #     print("model saved at ",means_eval,std_eval, targets_eval)
    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   means : ' % (
    # epoch, EPOCHS, loss.item(),
    # model_gp.gp.covar_module.base_kernel.lengthscale.item(),
    # ), means)
    if epoch%100 == 0:

        print('Iter %d/%d - Loss: %.3f   \n means : ' % (
        epoch, EPOCHS, loss.item(),
        ), means)
        print('True_val:',targets.cpu().numpy())
        print(' diff :  ',means-targets.cpu().numpy())#,'\n')
        print('std : ',std_print)

        # print(' Targets eval :' , means_eval,std_eval, targets_eval )


       
    model_gp.train()
    likelihood.train()

    loss_history.append({"train": loss})
# model_gp.load_state_dict(torch.load('model_weights.pth'))

#%%
######################################
#surrogate model
######################################
# get data from simulation:

# cluster = np.random.rand(5,3) # shape: [N_samples, N_atoms, spatial_dimension]
# y = np.random.rand(5,1)


# # select a kernel
# kernel = GPy.kern.MLP(cluster.shape[1],ARD = True)
        # optimizer.zero_grad()
        # output from model
# a trained GP model returns a `MultivariateNormal` containing theposterior mean and covariance



# find the best predicted energy so far (i.e. the minimum one)
def optimum_energy(model
first_attempt.get_suggestions(tradeoff = 1)
first_attempt.get_urns the mean as the exact target.
    model_gp.eval()
    likelihood.eval()
    means = -model_gp(Batch.from_data_list(data_list)).mean.cpu().detach().numpy()
    std = model_gp(Batch.from_data_list(data_list)).stddev.cpu().detach().numpy()
    # uncertainty

    best_so_far = means.max() # TODO: depending on the target, we need to adjust this to max/min  (with log normalization should be max)
    return best_so_far
# %%
# define expected_improvement and get_score helper functions

# expected improvement: 
#   gives a score to each point, trading off the prediction with the uncertainty. 
#   Goodness of a point is given by a combination of its prediction and uncertainty quality.
# NOTE: EI formula changes whether we are doing maximization or minimization
#       to check proof of EI, but I think that 
#       for max: mu - mu_sample_opt
#       for min: mu_sample_opt - mu
#       where mu is posterior mean of sample, mu_sample_opt is optimal prior mu so far (both obtained through GP)
def expected_improvement(mu, std, optimal_so_far, gpr, tradeoff=0.01):
    mu_sample_opt = optimal_so_far
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt + tradeoff
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
    return ei

# get_score
#   defines which acquisition function to use, whether pi, pe or ucb.
#   Goodness of a point is given by a combination of its prediction and uncertainty quality. 
#   Acquisition function combines them with a certain tradeoff.
def get_score(Xsamples, optimal_so_far, model, method = 'pe', tradeoff = 0.5 ):
    model.eval() 
    likelihood.eval()
    mu = []
    std = []
    test_loader = GeoLoader(Xsamples, batch_size = 1)
    for test_data in test_loader: #check layer x1 and x2 of the GP
        dist = model(test_data) # consider uing likelihood
        mu.append(dist.mean.cpu().detach().numpy())
        std.append(dist.stddev.cpu().detach().numpy())   # 

    mu, std = np.asarray(mu), np.asarray(std)
    std = np.reshape(std, (std.shape[0],1))
    mu = np.reshape(mu, (mu.shape[0],1))
    mu = -mu
    std = np.sqrt(std)
    if method == "pi":

        scores = norm.cdf((mu - optimal_so_far) / (std+1E-9)) 

    elif method == "pe":
        scores = expected_improvement(mu, std, optimal_so_far, model, tradeoff)
        
    elif method == "ucb":
        scores = mu - tradeoff*std 
    return scores

def points_on_sphere(center=None, N=100000, radius=1):
    '''generate N points on the surface of a sphere with given radius and center'''
    # spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, (N,))
    phi = np.random.uniform(0, np.pi, (N,))
    # cartesian coordinates
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    points = center + np.vstack((x,y,z)).T 
    return points
        # optimizer.zero_grad()
        # output from model

def is_feasible(points, hull):
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # if point is not in simplex
    return hull.find_simplex(points) < 0 



# get_suggestions
#   generate many random points (make sure they are all within the cubic cell)
#   get a score for each of these points
#   pick the 5 best points for running DFT on them
def get_suggestions(optimal_so_far, model, method = "ucb", tradeoff = 0.5, sample_size=10):
    Xsamples = [] # list of Data input objects to feed to GNNGP

    # radius = 1.43 # typical bond length of CO

    # NOTE: we could generate C depending on the convex hull of the adsorpant. Have to modify the is_feasible function and the generation of C points.
    hull = Delaunay(cluster) 
    # print(cluster)
    counter = 0
    Zn_samples = []
    while counter < sample_size:
        # pick an atom at random from the cluster
        n = np.random.randint(0, len(cluster)-1) # high is exclusive 
        # generate C atom within 1A of distance from the atom.
        
        atom_cluster = atoms[n]
        Flag = True
        if atom_cluster == 'Pt':
            Zn_sample = points_on_sphere(center=cluster[[n]], N=1, radius=3.14) 
        elif atom_cluster == 'O':
            Zn_sample = points_on_sphere(center=cluster[[n]], N=1, radius=3.54) 
        elif atom_cluster == 'C':   #C only applied weak interactions. So we don't want suggestions near C.
            continue   
        # Example
        # Zn_sample = [-5.0511 ,  9.28771,  8.64236]
        for i in range(0,len(cluster)):
            dis = np.linalg.norm(Zn_sample-cluster[i])
            if dis < 3.14:
                Flag = False
                break
        if Flag == True:
            if is_feasible(Zn_sample, hull):
                counter += 1
                Zn_samples.append(Zn_sample)
            

    Zn_samples = np.asarray(Zn_samples).reshape(sample_size,3)

    # create new cluster 
    for i in range(sample_size):
        new_X = np.concatenate((cluster, Zn_samples[i:i+1]), axis=0) 
        graph = get_graph(atoms, new_X)
        graph.x = graph.x.to(device)
        graph.edge_s = graph.edge_attr.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.mask = graph.mask.to(device)
        # print(graph.mask)
        Xsamples.append(graph.to(device)) 
    

    # scores = get_score(Xsamples, optimal_so_far, model, method = method, tradeoff = 0.5)
    a = int(SAMPLE_SIZE*0.5)
    b = a + int(SAMPLE_SIZE*0.3)
    # Xsamples = torch.Tensor(Xsamples).to(device)
    # plt.hist(model_gp(Batch.from_data_list(Xsamples)).stddev.detach().cpu().clone().numpy())
    # plt.show()
    scores_1 = get_score(Xsamples, optimal_so_far, model, method = method, tradeoff = -10)
    scores_2 = get_score(Xsamples, optimal_so_far, model, method = method, tradeoff = 10)
    # scores_3 = get_score(Xsamples[b:], optimal_so_far, model, method = method, tradeoff = 0.2)
    scores = np.concatenate((scores_1, scores_2), axis=0)
    
    # # do argsort 
    # ix_1 = np.argsort(scores_1, axis=0)[::-1][-1].reshape(-1)
    # ix_2 = np.argsort(scores_2, axis=0)[::-1][0].reshape(-1)
    # ix = np.concatenate((ix_1, ix_2), axis = 0 )
    # have to return only the positions of C and O
    ix1 = np.argsort(scores_1, axis=0)[::-1][:1].reshape(-1)
    ix2 = np.argsort(scores_2, axis=0)[::-1][:1].reshape(-1)
    ix = np.concatenate((ix1, ix2), axis=0)

    return Zn_samples[ix], scores[ix]

#%%

# print the optimum energy and other important data
opt = optimum_energy(model_gp, cluster)
Zn_pos_ucb, scores = get_suggestions(opt, model_gp, "ucb", 0.5, sample_size=SAMPLE_SIZE)
print('positions of Zn:')
print(Zn_pos_ucb)
print('scores:')
print(scores)
Zn_pos_pe, scores = get_suggestions(opt, model_gp, "pe", 0.5, sample_size=SAMPLE_SIZE)
print('positions of Zn:')
print(Zn_pos_pe)
# print(model_gp())
# print('\npositions of O:')
# print(O_pos)
print('scores:')
print(scores)

if save_pos is True:
    with open("output.txt", "w") as file:
        for i in range(2):
            file.write(f"Simulation {i+1}\n")
            for atom, coordinate in cluster_origin:
                line = f"{atom} {coordinate[0]} {coordinate[1]} {coordinate[2]}\n"
                file.write(line)
            file.write("Zn\t")
            np.savetxt(file, Zn_pos_ucb[[i]], fmt='%.7f')
            # file.write("O\n")
            # np.savetxt(file, O_pos[[i]], fmt='%.7f')
            file.write("\n######################################\n")
