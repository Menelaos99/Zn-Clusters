import os
import numpy as np
import warnings
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import get_graph
from models import BaseGNN, ExactGPModel
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# import matplotlib.pyplot as plt

# ignore the warning: Run   timeWarning: divide by zero encountered in true_divide
#   cm = (atomic_nums*atomic_nums.T) / pairwise_distances(pos)
warnings.filterwarnings("ignore", category=RuntimeWarning)
save_pos = True
NR_OF_ABSORBANTS = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
def optimum_energy(model_gp, data_list):
    best_so_far = np.inf
    # shouldnt this give the target y? if we evaluate on the train data, 
    # the posterior returns the mean as the exact target.

    means = -model_gp(Batch.from_data_list(data_list)).mean.cpu().detach().numpy()
    # uncertainty

    best_so_far = means.max() # TODO: depending on the target, we need to adjust this to max/min  (with log normalization should be max)
    return best_so_far



def points_on_sphere(center=None, N=5, radius=1):
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

 


class BO:
    def __init__(self, cluster,atoms,opt, likelihood, model_filenames, method = "ucb", tradeoff = 0, sample_size = 1 ,device = 'cpu'):
        self.cluster = cluster
        self.atoms = atoms
        self.opt = opt
        self.model_filenames = model_filenames
        self.method = method
        self.tradeoff = tradeoff
        self.sample_size = sample_size
        self.device = device
        self.likelihood = likelihood

    def create_samples(self):
        self.Xsamples = [] # list of Data input objects to feed to GNNGP

        # NOTE: we could generate C depending on the convex hull of the adsorpant. Have to modify the is_feasible function and the generation of C points.
        hull = Delaunay(self.cluster) 
        # print(cluster)
        counter = 0
        Zn_samples = []
        while counter < self.sample_size:
            # pick an atom at random from the cluster
            n = np.random.randint(0, len(self.cluster)) # high is exclusive 
            
            atom_cluster = self.atoms[n]
            Flag = True
            if atom_cluster == 'Pt':
                Zn_sample = points_on_sphere(center=self.cluster[[n]], N=1, radius=2.7)  
            elif atom_cluster == 'O':
                Zn_sample = points_on_sphere(center=self.cluster[[n]], N=1, radius=2.1)  
            elif atom_cluster == 'Ni':
                Zn_sample = points_on_sphere(center=self.cluster[[n]], N=1, radius=2.5)
            elif atom_cluster == 'C':   #C only applied weak interactions. So we don't want suggestions near C.
                continue   
            # Example
            for i in range(0,len(self.cluster)):
                dis = np.linalg.norm(Zn_sample-self.cluster[i])
                if dis < 2.1:  ##TODO CHANGE THIS VALUE
                    Flag = False
                    break
            if Flag == True:
                if is_feasible(Zn_sample, hull):
                    counter += 1
                    Zn_samples.append(Zn_sample)
                

        Zn_samples = np.asarray(Zn_samples).reshape(self.sample_size,3)

        # create new cluster 
        self.atoms.append('Zn')
        for i in range(self.sample_size):
            new_X = np.concatenate((self.cluster, Zn_samples[i:i+1]), axis=0) 
            graph = get_graph(self.atoms, new_X,distance = 6, domain='domain#')
            graph.x = graph.x.to(self.device)
            graph.edge_s = graph.edge_attr.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.mask = graph.mask.to(self.device)
            
            # print(graph.mask)
            self.Xsamples.append(graph.to(self.device)) 

    def use_model(self):
        self.mu = 0
        self.std = 0
        for filename in self.model_filenames:
            self.mu_temp = []
            self.std_temp = []
            model = torch.load(os.path.join(os.getcwd(), "best_models",filename))
            test_loader = GeoLoader(self.Xsamples, batch_size = 1)
            for test_data in test_loader: #check layer x1 and x2 of the GP
                test_predictions =self.likelihood(model(test_data))# consider uing likelihood)
                # test_predictions = self.likelihood(self.model(test_data))# consider uing likelihood)

                self.mu_temp.append(test_predictions.mean.cpu().detach().numpy())
                self.std_temp.append(test_predictions.stddev.cpu().detach().numpy())   # 

            self.mu_temp, self.std_temp = np.asarray(self.mu_temp), np.asarray(self.std_temp)
            self.std_temp = np.reshape(self.std_temp, (self.std_temp.shape[0],1))

            self.mu_temp = np.reshape(self.mu_temp, (self.mu_temp.shape[0],1))
            self.mu_temp = -self.mu_temp   #MULTIPLY MU WITH -1 BECAUSE WE WANT TO MAXIMIZE RATHER THAN MINIMIZE
            self.std_temp = np.sqrt(self.std_temp)
            self.mu += self.mu_temp
            self.std += self.std_temp
        self.mu /= len(self.model_filenames) ## TODO choose the point with lowest std 10000x17  after the func should be 10000x1 
        self.std /= len(self.model_filenames)

    def create_predictions(self):
        try:
            self.use_model()
        except:
            self.create_samples()
            self.use_model()
        plt.hist(self.std)
        plt.show()

    def expected_improvement(self):
        with np.errstate(divide='warn'):
            imp = self.mu - self.opt - self.tradeoff
            Z = imp / self.std
            ei = imp * norm.cdf(Z) + self.std * norm.pdf(Z)
            ei[self.std == 0.0] = 0.0
        return ei

    def get_score(self):
        if self.method == "pi":
            scores = norm.cdf((self.mu - self.opt) / (self.std+1E-9)) 

        elif self.method == "pe":
            scores = self.expected_improvement()
            
        elif self.method == "ucb":
            scores = self.mu + self.tradeoff*self.std 
        

        ix = np.argsort(scores, axis=0)[::-1][:1].reshape(-1)
        best_position = (self.Xsamples[ix[0]].pos[-1]).reshape(1,-1)
        return ix, best_position
