#%%
import numpy as np
from sklearn.metrics import pairwise_distances
import torch 
from torch_geometric.data import Data, Batch
import os
#%%

def calculate_edges(coordinates: np, atoms): 
    adjacency_matrix = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[0]):

            p1 = np.array(coordinates[i])   
            p2 = np.array(coordinates[j])
            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)
            
            if dist < 2.7:
                adjacency_matrix[i][j] = 1
            else:
                adjacency_matrix[i][j] = 0
    # np.fill_diagonal(adjacency_matrix, 0)
    ec_list = []
    ecoor_list = []
    for i in range(adjacency_matrix.shape[0]):
        e_connections = []
        e_coor = []
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i][j] == 1:
                e_connections.append(atoms[j])
                e_coor.append([i,j])
            if j == len(atoms) -1:
                ec_list.append(e_connections)
                ecoor_list.append(e_coor)
    bonds_matrix = np.copy(adjacency_matrix)
    for i in range(len(atoms)):
        if atoms[i] == 'C' and len(ec_list[i]) == 3:
            for j in range(len(ec_list[i])):
                if ec_list[i][j] == 'O':
                    bonds_matrix[ecoor_list[i][j][0]][ ecoor_list[i][j][1]] = 2
                    bonds_matrix[ecoor_list[i][j][1]][ ecoor_list[i][j][0]] = 2
    return adjacency_matrix, bonds_matrix


def save_atoms(root_path, txt_name, atoms):
    file_path = os.path.join(root_path, txt_name)
    if not os.path.isfile(file_path):
        np.savetxt(file_path, atoms, fmt='%s')
        print(f'File {txt_name} saved.')
    else:
        print(f'File {txt_name} exists')


def array_reshape(a):
    a = np.array(a)
    a = np.reshape(a, (a.shape[0]*a.shape[1],1))
    return a


################# DIFFERENT IMPLEMENTATION FO THE GET GRAPH #################
# def get_graph(atoms, pos, target=None):
#     '''
#     Get a Data graph from the numpy coordinates, the type of atom and the target.
#     '''
#         # edge index   
#     # we create edges that are fully connected. IE. the following lines
#     # will create a 2d vector with all-to-all connections
#     # i.e. if we have 3 nodes (atoms)
#     # the following commands will create the vector [0,0,0,1,1,1,2,2,2][0,1,2,0,1,2,0,1,2]    
#     # a = np.arange(len(atoms))    

#     #initialisation of edge indexes    ############################################################
#     # edges = np.array(np.meshgrid(a,a)).T.reshape(-1,2).T
#     # edges = torch.tensor(edges, dtype=torch.int64)

#     #here we will create some values for the nodes. These values will come from
#     # the properties of the atom that each node represents
#     atom_to_num = {'C': 6, 'O':8, 'Zn':30, 'Pt':78, 'Ni':28} # atom to atomic number
#     atom_to_en = {'C': 2.55, 'O':3.44, 'Zn':1.65, 'Pt':2.28, 'Ni':1.91} # atom to electronegativity
#     atom_to_r = {'C': 70, 'O':60, 'Zn':135, 'Pt':135, 'Ni':135} # atom to radius
#     atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[:, np.newaxis] # keep as numpy for later use
#     electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
#     atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)


#     # In the loop we extract the nodes' embeddings, edges connectivity 
#     # and label for a graph, process the information and put it in a Data
#     # object, then we add the object to a list

#     # Node features
#     # atomic number abd electronegativity 
#     # Edge features
#         # shape [N', D'] N': number of edges, D': number of edge features
#         # cm matrix and bond matrix   
    
#     adjacency_matrix, bonds_matrix= calculate_edges(pos, atoms)
#     adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float) 
#     edges = torch.nonzero(adjacency_matrix, as_tuple=False).t()
    
#     # print(edges)
#     # Here we calculate the coulomb matrix. This is a distance metric in a sense. 
#     # it shows how connected two nodes are (strength of electric interaction)    
#     pair_dist = pairwise_distances(pos)   
    
#     cm = (atomic_nums*atomic_nums.T) / pair_dist
#     np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
#     np.fill_diagonal(pair_dist, 1e-10)
    
#     cm = np.multiply(adjacency_matrix, cm)
#     pair_dist = np.multiply(adjacency_matrix, pair_dist)
    
#     cm_nonzero=torch.nonzero(cm, as_tuple=False)
#     cm_vec = cm[cm_nonzero[:, 0], cm_nonzero[:, 1]]

#     pd_nonzero=torch.nonzero(pair_dist, as_tuple=False)
#     pd_vec = pair_dist[pd_nonzero[:, 0], pd_nonzero[:, 1]]

#     cm_vec = cm_vec.reshape(-1, 1)
#     pd_vec = pd_vec.reshape(-1, 1)
#     # cm = cm.flatten()[:, np.newaxis]
#     # pair_dist = pair_dist.flatten()[:, np.newaxis]

#     # bonds_matrix = bonds_matrix.flatten()[:, np.newaxis]

#     # edge_attr = torch.tensor(cm, dtype=torch.float)

#     #initialisation of edge attributes    ############################################################
#     edge_attr = torch.cat([torch.tensor(cm_vec, dtype=torch.float), torch.tensor(pd_vec, dtype=torch.float)], dim = 1) #, torch.tensor(bonds_matrix, dtype=torch.float) #torch.tensor(pair_dist.flatten()[:, np.newaxis], dtype = torch.float)
    
#     # here we encode the molecule level energy
#     if target:
#         target = torch.tensor(target, dtype=torch.float)


#     # and here we package all the node attributes (electronegativity, atomic radius and atomic number into one array)
#     #initialisation of node attributes    ############################################################
#     node_attrs = torch.cat([torch.tensor(atomic_nums, dtype=torch.float), electroneg,atomic_radius], dim=1)
#     # print('nodeattr', node_attrs.shape)
#     # print('edges', edges.shape)
#     # print('edgeattr', edges.shape)
#     #the node attributes, edges, edge attributes and targets are packaged into one data object
#     #that is very handy to use for representing the graph
#     graph = Data(x=node_attrs,
#             edge_index=edges,
#             edge_attr=edge_attr, 
#             y=target)
    
#     return  graph

def get_graph(atoms, pos, target=None, distance = 8, domain = None):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
        # edge index   
    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a,a)).T.reshape(-1,2).T
    edges = torch.tensor(edges, dtype=torch.int64)

    # atomic number abd electronegativity # TODO: add atomic radius ####### DONE
    atom_to_num = {'C': 6, 'O':8, 'Zn':30, 'Pt':78, 'Ni':28} # atom to atomic number
    atom_to_en = {'C': 2.55, 'O':3.44, 'Zn':1.65, 'Pt':2.28, 'Ni':1.91} # atom to electronegativity
    atom_to_r = {'C': 70, 'O':60, 'Zn':135, 'Pt':135, 'Ni':135} 

    atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[:, np.newaxis] # keep as numpy for later use
    electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)

    # In the loop we extract the nodes' embeddings, edges connectivity 
    # and label for a graph, process the information and put it in a Data
    # object, then we add the object to a list

    # Node features

    # Edge features
        # shape [N', D'] N': number of edges, D': number of edge features
        # cm matrix and bond matrix   
    pair_dist = pairwise_distances(pos)   
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    # print('original cm', cm.shape)
    np.fill_diagonal(cm, 0)
    cm_max = cm.max()
    mask_matrix = np.zeros((cm.shape[0], cm.shape[1]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] > cm_max*0.1:
                mask_matrix[i,j] = 1
    # print(mask_matrix)
    #n_nodes = np.argwhere(cm > cm_max*0.3)
    #print("1 shape", n_nodes.shape)
    # n_nodes = np.array(list(set(n_nodes)))
    #print("2", n_nodes.shape)
    #mask = torch.zeros((cm.shape[0], cm.shape[0]), dtype=torch.bool)
    #print('mask shape', mask.shape)
    #mask[n_nodes] = 1
    #print('mask', mask)
    
    # print('final cm', cm)
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
    cm = cm.flatten()[:, np.newaxis]
    edge_attr = torch.tensor(cm, dtype=torch.float)
    edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), torch.tensor(pair_dist.flatten()[:, np.newaxis], dtype = torch.float)], dim = 1)
    if target:
        target = torch.tensor(target, dtype=torch.float)
    # belonging = torch.zeros(pos.shape[0],1)
    # belonging[:-2] = 1
    # belonging[:-2,0] = 1
    # belonging[-2:,1] = 1
    
    node_attrs = torch.cat([torch.tensor(atomic_nums, dtype=torch.float), electroneg,atomic_radius], dim=1)
    distances_co = pair_dist[-2:,:]
    nearby_nodes = np.argwhere(distances_co < distance)[:,1] #
    nearby_nodes = np.array(list(set(nearby_nodes)))
    mask = torch.zeros((pair_dist.shape[0]), dtype=torch.bool)
    mask[nearby_nodes] = 1

    
    graph = Data(x=node_attrs,
                 pos=pos,
            edge_index=edges,
            edge_attr=edge_attr, 
            y=target,
            mask = mask,
            domain =domain)
    
    return graph


def save_points(name, positions,cluster_origin, nr):
        with open(name+'.xyz', "w") as file:
            file.write(f" {positions.shape[0]+len(cluster_origin)}\n")
            file.write(f"Simulation {nr}\n")
            for atom, coordinate in cluster_origin:
                line = f"{atom} {coordinate[0]} {coordinate[1]} {coordinate[2]}\n"
                file.write(line)
            file.write("Zn\t")
            np.savetxt(file, positions, fmt='%.7f')
            file.write("\n######################################\n")


# %%
########### 
# random_array = np.random.rand(16, 3)
coordinates = np.array([[1.662463,-1.232242,0.459467],
[-0.579439,     -2.052090,      0.465018],
 [0.171796,     -0.512492,      2.178075],
 [3.346672,     -1.421256,     -0.078367],
 [0.716988,     -2.059383,     -0.966641],
[-1.878460 ,    -2.953070,     -0.328891],
[-1.601906,     -1.109658,     1.803799],
 [-0.096398,      0.417562,      3.661570],
[1.786992,      0.323842,      1.543286],
[4.442258,     -1.576264,     -0.408577],
[0.843685   ,  -2.413044 ,    -2.081435],
[-2.740458 ,    -3.448309 ,    -0.918755],
[-2.708046  ,   -1.025838 ,     2.197640],
[-0.281476  ,    1.056206 ,     4.606119],
[2.451552   ,   1.279002  ,    1.715110],
[1.074289 ,    -2.916704 ,     2.219852]])
atoms = ['Ni','Ni','Ni','C','C','C','C','C','C','O','O','O','O','O','O','Zn']
# atoms_unique, counts = np.unique(atoms, return_counts=True)
# print(counts)
# print(atoms_unique)
# unique_encode = {}

# for i in range(len(atoms_unique)):
#     unique_encode[atoms_unique[i]] = i
# atoms_encoded = []
# for i in range(len(atoms)):
#     atoms_encoded.append(unique_encode[atoms[i]])
# print('atoms encoded', atoms_encoded)

# atoms = np.array(atoms)
a,b = calculate_edges(coordinates, atoms)
# print(a)
adj = torch.tensor(a, dtype=torch.float) 
edge_index = torch.nonzero(adj, as_tuple=False)#.t()

# print(edge_index[:,0])
# print(edge_index[:,1])
# print(edge_index.shape)
# print(edge_index)
c = get_graph(atoms ,coordinates)
# %%
class MyClass:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Create an instance of the class
my_instance = MyClass(x=1, y=2, z=3)

# Print the original instance
print("Original instance:", my_instance.__dict__)

# Delete the 'y' attribute
del my_instance.y

# Print the modified instance
print("Modified instance:", my_instance.__dict__)
# %%
