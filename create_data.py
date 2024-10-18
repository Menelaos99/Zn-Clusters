#%%
import re
import numpy as np
import torch
from helpers import *
from pathlib import Path
#%%

def normalization(a):
    return (a - a.min(axis = 0))/(a.max(axis = 0) - a.min(axis = 0)), a.max(axis = 0), a.min(axis = 0)

# adjust this list for types you want to process
types_to_process = ['Pt6_', 'Pt9_', 'Ni3_', 'Ni6_','Pt3Ni3_', 'Ni3Pt2Ni_', 'Pt2Ni_PtNi2_', 'Pt3Pt2Ni_']  # , 'Pt13_'
types_to_predict = ['Pt2NiPt2Ni_']

raw_data_path = Path.cwd() / 'data_raw'
processed_data_path = Path.cwd() / 'data_processed'
prediction_data_path = Path.cwd() / 'data_prediction'

# default
np.seterr(divide='ignore', invalid='ignore')

# clear processed data folder
for file in processed_data_path.glob('*'):
    file.unlink()

# get all files with extension in directory
def files_with_extension(directory, extension):
    return sorted(directory.rglob(f'*{extension}'))

# read files and process them
def process_files(files, types_to_process):
    graph_list = []
    energies = []
    index = [0,3,-1] # which iterations of the files to pick: we pick the last ones, since they have the lowest energy
    file_indices = {t: 0 for t in types_to_process}  
    energy_adsorp_dict = {
        'Pt3_': (-1038.350912875803, -1778.725143985331),
        'Pt6_': (-2076.523502307076, -1778.477892),
        'Pt9_': (-3114.673164050372, -1778.477892), 
        'Ni6_': (-10408.7635, -1778.477892),
        'Pt3Ni3_': (-6242.6420586, -1778.477892),
        'Ni3Pt2Ni_': (-7631.360314157915, -1778.477892),
        'Ni3_': (-5204.455811711, -1778.477892),
        'Pt2Ni_PtNi2_': (-6242.649882303778, -1778.477892), 
        'Pt3Pt2Ni_': (-3465.231867146073, -1778.477892),
        'Pt13_': (-2913.752372557860, -1778.477892),
        'Pt2NiPt2Ni_':(-4853.940435, -1778.477892)
    }
    
    domain_dict ={}
    count = 0
    for filename in files:
        # print('filename',filename)
        # check filename for type
        type_key = next((key for key in types_to_process if filename.stem.startswith(key)), None) #"manual" for loop a =[1,2,...,n], print(next(a)) = 1, print(next(a)) = 2, ..., print(next(a)) = n
        #print(type(filename.stem))
        # print('type_key', type_key)
        if not type_key:
            continue
        file_index = file_indices[type_key] 
        isolated_en_cluster, isolated_en_mol = energy_adsorp_dict[type_key]
        e_adsorp = isolated_en_cluster + isolated_en_mol
        i = 0

        ######### Finding the regular expression of each cluster 

        # Define regular expressions to match 'Pt' and 'Ni' followed by numbers
        pt_pattern = r'Pt(\d*)'
        ni_pattern = r'Ni(\d*)'
        print(type(filename))
        # Find all matches of 'Pt' and 'Ni' with numbers
        pt_matches = re.findall(pt_pattern, str(filename))
        ni_matches = re.findall(ni_pattern, str(filename))

        # Initialize counts
        pt_count = 0
        ni_count = 0

        # Iterate through matches and sum up the counts
        for match in pt_matches:
            if match == '':
                pt_count += 1
            else:
                pt_count += int(match)

        for match in ni_matches:
            if match == '':
                ni_count += 1
            else:
                ni_count += int(match)
        domain_key = f'Pt{pt_count}Ni{ni_count}'
        
        if  domain_key in domain_dict.keys():
            domain =domain_dict[f'Pt{pt_count}Ni{ni_count}']
        else:
            domain_dict[f'Pt{pt_count}Ni{ni_count}'] = f'domain{count}'
            domain = f'domain{count}'
            count += 1
        
        with open(filename, 'r') as file:
            energy_temp, graph_temp = [], []
            for line in file:
                
                if line.strip().isdigit():
                    atom_num = int(line.strip())
                    atom_count = 0
                    atoms, coordinates = [], []
                    i+=1
                    continue
                
                energy_match = re.search(r'E\s+\s*([+-]?\d+\.\d+)', line)
                if energy_match:
                    e_value = float(energy_match.group(1))
                    energy_temp.append(e_value)
                    continue
                
                if re.match(r'^\s*[A-Z][a-z]?', line):
                    atom, x1, x2, x3 = line.split()
                    atoms.append(atom)
                    coordinates.append([float(x1), float(x2), float(x3)])
                    atom_count += 1
                    if atom_count == atom_num:
                        graph_temp.append(get_graph(np.array(atoms), np.array(coordinates), e_value, distance = 8, domain=domain)) 
            if i > 1: # some files contain many iterations, for these files we select the last iterations (i.e. index -1,-2, etc.)
                    graph_temp = [graph_temp[ind] for ind in index]
                    energy_temp = [energy_temp[ind] for ind in index]
            for i, graph in enumerate(graph_temp):
                graph_index = index[i] if i < len(index) else i 
                graph_list.append((type_key, graph, file_index, graph_index))
            for energy in energy_temp:
                    energies.append(energy)#- e_adsorp 
            save_atoms(processed_data_path, type_key + '_atoms.txt', atoms)
            file_indices[type_key] += 1
    return graph_list, energies

# extract all .xyz files
file_list = files_with_extension(raw_data_path, '.xyz')

graph_list, energies = process_files(file_list, types_to_process)
graph_list_pred, _ = process_files(file_list, types_to_predict)
# print(energies)

# normalize energies
energies = np.array(energies)
mean, std = np.mean(energies), np.std(energies)
energies = (energies - mean) / std  #
# energies ,_,_ = normalization(energies)

# shuffle data
for i, (type_key, graph, _, _) in enumerate(graph_list):
    graph.y = energies[i]

# plot histogram of energies
import matplotlib.pyplot as plt
plt.hist(energies)
plt.show()
# print(graph_list[:3])
# print(energies)

# save data
processed_data_path.mkdir(parents=True, exist_ok=True)
for type_key, graph, file_index, data_point_index in graph_list:
    filename = f'{type_key}_{file_index}_data_{data_point_index}.pt'
    torch.save(graph, processed_data_path / filename)

for type_key, graph, file_index, data_point_index in graph_list_pred:
    filename = f'{type_key}_{file_index}_data_{data_point_index}.pt'
    torch.save(graph, prediction_data_path / filename)
# %%

