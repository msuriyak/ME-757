from recursive_polynomial import *
import json
import os
from tqdm import tqdm
import numpy as np

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def make_weights(n1, n2):
    N_min, N_max = n1, n2+1        

    if not os.path.exists('nodes and weights') :
        os.makedirs('nodes and weights')

    if os.path.isfile('./nodes and weights/lobatto nodes') :
        f = open('./nodes and weights/lobatto nodes')
        lobatto_nodes = json.load(f)

        keys = [int(item) for item in list(lobatto_nodes.keys())]
        keys_to_iterate = diff([i for i in range(N_min, N_max)], keys)

        print('Lobatto nodes ...')
        for i in tqdm(keys_to_iterate):
            lobatto_nodes[str(i)] = lobatto_roots(i).tolist()
        
        lobatto_nodes = json.dumps(lobatto_nodes, sort_keys=True)
        f = open('./nodes and weights/lobatto nodes', 'w')
        f.write(lobatto_nodes)

    else : 
        lobatto_nodes = {}
        print('Lobatto nodes ...')
        for i in tqdm(range(N_min, N_max)):
            lobatto_nodes[i] = lobatto_roots(i).tolist()
        
        lobatto_nodes = json.dumps(lobatto_nodes)
        f = open('./nodes and weights/lobatto nodes', 'w')
        f.write(lobatto_nodes)

    if os.path.isfile('./nodes and weights/lobatto weights') :
        f = open('./nodes and weights/lobatto weights')
        lobatto_node_weights = json.load(f)

        keys = [int(item) for item in list(lobatto_node_weights.keys())]
        keys_to_iterate = diff([i for i in range(N_min, N_max)], keys)

        print('Lobatto weights ...')
        for i in tqdm(keys_to_iterate):
            lobatto_node_weights[str(i)] = lobatto_weights(i).tolist()
        
        lobatto_node_weights = json.dumps(lobatto_node_weights, sort_keys=True)
        f = open('./nodes and weights/lobatto weights', 'w')
        f.write(lobatto_node_weights) 
        
    else :
        lobatto_node_weights = {}
        print('Lobatto weights ...')
        for i in tqdm(range(N_min, N_max)):
            lobatto_node_weights[i] = lobatto_weights(i).tolist()
        
        lobatto_node_weights = json.dumps(lobatto_node_weights)
        f = open('./nodes and weights/lobatto weights', 'w')
        f.write(lobatto_node_weights)

    if os.path.isfile('./nodes and weights/legendre nodes') :
        f = open('./nodes and weights/legendre nodes')
        legendre_nodes = json.load(f)

        keys = [int(item) for item in list(legendre_nodes.keys())]
        keys_to_iterate = diff([i for i in range(N_min, N_max)], keys)

        print('Lobatto nodes ...')
        for i in tqdm(keys_to_iterate):
            legendre_nodes[str(i)] = legendre_roots(i).tolist()
        
        legendre_nodes = json.dumps(legendre_nodes, sort_keys=True)
        f = open('./nodes and weights/legendre nodes', 'w')
        f.write(legendre_nodes)       
    else :
        legendre_nodes = {}
        print('Leggendre nodes ...')
        for i in tqdm(range(N_min, N_max)):
            legendre_nodes[i] = legendre_roots(i).tolist()
        
        legendre_nodes = json.dumps(legendre_nodes)
        f = open('./nodes and weights/legendre nodes', 'w')
        f.write(legendre_nodes)

    if os.path.isfile('./nodes and weights/legendre weights') :
        f = open('./nodes and weights/legendre weights')
        legendre_node_weights = json.load(f)

        keys = [int(item) for item in list(legendre_node_weights.keys())]
        keys_to_iterate = diff([i for i in range(N_min, N_max)], keys)

        print('Lobatto weights ...')
        for i in tqdm(keys_to_iterate):
            legendre_node_weights[str(i)] = legendre_weights(i).tolist()
        
        legendre_node_weights = json.dumps(legendre_node_weights, sort_keys=True)
        f = open('./nodes and weights/legendre weights', 'w')
        f.write(legendre_node_weights)

    else :
        legendre_node_weights = {}
        print('Legendre weights ...')
        for i in tqdm(range(N_min, N_max)):
            legendre_node_weights[i] = legendre_weights(i).tolist()
        
        legendre_node_weights = json.dumps(legendre_node_weights)
        f = open('./nodes and weights/legendre weights', 'w')
        f.write(lobatto_node_weights)    

def get_lobatto_points(n):
    if n < 2:
        msg = 'N cannot be less than 2 %d < 2' %(n)
        raise ValueError(msg)

    if os.path.isfile('./nodes and weights/lobatto nodes'):
        f = open('./nodes and weights/lobatto nodes')
        lobatto_nodes = json.load(f)

        if not str(n) in lobatto_nodes.keys():
            make_weights(n, n)

        f = open('./nodes and weights/lobatto nodes')
        lobatto_nodes = json.load(f)
        return np.array(lobatto_nodes[str(n)])

    else : 
        make_weights(n, n)
        f = open('./nodes and weights/lobatto nodes')
        lobatto_nodes = json.load(f)
        return np.array(lobatto_nodes[str(n)])

def get_lobatto_weights(n):
    if n < 2:
        msg = 'N cannot be less than 2 %d < 2' %(n)
        raise ValueError(msg)
        
    if os.path.isfile('./nodes and weights/lobatto weights'):
        f = open('./nodes and weights/lobatto weights')
        lobatto_weights = json.load(f)

        if not str(n) in lobatto_weights.keys():
            make_weights(n, n)

        f = open('./nodes and weights/lobatto weights')
        lobatto_weights = json.load(f)
        return np.array(lobatto_weights[str(n)])

    else : 
        make_weights(n, n)
        f = open('./nodes and weights/lobatto nodes')
        lobatto_nodes = json.load(f)
        return np.array(lobatto_nodes[str(n)])

def get_legendre_points(n):
    if n < 2:
        msg = 'N cannot be less than 2 %d < 2' %(n)
        raise ValueError(msg)

    if os.path.isfile('./nodes and weights/legendre nodes'):
        f = open('./nodes and weights/legendre nodes')
        legendre_nodes = json.load(f)

        if not str(n) in legendre_nodes.keys():
            make_weights(n, n)

        f = open('./nodes and weights/legendre nodes')
        legendre_nodes = json.load(f)
        return np.array(legendre_nodes[str(n)])

    else : 
        make_weights(n, n)
        f = open('./nodes and weights/legendre nodes')
        legendre_nodes = json.load(f)
        return np.array(legendre_nodes[str(n)])

def get_legendre_weights(n):
    if n < 2:
        msg = 'N cannot be less than 2 %d < 2' %(n)
        raise ValueError(msg)
        
    if os.path.isfile('./nodes and weights/legendre weights'):
        f = open('./nodes and weights/legendre weights')
        legendre_weights = json.load(f)

        if not str(n) in legendre_weights.keys():
            make_weights(n, n)

        f = open('./nodes and weights/legendre weights')
        legendre_weights = json.load(f)
        return np.array(legendre_weights[str(n)])

    else : 
        make_weights(n, n)
        f = open('./nodes and weights/legendre nodes')
        legendre_nodes = json.load(f)
        return np.array(legendre_nodes[str(n)])

if __name__ == '__main__':
    make_weights(2, 64)