################################################################################
import numpy as np
import networkx as nx
from scipy import stats
import scipy as sp
from network_ei import *
import datetime
import sys
################################################################################

# October 26th
def get_macro(G, macro_mapping, p0=0):
    """
    Given an input graph and a micro-to-macro mapping, output a macro transition matrix.
    - param G: current micro graph
    - param macro_mapping: a dictionary with {micro_node:macro_node}
    - param p0: smallest probability
    
    - output: Macro, a macro-level transition-probability matrix
    """
    # might have to assert node_labels to be integers
    G = prepare_network(G,p0=p0)
    micro_network_size = G.number_of_nodes()
    micro_Wout = get_Wout_full(G, p0)
    micro_nodes = np.unique(list(macro_mapping.keys()))
    nodes_in_macro_network = np.unique(list(macro_mapping.values()))

    macro_network_size = len(nodes_in_macro_network)
    macro_nodes = nodes_in_macro_network[nodes_in_macro_network > micro_network_size-1]
    n_macro = len(macro_nodes)
    
    if n_macro==0:
        return micro_Wout
    
    micro_to_macro_list = []
    for macro_i in range(n_macro):
        micro_in_macro_i = [k for k,v in macro_mapping.items() if v==macro_nodes[macro_i]]
        micro_to_macro_list.append(micro_in_macro_i)
    
    # get new rows
    macro_row_list = []
    for macro_i in micro_to_macro_list:
        macro_row_list.append(sum(micro_Wout[macro_i,:]))
    
    macro_rows = np.vstack(macro_row_list)

    # get new cols
    macro_col_list = []
    for macro_i in micro_to_macro_list:
        macro_col_list.append(sum(micro_Wout.T[macro_i,:]))

    macro_cols = np.vstack(macro_col_list)    
    macro_cols = macro_cols.T
    
    # get stubby diagonal square in the bottom right
    macro_bottom_right = np.zeros((n_macro,n_macro))
    for macro_i in range(n_macro):
        for macro_j in range(n_macro):
            macro_bottom_right[macro_i, macro_j] = sum(macro_row_list[macro_i][micro_to_macro_list[macro_j]])

    # put them all together in a matrix that is too big
    too_big_macro = np.block([[micro_Wout, macro_cols], [macro_rows, macro_bottom_right]])
    
    macro_out = too_big_macro[nodes_in_macro_network,:][:,nodes_in_macro_network]
    
    Macro = macro_out / macro_out.sum(axis=1)[:, np.newaxis]
    
    
    return Macro

def causal_emergence(G, p0=0, thresh=0.0001, printt=True):
    """
    Given a micro-scale network, iterate through possible macro-groupings and look for causal emergence.
    - param G: a networkx object or adjacency matrix or TPM
    - param p0: smallest probability
    
    - output Gm: a macro-scale network object with higher EI than G
    """
    G = prepare_network(G, p0) # makes the network weighted, directed if it's not already
    current_ei = get_ei(G, p0)
    micro_nodes_left = list(G.nodes())
    micros_that_have_been_macroed = []
    macro_mapping = dict(zip(micro_nodes_left, micro_nodes_left))    
    if printt:
        print("Starting with this tpm:\n",np.round(get_Wout_full(G,p0), 4))
        print("\nSearch started... current_ei = %.4f"%current_ei)

    np.random.shuffle(micro_nodes_left)
    for node_i in micro_nodes_left:
        if printt:
            print("...",node_i,"...","macro size =",len(np.unique(list(macro_mapping.values()))))

        neighbors_i = set(list(G.successors(node_i))).union(set(list(G.predecessors(node_i))))
        for node_j in neighbors_i:
            neighbors_j = set(list(G.successors(node_j))).union(set(list(G.predecessors(node_j))))
            neighbors_i = neighbors_j.union(neighbors_i)
        macros_to_check = [i for i in list(neighbors_i) if i!=node_i]
        queue = macros_to_check.copy()

        node_i_macro = macro_mapping[node_i]
        if node_i_macro == node_i:
            node_i_macro = max(list(macro_mapping.values()))+1
            
        while len(queue) > 0:
            np.random.shuffle(queue)
            possible_macro = queue.pop()

            possible_mapping = macro_mapping.copy()
            possible_mapping[node_i]         = node_i_macro
            possible_mapping[possible_macro] = node_i_macro
            try:
                MACRO = get_macro(G, possible_mapping, p0)
                macro_ei = get_ei(MACRO, p0)
                Gm = prepare_network(MACRO, p0=p0)
            except:
                return [], macro_mapping, G

            if macro_ei - current_ei > thresh:
            # keep adding shit in the queue to the current_macro_grouping, once you get anything 
            # with a little extra EI
                current_ei = macro_ei
                macro_mapping = possible_mapping
                if printt:
                    print("just found successful macro grouping... current_ei = %.4f"%current_ei)
                    
                micros_that_have_been_macroed.append(node_i)
                micros_that_have_been_macroed.append(possible_macro)
                micros_that_have_been_macroed = list(set(micros_that_have_been_macroed))

                nodes_in_macro_i = [k for k, v in macro_mapping.items() if v==node_i_macro]
                
                for new_micro_in_macro_i in nodes_in_macro_i:
                    neighbors_Mi = set(list(
                        G.successors(new_micro_in_macro_i))).\
                        union(set(list(G.predecessors(new_micro_in_macro_i))))
                        
                    for node_Mj in neighbors_Mi:
                        if node_Mj not in queue and node_Mj != node_i:
                            queue.append(node_Mj)
    
    try:
        MACRO = get_macro(G, macro_mapping, p0)
        Gm = prepare_network(MACRO, p0=p0)
        return Gm, macro_mapping, G
    except:
        return [], macro_mapping, G

########################################################################################
# under this are not good functions
def macronode(G, nodebunch=[], nodebunch_size=2, p0=np.exp(-30), printt=False):
    """
    Takes a group of m nodes and turns them into a macronode, preserving edge weights.
    """
    weights = []
    if nx.get_node_attributes(G, 'id'):
        ids = nx.get_node_attributes(G, 'id')
    else: 
        ids = dict(zip(G.nodes(), np.array(list(G.nodes()), dtype=str)))
    if nx.get_edge_attributes(G, 'weight'):
        weights = nx.get_edge_attributes(G, 'weight')
    if nx.get_edge_attributes(G, 'edgetype'):
        edgetype = nx.get_edge_attributes(G, 'edgetype')
    if nx.get_edge_attributes(G, 'weight') == {}:
	    Wout0 = get_Wout_full(G, p0)
	    G = nx.from_numpy_array(Wout0, create_using=nx.DiGraph())
        
    G = prepare_network(G)
    W_out = get_Wout_full(G, p0)
    # if weights == []:
    # 	nx.set_edge_attributes(G, )
    W_in = get_Win(G, p0)

    if nodebunch == []: 
        nodebunch=get_nodebunch_random(G, nodebunch_size)
    nodebunch = np.array([int(i) for i in nodebunch])
    rows_to_keep = np.array(list(set(range(W_out.shape[0])) - set(nodebunch)))

    #####
    macro_id = ""
    for i in range(len(nodebunch)):
        if i+1!=len(nodebunch):
            macro_id += ids[nodebunch[i]] + " + "
        else:
            macro_id += ids[nodebunch[i]]

    for i in range(len(nodebunch)):
        del ids[nodebunch[i]]

    new_macro = max(ids.keys())+1
    ids[new_macro] = macro_id
    #####    
    
    tall = W_out.T[nodebunch]
    wide = W_out[nodebunch]
    center = W_out[nodebunch].T[nodebunch]

    keep_mat = W_out[rows_to_keep].T[rows_to_keep].T
    MACRO = np.zeros((keep_mat.shape[0]+1,keep_mat.shape[1]+1))

    new_col = list(tall.sum(axis=0)[rows_to_keep])
    new_col.append(0)
    new_col = np.array(new_col)
    new_row = list(wide.sum(axis=0)[rows_to_keep])
    new_row.append(center.sum())
    new_row = np.array(new_row)
    new_row = new_row/new_row.sum()

    MACRO[0:keep_mat.shape[0], 0:keep_mat.shape[1]] = keep_mat
    MACRO.T[-1] = new_col
    MACRO[-1] = new_row
    
    thresh = 0.0001/(MACRO.shape[0]+1)
    for i in range(MACRO.shape[0]):
        if len(MACRO[i][np.where(MACRO[i] > thresh)]) > 0:
            MACRO[i][np.where(MACRO[i]<thresh)[0]] = 0
        MACRO[i] = MACRO[i]/(sum(MACRO[i]))

    Gout = prepare_network(MACRO, thresh=True)
    newids = dict(zip(list(Gout.nodes()), ids.values()))
    nx.set_node_attributes(Gout, newids, 'id')
    
    return Gout

def prepare_network(G, thresh=True, p0=np.exp(-30)):
    """Takes a network and makes it ready for causal emergence"""
    if type(G)==np.ndarray or type(G)==np.matrixlib.defmatrix.matrix:
        M = G
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
    elif type(G)==nx.classes.graph.Graph:
        A = nx.to_numpy_array(G)
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    # if nx.get_networkx
    
    Gprep = G.copy() 
    return Gprep

def get_ei_diff(G, groupings, printt=False):
    """
    Returns a dictionary where the grouping is the key and the
    values are the ei_diff
    """
    
    success_edges = []
    ei_diff_edges = []
    ei_dict_diff = {}
    
    counter = 0
    if printt:
        print("Starting to span the possible groupings at:", datetime.datetime.now())

    for eij in np.array(groupings):
        counter += 1
        eij = tuple(eij)
        ei_dict_diff[eij] = {"Success":"No", "EI_diff":0}
        Gij = macronode(G, nodebunch=eij)
        ei_diff = get_ei(G) - get_ei(Gij)
        ei_diff_edges.append(ei_diff)
        ei_dict_diff[eij]["EI_diff"] = ei_diff
        
        if ei_diff < 0:
            success_edges.append(eij)
            ei_dict_diff[eij]["Success"] = "Yes"
    
        if printt:
            if counter % 1000==0:
                print("Done with", counter, "groupings out of", len(groupings),
                      "at:", datetime.datetime.now())

    out_dict = dict(zip(groupings, ei_diff_edges))
    
    return out_dict, ei_dict_diff

def get_nodebunch_rw(G, nodebunch_size=4, node_i=[], random_grouping=True):
    """
    Given a graph, this function picks a node and
    returns the tuple of the most overlapping neighbor.
    """
    if random_grouping:
        if node_i==[]:
            node_i = np.random.choice(list(G.nodes()))
    else: 
        deg = np.array(list(dict(G.degree()).values()))
        ps = deg/sum(deg)
        choice = np.random.multinomial(1, ps)
        node_i = np.nonzero(choice)[0][0]
        
    random_walks = [node_i]
    neih_i = list(G.neighbors(node_i))
    nodebunch_size = nodebunch_size-1
    
    while nodebunch_size > 0:
        node_j = np.random.choice(neih_i)
        random_walks.append(node_j)
        nodebunch_size = nodebunch_size-1
        node_i = node_j
        neih_i = list(G.neighbors(node_i))
    
    return list(set(random_walks))

def get_nodebunch_induced(G, nodebunch_size=4, node_i=[], random_grouping=True):
    """
    Given a graph, this function picks a node and returns
    the tuple of the most overlapping neighbor.
    """
    if random_grouping:
        if node_i==[]:
            node_i = np.random.choice(list(G.nodes()))
            
    else: 
        deg = np.array(list(dict(G.degree()).values()))
        ps = deg/sum(deg)
        choice = np.random.multinomial(1, ps)
        node_i = np.nonzero(choice)[0][0]
        
    random_walks = [node_i]
    neih_i = list(G.neighbors(node_i))
    nodebunch_size = nodebunch_size-1
    
    while nodebunch_size > 0:
        node_j = np.random.choice(neih_i)
        random_walks.append(node_j)
        nodebunch_size = nodebunch_size-1
        node_i = node_j
        neih_i = list(G.neighbors(node_i))
    
    return list(set(random_walks))

def get_nodebunch_random(G, nodebunch_size):
    """
    Returns a randomly-selected nodebunch of size nodebunch_size.
    """
    if nodebunch_size < 2: nodebunch_size = 2
    nodebunch = []
    
    for node in range(nodebunch_size):
        nodebunch.append(np.random.choice(list(set(G.nodes())-set(nodebunch))))
    
    return nodebunch

# def macronode(G, nodebunch_size=2, p0=np.exp(-30), nodebunch=[], printt=False):
#     """
#     Takes a group of m nodes and turns them into a
#     macronode, preserving edge weights.
#     """
#     G = check_network(G)

#     W_out0 = get_Wout_full(G, p0)
#     W_in_0 = get_Win(G, p0)

#     temp = np.vstack((W_out0.T,np.zeros(W_out0.shape[0])))
#     W_out_m = np.vstack((temp.T,np.zeros(temp.shape[0])))
#     W_in_m = np.append(W_in_0, np.zeros(1))
    
#     if nodebunch == []: 
#         nodebunch=get_nodebunch_random(G, nodebunch_size)
    
#     nodebunch = [int(i) for i in nodebunch]
#     W_out_macro_sum = sum([W_out0[i] for i in nodebunch])
    
#     if sum(W_out_macro_sum) > 0:
#         W_out_macro = W_out_macro_sum/sum(W_out_macro_sum)
#     else:
#         W_out_macro = np.zeros(len(W_out_macro_sum))
        
#     W_out_macro = np.append(W_out_macro, np.zeros(1))
#     W_out_m[(W_out_m.shape[0] - 1)] = W_out_macro
    
#     W_in_macro = []
    
#     for j in range(W_out0.shape[0]):
#         W_in_macro.append(sum(W_out0[j][nodebunch]))
#         W_out0[j][nodebunch] = np.zeros(len(nodebunch))
    
#     W_in_macro = np.append(np.array(W_in_macro), np.zeros(1))
#     W_out_m.T[(W_out_m.shape[0]-1)] = W_in_macro
    
#     W_out_m[-1][-1] = sum(W_out_m[-1][nodebunch])
#     W_m = np.delete(W_out_m, nodebunch, 0)
#     W_out = np.delete(W_m.T, nodebunch, 0).T
    
#     W = np.zeros(W_out.shape)
#     thresh = 1/(W.shape[0]+1)
#     for i in range(W.shape[0]):
#         if len(W_out[i][np.where(W_out[i] > thresh)]) > 0:
#             W[i][np.where(W_out[i]>thresh)[0]] = W_out[i][np.where(W_out[i]>thresh)[0]]

#     Gm = nx.from_numpy_array(W, create_using=nx.DiGraph())
    
#     if printt:
#         Gprint = np.array(np.round(get_Wout_full(G, p0),2), dtype=str)
#         for i in nodebunch:
#             strs = np.array(["XXX" if x=="0.0" else str(x) for x in Gprint[i]])
#             Gprint[i] = strs
#         print("Original network with size =",G0.number_of_nodes(),"\n",Gprint)
#         print()
#         print("Turns into a coarse-grained network after macronoding over nodes:",nodebunch)
#         print()
#         print("Macronoded network with size =",Gm.number_of_nodes(),"\n", 
#               np.array(np.round(nx.to_numpy_array(Gm), 2), dtype=str))

#     return Gm

def get_paths_i(G, node_i, d=2):
    """
    Gets an edgelist of all edges [node_i, node_k] made
    from paths of length d from node_i.
    """
    node_i = type(list(G.nodes())[0])(node_i)    
    nodes_j0 = list(G.neighbors(node_i))

    if d==1:
        out = list(zip([node_i]*len(nodes_j0), nodes_j0))

    elif d==2:
        nodes_k = []
        for node_j1 in nodes_j0:
            for node_k in list(G.neighbors(node_j1)):
                if node_k != node_i:
                    nodes_k.append(node_k)
        nodes_k = set(nodes_k)        
        out = list(zip([node_i]*len(nodes_k), nodes_k))

    elif d==3:
        nodes_k = []
        for node_j1 in nodes_j0:
            for node_j2 in list(G.neighbors(node_j1)):
                for node_k in list(G.neighbors(node_j2)):
                    if node_k != node_i:
                        nodes_k.append(node_k)
        nodes_k = set(nodes_k)
        out = list(zip([node_i]*len(nodes_k), nodes_k))

    elif d==4:
        nodes_k = []
        for node_j1 in nodes_j0:
            for node_j2 in list(G.neighbors(node_j1)):
                for node_j3 in list(G.neighbors(node_j2)):
                    for node_k in list(G.neighbors(node_j3)):
                        if node_k != node_i:
                            nodes_k.append(node_k)
        nodes_k = set(nodes_k)
        out = list(zip([node_i]*len(nodes_k), nodes_k))
    
    return out
        
def get_paths(G, d=2):
    """Gets all edges [node_i, node_k] made from paths of length d in the network."""
    all_length2_paths = []
    for node_i in G.nodes():
        paths_i = get_paths_i(G, node_i, d)
        for node_k in paths_i:
            if node_k != node_i:
                all_length2_paths.append(node_k)
        
        all_length2_paths = list(set(all_length2_paths))
            
    return list(set(all_length2_paths))

def get_communities(graph):
    """
        Wrapper for community detection algorithms.
        """
    return community.best_partition(graph)

def softmax(A,k=1.0):
    """
        Calculates the softmax of a distribution, modulated by the precision term, k
        """
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    A = A*k
    maxA = A.max()
    A = A-maxA
    A = np.exp(A)
    A = A/np.sum(A)
    return A

def normalized(A):
    """
        Normalizing a vector.
        """
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    
    return [float(i)/sum(A) for i in A]


