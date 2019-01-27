################################################################################
import numpy as np
import networkx as nx
from scipy import stats
import scipy as sp
################################################################################

def define_TPM(matType="copycopy", asnx=False, noise=0.0):
    """
    Defines a transition probability matrix, default is 
    TPM from two deterministic copy gates.
    """
    if matType=="copycopy":
        tpm = np.array([[1.0,0,0,0],
                        [0,0,1.0,0],
                        [0,1.0,0,0],
                        [0,0,0,1.0]])
    if matType=="andand":
        tpm = np.array([[1.0,0,0,0],
                        [1.0,0,0,0],
                        [1.0,0,0,0],
                        [0,0,0,1.0]])
        
    if matType=="oror":
        tpm = np.array([[1.0,0,0,0],
                        [0,0,0,1.0],
                        [0,0,0,1.0],
                        [0,0,0,1.0]])

    if matType=="orcopy":
        tpm = np.array([[1.0,0,0,0],
                        [0,0,1.0,0],
                        [0,0,0,1.0],
                        [0,0,0,1.0]])

    if matType=="star":
        tpm = np.array([[1.0,0,0,0],
                        [1.0,0,0,0],
                        [1.0,0,0,0],
                        [1.0,0,0,0]])

    if noise > 0.0:
        tpm += noise
        rowsums = np.sum(tpm, 1)
        tpm = np.array([tpm[i]/rowsums[i] for i in 
                        range(len(tpm))])
        
    if asnx:
        G = nx.from_numpy_matrix(tpm, create_using=nx.DiGraph())
        return G
    
    return tpm

def add_noise(G, noise=0.01, return_G=False):
    """
    Adds noise.
    """
    G = check_network(G)
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    A = A + noise
    rowsums = np.sum(A, 1)
    A = np.array([A[i]/rowsums[i] for i in range(len(A))])
    if return_G:
        return nx.from_numpy_array(A, create_using=nx.DiGraph())
    else:
        return A
    
def check_network(G, printt=False):
    """
    Checks to make sure G is a directed networkx object.
    """
    if type(G)==np.ndarray:
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
    if type(G)==nx.classes.graph.Graph:
        G = nx.DiGraph(G)
    if nx.get_edge_attributes(G, 'weight')=={}:
        if printt:
            print("Alert: im gonna add some edgeweights to G")
        weights = {}
        for i in G.nodes():
            out_edges = list(G.out_edges(i))
            k = len(out_edges)
            for eij in out_edges:
                weights[eij] = 1./k
        nx.set_edge_attributes(G, weights, 'weight')

    return G

def get_Wout(G, p0=np.exp(-30)):
    """
    Returns Wout, the transition probability matrix of a graph G, 
    only including nodes with outgoing edges.
    """
    G = check_network(G)
    A = nx.to_numpy_array(G) 
    A = A[~np.all(A == 0, axis=1)]
    
    Wout = A/A.sum(axis=1)[:,None]
    if Wout.shape[0] > 0:
        for i in range(Wout.shape[0]):
            temp = Wout[i] + np.random.uniform(0, p0, size=Wout[i].shape)
            Wout[i] = temp/temp.sum()

        return Wout
    else:
        return np.zeros((G.number_of_nodes(),G.number_of_nodes()))

def get_Wout_full(G, p0=np.exp(-30)):
    """
    Returns full Wout.
    """
    G = check_network(G)
    A = nx.to_numpy_array(G) 
    Wout = np.zeros(A.shape)
    for i in range(A.shape[0]):
        if A[i].sum()>0:
            Wout[i] = A[i]/A[i].sum()
    
    for i in range(Wout.shape[0]):
        if A[i].sum()>0:
            temp = Wout[i] + np.random.uniform(0, p0, size=Wout[i].shape)
            Wout[i] = temp/temp.sum()
    
    return Wout

def get_Win(G, p0=np.exp(-30)):
    """
    Returns W_in, the average probility that a random walker 
    transitions to a node_j in the next timestep.
    """
    W_out = get_Wout(G, p0)
    if sum(W_out.sum(axis=0))!=0.0:
        return W_out.sum(axis=0)/sum(W_out.sum(axis=0))
    else:
        return np.zeros(len(W_out[0]))

def get_ei_i(G, p0=np.exp(-20)):
    """
    Calculates effect information for each node in a given network.
    """
    W_out = get_Wout(G, p0)
    W_in  = get_Win( G, p0)
    N_out = W_out.shape[0]

    if N_out > 0 and sum(sum(W_out))>0:
        return np.array([sp.stats.entropy(W_outi, W_in, base=2) for W_outi in W_out])

    else:
        return np.zeros(W_out.shape[0])

def get_ei(G, p0=np.exp(-20)):
    """
    Calculates EI for a given network.
    """
    eis = get_ei_i(G,p0)
    Nout = len(eis)
    if Nout == 0 or sum(eis)==0:
        return 0.0

    return sum(eis)/Nout

def get_determinism(G, p0=np.exp(-30)):
    """
    Returns the determinism present in the network.
    """
    G = check_network(G)
    N = G.number_of_nodes()
    W_out = get_Wout(G, p0)
    N_out = W_out.shape[0]
    if W_out.shape[0] > 0 and sum(sum(W_out)) > 0:
        return np.log2(N) - sum([sp.stats.entropy(W_outi, base=2) 
                                 for W_outi in W_out])/N_out

    else:
        return 0.0
    
def get_degeneracy(G, p0=np.exp(-30)):
    """
    Returns the value for the degeneracy of the network.
    """
    G = check_network(G)
    N = G.number_of_nodes()
    W_in = get_Win(G, p0)
    if sum(W_in) > 0:
        return np.log2(N) - sp.stats.entropy(W_in, base=2)

    else:
        return 0.0

def get_ei_dd(G, p0=np.exp(-30)):
    """
    Returns the effective information of the network 
    from determinism & degeneracy.
    """
    G = check_network(G)
    return get_determinism(G, p0) - get_degeneracy(G, p0)