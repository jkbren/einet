"""
ei_net.py
--------------------
Network effective information code related to the paper:

Klein, B. & Hoel, E. (2019)
Uncertainty and causal emergence in complex networks.

author: Brennan Klein
email: brennanjamesklein at gmail dot com
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy


def check_network(G):
    """
    A pre-processing function that turns networkx objects into directed
    networks with edge weights, or turns np.ndarrays into directed networks.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question

    Returns
    -------
    G (nx.DiGraph): a directed, weighted version of G

    """

    if type(G) == np.ndarray:
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())

    if type(G) == nx.classes.graph.Graph:
        G = nx.DiGraph(G)

    if nx.get_edge_attributes(G, 'weight'):
        weights = {}

        for i in G.nodes():
            out_edges = list(G.out_edges(i, data=True))
            k = len(out_edges)
            weights_i = [out_edges[xx][2]['weight'] for xx in range(k)]
            weights_i_sum = sum(weights_i)

            for eij in out_edges:
                weights[(eij[0], eij[1])] = eij[2]['weight'] / weights_i_sum

        nx.set_edge_attributes(G, weights, 'weight')

    else:
        weights = {}

        for i in G.nodes():
            out_edges = list(G.out_edges(i))
            k = len(out_edges)

            for eij in out_edges:
                weights[eij] = 1./k

        nx.set_edge_attributes(G, weights, 'weight')

    old_node_labels = list(G.nodes())
    new_node_labels = list(range(G.number_of_nodes()))
    node_label_mapping = dict(zip(old_node_labels, new_node_labels))
    node_label_mapping_r = dict(zip(new_node_labels, old_node_labels))

    G = nx.relabel_nodes(G, node_label_mapping, copy=True)
    nx.set_node_attributes(G, node_label_mapping_r, 'label')

    return G


def W_out(G):
    """
    Returns Wout, the transition probability matrix of a graph G, only
    including nodes with outgoing edges.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.

    Returns
    -------
    Wout (np.ndarray): an $N x N$ transition probability matrix of random
                       walkers in the system.

    """

    G = check_network(G)
    return nx.to_numpy_array(G)


def W_in(G, intervention_distribution='Hmax'):
    """
    Returns Win, a vector of length N with elements that correspond to the
    expected distribution of random walkers after there is an intervention (the
    default is an intervention at maximum entropy) into the system (i.e. the
    introduction of random walkers).

    Previously, this has been referred to as the ``effect distribution'' of an
    intervention distribution because it's the expected distribution of effects
    or weights on a transition probability matrix.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    intervention_distribution (np.ndarray or str): if 'Hmax', this represents a
            uniform intervention into a system's states. Otherwise, it's a
            heterogeneous intervention, often used in causal emergence (because
            a coarse-graining can be interpreted as changing the kinds of
            interventions that are informative about a given system).

    Returns
    -------
    Win (np.ndarray): an $N x 1$ array where each element is the fraction of
                      random walkers that are expected to be on each node after
                      intervention has been performed onto the system.

    """

    Wout = W_out(G)

    if str(intervention_distribution) == 'Hmax':
        IntD = np.ones(Wout.shape[0])/Wout.shape[0]
    else:
        if sum(intervention_distribution) >= 0.0:
            IntD = intervention_distribution / sum(intervention_distribution)
        else:
            return np.zeros(Wout.shape[0])

    Win = Wout.T.dot(IntD)
    if sum(Win):
        return Win / sum(Win)

    else:
        return np.zeros(len(Win))


def effective_information(G):
    """
    Calculates the effective information (EI) of a network, $G$, according to
    the definition provided in Klein & Hoel, 2019. Here, we subtract the
    average entropies of the out-weights of nodes in a network, WOUT_average
    ($\langle H[W_i^{out}] \rangle$), from the entropy of the average out-
    weights in the network, WIN_entropy ($H[\langle W_i^{out} \rangle]$).

    $ EI = H[\langle W_i^{out} \rangle] - \langle H[W_i^{out}] \rangle $

    The first term in this subtraction is the maximum amount of information
    possible in a causal structure, whereas the second term corresponds to
    the amount of noise in the connections between a given node, $v_i$, and
    its neighbors, $v_j$.


    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.

    Returns
    -------
    EI (float): the effective information of a given network.

    """
    # make sure nodes in the network have edge weights that sum to 1.0
    G = check_network(G)
    N = G.number_of_nodes()

    # EI is only calculated over nodes with outgoing edges,
    # as sinks don't any information about causation.
    Nout = len(np.nonzero(list(dict(G.out_degree()).values()))[0])

    if Nout > 0:
        # Build empty arrays to fill either with probabilities or entropies
        Wout = np.zeros(N)
        Win = np.zeros(N)

        # for each node, calculate its contribution to WOUT and WIN
        for i, node_i in enumerate(G.nodes()):
            # get the out-weights of node_i, and add its entropy to WOUT
            Wout_i = [node_j['weight'] for node_j in G[node_i].values()]
            Wout[i] = entropy(Wout_i, base=2)

            # add all of its out-weights to the vector of 'in-weights'
            # (aka the vector of average out-weights), and normalize by
            # the number of nodes with output, Nout.
            for j, w_ij in G[node_i].items():
                Win[j] += w_ij['weight'] / Nout

        # EI is defined by a subtraction involving two quantities:
        #   1. the average entropy of out-weights in the network,
        #      WOUT_average, corresponding to "noise" in the outgoing
        #      connections from each node.
        #   2. the entropy of average out-weights (careful here--the
        #      precise wording is needed), WIN_entropy, which
        #      represents the maximum possible information you could
        #      get about causation given this network structure.

        Wout_average = np.sum(Wout) / Nout
        Win_entropy = entropy(Win, base=2)

        # EI = WIN_entropy - WOUT_average
        return Win_entropy - Wout_average

    # if there are no nodes with outgoing edges, the EI = 0.0
    else:
        return 0.0


def effect_information_i(G, node_i=[], intervention_distribution='Hmax'):
    """
    Calculates the effect information (EI) of a node_i in a network,
    $G$, according to an intervention distribution.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question
    node_i (list or int): if node_i = [], this function returns a dictionary of
                    {node_1: EI_1, node_2: EI_2...} but if node_i is specified,
                    it returns the effect information $EI_i$ of node_i.
    intervention_distribution (np.ndarray or str): if 'Hmax', this represents a
            uniform intervention into a system's states. Otherwise, it's a
            heterogeneous intervention, often used in causal emergence (because
            a coarse-graining can be interpreted as changing the kinds of
            interventions that are informative about a given system).

    Returns
    -------
    EI_i (float or dict): the effect information of a given node_i or a
                          dictionary with each node and its effect information
                          contribution to the network's effective information.

    """

    if type(node_i) != list:
        node_i = [node_i]

    if len(node_i) == 0:
        node_i = list(G.nodes())

    node_name_mapping = {i: idx for idx, i in enumerate(list(G.nodes()))}
    node_name_mapping_i = {i: idx for idx, i in node_name_mapping.items()
                           if idx in node_i}

    EI_i = {i: 0 for i in node_i}

    Wout = W_out(G)
    Win = W_in(G)
    nonzeros_in = list(np.nonzero(Win)[0])

    Wout_i = np.atleast_2d(Wout[list(node_name_mapping_i.keys())])

    for idx, i in enumerate(node_i):
        nonzeros_out = list(np.nonzero(Wout_i[idx])[0])
        if len(nonzeros_out) > 0:
            nonzeros = list(np.unique(nonzeros_in + nonzeros_out))
            ei_i = entropy(Wout_i[idx][nonzeros], Win[nonzeros], base=2)
        else:
            ei_i = 0

        EI_i[i] = ei_i

    if len(node_i) == 1:
        return ei_i

    else:
        return EI_i


def determinism(G, intervention_distribution='Hmax'):
    """
    The determinism is the uncertainty in outcomes following an intervention
    into its states. Under a uniform intervention into a system's states,
    this equation becomes:

    $ det = log2(N) - frac{1}{Nout} sum_i^Nout H[W_i^{out}] $

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    intervention_distribution (np.ndarray or str): if 'Hmax', this represents a
            uniform intervention into a system's states. Otherwise, it's a
            heterogeneous intervention, often used in causal emergence (because
            a coarse-graining can be interpreted as changing the kinds of
            interventions that are informative about a given system).


    Returns
    -------
    det (float): the determinism of the network.

    """

    Wout = W_out(G)

    # Wout_noise is only calculated in nodes that have outputs
    eligible_nodes = np.nonzero(Wout.sum(axis=1))[0]

    Nout = len(eligible_nodes)

    if str(intervention_distribution) == 'Hmax':
        IntD = np.ones(Wout.shape[0])/Wout.shape[0]
    else:
        if sum(intervention_distribution) >= 0.0:
            IntD = intervention_distribution / sum(intervention_distribution)
        else:
            return np.zeros(Wout.shape[0])

    if Nout > 0 and sum(sum(Wout)) > 0:
        det = sum([entropy(Wout_i, IntD, base=2)
                   for Wout_i in Wout[eligible_nodes]])

        return det / Nout

    else:
        return 0.0


def degeneracy(G, intervention_distribution='Hmax'):
    """
    The degeneracy is the heterogeneity of the effect distribution following an
    intervention into a system's states. Under a uniform intervention into a
    system's states, this becomes:

    $ deg = log2(N) - H[W_i^{out}] $

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    intervention_distribution (np.ndarray or str): if 'Hmax', this represents
            a uniform intervention into a system's states. Otherwise, it's a
            heterogeneous intervention, often used in causal emergence (because
            a coarse-graining can be interpreted as changing the kinds of
            interventions that are informative about a given system).

    Returns
    -------
    deg (float): the degeneracy of the network.

    """

    Wout = W_out(G)

    if str(intervention_distribution) == 'Hmax':
        IntD = np.ones(Wout.shape[0])/Wout.shape[0]
    else:
        if sum(intervention_distribution) >= 0.0:
            IntD = intervention_distribution / sum(intervention_distribution)
        else:
            return np.zeros(Wout.shape[0])

    Win = W_in(G, IntD)
    nodes_with_input = np.nonzero(Win)[0]

    if len(nodes_with_input) > 0:
        intervened = np.nonzero(IntD)[0]
        deg = entropy(Win[intervened], IntD[intervened], base=2)
        return deg
    else:
        return 0.0


def effective_information_detdeg(G, intervention_distribution='Hmax'):
    """
    Calculates the effective information (EI) of a network, $G$, based on the
    determinism and the degeneracy of a given network following an intervention
    into a system's states.

    $EI = determinism - degeneracy $

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    intervention_distribution (np.ndarray or str): if 'Hmax', this represents a
            uniform intervention into a system's states. Otherwise, it's a
            heterogeneous intervention, often used in causal emergence (because
            a coarse-graining can be interpreted as changing the kinds of
            interventions that are informative about a given system).

    Returns
    -------
    EI (float): the effective information of a given network.

    """
    IntD = intervention_distribution
    return determinism(G, IntD) - degeneracy(G, IntD)


def stationary_distribution(G, smallest=1e-10):
    """
    Return a stationary probability vector of a given network

    x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    smallest (float): magnitude of probability that should be set to zero.

    Returns
    -------
    P (np.ndarray): vector of stationary probabilities of random walkers.

    """

    A = W_out(G)
    N = A.shape[0]
    a = np.eye(N) - A
    a = np.vstack((a.T, np.ones(N)))
    b = np.matrix([0] * N + [1]).T

    P = np.linalg.lstsq(a, b, smallest)[0]
    P[P < smallest] = 0

    if sum(P) != 1.0 and sum(P) != 0:
        P = P / sum(P)

    return np.array(P).reshape(N)


def random_walker_distribution_t(G, t=1, smallest=1e-10):
    """
    Return a probability vector of a given network after
    $t$ steps of a random walker.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    smallest (float): magnitude of probability that should be set to zero.

    Returns
    -------
    P (np.ndarray): vector of stationary probabilities of random walkers.

    """
    G = check_network(G)
    W = W_out(G)
    N = W.shape[0]
    P = W.copy()

    if t == 0:
        return W_in(W)

    for _ in range(t):
        P = P.dot(W)

    ps = W_in(P)

    if sum(ps) != 1.0 and sum(ps) != 0:
        ps = ps / sum(ps)

    ps = np.array(ps).reshape(N)

    return ps
