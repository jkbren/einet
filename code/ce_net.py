"""
ce_net.py
--------------------
Causal emergence code related to the paper:

Klein, B. & Hoel, E. (2019)
Uncertainty and causal emergence in complex networks.

author: Brennan Klein
email: brennanjamesklein at gmail dot com

With development contributions from
author: Ross Griebenow
email: rossgriebenow at gmail dot com
"""

import numpy as np
from ei_net import check_network
from ei_net import stationary_distribution
from ei_net import effective_information
from ei_net import W_out
import scipy as sp
import warnings
from sklearn.cluster import OPTICS, cluster_optics_dbscan


def create_macro(G, macro_mapping, macro_types={}):
    r"""
    Coarse-grains a network according to the specified macro_mapping and
    the types of macros that each macro is associated with.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    macro_mapping (dict): a dictionary where the keys are the microscale nodes
             and the values represent the macronode that they are assigned to.
    macro_types (dict): the corresponding macro_types dictionar associated
                    with the type of macronode in the "winning" G_macro.

    Returns
    -------
    M (np.ndarray): coarse-grained network according to the mapping of micro
                    nodes onto macro nodes, given by macro_mapping.

    """

    G_micro = check_network(G)
    Wout_micro = W_out(G_micro)
    # if macro_mapping == {}:
    #     return Wout_micro

    if macro_types == {}:
        macro_types = {j:'spatem1' for i, j in macro_mapping.items() if i != j}

    # the size of the whole microscale network
    micro_stationary = stationary_distribution(G_micro)

    # list of nodes that are in the macroscale network
    nodes_in_macro_network = list(np.unique(list(macro_mapping.values())))

    # this is the max index of the non-spatem2 macros, every index above this
    # then will be considered as a spatem2.
    non_spatem2_max_index = max(nodes_in_macro_network) + 1
    n_macros_spatem2 = len([i for i, j in macro_types.items()
                            if j == 'spatem2'])
    new_max_index_tmp = non_spatem2_max_index+n_macros_spatem2
    macro_id_spatem2 = list(range(non_spatem2_max_index, new_max_index_tmp))

    # now: these are the indices of the nodes in the TOO_BIG_MACRO
    nodes_in_macro_network = nodes_in_macro_network + macro_id_spatem2
    # the size of the TOO_BIG_MACRO that we will be filling in
    n_TOO_BIG_MACRO = max(nodes_in_macro_network) + 1

    # dictionary where the keys are the indices of the TOO_BIG_MACRO
    # and the values are the indices of where they will be in the final
    # matrix returned at the end, M
    nodes_in_macro_network_mapping = {j: i for i, j in
                                      enumerate(nodes_in_macro_network)}

    # and make the big empty matrix
    TOO_BIG_MACRO = np.zeros((n_TOO_BIG_MACRO, n_TOO_BIG_MACRO))

    all_final_node_types = {i: 'micro' for i in nodes_in_macro_network if
                            i < G_micro.number_of_nodes()}

    for macro_i in nodes_in_macro_network_mapping.keys():
        if macro_i not in all_final_node_types.keys():
            all_final_node_types[macro_i] = 'spatem1'

    macro_mumu_pairings = {}
    spt2_ind_tmp = 0

    for k, v in macro_types.items():
        all_final_node_types[k] = v
        if v == 'spatem2':
            mu_mu = macro_id_spatem2[spt2_ind_tmp]
            all_final_node_types[mu_mu] = 'mu_mu'
            macro_mumu_pairings[k] = mu_mu
            spt2_ind_tmp += 1

    # now the goal is to fill in the TOO_BIG_MACRO with the correct values
    # and, importantly, at the correct indices. do this with a for loop

    for final_node_i in nodes_in_macro_network:
        # this will be the vector that will be added to the TOO_BIG_MACRO
        W_i_out_final = np.zeros(n_TOO_BIG_MACRO)

        # we want to check every node for its type:
        final_node_i_type = all_final_node_types[final_node_i]

        ##########
        # MICROS #
        ##########
        if final_node_i_type == 'micro':
            # get the indices of the MICRO nodes that this particular micro
            # node leads to, and the weights associated with those links
            out_indices = np.nonzero(Wout_micro[final_node_i])[0]
            out_weights = Wout_micro[final_node_i][out_indices]

            # have a way to convert between the micro nodes it leads to and
            # the macro index that they will eventually be associated with
            new_indices = [macro_mapping[i] for i in out_indices]

            for wij_ind, wij in enumerate(out_weights):
                W_i_out_final[new_indices[wij_ind]] += wij

            TOO_BIG_MACRO[final_node_i, :] = W_i_out_final

        ###########
        # SPATIAL #
        ###########
        if final_node_i_type == 'spatial':
            micros_in_macro_i = [i for i, m in macro_mapping.items()
                                 if m == final_node_i]
            macro_row_sum = np.zeros(Wout_micro.shape[0])

            # get the rows of the Wout_micro associated with the micros
            # in the macro node that we are currently on
            Wout_macro_subgraph = Wout_micro[micros_in_macro_i, :]

            # indices of the FINAL macro network that are not in the current
            # macro. note that "nodes_in_macro_network" is the length of the
            # size_of_final_macro, and its elements correspond to the *values*
            # in macro_mapping
            nodes_outside_macro_i = [i for i in nodes_in_macro_network if
                                     i not in micros_in_macro_i and
                                     i != final_node_i]

            # same list as above, but this one uses microscale indices as its
            # elements so that you can get the right exit rates from Wout_micro
            nodes_outside_macro_mic_index = [i for i, j in
                                             macro_mapping.items() if j in
                                             nodes_outside_macro_i]

            input_probs_to_macro = Wout_micro.T[micros_in_macro_i].T

            # get the input to the nodes in the macro
            for i in range(len(macro_row_sum)):
                macro_row_sum += Wout_macro_subgraph.T.dot(
                                 input_probs_to_macro[i].T)

            if sum(macro_row_sum) == 0:
                # then there are no inputs from outside
                W_i_out_final[final_node_i] = 1
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final

            # chances are, it's not some huge isolated clique, so we need to
            # calculate the out-weights of the nodes that make up the macro

            else:
                # find the indices of micro nodes with input from macro scale
                out_indices = np.nonzero(
                                         sum(Wout_micro[micros_in_macro_i])
                                         [nodes_outside_macro_mic_index])[0]

                # and how do those indices map onto the new indices?
                new_indices = [nodes_outside_macro_mic_index[i] for
                               i in out_indices]

                # now, we can fill the row of TOO_BIG_MACRO (corresponding
                # to the current macro node) with correct edge weight values.
                for i, wij_ind in enumerate(new_indices):
                    # old_i   = out_indices[i]
                    wij_out = macro_mapping[wij_ind]
                    W_i_out_final[wij_out] += macro_row_sum[wij_ind]

                # and lastly, make a self-loop if there is any.
                selfloop = sum(macro_row_sum[micros_in_macro_i])
                if selfloop < 0:
                    selfloop = 0

                W_i_out_final[final_node_i] = selfloop
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final /\
                    sum(W_i_out_final)

        ###########
        # SPATEM1 #
        ###########
        if final_node_i_type == 'spatem1':
            # these are the indices of the micros inside this spatem1 macro
            micros_in_macro_i = [i for i, m in macro_mapping.items()
                                 if m == final_node_i]

            # get the rows of the Wout_micro associated with those indices
            Wout_macro_subgraph = Wout_micro[micros_in_macro_i, :]

            # get the stationary dist values of the micro nodes in this macro
            macro_i_stationary = micro_stationary[micros_in_macro_i]

            # get the total stationary probability inside this macro
            macro_i_stationary_sum = sum(macro_i_stationary)

            # weight the Wout_macro_subgraph by the stationary distribution
            Wout_macro_subgraph_weighted = Wout_macro_subgraph.copy()
            for j, W_j_out in enumerate(Wout_macro_subgraph):
                Wout_macro_subgraph_weighted[j] = W_j_out*macro_i_stationary[j]

            # get the (FINAL macro net.) indices of nodes not inside this macro
            nodes_outside_macro_i = [i for i in nodes_in_macro_network if
                                     i not in micros_in_macro_i and
                                     i != final_node_i]

            # same list as above but this one uses microscale indices as its
            # elements so that you can get the right exit rates from Wout_micro
            nodes_outside_macro_mic_index = [i for i, j in
                                             macro_mapping.items()
                                             if j in nodes_outside_macro_i]

            # a len(nodes_outside_macro_mic_index)x1 vector of exitrates
            Wout_macro_i_exitrates = sum(Wout_macro_subgraph_weighted[:,
                                         nodes_outside_macro_mic_index])

            # total exit rate probability from the macro in question
            Wout_macro_i_exitrates_sum = sum(Wout_macro_i_exitrates)

            # if there are no edges w/ weights outside the macro, add selfloop
            if Wout_macro_i_exitrates_sum == 0:
                W_i_out_final[final_node_i] = 1
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final

            # chances are, it's not some huge isolated clique, so we
            # calculate the out-weights of the nodes that make up the macro

            else:
                # normalize Wout_macro_i_exitrates given macro_i_stationary_sum
                Wout_macro_i_exitrates_norm = Wout_macro_i_exitrates.copy()
                for i, wij in enumerate(Wout_macro_i_exitrates):
                    Wout_macro_i_exitrates_norm[i] = wij/macro_i_stationary_sum

                # micro indices of nodes with inputs from nodes in this macro
                out_indices = np.nonzero(
                                sum(Wout_micro[micros_in_macro_i])
                                [nodes_outside_macro_mic_index])[0]

                # and how do those indices map onto the new indices?
                new_indices = [nodes_outside_macro_mic_index[i] for
                               i in out_indices]

                # now, we can fill the row of TOO_BIG_MACRO (corresponding
                # to the current macro node) with correct edge weight values.
                for i, wij_ind in enumerate(new_indices):
                    old_i = out_indices[i]
                    wij_ind = macro_mapping[wij_ind]
                    W_i_out_final[wij_ind] += \
                        Wout_macro_i_exitrates_norm[old_i]

                # and lastly, make a self-loop if there is any.
                selfloop = 1 - sum(W_i_out_final)
                if selfloop < 0:
                    selfloop = 0

                W_i_out_final[final_node_i] = selfloop
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final

        ###########
        # SPATEM2 #
        ###########
        if final_node_i_type == 'spatem2':
            # for the mu_mu node, we need to add another row. This extra
            # W_i_out_final node will just have a single 1.0 edge weight
            # to the mu_mu macro node. This bit is primarily for finding
            # the edge weights from mu_mu.
            mu_mu_index = macro_mumu_pairings[final_node_i]
            W_mu_out_final = np.zeros(n_TOO_BIG_MACRO)

            # but first we need to find out which nodes are in the macro
            micros_in_macro_i = [i for i, m in macro_mapping.items()
                                 if m == final_node_i]

            # get the rows of the Wout_micro associated with the micros
            # in the macro node that we are currently on
            Wout_macro_subgraph = Wout_micro[micros_in_macro_i, :]

            # get the stationary dist values of the micro nodes in this macro
            macro_i_stationary = micro_stationary[micros_in_macro_i]

            # get the total stationary probability inside this macro
            macro_i_stationary_sum = sum(macro_i_stationary)

            # weight the Wout_macro_subgraph by the stationary
            Wout_macro_subgraph_weighted = Wout_macro_subgraph.copy()
            for j, W_j_out in enumerate(Wout_macro_subgraph):
                Wout_macro_subgraph_weighted[j] = W_j_out*macro_i_stationary[j]

            # indices of the FINAL macro network that do not include
            nodes_outside_macro_i = [i for i in nodes_in_macro_network if
                                     i not in micros_in_macro_i and
                                     i != final_node_i]

            # same list as above but this uses the microscale indices as its
            # elements so that you can get the right exit rates from Wout_micro
            nodes_outside_macro_mic_index = [i for i, j in
                                             macro_mapping.items()
                                             if j in nodes_outside_macro_i]

            # a len(nodes_outside_macro_mic_index)x1 vector of exitrates
            Wout_macro_i_exitrates = sum(
                Wout_macro_subgraph_weighted[:, nodes_outside_macro_mic_index])

            # total exit rate probability from the macro in question
            Wout_macro_i_exitrates_sum = sum(Wout_macro_i_exitrates)

            # if there are no edges with weights outside the macro,
            # then give it a self-loop
            if Wout_macro_i_exitrates_sum == 0:
                W_i_out_final[final_node_i] = 1
                W_mu_out_final[mu_mu_index] = 1
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final
                TOO_BIG_MACRO[mu_mu_index, :] = W_mu_out_final

            else:
                # normalize Wout_macro_i_exitrates given macro_i_stationary_sum
                Wout_macro_i_exitrates_norm = Wout_macro_i_exitrates.copy()
                for i, wij in enumerate(Wout_macro_i_exitrates):
                    ##########################
                    # DIFFERENT THAN SPATEM1 #
                    denom = macro_i_stationary_sum - Wout_macro_i_exitrates_sum
                    Wout_macro_i_exitrates_norm[i] = wij / denom
                    ##########################

                # micro indices of nodes with inputs from nodes in this macro
                out_indices = np.nonzero(
                                    sum(Wout_micro[micros_in_macro_i])
                                    [nodes_outside_macro_mic_index])[0]

                # and how do those indices map onto the new indices?
                new_indices = [nodes_outside_macro_mic_index[i] for
                               i in out_indices]

                # now, we can fill the row of TOO_BIG_MACRO (corresponding
                # to the current macro node) with correct edge weight values.
                for i, wij_ind in enumerate(new_indices):
                    old_i = out_indices[i]
                    wij_ind = macro_mapping[wij_ind]
                    W_mu_out_final[wij_ind] += \
                        Wout_macro_i_exitrates_norm[old_i]

                # and lastly, make a self-loop if there is any.
                W_i_out_final[mu_mu_index] = 1

                # do the same check as the spatem1 nodes above
                mu_selfloop = 1 - sum(W_mu_out_final)
                if mu_selfloop < 0:
                    mu_selfloop = 0
                W_mu_out_final[mu_mu_index] = mu_selfloop

                # add the two vectors to TOO_BIG_MACRO
                TOO_BIG_MACRO[final_node_i, :] = W_i_out_final
                TOO_BIG_MACRO[mu_mu_index, :] = W_mu_out_final

        if final_node_i_type == 'mu_mu':
            # if it's a mu_mu node, it should have already been accounted for,
            # so skip it during this section of the code, and simply continue.
            continue

    M = TOO_BIG_MACRO[nodes_in_macro_network, :][:, nodes_in_macro_network]

    return M


def select_macro(G_micro, node_i_macro, possible_mapping, macro_types, F=True):
    r"""
    Given a current macro_mapping of a micro scale network, and given a new
    node that is being considered for a macro node, this function selects the
    type of macro node that the new nide should be assigned to, such that the
    resulting macro scale network has maximal accuracy.

    Note: this version of select_macro hastily decides which type of macro to
          assign to the candidate node_i_macro, as opposed to the exhaustive
          method in previous versions of this function.

    Parameters
    ----------
    G_micro (nx.Graph or np.ndarray): the micro network in question.
    node_i_macro (int): the index of the new potential macro node.
    possible_mapping (dict): a dictionary where the keys are the microscale
                nodes and the values represent the *potential* macronode
                that they are assigned to. This is used to create three
                possible new mappings, corresponding to the three macro types.
    macro_types (dict): the current dictionary with information about the
                current macronodes in the network, along with their macro type.
                This is used, again, to create three new macro_types_X dicts
                that will be used to evaluate the proposed new G_macro.
    F (bool): F stands for "fast"--currently this parameter is not in use, but
              when the final version of this codebase is released, if F=False,
              the function will proceed with an exhaustive selection mechanism

    Returns
    -------
    G_macro (nx.Graph): the macro scale network with the least inaccuracy. It
                        corresponds to the "winning" type of macro node to be
                        added to the network.
    macro_types (dict): the corresponding macro_types dictionar associated
                        with the type of macronode in the "winning" G_macro.

    """

    G_micro = check_network(G_micro)
    # Wout_micro = W_out(G_micro)
    # micro_stationary = stationary_distribution(G_micro)

    # micronode ids for nodes within the macro
    nodes_in_new_macro = [k for k, v in possible_mapping.items()
                          if v == node_i_macro]
    # micronode ids for nodes NOT within the macro
    rest_of_nodes = [k for k, v in possible_mapping.items()
                     if v != node_i_macro]

    # two dictionaries where the keys are micro nodes in macro and the
    # values are either the nodes that THEY LEAD TO...
    edges_from_micro_in_macro = {i: list(zip(*list(G_micro.out_edges(i))))[1]
                                 for i in nodes_in_new_macro}
    # ...or the nodes that LEAD TO THEM...
    edges_to_micro_in_macro = {i: list(zip(*list(G_micro.in_edges(i))))[0]
                               for i in nodes_in_new_macro}

    # and the nodes inside the macro that are involved with the above dicts...
    nodes_in_macro_with_outside_output = [
                i for i, j in edges_from_micro_in_macro.items() if
                len(set(rest_of_nodes).intersection(set(list(j)))) > 0]

    nodes_in_macro_with_outside_input = [
                i for i, j in edges_to_micro_in_macro.items() if
                len(set(rest_of_nodes).intersection(set(list(j)))) > 0]

    macro_types_spatem1 = macro_types.copy()
    macro_types_spatem1[node_i_macro] = "spatem1"
    macro_types_spatem2 = macro_types.copy()
    macro_types_spatem2[node_i_macro] = "spatem2"

    #######################
    # For spatiotemporal2 #
    #######################
    cond = set(nodes_in_macro_with_outside_input).intersection(
           set(nodes_in_macro_with_outside_output))
    # if there is no intersection between the two sets above, make spatem2
    if len(cond) == 0:
        G_macro = create_macro(G_micro, possible_mapping, macro_types_spatem2)
        return G_macro, macro_types_spatem2

    #######################
    # For spatiotemporal1 #
    #######################
    else:
        G_macro = create_macro(G_micro, possible_mapping, macro_types_spatem1)

        return G_macro, macro_types_spatem1

    return G_micro, macro_types


def causal_emergence(G, span=-1, thresh=1e-4, t=500,
                     types=False, check_inacc=False, printt=True):
    r"""
    Given a microscale network, $G$, this function iteratively checks different
    coarse-grainings to see if it finds one with higher effective information.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    span (int): defaults at -1, which means that the entire network will be
                searched. Positive integers means only a fraction of the
                possible coarse grains will be searched.
    thresh (float): if the difference between the micro and macro EI values is
                    greater than this threshold, we will admit the macro node
                    into the coarse-grained network.
    t (int): default to 10, this the number of timesteps over which inaccuracy
             is evaluated.
    types (bool): if True, this function will store the kinds of macro_types
                  that are selected for each run of causal emergence. Otherwise
                  this function creates macronodes based on the stationary
                  distribution of the underlying micronodes.
    check_inacc (bool): will check the inaccuracy following the addition of
                        each newly-added macro-node, to ensure that only
                        accurate macros are added.
    printt (bool): if True, this will print out progress of the algorithm

    Returns
    -------
    CE (dict): a dictionary with the following information
      - G_macro (nx.Graph): a coarse-grained description of the micro network.
      - G_micro (nx.Graph): the original microscale network.
      - mapping (dict): the mapping that most-successfully increased the
                        effective information of the network.
      - macro_types (dict): the dictionary associated with each type of
                            macronode in the "winning" G_macro. If types==False
                            then this just is a dictionary with every value
                            set to 'spatem1'.
      - inaccuracy (np.ndarray): a sequence of inaccuracy values associated
                                 with the successful macro_mapping.
      - EI_micro (float): the effective information of the micro scale network
      - EI_macro (float): the effective information of the macro scale network

    """

    G_micro = check_network(G)
    Wout = W_out(G_micro)
    MB = markov_blanket(G_micro)

    # will search these nodes. if span is > 1, we will search a sample of the
    # network for good coarse grains, but if it's default (-1), search the
    # entire network for causal emergence
    micro_nodes_left = list(MB.keys())
    macro_mapping = dict(zip(micro_nodes_left, micro_nodes_left))

    # begin by initializing a "macro_types" dictionary. This will store the
    # various types of macros associated with a given mapping.
    macro_types = {}

    np.random.shuffle(micro_nodes_left)
    if span > 1:
        micro_nodes_left = micro_nodes_left[:span]

    # original microscale EI, with uniform intervention
    EI_micro = effective_information(G_micro)
    EI_current = EI_micro

    # initialize the mapping as a 1-to-1 mapping (i.e. all nodes are micro)
    micros_already_macroed = []
    # inaccurate_macro_pairs = 0
    # accurate_macro_pairs   = 0

    if printt:
        print("Starting with this TPM:\n", np.round(Wout, 4))
        print("\nSearch started ... EI_micro = %.4f" % EI_micro)
        print()
        curr_count = 0
        out_of = len(micro_nodes_left)

    # if you want to have a List_of_Mappings
    # List_of_Mappings = []
    for node_i in micro_nodes_left:

        if printt:
            print("Checking node %05i (%.1f%% done)..." %
                  (node_i, 100*(curr_count/out_of)),
                  "coarse-grained network size = %05i" %
                  len(np.unique(list(macro_mapping.values()))))
            curr_count += 1

        macros_to_check = update_markov_blanket(MB, micros_already_macroed)
        macros_to_check = macros_to_check[node_i]

        if len(macros_to_check) < 1:
            continue

        # make a queue of nodes that need to be checked
        queue = macros_to_check.copy()

        # node_i is currently assigned to this macro
        node_i_macro = macro_mapping[node_i]

        # if not yet assigned to a macro, set to next highest macro index
        if node_i_macro == node_i:
            node_i_macro = max(list(macro_mapping.values())) + 1

        # now start a loop of EI comparisons
        while len(queue) > 0:
            np.random.shuffle(queue)

            # here's a possible micro_node to attempt to group
            # with node_i in order to make a new macro node
            possible_macro = queue.pop()

            # this is the hypothetical mapping that we'll compare to
            possible_mapping = macro_mapping.copy()
            possible_mapping[node_i] = node_i_macro
            possible_mapping[possible_macro] = node_i_macro

            # We want to create a variable, G_macro, a candidate macro network
            if types:
                G_macro, macro_types_tmp = select_macro(
                    G_micro, node_i_macro, possible_mapping, macro_types)
            else:
                macro_types_tmp = macro_types.copy()
                macro_types_tmp[node_i_macro] = "spatem1"
                G_macro = create_macro(G_micro, possible_mapping,
                                       macro_types_tmp)

            G_macro = check_network(G_macro)
            EI_macro = effective_information(G_macro)
            if np.isinf(EI_macro):
                return G_macro

            inacc = np.zeros(t)
            if check_inacc:
                inacc = macro_inaccuracy(G_micro, G_macro, possible_mapping,
                                         macro_types_tmp, t)['inaccuracies']

            if EI_macro - EI_current > thresh and sum(inacc[-4:]) < 1e-3:

                # accurate_macro_pairs  += 1
                # keep adding nodes in the queue to the current macro
                # grouping once you get anything with a little extra EI
                EI_current = EI_macro
                macro_mapping = possible_mapping
                macro_types = macro_types_tmp.copy()

                if printt:
                    print("\tJust found a successful macro grouping ...",
                          "the EI_current = %.4f" % EI_current)

                # avoid inefficient redundant searches
                micros_already_macroed.append(node_i)
                micros_already_macroed.append(possible_macro)
                micros_already_macroed = list(set(micros_already_macroed))

                nodes_in_macro_i = [k for k, v in macro_mapping.items()
                                    if v == node_i_macro]

                # plus we have to bring in any nodes that node_j might have
                # that would be relevant
                for new_micro_i in nodes_in_macro_i:
                    children_i_M = list(G_micro.successors(new_micro_i))
                    parents_i_M = list(G_micro.predecessors(new_micro_i))
                    neighbors_i_M = set(children_i_M).union(set(parents_i_M))

                    for node_j_M in neighbors_i_M:
                        if node_j_M not in queue and node_j_M != node_i:
                            queue.append(node_j_M)

    CE = {}
    G_macro = create_macro(G_micro, macro_mapping, macro_types)
    G_macro = check_network(G_macro)
    EI_macro = effective_information(G_macro)
    if macro_types == {}:
        EI_macro = EI_micro
        G_macro = G_micro.copy()

    CE['G_macro'] = G_macro
    CE['G_micro'] = G_micro
    CE['mapping'] = macro_mapping
    CE['macro_types'] = macro_types
    CE['EI_micro'] = EI_micro
    CE['EI_macro'] = EI_macro

    if check_inacc:
        inaccuracies = macro_inaccuracy(G_micro, G_macro, macro_mapping,
                                        macro_types, t)
        CE['inaccuracy'] = inaccuracies['inaccuracies']

    return CE


def macro_inaccuracy(G_micro, G_macro, macro_mapping, macro_types, t=500):
    r"""
    Here, we consider only the inaccuracy associated with a macro scale mapping
    through the introduction of random walkers on micronodes that have not been
    grouped into macro nodes. From this, the inaccuracy associated with a
    macro mapping is the KL Divergence of the expected location w/ distribution
    of random walkers following an intervention distribution on the micro and
    macro representation of the network.

    Parameters
    ----------
    G_micro (nx.Graph or np.ndarray): the micro network in question.
    G_macro (nx.Graph or np.ndarray): the macro network that you want to
                                      calculate the inaccuracy of.
    macro_mapping (dict): a dictionary where the keys are the microscale nodes
             and the values represent the macronode that they are assigned to.
    macro_types (dict): the corresponding macro_types dictionar associated
                    with the type of macronode in the "winning" G_macro.
    t (int): timesteps in the future (default is t+1, but if more Win indicates
             likely positions of random walkers at t = t+x steps in the future)
             (must be between 1 and T)

    Returns
    -------
    inaccuracy_dict (dict): dictionary with tensors of TPMs for micro_out,
                    macro_out, micro_in, micro_out, and a list of inaccuracies.
    """

    inaccuracy_dict = {}
    Wout_micro = W_out(G_micro)
    Wout_macro = W_out(G_macro)

    # get a list of the macro nodes
    nodes_in_micro_network = np.unique(list(macro_mapping.keys()))
    nodes_in_macro_network = np.unique(list(macro_mapping.values()))

    # the size of the whole microscale network
    N_micro = len(nodes_in_micro_network)

    # identify the macro nodes
    macro_nodes = nodes_in_macro_network[nodes_in_macro_network > N_micro-1]

    # get the total amount of macro nodes once HOMs are added in
    # (assuming there are in fact higher-order macros)
    # N_macro = len(macro_nodes)
    spatem2_count = 0
    for x in range(len(macro_nodes)):
        macro_type = macro_types[macro_nodes[x]]
        if macro_type == "spatem2":
            spatem2_count = spatem2_count + 1

    # the remaining micro nodes in the macroscale network, after coarsening
    remaining_micro_nodes = nodes_in_macro_network[
                            nodes_in_macro_network < N_micro]

    # distribution over micro nodes at microscale
    distribution_over_micro = np.zeros(N_micro)
    for x in range(N_micro):
        for y in range(len(remaining_micro_nodes)):
            if nodes_in_micro_network[x] == remaining_micro_nodes[y]:
                distribution_over_micro[x] = 1

    distribution_over_macro = np.zeros(
                len(nodes_in_macro_network) + spatem2_count)

    for y in range(len(remaining_micro_nodes)):
        distribution_over_macro[y] = 1

    # pre-populate a list of transition probability matrices for the micro
    # at t = ti steps in the future.
    list_of_micros = [Wout_micro]
    Wout_micro_t = Wout_micro.copy()
    for ti in range(1, t):
        Wout_micro_t = Wout_micro_t.dot(Wout_micro)  # transitions over 1 step
        list_of_micros.append(Wout_micro_t)

    # pre-populate a list of transition probability matrices for the macro
    # at t = ti steps in the future.
    list_of_macros = [Wout_macro]
    Wout_macro_t = Wout_macro.copy()
    for ti in range(1, t):
        Wout_macro_t = Wout_macro_t.dot(Wout_macro)  # transitions over 1 step
        list_of_macros.append(Wout_macro_t)

    inaccuracy_dict['Wout_micro_list'] = list_of_micros
    inaccuracy_dict['Wout_macro_list'] = list_of_macros

    # multiply by the distribution_over_micro
    list_of_just_micros = []
    for micro in list_of_micros:
        Win_j = micro.T.dot(distribution_over_micro)
        ED_micro = Win_j / sum(Win_j)

        # get only the micro nodes that are not involved in the macro
        ED_micro_just_micro = []
        for x in range(len(ED_micro)):
            for y in remaining_micro_nodes:
                if x == y:
                    ED_micro_just_micro.append(ED_micro[x])

        # append to the end of ED_micro_just_micro the 1-sum
        ED_micro_just_micro.append(1-sum(ED_micro_just_micro))

        # add to list
        list_of_just_micros.append(ED_micro_just_micro)

    # multiply by the distribution_over_macro
    list_of_just_micros_macro = []
    for macro in list_of_macros:
        Win_j = macro.T.dot(distribution_over_macro)
        ED_macro = Win_j / sum(Win_j)
        # EffectDist_macro = ED_macro

        # get only the initial micro nodes (all in front)
        ED_micro_just_micro = []
        for y in range(len(remaining_micro_nodes)):
            ED_micro_just_micro.append(ED_macro[y])

        # append to the end of ED_micro_just_micro the 1-sum
        ED_micro_just_micro.append(1-sum(ED_micro_just_micro))

        # add to list
        list_of_just_micros_macro.append(ED_micro_just_micro)

    inaccuracy_dict['EffectDist_micro_just_micro'] = list_of_just_micros
    inaccuracy_dict['EffectDist_macro_just_micro'] = list_of_just_micros_macro

    def KLD_for_inaccuracy(distr1, distr2):
        # this can be negative
        # because the distributions aren't normalized
        # so normalize before sending them in

        import math
        total_difference = 0

        for x in range(len(distr1)):
            difference = 0
            p = distr1[x]
            q = distr2[x]

            if p > 0 and q > 0:  # if p != 0 and q !=0:
                difference = p * math.log((p/q), 2)
            if p != 0 and q == 0:
                difference = math.inf

            total_difference = total_difference + difference
        return total_difference

    # now calculate the final inaccuracies
    inaccuracies = []
    for i in range(len(list_of_just_micros_macro)):
        distr1 = list_of_just_micros_macro[i]
        distr2 = list_of_just_micros[i]
        inaccuracy = KLD_for_inaccuracy(distr1, distr2)
        inaccuracies.append(inaccuracy)

    inaccuracies[inaccuracies < 1e-8] = 0
    inaccuracy_dict['inaccuracies'] = np.array(inaccuracies)

    return inaccuracy_dict


def markov_blanket(G, internal_nodes=[]):
    r"""
    Given a graph and a specified (list of) internal node(s), return
    the parents, the children, and the parents of the children of the
    internal node(s).

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    internal_nodes (int or list): the nodes around which to build a
                Markov blanket. If this value is an empty list [],
                return a dictionary where the keys are the nodes in
                the original graph, and the values are the nodes
                that constitute their Markov Blanket.

    Returns
    -------
    MB (list or dict): if internal_node(s) were specified, this function
                       returns a list. Otherwise, it returns a dictionary
                       where the keys are the nodes in the original graph,
                       and the values are the nodes that constitute their
                       Markov Blanket

    """

    G = check_network(G)
    if type(internal_nodes) == int:
        internal_nodes = [internal_nodes]

    if internal_nodes == []:
        internal_nodes = list(G.nodes())

    MB = {}
    for node_i in internal_nodes:
        MB[node_i] = []

        # get the parents and the children
        parents_i = list(G.predecessors(node_i))
        children_i = list(G.successors(node_i))
        MB_i = set(children_i).union(set(parents_i))

        # ...and get their neighbors as well
        for node_j in children_i:
            parents_j = list(G.predecessors(node_j))
            MB_i = set(parents_j).union(MB_i)

        MB[node_i] = [i for i in list(MB_i) if i != node_i]

    if len(internal_nodes) == 1:
        return {internal_nodes[0]: MB[internal_nodes[0]]}  # fix this

    else:
        return MB


def update_markov_blanket(MB, remove_nodes=[]):
    r"""
    Given a Markov Blanket dict and a (list of) node(s) that need to be
    updated in the blanket (i.e. they were recruited into a macro-node
    and should therefore not be searched), return a new Markov Blanket
    that has taken out the remove_nodes in question

    Parameters
    ----------
    MB (dict): Dictionary where the keys are the nodes in a graph, G,
               and the values are the nodes that constitute each node's
               Markov Blanket.
    remove_nodes (int or list): the nodes that have been recruited to
                    join a macro_node and that thus should not be
                    searched when trying to find other nodes to turn
                    into macro_nodes.

    Returns
    -------
    MB_new (list or dict): if remove_nodes was specified, this function
                       returns an updated Markov Blanket that lacks the
                       remove_nodes in the values of all the nodes.
                       Otherwise, it returns the original Markov Blanket.

    """

    if remove_nodes == []:
        return MB
    if type(remove_nodes) == int:
        remove_nodes = [remove_nodes]

    MB_new = {}
    for node_i, blanket_j in MB.items():
        MB_new[node_i] = [b for b in blanket_j if b not in remove_nodes]

    return MB_new


def all_possible_mappings(G):
    r"""
    This function will return a list of dictionaries, each containing a
    macro_mapping of the network in question. Be careful with this function
    though, as the size of the network grows to beyond N = 10, there are too
    many possible mappings to handle.

    Uses a function for computing the many permutations "algorithm_u" which
    I found at this stackoverflow page:

    https://codereview.stackexchange.com/questions/1526/finding-
    all-k-subset-partitions

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.

    Returns
    -------
    mapping_list (list): list of dictionaries of macro_mapping objects

    """
    def algorithm_u(ns, m):
        """
        A very efficient algorithm (Algorithm U) is described by Knuth in the
        Art of Computer Programming, Volume 4, Fascicle 3B to find all set
        partitions with a given number of blocks. Since Knuth's algorithm isn't
        very concise, its implementation is lengthy as well. Note that the
        implementation below moves an item among the blocks one at a time and
        need not maintain an accumulator containing all partial results.
        For this reason, no copying is required.
        """

        def visit(n, a):
            ps = [[] for i in range(m)]
            for j in range(n):
                ps[a[j + 1]].append(ns[j])
            return ps

        def f(mu, nu, sigma, n, a):
            if mu == 2:
                yield visit(n, a)
            else:
                for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v
            if nu == mu + 1:
                a[mu] = mu - 1
                yield visit(n, a)
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    yield visit(n, a)
            elif nu > mu + 1:
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = mu - 1
                else:
                    a[mu] = mu - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] > 0:
                    a[nu] = a[nu] - 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v

        def b(mu, nu, sigma, n, a):
            if nu == mu + 1:
                while a[nu] < mu - 1:
                    yield visit(n, a)
                    a[nu] = a[nu] + 1
                yield visit(n, a)
                a[mu] = 0
            elif nu > mu + 1:
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                while a[nu] < mu - 1:
                    a[nu] = a[nu] + 1
                    if (a[nu] + sigma) % 2 == 1:
                        for v in f(mu, nu - 1, 0, n, a):
                            yield v
                    else:
                        for v in b(mu, nu - 1, 0, n, a):
                            yield v
                if (mu + sigma) % 2 == 1:
                    a[nu - 1] = 0
                else:
                    a[mu] = 0
            if mu == 2:
                yield visit(n, a)
            else:
                for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                    yield v

        n = len(ns)
        a = [0] * (n + 1)
        for j in range(1, m + 1):
            a[n - m + j] = j - 1
        return f(m, n, 0, n, a)

    G = check_network(G)
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    if len(nodes) > 10:
        print("Comment this out if you really want to run a network that big.")
        return []

    mapping_list = []
    list_of_lists_of_mappings = [[nodes]]

    for i in range(2, N+1):  # this must be from 2! if from 1, it breaks
        x = list(algorithm_u(nodes, i))  # key step

        for j in x:
            list_of_lists_of_mappings.append(j)

    for mapping in list_of_lists_of_mappings:
        out_dict = dict(zip(nodes, [-1]*len(nodes)))
        MacroID = max(nodes)  # or len(nodes)

        for mac in mapping:
            if len(mac) > 1:
                MacroID += 1
                for i in mac:
                    out_dict[i] = MacroID

            else:
                for i in mac:
                    out_dict[i] = i

        mapping_list.append(out_dict)

    return mapping_list


def intervention_distribution(G, macro_mapping,
                              scale='macro', conditional=True):
    r"""
    Given a network and a macro_mapping, this function returns the intervention
    distribution, depending on the scale of the network being input. Note: if
    scale='micro', this function outputs an intervention distribution where
    the indices are formatted as follows:
    [micro_0, micro_1,... micro_N, macro_0,...]

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question (either micro or macro)
    macro_mapping (dict): a dictionary where the keys are the microscale nodes
             and the values represent the macronode that they are assigned to.
    scale (str): either 'macro' or 'micro'. If scale='macro', return a maximum
                 entropy distribution of size N_macro.
    conditional (bool): is the coarse-graining happening in a naive way or does
                        it take into account conditional random walks? Note:
                        more often than not conditional coarse-grained networks
                        perform better with respect to the accuracy of random
                        walkers placed onto the network.

    Returns
    -------
    IntD (np.ndarray): the intervention distribution (Nx1), taking into account
                       the input weights to the nodes being coarse grained if
                       conditional=True. Otherwise the intervention is based on
                       uniformly distributed input weights.

    """

    G = check_network(G)
    if scale == 'macro':
        N_macro = G.number_of_nodes()
        return np.ones(N_macro) / N_macro

    # list of nodes that are in the microscale network
    nodes_in_micro_network = np.unique(list(macro_mapping.keys()))
    # list of nodes that are in the macroscale network
    nodes_in_macro_network = np.unique(list(macro_mapping.values()))

    # the size of the whole microscale network
    N_micro = len(nodes_in_micro_network)
    # the size of the whole macroscale network
    N_macro = len(nodes_in_macro_network)

    # the unique macro nodes in the macroscale network
    macro_nodes = nodes_in_macro_network[nodes_in_macro_network > N_micro-1]
    N_macronodes = len(macro_nodes)  # number of unique macro nodes

    # this will make a list of micro nodes that will not be present in the
    # macro scale network -- micro_to_macro_list is a list of lists.
    micro_to_macro_list = []
    for macronode_i in range(N_macronodes):
        micro_in_macro_i = [k for k, v in macro_mapping.items()
                            if v == macro_nodes[macronode_i]]
        micro_to_macro_list.append(micro_in_macro_i)

    IntD = np.ones(N_micro) / N_macro

    # input probs are 1/k for not conditional
    if len(micro_to_macro_list) > 0:
        if not conditional:
            for macro_i in micro_to_macro_list:
                input_probs = np.ones(len(macro_i)) / len(macro_i)  # uniform
                IntD[macro_i] = IntD[macro_i] * input_probs

            return IntD

        else:
            Wout_micro = W_out(G)

            for macro_i in micro_to_macro_list:
                denom = sum(Wout_micro.T[macro_i].sum(axis=1))
                input_probs = 0
                if denom > 0:
                    input_probs = Wout_micro.T[macro_i].sum(axis=1) / denom

                IntD[macro_i] = IntD[macro_i] * input_probs

            return IntD


def reorder_elements(Win_micro, macro_mapping):
    r"""
    The function for finding the intervention distribution returns
    a vector of length N_micro. This needs to be reshaped in the
    calculation of inaccuracy, meaning macro nodes should be in the
    final positions of the vector, preceded by micro nodes in order.

    Parameters
    ----------
    Win_micro (np.ndarray): the micro-scale effect distribution (although)
                            this really could be any vector that needs to be
                            re-ordered into a macro grouping.
    macro_mapping (dict): a dictionary where the keys are the microscale nodes
             and the values represent the macronode that they are assigned to.

    Returns
    -------
    Win_micro_given_macro (np.ndarray): micro effect distribution, reordered.

    """

    # list of nodes that are in the microscale network
    nodes_in_micro_network = np.unique(list(macro_mapping.keys()))
    # list of nodes that are in the macroscale network
    nodes_in_macro_network = np.unique(list(macro_mapping.values()))

    # the size of the whole microscale network
    N_micro = len(nodes_in_micro_network)
    # the size of the whole macroscale network
    N_macro = len(nodes_in_macro_network)

    if N_macro == 1:
        return np.array([1.])

    # the unique macro nodes in the macroscale network
    macro_nodes = nodes_in_macro_network[nodes_in_macro_network > N_micro-1]
    N_macronodes = len(macro_nodes)  # number of unique macro nodes

    # this will make a list of micro nodes that will not be present in the
    # macro scale network -- micro_to_macro_list is a list of lists.
    micro_to_macro_list = []
    for macronode_i in range(N_macronodes):
        micro_in_macro_i = [k for k, v in macro_mapping.items()
                            if v == macro_nodes[macronode_i]]
        micro_to_macro_list.append(micro_in_macro_i)

    Win_micro_given_macro = np.zeros(N_macro)

    overlap = list(set(
                nodes_in_micro_network).intersection(nodes_in_macro_network))

    # add micro node values to the output
    if len(overlap) != 0:
        for ind, node_i in enumerate(overlap):
            Win_micro_given_macro[ind] = Win_micro[node_i]

    else:
        ind = len(overlap) - 1

    ind += 1
    for macro_i in micro_to_macro_list:
        Win_micro_given_macro[ind] = sum(Win_micro[macro_i])
        ind += 1

    return Win_micro_given_macro


def construct_distance_matrix(G_micro, nonz=1e-3, dist_add=1e3):
    r"""
    Make distance matrix for OPTICS algorithm for spectral causal emergence.
    This is done through an eigendecomposition of the transition probability
    matrix (Wout) of G_micro, the original microscale network.

    The eigenvalues and eigenvectors of Wout is computed via

        $$ W_{out} = E \Lambda E^T $$

    where columns in $E$ corresponds to the eigenvectors of nodes in G_micro,
    weighted by the eigenvalue they are associated with.


        Development work contributed by Ross Griebenow.
            email: rossgriebenow at gmail dot com

    Parameters
    ----------
    G_micro (nx.Graph or np.ndarray): the microscale network in question.
    nonz (float): Simple parameter governing the minimum size that an
                  eigenvalue can be in order to be included in the distance
                  calculation
    dist_add (float): Should technically be infinity, but for practical
                      purposes, only ~1000 is needed in order to only measure
                      the distance between nodes within each nodes' Markov
                      blankets.

    Returns
    -------
    dist (np.ndarray): the distance matrix upon which the OPTICS algorithm
                       will perform spectral clustering with different distance
                       thresholds, $\epsilon$

    """

    Wout = W_out(G_micro)
    lam, eig = np.linalg.eig(Wout)
    span = np.nonzero(np.abs(np.real(lam)) > nonz)[0]

    # weight the eigenvectors by their corresponding eigenvalues
    M = np.real(eig)[:, span] * np.real(lam)[span]

    # create values for a distance matrix, which will become the output
    distance_vector = sp.spatial.distance.pdist(M, metric='cosine')

    dist = sp.spatial.distance.squareform(distance_vector)
    dist[np.isnan(dist)] = 0

    MB = markov_blanket(G_micro)
    for i in range(G_micro.number_of_nodes()):
        dist[:, i] += dist_add
        dist[MB[i], i] -= dist_add

    return dist


def find_epsilon_mapping(reach, core, order, G_micro, depth=4,
                         min_ep=1e-4, max_ep=9.99e-1, scale=1e-4):
    r"""
    Binary search down the tree of possible epsilon values for finding the one
    that returns a macroscale mapping that maximizes the effective information
    of the resulting macroscale network. The spectral algorithm OPTICS creates
    a mapping of of points to clusters based on a distance matrix, which itself
    was generated by eigendecomposing the transition probability matrix of
    G_micro, the original graph.

    Algorithm adapted from the paper:
        Mihael Ankerst. Markus M. Breunig, Hans-Peter Kriegel, & Jörg Sander
        “OPTICS: Ordering points to identify the clustering structure”.
        Proc. ACM SIGMOD’99 Int. Conf. on Management of Data. ACM Press, 1999

    Development work contributed by Ross Griebenow.
        email: rossgriebenow at gmail dot com

    Parameters
    ----------
    reach, core, order (optics parameters): outputs from the original run of
                        the OPTICS algorithm.
    G_micro (nx.Graph or np.ndarray): the microscale network in question.
    depth (int): How many iterations deep the algorithm should run.
    min_ep (float): the minimum value to check for grouping nodes into macros
    max_ep (float): the maximum value to check for grouping nodes into macros
    scale (float): the smallest value to use for re-updating the epsilon range

    Returns
    -------
    if depth==0:
        EI_macro (float): the $EI$ of the macro_mapping that was selected
        macro_mapping (dict): the macroscale mapping that maximizes the $EI$
                              of the resulting macroscale network, G_macro.

    """

    eps_range = (max_ep - min_ep)*scale
    epsilon_ei = []
    epsilon_range = np.linspace(min_ep, max_ep, 3)
    epsilon_mappings = []

    for eps in epsilon_range:

        labs_e = cluster_optics_dbscan(reach, core, order, eps)

        macro_mapping_e = {i: i if lab == -1 else (len(labs_e)+lab)
                           for i, lab in enumerate(labs_e)}
        macro_types_e = {i: 'spatem1' for i in macro_mapping_e.values()
                         if i > max(macro_mapping_e.keys())}

        Gm_e = create_macro(G_micro, macro_mapping_e, macro_types_e)
        G_macro_e = check_network(Gm_e)

        EI_macro_e = effective_information(G_macro_e)
        epsilon_ei.append(EI_macro_e)
        epsilon_mappings.append(macro_mapping_e)

    if depth == 0:
        ind = np.argmax(epsilon_ei)
        return epsilon_ei[ind], epsilon_mappings[ind]

    else:
        if epsilon_ei[1] >= epsilon_ei[2] and epsilon_ei[1] >= epsilon_ei[0]:
            new_max = (epsilon_range[1] + epsilon_range[2]) / 2 + eps_range
            new_min = (epsilon_range[1] + epsilon_range[0]) / 2 - eps_range

        elif epsilon_ei[0] >= epsilon_ei[2] and epsilon_ei[0] >= epsilon_ei[1]:
            new_max = epsilon_range[1] + eps_range
            new_min = epsilon_range[0] - eps_range

        else:
            new_max = epsilon_range[2] + eps_range
            new_min = epsilon_range[1] - eps_range

        return find_epsilon_mapping(reach, core, order, G_micro,
                                    depth=depth-1, min_ep=new_min,
                                    max_ep=new_max, scale=scale)


def causal_emergence_spectral(G, check_inacc=False, t=500):
    r"""
    Given a microscale network, G, this function computes a macroscale mapping,
    macro_mapping, using a spectral clustering method such that when G is
    recast as a coarse-grained representation, G_macro, the resulting graph's
    $EI$ is higher than the original network.

    Parameters
    ----------
    G (nx.Graph or np.ndarray): the network in question.
    check_inacc (bool): will check the inaccuracy following the addition of
                        each newly-added macro-node, to ensure that only
                        accurate macros are added.
    t (int): default to 10, this the number of timesteps over which inaccuracy
             is evaluated.

    Returns
    -------
    CE (dict): a dictionary with the following information
      - G_macro (nx.Graph): a coarse-grained description of the micro network.
      - G_micro (nx.Graph): the original microscale network.
      - mapping (dict): the mapping that most-successfully increased the
                        effective information of the network.
      - macro_types (dict): the dictionary associated with each type of
                            macronode in the "winning" G_macro. If types==False
                            then this just is a dictionary with every value
                            set to 'spatem1'.
      - EI_micro (float): the effective information of the micro scale network
      - EI_macro (float): the effective information of the macro scale network
      if check_inacc==True:
          - inaccuracy (np.ndarray): a sequence of inaccuracy values associated
                                 with the successful macro_mapping.

    """

    G_micro = check_network(G)
    EI_micro = effective_information(G_micro)

    dist = construct_distance_matrix(G_micro)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optics = OPTICS(min_samples=2, max_eps=1.0, metric='precomputed')
        _ = optics.fit_predict(dist)  # might need to fix

    reach = optics.reachability_
    core = optics.core_distances_
    order = optics.ordering_

    EI_macro, macro_mapping = find_epsilon_mapping(reach, core, order, G_micro)

    macro_types = {i: 'spatem1' for i, j in macro_mapping.items() if i != j}

    CE = {}
    if macro_types == {}:
        EI_macro = EI_micro
        G_macro = G_micro.copy()

    else:
        G_macro = create_macro(G_micro, macro_mapping, macro_types)
        G_macro = check_network(G_macro)

    CE['G_macro'] = G_macro
    CE['G_micro'] = G_micro
    CE['mapping'] = macro_mapping
    CE['macro_types'] = macro_types
    CE['EI_micro'] = EI_micro
    CE['EI_macro'] = EI_macro

    if check_inacc:
        inaccuracies = macro_inaccuracy(G_micro, G_macro, macro_mapping,
                                        macro_types, t)
        CE['inaccuracy'] = inaccuracies['inaccuracies']

    return CE
