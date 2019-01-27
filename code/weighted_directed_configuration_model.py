
import numpy as np
import networkx as nx
import random
import  math 
import bisect


def my_directed_configuration_model(input_G, weight=None,weight_value=None,
                                 create_using=None,seed=None):
      
    """This function is just a modification of the directed_configuration_model algorithm that can be found in
    the networkx module (https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.degree_seq.directed_configuration_model.html)
   
    In addition to the networkx one, this one keeps the node name rather than assigning a new sequence of integers
    in the moment of the creation of the new network. Furthermore, I introduced a weight parameter that allows to 
    assign the name of the attribute and the constant value -- weight_value -- that will be assigned to all the
    edges of the output network.
    """   
        
    in_degree = dict(input_G.in_degree)
    out_degree = dict(input_G.out_degree)
    in_degree_sequence = in_degree.values()
    out_degree_sequence = out_degree.values()
        
        
    if not sum(in_degree_sequence) == sum(out_degree_sequence):
        raise nx.NetworkXError('Invalid degree sequences. '
                               'Sequences must have equal sums.')

    if create_using is None:
        create_using = nx.MultiDiGraph()

    if not seed is None:
        random.seed(seed)

    nin=len(in_degree_sequence)
    nout=len(out_degree_sequence)

    # pad in- or out-degree sequence with zeros to match lengths
    if nin>nout:
        out_degree_sequence.extend((nin-nout)*[0])
    else:
        in_degree_sequence.extend((nout-nin)*[0])

    # create empty graph with same nodes
    G=nx.DiGraph()
    G.add_nodes_from(input_G.nodes())
    N=input_G.order()
    
    if N==0 or max(in_degree_sequence)==0: # done if no edges
        return G

    # build stublists of available degree-repeated stubs
    # e.g. for degree_sequence=[3,2,1,1,1]
    # initially, stublist=[1,1,1,2,2,3,4,5]
    # i.e., node 1 has degree=3 and is repeated 3 times, etc.
    in_stublist=[]
    for n in G:
        for i in range(input_G.in_degree(n)):
            in_stublist.append(n)

    out_stublist=[]
    for n in G:
        for i in range(input_G.out_degree(n)):
            out_stublist.append(n)

    # shuffle stublists and assign pairs by removing 2 elements at a time
    random.shuffle(in_stublist)
    random.shuffle(out_stublist)
    while in_stublist and out_stublist:
        source = out_stublist.pop()
        target = in_stublist.pop()
        G.add_edge(source,target)
        G[source][target][weight]=weight_value
        
    #removing self loops:
    G.remove_edges_from(G.selfloop_edges())  

    G.name="directed configuration_model %d nodes %d edges"%(G.order(),G.size())
    return G



def split_chunks(l, k):
    """ Splits l in k successive chunks."""
    if k < 1:
        yield []
        raise StopIteration
    n = len(l)
    avg = n/k
    remainders = n % k
    start, end = 0, avg
    while start < n:
        if remainders > 0:
            end = end + 1
            remainders = remainders - 1
        yield l[start:end]
        start, end = end, end+avg
  
 


def get_log_binned_chunks(edges, weight='weight', n_chunks=100):
    weight_list = [e[2][weight] for e in edges]
    logmax_weight = int(math.ceil(np.log10(max(weight_list))))
    logbins = np.logspace(0,logmax_weight,n_chunks)
    
    #dividing the edge list into chunks [[], [], []]
    chunks = [[] for n in range(n_chunks)] #each chunk is a list of edges 
    for e in edges:
        thisw = e[2][weight]
        chunk_index = bisect.bisect(logbins,thisw)
        chunks[chunk_index].append(e)
     
    #some bins are empty, I remove them
    not_empty_chunks = [chunk for chunk in chunks if len(chunk)!=0]
    return not_empty_chunks



def my_weighted_directed_configuration_model(G, weight='weight', n_chunks=100, mode='quantiles'):
    """Returns a directed weighted random graph with the same degree sequence of the input graph
       and a 'medianised' weight distribution'.
       
       1. The edges of the original graph are initially split into n different chunks, representing
          n different classes of weight. Each class defines a sub-graph.
       2. For each sub-graph we run the unweighted directed configuration model (based only on the degree
          sequence) and then we assign the median of the weights of the initial sub-graph to the edges of the 
          resulting random graph.
       3. The n weighted random graphs are finally reassembled together into one unique output graph.
        
        Args
        ----
        G: NetworkX DiGraph
           
        weight: string (default='weight')
            Key for edge data used as the edge weight w_ij.
            WARNING: give the definition above, the weight value has to be a nonzero.
            
        n_chunks: int (degault=100)
            The number of chunks used to split the edges of the original graph into different
            'classes of weight'. 
        
        mode: string (default='quantiles')
            Keywork for the two modalities of chunk splitting
            + quantiles:
                The edges are sorted and then equally split into n_chunks quantiles of same size.
            + logbinning:
                The edges are split into n_chunks log-bins. Empty bins are finally removed.
        
        Returns
        -------
        CM: NetworkX DiGraph
            
    """    
    
    G = G.copy()
    
    #sorting edges according to the weight
    edges = []
    for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight']):
        edges.append((a,b,{weight:data[weight]}))
    # for a, b, data in sorted(G.edges(data=True), key=lambda (a, b, data): data[weight]):
        # edges.append((a,b,{weight:data[weight]}))
    
    #splitting the edge list into chunks 
    if mode=='quantiles': chunks = split_chunks(edges,n_chunks)
    elif mode=='logbinning': chunks = get_log_binned_chunks(edges, weight=weight, n_chunks=n_chunks)
    
    CM = nx.DiGraph()
    CM.add_nodes_from(G.nodes())
    for chunk in chunks:
        weights = [e[2][weight] for e in chunk]
        median_weight = np.median(weights)
        thisG=nx.DiGraph()
        for e in chunk:
            thisG.add_edge(e[0],e[1])
    
        thisCM = my_directed_configuration_model(thisG, weight=weight,
                                                 weight_value=median_weight,
                                                 create_using=nx.DiGraph())
    
        for u, v, d in thisCM.edges_iter(data=True):
            if CM.has_edge(u,v): CM[u][v][weight] += d[weight]
            else:
                CM.add_edge(u,v)
                CM[u][v][weight]=d[weight]
    
    CM.name="weighted directed configuration_model %d nodes %d edges"%(CM.order(),CM.size())
    return CM
