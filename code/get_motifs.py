import networkx as nx
import numpy as np
from collections import defaultdict
import community
import argparse
import sys
from network_ei import *

def getMotifs(n=3):
    """Returns a dictionary of all 3-node motifs."""
    if n==3:
        G01 = nx.DiGraph()
        G02 = nx.DiGraph()
        G03 = nx.DiGraph()
        G04 = nx.DiGraph()
        G05 = nx.DiGraph()
        G06 = nx.DiGraph()
        G07 = nx.DiGraph()
        G08 = nx.DiGraph()
        G09 = nx.DiGraph()
        G10 = nx.DiGraph()
        G11 = nx.DiGraph()
        G12 = nx.DiGraph()
        G13 = nx.DiGraph()

        G01.add_nodes_from([0,1,2])
        G02.add_nodes_from([0,1,2])
        G03.add_nodes_from([0,1,2])
        G04.add_nodes_from([0,1,2])
        G05.add_nodes_from([0,1,2])
        G06.add_nodes_from([0,1,2])
        G07.add_nodes_from([0,1,2])
        G08.add_nodes_from([0,1,2])
        G09.add_nodes_from([0,1,2])
        G10.add_nodes_from([0,1,2])
        G11.add_nodes_from([0,1,2])
        G12.add_nodes_from([0,1,2])
        G13.add_nodes_from([0,1,2])

        G01.add_edges_from([(0,1),(0,2)])
        G02.add_edges_from([(0,1),(2,0)])
        G03.add_edges_from([(0,1),(0,2),(2,0)])
        G04.add_edges_from([(1,0),(2,0)])
        G05.add_edges_from([(1,0),(1,2),(0,2)]) # e. coli 
        G06.add_edges_from([(1,0),(1,2),(0,2),(0,1)])
        G07.add_edges_from([(1,0),(0,2),(2,0)])
        G08.add_edges_from([(1,0),(0,1),(0,2),(2,0)])
        G09.add_edges_from([(1,0),(0,2),(2,1)])
        G10.add_edges_from([(1,0),(2,0),(1,2),(0,1)])
        G11.add_edges_from([(1,0),(2,0),(2,1),(0,1)])
        G12.add_edges_from([(0,1),(1,0),(1,2),(2,1),(0,2)])
        G13.add_edges_from([(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)])

        motif_dict = {"Motif 01": {"G":G01, "edges":str(list(G01.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 02": {"G":G02, "edges":str(list(G02.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 03": {"G":G03, "edges":str(list(G03.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 04": {"G":G04, "edges":str(list(G04.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 05": {"G":G05, "edges":str(list(G05.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 06": {"G":G06, "edges":str(list(G06.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 07": {"G":G07, "edges":str(list(G07.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 08": {"G":G08, "edges":str(list(G08.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 09": {"G":G09, "edges":str(list(G09.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 10": {"G":G10, "edges":str(list(G10.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 11": {"G":G11, "edges":str(list(G11.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 12": {"G":G12, "edges":str(list(G12.edges())), "EI":0, "CI":0, "DD":0},
 					  "Motif 13": {"G":G13, "edges":str(list(G13.edges())), "EI":0, "CI":0, "DD":0}}
        
        return motif_dict