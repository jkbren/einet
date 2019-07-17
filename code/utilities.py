"""
utilities.py
--------------------
Plotting and other utility code related to the paper:

Klein, B. & Hoel, E. (2019) Effective information quantifies causal structure
                            and causal emergence in complex networks.

author: Brennan Klein 
email: brennanjamesklein at gmail dot com
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
import scipy as sp
from ei_net import *
from ce_net import *


def show_values(pc, ax, fontsize=16, fmt="%.3f", **kw):
    """
    For bar charts, show the value of the height of the bar.
    """
    pc.update_scalarmappable()
    for p, color, value in zip(pc.get_paths(), 
    						   pc.get_facecolors(), 
    						   pc.get_array()):

        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", 
        		size=fontsize, color=color, **kw)

        
def plot_TPM_micromacro(TPMicro, TPM_grouping, mult=5, save=False, fn=1):
    """
    Make pretty TPMs
    """ 
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    			'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    			'U', 'V', 'W', 'X', 'Y', 'Z']
    greek    = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 
    			'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 
    			'χ', 'ψ', 'ω']

    
    TPMacro_c = MACRO(TPMicro, TPM_grouping, conditional=True )
    TPMacro_n = MACRO(TPMicro, TPM_grouping, conditional=False)
    
    micro_network_size = TPMicro.shape[0]
    nodes_in_macro_network = np.unique(list(TPM_grouping.values()))
    macro_network_size = len(nodes_in_macro_network)
    macro_nodes = nodes_in_macro_network[nodes_in_macro_network > \
    									 micro_network_size-1]
    n_macro = len(macro_nodes)
    micro_to_macro_list = []
    for macro_i in range(n_macro):
        micro_in_macro_i = [k for k,v in TPM_grouping.items() \
        					if v==macro_nodes[macro_i]]
        micro_to_macro_list.append(micro_in_macro_i)

    micronode_xlabels = alphabet[0:TPMicro.shape[0]]
    micronode_ylabels = alphabet[0:TPMicro.shape[1]]
    macronode_xlabels = []
    macronode_ylabels = []
    for macc in micro_to_macro_list:
        for i in micronode_xlabels:
            if i not in np.array(micronode_xlabels)[macc]:
                macronode_xlabels.append(i)
                macronode_ylabels.append(i)
    for macc in range(n_macro):
        macronode_xlabels.append("%s|%s"%(greek[macc],alphabet[macc]))
        macronode_ylabels.append("%s|%s"%(greek[macc],alphabet[macc]))

    fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(3*mult,1*mult))

    c0 = ax0.pcolor(np.arange(-0.5, TPMicro.shape[0], 1), 
    				np.arange(-0.5, TPMicro.shape[0], 1), 
               TPMicro,   edgecolors='#999999', linewidths=3.0, cmap='bone_r')
    ax0.invert_yaxis()

    show_values(c0, ax=ax0, fontsize=mult*2.3)

    c1 = ax1.pcolor(np.arange(-0.5, TPMacro_c.shape[0], 1), 
    				np.arange(-0.5, TPMacro_c.shape[0], 1), 
               TPMacro_c, edgecolors='#999999', linewidths=3.0, cmap='bone_r')
    ax1.invert_yaxis()
    show_values(c1, ax=ax1, fontsize=mult*2.3)

    ax0.set_xticks(np.arange(0, TPMicro.shape[0], 1))
    ax0.set_yticks(np.arange(0, TPMicro.shape[1], 1))
    ax1.set_xticks(np.arange(0, TPMacro_c.shape[0], 1))
    ax1.set_yticks(np.arange(0, TPMacro_c.shape[1], 1))

    ax0.set_xticklabels(micronode_xlabels, fontsize=3*mult)
    ax0.set_yticklabels(micronode_ylabels, fontsize=3*mult)
    ax1.set_xticklabels(macronode_xlabels, fontsize=3*mult)
    ax1.set_yticklabels(macronode_ylabels, fontsize=3*mult)

    ax0.set_xticks(np.arange(-0.5, TPMicro.shape[0]-0.5, 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, TPMicro.shape[1]-0.5, 1), minor=True)
    ax1.set_xticks(np.arange(-0.5, TPMacro_c.shape[0]-0.5, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, TPMacro_c.shape[1]-0.5, 1), minor=True)

    ax0.xaxis.tick_top()
    ax1.xaxis.tick_top()

    ax0.set_title('Micro TPM',               fontsize=3.5*mult, pad=30)
    ax1.set_title('Macro TPM (conditional)', fontsize=3.5*mult, pad=30)
    
    timee = 10
    
    inaccuracy_dict_n = INACCURACY(TPMicro, TPM_grouping, 
    								conditional=False, t=timee)
    inaccuracy_dict_c = INACCURACY(TPMicro, TPM_grouping, 
    								conditional=True,  t=timee)
    inaccs_n = inaccuracy_dict_n['inaccuracies']
    inaccs_c = inaccuracy_dict_c['inaccuracies']
    
    min_n = min(np.array(inaccs_n))+np.exp(-8)
    min_c = min(np.array(inaccs_c))+np.exp(-8)
    inaccs_n_plot = np.array(inaccs_n) + \
    				min_n*np.random.uniform(-0.001,0.001,len(inaccs_n))
    inaccs_c_plot = np.array(inaccs_c) + \
    				min_c*np.random.uniform(-0.001,0.001,len(inaccs_c))
        
    minz  = min(min(inaccs_n),min(inaccs_c))
    maxz  = max(max(inaccs_n),max(inaccs_c))
    if minz==maxz:
        ylimzz = np.linspace(-0.01, 0.09, 11)
    else:
        ylimzz = np.linspace(-np.abs(minz), maxz+np.abs(minz), 11)

    ax2.plot([-10,20], [0,0], color='k', linewidth=4.5, alpha=0.8, 
    		 label='Zero inaccuracy', linestyle=':')
    ax2.plot(inaccs_n_plot, color='#ff235c', linewidth=2.75, linestyle='-.', 
    		 marker='o', alpha=0.8, label='Macro inaccuracy (naive)')
    ax2.plot(inaccs_c_plot, color='#028bce', linewidth=2.75, linestyle='--',
    		 marker='s', alpha=0.8, label='Macro inaccuracy (conditional)')
    
    if min_n < 0 or min_c < 0:
        ax2.plot([-10,20], [0,0], color='#ca5d46', alpha=0.8, 
        		 label='(Inaccuracy goes to infinity)', 
        		 linestyle='', marker='o', markersize=8.0)
    
    ax2.set_xticks(np.linspace(0,timee,timee+1))
    ax2.set_xticklabels(np.arange(0,timee+1,1), fontsize=2*mult)
    ax2.set_yticks(ylimzz)
    ax2.set_yticklabels(["%.5f"%i for i in ylimzz], fontsize=2*mult)
    ax2.set_xlim(-0.5,timee-0.5)
    ax2.set_xlabel("Time", size=3*mult)
    ax2.set_title("Inaccuracy", size=3.5*mult, pad=30)
    ax2.grid(linewidth=1.5, linestyle='--', color='#999999', alpha=0.3)
    ax2.legend(fontsize=2.3*mult)

    if save==True:
        plt.savefig("../figs/pngs/Conditional_Macros_%03i.png"%int(fn), 
                    bbox_inches='tight', dpi=425)

    else:
        plt.show()    


def softmax(A, k=1.0):
    """
    Calculates the softmax of a distribution, modulated by a precision term, k.
    
	Params
	------
	A (np.ndarray): a vector of real-valued numbers
	k (float): a factor that modulates how precise the output softmaxed vector
			   will end up being (k=1.0 is standard, k=0.0 makes it uniform).

	Returns
	-------
	A (np.ndarray): a softmaxed version of the original vector

    """
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    A = A*k
    maxA = A.max()
    A = A - maxA
    A = np.exp(A)
    A = A/np.sum(A)

    return A
