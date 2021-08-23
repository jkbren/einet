# Effective information and causal emergence in python

[![DOI](https://zenodo.org/badge/167852490.svg)](https://zenodo.org/badge/latestdoi/167852490)

Python code for calculating *effective information* in networks. This can 
then be used to search for macroscale representations of a network such 
that the coarse grained representation has more effective information than 
the microscale, a phenomenon known as *causal emergence*. This code 
accompanies the recent paper: 

**The emergence of informative higher scales in complex networks**\
Brennan Klein and Erik Hoel, 2020. Complexity.\
[doi:10.1155/2020/8932526](https://doi.org/10.1155/2020/8932526)

- - - -

<p align="center">
<img src="figs/pngs/ei_ER_PA.png" alt="EI in ER and PA networks" width="95%"/>
</p>

**<p align="center">Fig. 1: Effective information vs network size.**

<p align="center">
<img src="figs/pngs/CE_PA.png" alt="EI in ER and PA networks" width="90%"/>
</p>

**<p align="center">Fig. 2: Causal emergence vs preferential attachment.</center>**

- - - -

## Tutorial Notebooks (works in progress...)
1. [Chapter 01 - Network Effective Information](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2001%20-%20Network%20Effective%20Information.ipynb)
2. [Chapter 02 - Network Size and Effective Information](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2002%20-%20Network%20Size%20and%20Effective%20Information.ipynb)
3. [Chapter 03 - Determinism and Degeneracy](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2003%20-%20Determinism%20and%20Degeneracy.ipynb)
4. [Chapter 04 - Effective Information in Real Networks](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2004%20-%20Effective%20Information%20in%20Real%20Networks.ipynb)
5. [Chapter 05 - Causal Emergence in Preferential Attachment and SBMs](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2005%20-%20Causal%20Emergence%20in%20Preferential%20Attachment%20and%20SBMs.ipynb)
6. [Chapter 06 - Causal Emergence and the Emergence of Scale](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2006%20-%20Causal%20Emergence%20and%20the%20Emergence%20of%20Scale.ipynb)
7. [Chapter 07 - Estimating Causal Emergence in Real Networks](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2007%20-%20Estimating%20Causal%20Emergence%20in%20Real%20Networks.ipynb)
8. [Chapter 08 - Miscellaneous](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2008%20-%20Miscellaneous.ipynb)
9. [Chapter 09 - Spectral Causal Emergence](https://nbviewer.jupyter.org/github/jkbren/einet/blob/master/code/Chapter%2009%20-%20Spectral%20Causal%20Emergence.ipynb)

## Installation and Usage

In order to use this code, first clone/download the repository. 
Below is a simple example usage. Please feel free to reach 
out if you find any bugs, have any questions, or if for some reason
the code does not run. 

```Python
>>> from ei_net import *
>>> import networkx as nx
>>> G = nx.karate_club_graph()
>>> print("effective_information(G) =", effective_information(G))
```

```text
EI(G) = 2.3500950888734686
```

The tutorial notebooks are designed to walk through some of the 
main results from the [paper above](https://arxiv.org/abs/1907.03902), 
in addition to several in-depth analyses that were not included in 
the original paper.

## Requirements  <a name="requirements"/>

This code is written in [Python 3.x](https://www.python.org) and uses 
the following packages:

* [NetworkX](https://networkx.github.io)
* [Scipy](http://www.scipy.org/)
* [Numpy](http://numpy.scipy.org/)
* And for replicating figures, you will need:
    + [matplotlib](https://matplotlib.org)
    + [Pandas](https://pandas.pydata.org/)

The colormaps in the paper are from [https://matplotlib.org/cmocean/](https://matplotlib.org/cmocean/)
and the named colors are from [https://medialab.github.io/iwanthue/](https://medialab.github.io/iwanthue/).

## Citation   <a name="citation"/>

If you use these methods and this code in your own research, please cite our paper:

Klein, B. & Hoel, E. (2020). **The emergence of informative higher scales in complex networks**. 
_Complexity_, no. 8932526. doi:[10.1155/2020/8932526](https://doi.org/10.1155/2020/8932526).

Bibtex: 
```text
@article{Klein2020causalemergence,
    title = {{The emergence of informative higher scales in complex networks}},
    author = {Klein, Brennan and Hoel, Erik},
    journal = {Complexity},
    year = {2020},
    pages = {1--12},
    volume = {2020},
    arxivId = {1907.03902v2},
    doi = {10.1155/2020/8932526}
}
```

## See also:

* Hoel, E. (2017). **When the map is better than the territory**. 
*Entropy*. 19(5), 188; doi: [10.3390/e19050188](https://www.mdpi.com/1099-4300/19/5/188).
    + recent work making explicit connections between causal emergence 
    and the channel capacity of a model.
* Hoel, E., Albantakis, L., & Tononi, G. (2013). **Quantifying causal 
emergence shows that macro can beat micro**. *Proceedings of the 
National Academy of Sciences*. 110 (49) 19790-19795.
doi: [10.1073/pnas.1314922110](https://www.pnas.org/content/110/49/19790).
    + the first work to quantify causal emergence, showing how and why 
    certain coarse-grained models can have more effective information.
