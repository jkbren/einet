# Effective information and causal emergence in networks

Python code for calculating *effective information* in networks. This can 
then be used to search for macroscale representations of a network such 
that the coarse grained representation has more effective information than 
the microscale, a phenomenon known as *causal emergence*.

<p align="center">
<img src="ei_ER_PA.png" alt="EI in ER and PA networks" width="85%"/>
</p>

This code accompanies the recent paper: 

**Uncertainty and causal emergence in complex networks**\
Brennan Klein and Erik Hoel, 2019.\
[arXiv:1907.03902](https://arxiv.org/abs/1907.03902)

## Installation and Usage

In order to use this code, clone/download the repository. The tutorial 
notebooks are designed to walk through some of the main results from the
[paper above](https://arxiv.org/abs/1907.03902), in addition to several
in-depth analyses that were not included in the original paper.

For starters, follow along in the 
```Chapter 01 - Chapter 01 - Network Effective Information.ipynb``` notebook.



## Requirements  <a name="requirements"/>

* [Python 3.x](https://www.python.org) with packages:
    + [Numpy](http://numpy.scipy.org/)
    + [Scipy](http://www.scipy.org/)
    + [NetworkX](https://networkx.github.io)

A recent install of [Anaconda Python](https://www.anaconda.com) should come with everything you need.


## Citation   <a name="citation"/>

If you use these methods and this code in your own research, please cite our paper:

Brennan Klein and Erik Hoel, *Uncertainty and causal emergence in complex networks* (2019)
[arXiv:1907.03902](https://arxiv.org/abs/1907.03902)

Here is a bibtex entry:
```text
@article{klein2019causalemergence,
  title={Uncertainty and causal emergence in complex networks},
  author={Klein, Brennan and Hoel, Erik},
  journal={arXiv preprint arXiv:1907.03902},
  year={2019}
}
```

### See also:

*[When the Map is Better than the Territory](https://www.mdpi.com/1099-4300/19/5/188)*
&mdash; recent work comparing effective information and causal emergence to the channel capacity.
*[Quantifying causal emergence shows that macro can beat micro](https://www.pnas.org/content/110/49/19790)*
&mdash; the first work to quantify causal emergence.
