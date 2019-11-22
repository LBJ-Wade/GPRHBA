GPRHBA - A Hierachical Bayesian code for population inference with Gaussian Process
===================================================================================



Disclaimer: dataCompress.py is a module taken from https://github.com/stevertaylor/gw_catalog_mining, please follow instruction on their repository to give credit to their work correctly. 

Installation 
============

To install this code, run ``python setup.py install --user`` in the root directory.

Examples
========

There are two scripts in the **script** folder,*GPRtutorial.py* and *HierarchicalMain.py*.
The *GPRtutorial.py* serves as a quick example to the Gaussian process part of the code
run the following command to check if the Gaussian process part of the code is functioning correctly:

``python GPRtutorial.py --SimFile ../data/pretrack_binned_time_uniform_Mc_z_design.npz``

*HierarchicalMain.py* is a more comprehensive code which run the entire inference pipeline.
Run the following command to run a test run of the inference:

``python HierarchicalMain.py --PosteriorFile ../data/injection_sigma_100_N100.npz --SimulationFile ../data/pretrack_binned_time_uniform_Mc_z_design.npz --OutputDirectory ./``
