# Deep Mercer Gaussian Process (DMGP) Regression
We provide the code used in our paper [Faster Gaussian Processes via Deep Embeddings](https://arxiv.org/abs/2004.01584).

### Prerequisites
TensorFlow version 2.1.0  
TensorFlow Probability version 0.9.0  
GPflow version 2.0.0 or newer  

### Source code

The following files can be found in the **src** directory :  

- *dmgp_model.py*: implementation of the DMGP model
- *helper.py*: various utility gunctions
- *hermite_coeff.npy*: a numpy array containing the Hermite polynomial coefficients needed for the DMGP model
- *main_realworld.py*: code for replicating the results over the real-world datasets
- *main_simulated_data.py*: code for replicating the results over the two simulated datasets
