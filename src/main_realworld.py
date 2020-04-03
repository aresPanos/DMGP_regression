#!/usr/bin/python3
# Copyright 2020 Aristeidis Panos

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script produces the results for all the real world datasets used in the paper


import numpy as np
import tensorflow as tf
from dmgp_model import DMGP_Regressor
from helper import *
tf.random.set_seed(1) # used for reproducibility

# Parameters used in the paper for each of the real-world datasets
data_set_names = ['elevators', 'protein', 'sarcos', '3droad', 'song', 'buzz', 'electric'] # the real-world datasets used for the experiments
lr_datasets = {'elevators': .0001, 'protein': .002, 'sarcos': .001, '3droad': .01, 'song': .00001, 'buzz': .001, 'electric': .005} # learning rates used 
var_f_datasets = {'elevators': 1., 'protein': 1., 'sarcos': 1., '3droad': 8., 'song': 0.2, 'buzz': 7., 'house_electric': 10.} # initial value for the signal variance
var_n_datasets = {'elevators': .1, 'protein': .1, 'sarcos': .1, '3droad': 1., 'song': 3., 'buzz': .5, 'house_electric': 1.} # initial value for the noise variance

num_epochs = 5000

for dataset_name in data_set_names:
    print('\n\n*** Dataset: {} ****' .format(dataset_name))
    x_all, y_all, perm, step = load_dataset(dataset_name)
    num_splits = 10 if x_all.shape[0] < 100000 else 5 # number of splits to run DMGP for each dataset; 10 for small-scale datasets and 10 for large-scale ones
    rmse_vec = np.zeros(num_splits)
    nlpd_vec = np.zeros(num_splits)
        
    for split_id in range(num_splits):        
        x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, perm, step, split_id)
        x_train, y_train, x_test, y_test = standardize_dataset(dataset_name, x_train, y_train, x_test, y_test)
        
        N, D = x_train.shape
        adam_keras_hermite = tf.optimizers.Adam(learning_rate=lr_datasets[dataset_name])
        model_dmgp = DMGP_Regressor(data=(x_train, y_train), sigma_f_sq=var_f_datasets[dataset_name], sigma_n_sq=var_n_datasets[dataset_name]) # The DMGP model initialized with 0.1 noise variance

        @tf.function(autograph=False)
        def optimization_step():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model_dmgp.trainable_variables)
                objective = model_dmgp.neg_log_likelihood()
                grads = tape.gradient(objective, model_dmgp.trainable_variables)
            adam_keras_hermite.apply_gradients(zip(grads, model_dmgp.trainable_variables))
            return objective

        for epoch in range(num_epochs):  
            loss_value = optimization_step()
            if epoch % 200 == 0:
                print('\nEpoch: {}/{}   log-marginal-likl: {:.3f}' .format(epoch+1, num_epochs, -0.5*(loss_value.numpy() + N*np.log(2*np.pi))))              

        f_mean, f_var = model_dmgp.predict_f(x_test)
        f_mean, f_var = f_mean.numpy(), f_var.numpy() + model_dmgp.sigma_n_sq.value().numpy()   
        error_test, nlpd_test = compute_predictive_perform(y_test, f_mean, f_var)
        rmse_vec[split_id] = error_test
        nlpd_vec[split_id] = nlpd_test

    print('avg RMSE = %.3f  +/- %.3f' %(rmse_vec.mean(), rmse_vec.std()))
    print('avg NLPD = %.3f  +/- %.3f' %(nlpd_vec.mean(), nlpd_vec.std()))
    