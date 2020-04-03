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

# This script produces the results for the 1D-Data and 4D-Data simulated datasets used in the paper

import numpy as np
import gpflow
import tensorflow as tf
from tensorflow.keras import optimizers
from helper import *
from dmgp_model import DMGP_Regressor
from datetime import datetime
tf.random.set_seed(1) # used for reproducibility

use_1d_data = True # switch to False to use the 4D-Data dataset 
m_in = 20 # number of eigenfunctions used > 0
d_in = 1 # number of outputs of the DNN > 0
x_train, y_train, x_test, y_test = generate_1d() if use_1d_data else generate_4d() # generates data of either the "DATA-1D" or "DATA-4D" dataset
dir_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

model_dmgp = DMGP_Regressor(data=(x_train, y_train), m=m_in, d=d_in, simple_dnn=True)

@tf.function(autograph=False)
def objective_closure():
    return model_dmgp.neg_log_likelihood()

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(objective_closure, model_dmgp.trainable_variables, options=dict(maxiter=500))
neg_log_lkl_opt = 0.5*(model_dmgp.neg_log_likelihood().numpy() + y_train.size*np.log(2*np.pi))

mean_f, var_f = model_dmgp.predict_y(x_test)
mean_f, var_f = mean_f.numpy(), var_f.numpy()

error_test, nlpd_test = compute_predictive_perform(y_test, mean_f, var_f)

print('DMGP RMSE: %.4f   NLPD: %.4f' %(error_test, nlpd_test))
print('Optimized negative log-marginal-lkl = %.4f' % neg_log_lkl_opt )

if use_1d_data:
    range_np = np.arange(2, 2 + 10, 4)
    neg_log_lkl_vec = np.zeros(len(range_np))
    rmse_vec = np.zeros(len(range_np))
    nlpd_vec = np.zeros(len(range_np))
    time_tr_vec = np.zeros(len(range_np))
    time_pred_vec = np.zeros(len(range_np))
    N_tr = y_train.size
    count = 0
    
    for m in range_np:
        model_dmgp = DMGP_Regressor(data=(x_train, y_train), m=m, simple_dnn=True)

        @tf.function(autograph=False)
        def objective_closure_1d():
            return model_dmgp.neg_log_likelihood()

        opt = gpflow.optimizers.Scipy()
        start_time_tr = datetime.now() 
        opt_logs = opt.minimize(objective_closure_1d, model_dmgp.trainable_variables, options=dict(maxiter=500))
        end_time_tr = datetime.now()
        neg_log_lkl_opt = 0.5*(model_dmgp.neg_log_likelihood().numpy() + N_tr*np.log(2*np.pi))

        start_time_pred = datetime.now()
        mean_f, var_f = model_dmgp.predict_y(x_test)
        end_time_pred = datetime.now()
        mean_f, var_f = mean_f.numpy(), var_f.numpy()
        error_test, nlpd_test = compute_predictive_perform(y_test, mean_f, var_f)

        rmse_vec[count] = error_test
        nlpd_vec[count] = nlpd_test
        time_tr_vec[count] = (end_time_tr - start_time_tr).total_seconds()
        time_pred_vec[count] = (end_time_pred - start_time_pred).total_seconds()
        neg_log_lkl_vec[count] = neg_log_lkl_opt
        
        std_f = np.sqrt(var_f)
        
        plt.figure()
        plt.plot(x_test.squeeze(), y_test, 'k', label='f(x)')
        plt.plot(x_test.squeeze(), mean_f, 'r', label='DMGP - m=' + str(m))
        plt.fill_between(x_test.squeeze(), mean_f - 1.96*std_f, mean_f + 1.96*std_f, color='blue', alpha=0.2)
        plt.xlabel('$x$')
        plt.legend()
        plt.savefig(dir_parent + '/plots/dnn_mercer_1d_m=' + str(m) + '.pdf', format='pdf')
        
        count += 1
        
    x_nn_tr = model_dmgp.neural_net(x_train).numpy().squeeze()
    x_nn_tr = (x_nn_tr - x_nn_tr.mean())/x_nn_tr.std()
    Lambda_Herm = model_dmgp.eigen_fun(x_nn_tr)[0].numpy().squeeze()/model_dmgp.sigma_f_sq.value().numpy()
    
    model_gp = gpflow.models.GPR(data=(x_train, y_train[:, None]), kernel=gpflow.kernels.SquaredExponential(), mean_function=None)

    @tf.function(autograph=False)
    def objective_closure_gp():
        return - model_gp.log_marginal_likelihood()
        
    opt = gpflow.optimizers.Scipy()
    start_time_tr = datetime.now()
    opt_logs = opt.minimize(objective_closure_gp, model_gp.trainable_variables, options=dict(maxiter=250))
    end_time_tr = datetime.now()
    neg_log_lkl_GP_opt = -model_gp.log_likelihood().numpy()

    start_time_pred = datetime.now()
    mean_f, var_f = model_gp.predict_y(x_test)
    end_time_pred = datetime.now()
    mean_f, var_f = mean_f.numpy().squeeze(), var_f.numpy().squeeze()

    error_test_gp, nlpd_test_gp = compute_predictive_perform(y_test, mean_f, var_f)
    GP_time_tr = (end_time_tr - start_time_tr).total_seconds()
    GP_time_pred = (end_time_pred - start_time_pred).total_seconds()
        
    plot_1D_data(rmse_vec, nlpd_vec, time_tr_vec, time_pred_vec, neg_log_lkl_vec, dir_parent, Lambda_Herm, range_np, neg_log_lkl_GP_opt, 
                 error_test_gp, nlpd_test_gp, GP_time_tr, GP_time_pred)