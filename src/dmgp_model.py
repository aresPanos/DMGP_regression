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

from typing import Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K_bd
from itertools import product
import os 

from gpflow.base import Module, Parameter
from gpflow.config import default_float, set_default_float
from gpflow.utilities import ops, positive


class DMGP_Regressor(Module):
    '''
    This is the Deep Mercer Gaussian Process (DMGP) model for regression
    
    data: a tuple of two tensors. First tuple corresponds to the design matrix with shape [N, D] 
          and the second tensor is an N-dimensional array with the targets
    m: number of eigenfunctions used for the approximation; an integer greater than 1
    d: number of outputs for the neural net; a positive integer 
    alpha: normalization constant related to the variance of the embedding space
    eps_sq: initial value of the epsilon square
    sigma_n_sq: initial value of the noise variance
    sigma_f_sq: initial value of the signal variance
    simple_dnn: if True a single layer DNN is used otherwise a 4-layer DNN
    '''

    def __init__(self, 
                 data: Tuple[tf.Tensor, tf.Tensor], 
                 m: int = 20, 
                 d: int = 1,
                 alpha: np.float = 1./np.sqrt(2.), 
                 eps_sq: np.float = 1,
                 sigma_n_sq: np.float = 1,
                 sigma_f_sq: np.float = 1,
                 simple_dnn: bool = False):
                    
        if data[1].dtype == np.float64:
            K_bd.set_floatx('float64')
        else:
            set_default_float(np.float32)
            
        self.num_data = tf.cast(data[1].shape[0], default_float())
        self.data = (tf.cast(data[0], default_float()), tf.cast(data[1], default_float()))
               
        self.flag_1d = d == 1
        self.alpha = tf.cast(alpha, default_float())
        self.alpha_sq = tf.square(self.alpha)
        self.m = tf.cast(m, default_float())
        self.this_range = tf.constant(np.asarray(list(product(range(1, m + 1), repeat=d))).squeeze(), dtype=default_float())
        self.this_range_1 = self.this_range - 1.
        self.this_range_1_2 = self.this_range_1 if self.flag_1d else tf.range(m, dtype=default_float())
        self.this_range_1_int = tf.cast(self.this_range_1, tf.int32)
        self.tf_range_dnn_out = tf.range(d)
        self.this_range_1_ln2 = np.log(2.)*self.this_range_1
        
        self.vander_range = tf.range(m+1, dtype=default_float())
        self.eye_k = tf.eye(m**d, dtype=default_float())
        self.yTy = tf.reduce_sum(tf.math.square(self.data[1])) 
        self.coeff_n_tf = tf.constant(np.load(os.path.dirname(os.path.realpath(__file__)) + '\hermite_coeff.npy')[:m, :m], dtype=default_float())
        
        eps_sq = eps_sq*np.ones(d) if d > 1 else eps_sq
        self.eps_sq = Parameter(eps_sq, transform=positive(), dtype=default_float())
        self.sigma_f_sq = Parameter(sigma_f_sq, transform=positive(), dtype=default_float())
        self.sigma_n_sq = Parameter(sigma_n_sq, transform=positive(), dtype=default_float())
        
        model = models.Sequential()
        if simple_dnn:
            model.add(layers.Dense(1, activation='tanh', input_dim=data[0].shape[1]))  
        else:
            model.add(layers.Dense(256, activation='tanh', input_dim=data[0].shape[1]))     
            model.add(layers.Dense(128, activation='tanh'))
            model.add(layers.Dense(64, activation='tanh'))
            model.add(layers.Dense(32, activation='tanh'))
        
        model.add(layers.Dense(d))  
        self.neural_net = model
       
                          
    def neg_log_likelihood(self) -> tf.Tensor:
        '''
        It computes the negative log marginal likelihood up to a constant
        '''
        x_nn_tr = tf.squeeze(self.neural_net(self.data[0]))
        x_nn_tr = (x_nn_tr - tf.math.reduce_mean(x_nn_tr,0))/tf.math.reduce_std(x_nn_tr,0)
        
        inv_sigma_sq = 1/self.sigma_n_sq 
        Lambda_Herm, V_herm = self.eigen_fun(x_nn_tr) # [k, None], [N, k]

        V_lambda_sqrt = V_herm*tf.math.sqrt(Lambda_Herm) # [N, k]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, self.data[1], transpose_a=True) # [k, None]
        
        low_rank_term = self.eye_k + inv_sigma_sq*tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [k, k]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [k, 1]
        data_fit = inv_sigma_sq*(self.yTy - inv_sigma_sq*tf.reduce_sum(tf.math.square(L_inv_V_y)))
        return data_fit + self.num_data*tf.math.log(self.sigma_n_sq) + 2.*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(low_rank_L))) 

        
    def eigen_fun(self, data_x) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        It computes the eigenvalues and eigenfunctions evaluated at the training inputs.
        ''' 
        beta_tmp = 1. + 4.*self.eps_sq/self.alpha_sq # [d, None]
        beta = tf.pow(beta_tmp, .25) # [d, None]
        beta_sq = tf.math.sqrt(beta_tmp) # [d, None]
        delta_sq = .5*self.alpha_sq*(tf.square(beta) - 1.) # [d, None]
        log_a_d_ep_sq = tf.math.log(self.alpha_sq + delta_sq + self.eps_sq) # [d, None]
        
        log_term = .5*(tf.math.log(self.alpha_sq) - log_a_d_ep_sq) + self.this_range_1*(tf.math.log(self.eps_sq) - log_a_d_ep_sq) # [n_eigen^d, d]
        
        gamma = tf.exp(.5 * (tf.math.log(beta) - self.this_range_1_ln2 - tf.math.lgamma(self.this_range))) # [n_eigen^d, d]
        tmp_exp = tf.exp(-delta_sq*(tf.square(data_x))) # [N, d]
        x_data_upgr = self.alpha*beta*data_x # [N, d]
            
        vander_tf = tf.math.pow(tf.expand_dims(x_data_upgr, axis=-1), self.this_range_1_2[None, :]) # [N, n_eigen] or [N, d, n_eigen]
        final_mtr = tf.linalg.matmul(vander_tf, self.coeff_n_tf, transpose_b=True) # [N, n_eigen] or [N, d, n_eigen] 
        
        if self.flag_1d:
            tf_lambda_n = tf.exp(log_term)
            tf_phi_n = gamma*tmp_exp[:, None]*final_mtr # [N, n_eigen]
        else:
            tf_lambda_n = tf.exp(tf.reduce_sum(log_term, 1))           
            eigen_fun_tmp = tf.map_fn(lambda x: tf.gather(final_mtr[:, x], self.this_range_1_int[:, x], axis=-1), self.tf_range_dnn_out, dtype = tf.float64) # [d, N, n_eigen^d]
            eigen_fun_tmp = tf.einsum('ed,nd,dne->dne', gamma, tmp_exp, eigen_fun_tmp) # [d, N, n_eigen^d]
            tf_phi_n = tf.reduce_prod(eigen_fun_tmp, 0) # [N, n_eigen^d]
        
        return self.sigma_f_sq*tf_lambda_n, tf_phi_n
    
       
    def predict_y(self, x_test: tf.Tensor, full_cov: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        It computes the mean and variance of the held-out data x_test.
        
        :x_test: tf.Tensor
            Input locations of the held-out data with shape=[Ntest, D]
            where Ntest is the number of rows and D is the input dimension of each point.
        :full_cov: bool
            If True, compute and return the full Ntest x Ntest test covariance matrix. Otherwise return only the diagonal of this matrix.
        '''    
        inv_sigma_sq = 1/self.sigma_n_sq 
        x_nn_tr = tf.squeeze(self.neural_net(self.data[0]))
        mean_x_nn_tr, std_x_nn_tr = tf.math.reduce_mean(x_nn_tr, 0), tf.math.reduce_std(x_nn_tr, 0)
        x_nn_tr = (x_nn_tr - mean_x_nn_tr)/std_x_nn_tr
        x_nn_test = tf.squeeze(self.neural_net(x_test))
        x_nn_test = (x_nn_test - mean_x_nn_tr)/std_x_nn_tr
        
        Lambda_Herm, V_herm = self.eigen_fun(x_nn_tr) # [N, k]
        V_herm_test = self.eigen_fun(x_nn_test)[1] # [Ntest, k]
        sqrt_Lambda_Herm = tf.math.sqrt(Lambda_Herm) # [k, None]
        
        V_lambda_sqrt = V_herm*sqrt_Lambda_Herm # [N, k]
        V_lambda_sqrt_y = tf.linalg.matvec(V_lambda_sqrt, self.data[1], transpose_a=True) # [k, None]

        V_test_lambda_sqrt = V_herm_test*sqrt_Lambda_Herm # [Ntest, k]
        K_Xtest_X_y =  tf.linalg.matvec(V_test_lambda_sqrt, V_lambda_sqrt_y) # [Ntest, None]
        VT_V = tf.linalg.matmul(V_lambda_sqrt, V_lambda_sqrt, transpose_a=True) # [k, k]
        
        low_rank_term = self.eye_k + inv_sigma_sq*VT_V # [k, k]
        low_rank_L = tf.linalg.cholesky(low_rank_term)
        L_inv_V_y = tf.linalg.triangular_solve(low_rank_L, V_lambda_sqrt_y[:, None], lower=True) # [k, 1]
        
        V_K_Xtest = tf.linalg.matmul(VT_V, V_test_lambda_sqrt, transpose_b=True) # [k, N_test]
        tmp_inv = tf.linalg.triangular_solve(low_rank_L, V_K_Xtest, lower=True) # [k, N_test]
        mean_f = inv_sigma_sq*(K_Xtest_X_y - inv_sigma_sq*tf.linalg.matvec(tmp_inv, tf.squeeze(L_inv_V_y), transpose_a=True))
            
        if full_cov:     
            tmp_matmul = tf.linalg.matmul(V_test_lambda_sqrt, V_K_Xtest) # [Ntest, Ntest]
            K_Xtest_herm = tf.linalg.matmul(V_test_lambda_sqrt, V_test_lambda_sqrt, transpose_b=True) # [Ntest, Ntest]
            var_f = K_Xtest_herm - inv_sigma_sq*(tmp_matmul - inv_sigma_sq*tf.linalg.matmul(tmp_inv, tmp_inv, transpose_a=True))  
            diag = tf.linalg.diag_part(var_f) + self.sigma_n_sq
            var_f = tf.linalg.set_diag(var_f, diag)            
        else:
            var_f = self.sigma_f_sq + self.sigma_n_sq - inv_sigma_sq*(tf.einsum('kn,nk->n', V_K_Xtest, V_test_lambda_sqrt) - inv_sigma_sq*tf.reduce_sum(tf.math.square(tmp_inv), 0))
            
        return mean_f, var_f
     
