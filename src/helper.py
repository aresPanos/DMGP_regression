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

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve_triangular
import os 
import matplotlib.pyplot as plt

def fun_1d(x_in):
    return 1.5*np.sin(x_in/.5) + 0.5*np.cos(x_in/.1) + x_in/8.
    
    
def generate_1d(N_tr: int = 1500, N_test: int = 200, random_gen: int = 1 ):
    np.random.seed(random_gen)
    x_train = 2*np.random.rand(N_tr)
    x_train.sort()
    y_train = fun_1d(x_train) + 0.1*np.random.randn(N_tr)

    x_test = 2.4*np.random.rand(N_test) - 0.3
    x_test.sort()
    y_test = fun_1d(x_test)
    
    return x_train[:, None], y_train, x_test[:, None], y_test
    
    
def generate_4d(N_tr: int = 1000, N_test: int = 300, random_gen: int = 1 ):
    np.random.seed(random_gen)
    D = 3
    eps_sq_true = .1
    sigma_sq_n_true = 0.01 
    sigma_f = 1.5

    x_train_org = np.zeros((N_tr, D ))
    x_test_org = np.zeros((N_test, D ))

    x_train_org[:, 0] = np.random.randn(N_tr)
    x_train_org[:, 1] = 5. + 2.*np.random.randn(N_tr)
    x_train_org[:, 2] = 2. + 3.*np.random.randn(N_tr)

    x_test_org[:, 0] = np.random.randn(N_test)
    x_test_org[:, 1] = 5. + 2.*np.random.randn(N_test)
    x_test_org[:, 2] = 2. + 3.*np.random.randn(N_test)

    x_train_eps = np.sqrt(eps_sq_true)*x_train_org
    diag_X = np.square(x_train_eps).sum(1)
    dist_X_my = diag_X[:, None] - 2.*x_train_eps@x_train_eps.T + diag_X[None, :]

    K_x = sigma_f*np.exp(-dist_X_my)
    K_x_hat = K_x + sigma_sq_n_true*np.eye(N_tr) # [N, N]
    L_hat = np.linalg.cholesky(K_x_hat) # [N, N]

    Normal_iid_tr = np.random.randn(N_tr)
    y_train = L_hat@Normal_iid_tr
    alpha_y = solve_triangular(L_hat, y_train, lower=True)
    log_marginal_lkl_true = -.5*(np.square(alpha_y).sum() + N_tr*np.log(2*np.pi)) - np.log(np.diag(L_hat)).sum()

    x_test_eps = np.sqrt(eps_sq_true)*x_test_org
    diag_X_ntest = np.square(x_test_eps).sum(1)
    dist_X_my_ntest = diag_X_ntest[:, None] - 2.*x_test_eps@x_test_eps.T + diag_X_ntest[None, :]
    K_x_ntest = sigma_f*np.exp(-dist_X_my_ntest) + sigma_sq_n_true*np.eye(N_test)
    L_hat_ntest = np.linalg.cholesky(K_x_ntest) # [N, N]

    Normal_iid_test = np.random.randn(N_test)
    y_test = L_hat_ntest@Normal_iid_test

    x_train = np.zeros((N_tr, D + 1))
    x_test = np.zeros((N_test, D + 1))

    delta = np.random.rand(N_tr)
    x_train[:, :D-1] = x_train_org[:, :D-1]
    x_train[:, D-1] = delta*x_train_org[:, D-1]
    x_train[:, D] = (1-delta)*x_train_org[:, D-1]

    delta_test = np.random.rand(N_test)
    x_test[:, :D-1] = x_test_org[:, :D-1]
    x_test[:, D-1] = delta_test*x_test_org[:, D-1]
    x_test[:, D] = (1-delta_test)*x_test_org[:, D-1]
    
    return x_train, y_train, x_test, y_test


def load_dataset(dataset_name, train_ratio = 0.9, random_gen: int = 1):
    dir_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

    data_all = np.load(dir_parent + '/data/' + dataset_name + '_all.npy', allow_pickle=True)
    x_all, y_all = data_all[()]['x_all'], data_all[()]['y_all']
    
    np.random.seed(random_gen)
    perm = np.random.permutation(np.arange(y_all.shape[0]))
    step = y_all.shape[0] - int(train_ratio*y_all.shape[0])
    
    return x_all, y_all, perm, step


def train_test_split(x_all, y_all, perm, step, split_id):
    start = split_id*step
    train_ind = np.concatenate((perm[:start], perm[start+step:]))
    test_ind = perm[start:start+step]
    
    return x_all[train_ind], y_all[train_ind], x_all[test_ind], y_all[test_ind]
    
    
def standardize_dataset(dataset_name, x_train, y_train, x_test, y_test):
    if dataset_name == 'house_electric':
        col_standarize = np.arange(13, 19)
        for col in col_standarize:
            mean_col = np.mean(x_train[:, col])
            std_col = np.std(x_train[:, col])
            
            x_train[:, col] = (x_train[:, col] - mean_col)/std_col
            x_test[:, col] = (x_test[:, col] - mean_col)/std_col
            
    elif dataset_name != 'sarcos':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    y_train_mean, y_train_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_train_mean)/y_train_std
    y_test = (y_test - y_train_mean)/y_train_std
    return x_train, y_train, x_test, y_test
    
    
def compute_predictive_perform(y_true, y_pred, var_test):
    mse_term = np.square(y_true - y_pred)
    nlpd = 0.5*(np.log(2.*np.pi*var_test) + mse_term/var_test)
    rmse = np.sqrt(mse_term.mean())

    return rmse, nlpd.mean()
    
    
def plot_1D_data(rmse_vec, nlpd_vec, time_tr_vec, time_pred_vec, neg_log_lkl_vec, dir_parent, Lambda_Herm, range_np, neg_log_lkl_GP_opt, 
                 error_test_gp, nlpd_test_gp, GP_time_tr, GP_time_pred):
    ones_range = np.ones(len(range_np))
    
    plt.figure()
    plt.plot(range_np, rmse_vec, 'b', label='DMGP RMSE')
    plt.plot(range_np, error_test_gp*ones_range, 'r', label='Exact GP RMSE')
    plt.xlabel('$m$')
    plt.ylabel('$RMSE$')
    plt.legend()
    plt.grid()
    plt.savefig(dir_parent + '/plots/dnn_mercer_rmse_1d.pdf', format='pdf')

    plt.figure()
    plt.plot(range_np, nlpd_vec, 'b', label='DMGP NLPD')
    plt.plot(range_np, nlpd_test_gp*ones_range, 'r', label='Exact GP NLPD')
    plt.xlabel('$m$')
    plt.ylabel('$NLPD$')
    plt.legend()
    plt.grid()
    plt.savefig(dir_parent + '/plots/dnn_mercer_nlpd_1d.pdf', format='pdf')

    plt.figure()
    plt.plot(range_np, neg_log_lkl_vec, 'b', label='DMGP loss')
    plt.plot(range_np, neg_log_lkl_GP_opt*ones_range, 'r', label='Exact GP loss')
    plt.xlabel('$m$')
    plt.ylabel('$Loss$')
    plt.legend()
    plt.grid()
    plt.savefig(dir_parent + '/plots/dnn_mercer_neg_log_marginal_lkl_1d.pdf', format='pdf')
    
    plt.figure()
    plt.plot(range_np, time_tr_vec, 'b+', label='DMGP training time')
    plt.plot(range_np, GP_time_tr*ones_range, 'r+', label='Exact GP training time')
    plt.plot(range_np, time_pred_vec, 'b*', label='DMGP prediction time')
    plt.plot(range_np, GP_time_pred*ones_range, 'r*', label='Exact GP prediction time')
    plt.xlabel('$m$')
    plt.ylabel('$Time (sec)$')
    plt.legend()
    plt.grid()
    plt.savefig(dir_parent + '/plots/dnn_mercer_tr_pred_time.pdf', format='pdf')
        
    plt.figure()
    plt.plot(Lambda_Herm, 'b', label='DMGP eigenvalues')
    plt.xlabel('$n$')
    plt.ylabel('$\lambda_n$')
    plt.legend()
    plt.grid()
    plt.savefig(dir_parent + '/plots/dnn_mercer_eigenv.pdf', format='pdf')
            