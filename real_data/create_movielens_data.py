import os
import pickle
import numpy as np
from tqdm import trange
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import Dataset
from utils import *


def create_movielens_data(d=12, gamma=0.5, V_epsilon=0.0, p_epsilon=0.0, rho=0.0):
    seed = 3
    np.random.seed(seed)

    n_markets = 240
    n_splits = 6
    n_seeds = 6
    m = 20  # num items
    n = 20  # num users
    d_prime = d // 2

    data = Dataset.load_builtin('ml-100k', prompt=False)
    data = data.build_full_trainset()
    algo = NMF(n_factors=d)
    algo.fit(data)

    users_indices = np.random.choice(range(data.n_users), n * n_markets)
    all_B = algo.pu[users_indices]
    # derive T from all markets
    all_U, T = factorize(all_B, n_components=d_prime)

    all_U = all_U.reshape((n_markets, n, d_prime))
    all_B = all_B.reshape((n_markets, n, d))

    for items_seed in range(n_seeds):
        params_str = ''.join([param_str('gamma', gamma, 0.5),
                              param_str('V_eps', V_epsilon, 0.0),
                              param_str('p_eps', p_epsilon, 0.0),
                              param_str('rho', rho, 0.0)
                              ])
        filename = f'movielens__items_seed{str(items_seed)}__d{d}{params_str}.pkl'
        if os.path.exists(os.path.join('pickles', 'data', filename)):
            print(f'skipped {filename} - already exist')
            continue
        
        np.random.seed(items_seed)
        items_indices = np.random.choice(range(data.n_items), m)
        X = algo.qi[items_indices] / data.rating_scale[1]
        all_p = []

        for i in trange(n_markets):
            U = all_U[i]
            B = U @ T
            # calculate V
            V = B @ X.T
            # compute prices according to full information (V)
            p = set_prices(V, gamma, V_epsilon, p_epsilon, rho)    
            all_p.append(p)
        all_p = np.array(all_p)

        # ctrate splits
        indices = np.arange(n_markets).astype(int)
        np.random.shuffle(indices)
        splits = np.split(indices, n_splits)
        
        with open(os.path.join('pickles', 'data', filename), "wb") as f:
            pickle.dump((X, all_U, all_p, T, splits), f)
            
            
if __name__ == '__main__':
    os.makedirs(os.path.join('pickles', 'data'), exist_ok=True)
    
    # d = 12
    # vary gamma
    for gamma in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]:
        create_movielens_data(gamma=gamma)

    # vary epsilon
    for epsilon in [0.0, 1e-2, 3e-2, 1e-1, 3e-1]:
        create_movielens_data(V_epsilon=epsilon)  # V_eps
        create_movielens_data(p_epsilon=epsilon)  # p_eps
    
    # vary rho
    for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        create_movielens_data(rho=rho)
    
    # d = 100
    d = 100
    create_movielens_data(d=d)
