import warnings
import os
import threading
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_exp(df, i, k, proxy_lam, d, V_eps, p_eps, gamma, proxy_lam_name, rho, threads):
    for items_seed in range(6):
        params = {'id': i,
                  'k': k,
                  'd': d,
                  'proxy_lam': proxy_lam,
                  'proxy_lam_name': proxy_lam_name,
                  'items_seed': items_seed,
                  'V_eps': V_eps,
                  'p_eps': p_eps,
                  'gamma': gamma,
                  'rho': rho,
                 }
        cols = [col for col in df.columns if col != 'id']
        if items_seed == 0 and (df[cols] == pd.Series(params)[cols]).all(axis=1).sum() == 1:
            print(f'{params} already exists')
            break
            
        df.loc[df.shape[0]] = params

        for split_num in range(6):
            # run experiment
            command = './py-sbatch.sh full_framework.py'
            command += f' -d {d} -k {k} -exp-id {i} -split-num {split_num} -proxy-lam {proxy_lam} -items-seed {items_seed}'  # arguments
            command += f' --gamma {gamma} --V-eps {V_eps} --p-eps {p_eps} --rho {rho}'  # optional arguments
            t = threading.Thread(target=lambda: os.system(command))
            t.start()
            threads.append(t)
        i += 1
                
    return df, threads, i
        

def run_all_experiments():
    df = pd.DataFrame(columns=['id', 'k', 'proxy_lam', 'items_seed', 'd', 'proxy_lam_name',
                               'gamma', 'V_eps', 'p_eps', 'rho'])
    threads = []
    
    i = 0
    V_eps = 0.0
    p_eps = 0.0
    gamma = 0.5
    rho = 0.0
    
    # d = 12 main experiment
    d = 12
    lam_list = list(np.arange(0, 1.01, 0.25)) + ['1-k/2d']
    for proxy_lam_name in lam_list:    
        for k in range(1, d + 1, 1):
            proxy_lam = 1 - (k / (2 * d)) if proxy_lam_name == '1-k/2d' else proxy_lam_name
            df, threads, i = run_exp(df, i, k, proxy_lam, d, V_eps, p_eps, gamma, proxy_lam_name, rho, threads)

    # d = 100 experiment
    d = 100
    lam_list = [0.5, '1-k/2d']
    for proxy_lam_name in lam_list: 
        for k in range(10, d, 10):
            proxy_lam = 1 - (k / (2 * d)) if proxy_lam_name == '1-k/2d' else proxy_lam_name
            df, threads, i = run_exp(df, i, k, proxy_lam, d, V_eps, p_eps, gamma, proxy_lam_name, rho, threads)

    # prices experiments
    d = 12
    k = 6
    proxy_lam_name = '1-k/2d'
    proxy_lam = 1 - (k / (2 * d))

    # vary gamma
    epsilon = 0.0
    rho = 0.0
    for gamma in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]:
        df, threads, i = run_exp(df, i, k, proxy_lam, d, V_eps, p_eps, gamma, proxy_lam_name, rho, threads)

    # vary epsilon
    gamma = 0.5
    rho = 0.0
    default_eps = 0.0
    for epsilon in [0.0, 1e-2, 3e-2, 1e-1, 3e-1]:
        df, threads, i = run_exp(df, i, k, proxy_lam, d, epsilon, default_eps, gamma, proxy_lam_name, rho, threads)
        df, threads, i = run_exp(df, i, k, proxy_lam, d, default_eps, epsilon, gamma, proxy_lam_name, rho, threads)
        
    # vary rho
    gamma = 0.5
    epsilon = 0.0
    for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        df, threads, i = run_exp(df, i, k, proxy_lam, d, epsilon, epsilon, gamma, proxy_lam_name, rho, threads)

    for t in threads:
        t.join()

    # save
    df.to_csv('all_exp.csv', index=False)

    
if __name__ == '__main__':
    run_all_experiments()
    