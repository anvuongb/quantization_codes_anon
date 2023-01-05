import argparse
import numpy as np
import os
import ast
import scipy
from helpers import *
# import yaml
import time


# def read_yaml(file_path):
#     with open(file_path, "r") as f:
#         return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser(description="This script finds the optimal quantization and input distribution",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=int, default=3,
                        help='''1->find optimal quantizer given input distribution
                                2->find optimal input distribution given quantizer using DP
                                3->alternating to find both
                                4->find optimal input distribution given quantizer using exhaustive search
                            ''')
    parser.add_argument("--input-levels", type=str, default='''[-3, 0, 3]''',
                        help="input levels")
    parser.add_argument("--search-interval", type=str, default='''[-8, 8]''',
                        help="region to search for optimal values")
    parser.add_argument("--num-interval", type=int, default=200,
                        help="number of intervals to divide")
    parser.add_argument("--sigma", type=float, default=0.75,
                        help="channel noise level")
    parser.add_argument("--input-dist", type=str,
                        help="input distribution")
    parser.add_argument("--quantizer", type=str,
                        help="thresholds")   
    parser.add_argument("--seed", type=int, 
                        help="random seed for Px generation")        
    parser.add_argument("--stop-thres", type=float, default=0.0001,
                        help="when to stop alternating") 
    parser.add_argument("--max-iter", type=int, default=10,
                        help="maximum number of iterations to run")                             
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    mode = args.mode
    
    X = np.array(ast.literal_eval(args.input_levels))
    Q = len(X)
    N = args.num_interval
    search_interval = ast.literal_eval(args.search_interval)
    int_start = search_interval[0]
    int_end = search_interval[1]
    step = (int_end-int_start)/N
    S = np.linspace(int_start, int_end, N+1)
    M = Q

    sigma = args.sigma
    Y = X + np.random.randn(Q)*sigma
    Phi = [scipy.stats.norm(loc=X[i], scale=sigma) for i in range(Q)]
    
    print("loaded config")
    print("X={}, N={}, M={}, Sigma={}".format(X, N, M, sigma))
    print("start={}, end={}".format(int_start, int_end))
    print("max_iter={}, stop_thres={}".format(args.max_iter, args.stop_thres))
    print("mode={}".format(mode))
    
    if not os.path.exists("logs/sigma_{}/{}_levels".format(sigma, Q)):
        os.makedirs("logs/sigma_{}/{}_levels".format(sigma, Q))
    log_fn = "logs/sigma_{}/{}_levels/mode{}.sigma{}.seed{}.time{}.log".format(sigma, Q, args.mode, args.sigma, args.seed, int(time.time()))
    with open(log_fn, "w") as f:
        f.writelines("X={}, N={}, M={}, Sigma={}\n".format(X, N, M, sigma))
        f.writelines("start={}, end={}\n".format(int_start, int_end))
        f.writelines("max_iter={}, stop_thres={}\n".format(args.max_iter, args.stop_thres))
        f.writelines("mode={}\n".format(mode))
        f.writelines("\n")
    
    if mode == 1 or mode == 4:
        Px = np.array(ast.literal_eval(args.input_dist))
        print("Px={}".format(Px))
    if mode == 2:
        Hx = np.array(ast.literal_eval(args.quantizer))
        print("Hx={}".format(Hx))
    if mode == 3:
        print("seed={}".format(args.seed))
        if args.seed is not None:
            np.random.seed(args.seed)
        Px = np.random.randint(0, 101, size = Q)
        Px = Px/np.sum(Px)
        print("Px={}".format(Px))
        
    # print("read yaml")
    # config = read_yaml("config.yaml")
    # print(config)
    print("\n Start running")
    if mode == 1:
        start = time.time()
        Azx = np.zeros((M, Q))
        for j in range(Q):
            for i in range(M):
                Azx[i, j] = Phi[j].cdf(Hx[i+1]) - Phi[j].cdf(Hx[i])
        status, obj_value, px_value = calculate_optimal_distribution(Q, M, Azx)
        stop = time.time()
        print("Finished")
        print("    optimal input distribution {}".format(px_value))
        print("    optimal I(X;Z) {}".format(obj_value))
        print("    took {:.4f}s".format(stop-start))
        with open(log_fn, "a") as f:
            f.writelines("Finished\n")
            f.writelines("    optimal input distribution {}\n".format(px_value))
            f.writelines("    optimal I(X;Z) {}\n".format(obj_value))
            f.writelines("    took {:.4f}s\n".format(stop-start))
    
    if mode == 2:
        start = time.time()
        Ayx, Axy, Py, Pxy = calculate_transition_matrix(Px, N, Q, S, Phi)
        opt_H, opt_value = dp_optimal_quantizer(N, M, Pxy, Py)
        stop = time.time()
        print("Finished")
        print("    optimal quantizer {}".format(S[opt_H]))
        print("    optimal I(X;Z) {}".format(calculate_I(Px, Q, M, Phi, S[opt_H])))
        print("    took {:.4f}s".format(stop-start))
        with open(log_fn, "a") as f:
            f.writelines("Finished\n")
            f.writelines("    optimal quantizer {}\n".format(S[opt_H]))
            f.writelines("    optimal I(X;Z) {}\n".format(calculate_I(Px, Q, M, Phi, S[opt_H])))
            f.writelines("    took {:.4f}s\n".format(stop-start))
    
    if mode == 4:
        start = time.time()
        if Q == 5:
            current_best, current_best_thres, _ = exhaustive_search_quantization_5_level(Px, N, Q, M, S, Phi)
        if Q == 4:
            current_best, current_best_thres, _ = exhaustive_search_quantization_4_level(Px, N, Q, M, S, Phi)
        if Q == 3:
            current_best, current_best_thres, _ = exhaustive_search_quantization_3_level(Px, N, Q, M, S, Phi)
        stop = time.time()
        thres = [S[0]] + list(current_best_thres) + [S[-1]]
        print("Finished")
        print("    optimal quantizer {}".format(thres))
        print("    optimal I(X;Z) {}".format(current_best))
        print("    took {:.4f}s".format(stop-start))
        with open(log_fn, "a") as f:
            f.writelines("Finished\n")
            f.writelines("    optimal quantizer {}\n".format(thres))
            f.writelines("    optimal I(X;Z) {}\n".format(current_best))
            f.writelines("    took {:.4f}s\n".format(stop-start))
        
        
    if mode == 3:
        I_prev = -100
        for it in tqdm.tqdm(range(args.max_iter)):
            start = time.time()
            
            # dp to find optimal quantizer
            Ayx, Axy, Py, Pxy = calculate_transition_matrix(Px, N, Q, S, Phi)
            opt_H, opt_value = dp_optimal_quantizer(N, M, Pxy, Py, Q)

            # cvxopt to find optimal input distribution
            # compute Azx
            Hx = S[opt_H]
            Azx = np.zeros((M, Q))
            for j in range(Q):
                for i in range(M):
                    Azx[i, j] = Phi[j].cdf(Hx[i+1]) - Phi[j].cdf(Hx[i])
            status, obj_value, px_value = calculate_optimal_distribution(Q, M, Azx)
            
            stop = time.time()

            if status == "optimal":
                print("iter {}".format(it))
                print("    given input distribution {}".format(Px))
                print("    optimal quantizer {}".format(S[opt_H]))
                print("    optimal input distribution {}".format(px_value))
                print("    optimal I(X;Z) {}".format(obj_value))
                print("    took {:.4f}s".format(stop-start))
                with open(log_fn, "a") as f:
                    f.writelines("iter {}\n".format(it))
                    f.writelines("    given input distribution {}\n".format(Px))
                    f.writelines("    optimal quantizer {}\n".format(S[opt_H]))
                    f.writelines("    optimal input distribution {}\n".format(px_value))
                    f.writelines("    optimal I(X;Z) {}\n".format(obj_value))
                    f.writelines("    took {:.4f}s\n".format(stop-start))
                Px = px_value
            else:
                print("cvxopt failed")
                with open(log_fn, "a") as f:
                    f.writelines("cvxopt failed")
                break
                
            if abs((obj_value - I_prev)/I_prev) <= args.stop_thres:
                print("stopping criterion met")
                with open(log_fn, "a") as f:
                    f.writelines("stopping criterion met")
                break
            I_prev = obj_value