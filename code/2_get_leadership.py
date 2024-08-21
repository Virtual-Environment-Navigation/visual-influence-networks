'''

CI values are too computationally heavy, so I ran this script using supercomputer for each trial/segment (we used Oscar:(https://docs.ccv.brown.edu/oscar).

'''

##  init 
import numpy as np
import pickle
import time
import multiprocessing

from utils.get_leadership import get_NI, get_NBI, get_CI, get_CBI, get_rank

## ====== change ONLY HERE ============
prune_type = 'pruned'       # unpruned or pruned
ntwk_window_size = 0.5

# specify trials
trials = [7]

# whether individual segments should be saved
div_segs = False     # useful for trial 7
# define segs to be used if div_segs == True
segs = [7] #[1,2,5]
## ====================================

if __name__ == "__main__":
    ## init
    ntwk_window_size_ms = int(ntwk_window_size * 1000)
    if prune_type=='unpruned':
        weights = pickle.load( open(f'../data/pickle/sayles_weights_unpruned_{ntwk_window_size_ms}ms.p', 'rb') )
    elif prune_type=='pruned':
        weights = pickle.load( open(f'../data/pickle/sayles_weights_pruned_{ntwk_window_size_ms}ms.p', 'rb') )
    else:
        raise ValueError('prune type error')

    measure_types = ['NI', 'NBI', 'CI', 'CBI']

    print('Computing leadership measure types', measure_types)
    print('for', prune_type, 'networks in trials', trials, '(network window size:', ntwk_window_size, 's)\n')
    start_time = time.time()

    for trial in trials:
        print('trial', int(trial))
        leadership = {}
        for measure_type in measure_types:
            leadership[measure_type] = {}
            leadership[measure_type+'rank'] = {}

        if ~div_segs:
            segs = weights[trial].keys()

        for seg in segs:
            print('-- segment', seg)
            num_networks, N, _ = weights[trial][seg].shape    # (num_networks, N, N)

            for measure_type in measure_types:
                print('----', measure_type, end=':')
                leadership[measure_type][seg] = np.zeros((num_networks, N))
                leadership[measure_type+'rank'][seg] = np.zeros((num_networks, N))
                if measure_type=='NI':
                    get_leadership = get_NI
                elif measure_type=='NBI':
                    get_leadership = get_NBI
                elif measure_type=='CI':
                    get_leadership = get_CI
                elif measure_type=='CBI':
                    get_leadership = get_CBI
                else:
                    raise ValueError('measure type error')

                # --- get leadership ---
                # create a pool of worker processes
                pool = multiprocessing.Pool()
                # compute leadership values in parallel using multiprocessing.Pool.imap()
                results = pool.imap(get_leadership, [(weights[trial][seg][ntwk,:,:]) for ntwk in range(num_networks)])
                # Collect results in the correct order
                for idx, result in enumerate(results):
                    leadership[measure_type][seg][idx,:] = result
                # Close the pool to release resources
                pool.close()
                pool.join()

                # --- get ranking ---
                for ntwk in range(num_networks):
                    print(ntwk+1, end=' ')
                    # leadership[measure_type][seg][ntwk,:] = get_leadership(weights[trial][seg][ntwk,:,:], order="ij")  # (N,)
                    leadership[measure_type+'rank'][seg][ntwk,:] = get_rank(leadership[measure_type][seg][ntwk,:])  # (N,)

                print(end='\n')

        if ~div_segs:
            filename = f'sayles_{prune_type}_trial{trial}_{ntwk_window_size_ms}ms_newnew.p'
            # e.g., leadership['NIrank'][seg][ntwk,i]
            pickle.dump(leadership, open(f'../data/sayles_leadership_Oscar/{filename}', 'wb'))
            print(filename, 'saved')

    end_time = time.time()
    print(f"\nall saved! (time: {round(end_time - start_time)}s)")