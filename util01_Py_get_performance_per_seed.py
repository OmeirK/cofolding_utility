import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--metric_tsv', '-m')

args = parser.parse_args()

def parse_metrics(metric_df):
    data = {}

    for i, target in enumerate(metric_df['target']):
        method = metric_df['method'].iloc[i]
        if method not in data:
            data[method] = {}


        if metric_df['is_proper'].iloc[i] == False:
            continue

        try:
            xx = float(metric_df['lig_rmsd'].iloc[i])
        except:
            continue

        if target not in data[method]:
            data[method][target] = {}

        seed = metric_df['seed'].iloc[i]
        
        if seed not in data[method][target]:
            data[method][target][seed] = []

        is_succ = metric_df['is_succ'].iloc[i]

        data[method][target][seed].append(is_succ)

    return data

def check_success_rates(data):
    for m in data:
        seed_succ_data = []
        succ_model_data = np.zeros(26)

        for t in data[m]:
            print(t)
            n_succ_seed = 0
            n_succ_models = 0
            for s in data[m][t]:
                n_true = data[m][t][s].count(True)
                if n_true > 0:
                    n_succ_seed += 1

                print('\t', s, n_true)
                n_succ_models += n_true

            succ_model_data[n_succ_models] += 1
            seed_succ_data.append(n_succ_seed)
            print('\t', n_succ_seed, n_succ_models)

        print(succ_model_data, len(succ_model_data))

        fig, ax = plt.subplots(dpi=300, tight_layout=True)

        x_pos = np.arange(0,26)
        ax.bar(x_pos, succ_model_data)
        ax.set_xlabel('Number of Accurate Models\n(ligRMSD ≤ 2.0, LDDT-PLI ≥ 0.8)')
        ax.set_ylabel(f'Number of Targets')
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_pos, fontsize=8)

        plt.savefig(f'png-bar_succ_models_per_target_{m}.png')

def bootstrap_succ_rate(bool_list, n_bootstrap=1000):
    
    bool_list = np.array(bool_list)
    bootstrap_means = []
    for n in range(n_bootstrap):
        bootstrap_sample = np.random.choice(
                bool_list.astype(float), size=len(bool_list), replace=True)
        
        mean_val = np.mean(bootstrap_sample)
        bootstrap_means.append(mean_val)

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    #succ_rate = bool_list.Count(True)/len(bool_list)
    succ_rate = np.mean(bool_list.astype(float))

    return succ_rate, ci_lower, ci_upper

def main():

    df = pd.read_csv(args.metric_tsv, delimiter='\t')
    data = parse_metrics(df)
    check_success_rates(data)

if __name__=='__main__':
    main()
