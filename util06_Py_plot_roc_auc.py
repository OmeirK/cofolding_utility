import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser()

parser.add_argument('--score_tsv', '-s', help='A .tsv file with pair_iptm, mcs_overlap, and color_overlap scores (generated via util04_Py_extract_performance_metrics.py)')
parser.add_argument('--outfile', '-o', help='Name of the output ROC-AUC curve plot png')

args = parser.parse_args()

# For each target, select the best model by a given metric
def get_df_subset(df, filter_metric, exclude_list=[]):
    target_l = list(set(df['target']))
    
    out_data = {'vals': [], 'is_binder': []}
    for target in target_l:
        if target in exclude_list:
            continue


        t_df = df.loc[df['target'] == target]
        idx = t_df[filter_metric].idxmax()
        
        val = df[filter_metric].iloc[idx]
        is_binder = df['is_binder'].iloc[idx]
        #test = df['color_overlap'].iloc[idx]
        #print(t_df)
        #print(test, filter_metric, idx, is_binder)
        out_data['vals'].append(val)
        out_data['is_binder'].append(is_binder)


    return out_data


def main():
    df = pd.read_csv(args.score_tsv, delimiter='\t')

    #get_df_subset(df, 'mcs_color_avg')
    #return
            

    fig, ax = plt.subplots(figsize=(7, 5))
    
    metric_l = 'pair_iptm   mcs_overlap color_overlap  mcs_color_avg  mcs_color_prod'.split()

    for metric in metric_l:
        plot_data = get_df_subset(df, metric)
        fpr, tpr, _ = roc_curve(plot_data['is_binder'], plot_data['vals'])
        roc_auc = auc(fpr, tpr)
        print(metric, roc_auc)
        ax.plot(fpr, tpr, label=f'{metric} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()

    plt.savefig(args.outfile)


if __name__=='__main__':
    main()
