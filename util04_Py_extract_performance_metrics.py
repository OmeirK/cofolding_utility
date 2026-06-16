import pandas as pd
import numpy as np
import argparse
import tqdm
import json
import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument('--results', '-r', help='Directory where OF3 results are stored')
parser.add_argument('--mcs_tsv', '-t', help='.tsv file with MCS-RMSD and color overlap metrics')
parser.add_argument('--outfile', '-o', help='Name of output .tsv file with pair-iptm and overlap metrics compiled')

args = parser.parse_args()

def parse_mcs_df(mcs_tsv):
    df = pd.read_csv(mcs_tsv, delimiter='\t')
    
    out_data = {}
    for i, mol_name in enumerate(df['mol_name']):
        mcs_cov = df['low_rmsd_mcs_coverage'].iloc[i]
        color_ov = df['color_overlap'].iloc[i]
        
        name_data = mol_name.split('_')
        case = name_data[0]
        seed = int(name_data[2])
        sample = int(name_data[4])
        ligid = name_data[-1]
        lig_ch = ligid.split('-')[2]

        if case not in out_data:
            out_data[case] = {}
        if seed not in out_data:
            out_data[case][seed] = {}
        if sample not in out_data[case][seed]:
            out_data[case][seed][sample] = {}
        if lig_ch not in out_data[case][seed][sample]:
            out_data[case][seed][sample][lig_ch] = {}

        out_data[case][seed][sample][lig_ch] = {'color_ov': color_ov,
                                                'mcs_ov': mcs_cov }
        
        #print(mol_name, case, seed, ligid, lig_ch, sample, mcs_cov, color_ov)

    return out_data

# Get average pair iptm between ligand and non-ligand chains
# for each ligand 
def read_confidence_scores(conf_json, lig_chains):
    with open(conf_json) as f:
        data = json.load(f)
    
    iptm_data = {}
    for lc in lig_chains:
        iptm_data[lc] = []

    for pair in data['chain_pair_iptm']:
        tmp = pair[1:-1].split(',')
        n_ligs = 0
        curr_lc = ''
        for t in tmp:
            ch = t.strip()
            if ch in lig_chains:
                n_ligs += 1
                curr_lc = ch

        if n_ligs == 1:
            #print('\t', pair, data['chain_pair_iptm'][pair], curr_lc)
            iptm_data[curr_lc].append(data['chain_pair_iptm'][pair])
        
    for lc in iptm_data:
        iptm_data[lc] = np.average(iptm_data[lc])
        #print(lc, iptm_data[lc])

    return iptm_data

def main():
    mcs_data = parse_mcs_df(args.mcs_tsv)

    outlines = ['target\tseed\tsample\tlig_id\tpair_iptm\tmcs_overlap\tcolor_overlap\tmcs_color_avg\tmcs_color_prod']
    case_l = os.listdir(args.results)
    for case in tqdm.tqdm(case_l):
        case_dir = f'{args.results}/{case}/'
        if os.path.isdir(case_dir) == False:
            continue

        for seed in os.listdir(case_dir):
            seed_dir = f'{case_dir}/{seed}/'
            seed_id = int(seed.split('_')[1])

            conf_json_l = glob.glob(f'{seed_dir}/*confidences_aggregated.json')

            for cj in conf_json_l:
                sample = int(os.path.basename(cj).split('_')[4])
                lig_sdfs = glob.glob(f'{seed_dir}/{case}_{seed}_sample_{sample}*.sdf')
                # Get lig chains for LIG residues
                lig_chain_l = []
                relevant_ligs = []
                for lsdf in lig_sdfs:
                    ligid = lsdf.split('_')[-1][:-4]
                    lig_data = ligid.split('-')
                    lig_resn = lig_data[0]
                    lig_ch = lig_data[2]
                    lig_chain_l.append(lig_ch)

                    if lig_resn.startswith('LIG'):
                        relevant_ligs.append(ligid)
                pair_iptm_data = read_confidence_scores(cj, lig_chain_l)
                
                for ligid in relevant_ligs:
                    r_lc = ligid.split('-')[2]
                    try:
                        mcs_ov = mcs_data[case][seed_id][sample][r_lc]['mcs_ov']
                        color_ov = mcs_data[case][seed_id][sample][r_lc]['color_ov']
                    except:
                        #print(f'{case} {seed} {sample} {r_lc} no mcs overlap')
                        mcs_ov = 0
                        color_ov = 0
                    
                    avg_ov = np.average([mcs_ov, color_ov])
                    prod_ov = mcs_ov*color_ov

                    #print(case, seed, sample, r_lc, mcs_ov, color_ov, avg_ov, prod_ov)
                    outlines.append(f'{case}\t{seed}\t{sample}\t{ligid}\t{pair_iptm_data[r_lc]}\t{mcs_ov}\t{color_ov}\t{avg_ov}\t{prod_ov}')
                        
    with open(args.outfile, 'w') as fo:
        fo.write('\n'.join(outlines))



if __name__=='__main__':
    main()
