'''
Align OF3 predictions to a fragalysis structure, and calculate MCS RMSD metrics
for predicted poses with respect to fragment hits

NOTE: This has only been tested for monomeric proteins
'''

import os
import json
import tqdm
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from pymol import cmd
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import rdFMCS, ChemicalFeatures
#from calc_sucos_mod import main as calc_sucos

parser = argparse.ArgumentParser()

parser.add_argument('--ref_rec', '-r', help='Path to a reference .pdb file for a fragment screening hits. All OF3 predictions will be aligned to this structure')
parser.add_argument('--fragment_sdf', '-fsdf', help='Path to an sdf with the subset of fragment structures to use for mcs_cov calculations. NOTE: all structures should be prealigned to --ref_rec.', default=None, required=True)
parser.add_argument('--of3_results_dir', '-of3_r', help='Path to the directory containing OF3 predictions. NOTE: OF3 predictions should have ligands converted to .sdf format')
parser.add_argument('--outdir', '-o', help='Path to directory to store mcs_scoring_outputs (default = mcs-rmsd_score/', default='mcs-rmsd_score/')
parser.add_argument('--tmpdir', '-tmp', help='Path to directory to store temporary files (default = tmpmols/', default='tmpmols/')

args = parser.parse_args()

os.makedirs(args.tmpdir, exist_ok=True)

def get_frag_pdbs(frag_list):
    with open(frag_list) as f:
        ll = f.readlines()
    
    frags = ll[0].split(',')
    return frags

def get_frag_paths(frag_list):
    paths = []
    with open(frag_list) as f:
        for l in f:
            paths.append(l.strip())

    return paths
            
        

# Map atoms in the target ligand to atoms in the template
def cs_sym_mappings(target_mol, template_mol, cs_smarts): # accounts for target_mol symmetry
    cs_patt = Chem.MolFromSmarts(cs_smarts)
    target_cs_matches = target_mol.GetSubstructMatches(cs_patt, uniquify=False)
    template_cs_matches = template_mol.GetSubstructMatches(cs_patt, uniquify=False)

    # Debugging #
    #print(target_cs_matches)
    #print(template_cs_matches)
    #print(Chem.MolToSmiles(target_mol), Chem.MolToSmiles(template_mol), cs_smarts)
    #print(template_mol.HasSubstructMatch(cs_patt))
    
    mappings = set()
    for target_cs_match in target_cs_matches:
        for template_cs_match in template_cs_matches:
            mapping = tuple(sorted(zip(target_cs_match, template_cs_match), key=lambda x: x[1]))
            mappings.add(mapping)
    
    mol_sym_matches = target_mol.GetSubstructMatches(target_mol, uniquify=False)
    mappings_reduced = []
    while len(mappings) > 0:
        mapping = list(mappings.pop())
        mappings_reduced.append(mapping)
        redundant_mappings = {tuple((mol_sym_match[i], j) for i, j in mapping) for mol_sym_match in mol_sym_matches}
        mappings -= redundant_mappings
    return mappings_reduced


# Find appropriate tempalte fragments using maximum common substructure
# Only consider a fragment as a template if the mapped MCS atoms have a low RMSD
def get_mcs_cov(frag_mols, all_mols, max_rmsd=1.0):
    
    mcs_cov_data = {}
    for m1 in all_mols:
        #m1_name = m1.GetProp('_Name')
        m1_name = os.path.basename(m1.GetProp('path'))

        mcs_cov_data[m1_name] = {}
        m1_size = m1.GetNumHeavyAtoms()
        for f1 in frag_mols:
            f1_name = f1.GetProp('_Name')
            f1_size = f1.GetNumHeavyAtoms()

            # Complete rings required for match
            #res=rdFMCS.FindMCS([m1, f1], bondCompare=rdFMCS.BondCompare.CompareOrderExact, completeRingsOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom)

            # Incomplete rings allowed
            res=rdFMCS.FindMCS([m1, f1], bondCompare=rdFMCS.BondCompare.CompareOrderExact, completeRingsOnly=False, atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom)
            n_mcs = res.numAtoms
            
            # Get all potential MCS mappings
            sym_mappings = cs_sym_mappings(m1, f1, res.smartsString)
            
            # Check MCS rmsd wrt the fragment structure
            # for all MCS mappings. Only keep mappings
            # that satisfy the {max_rmds} threshold
            # NOTE: Fragalysis outputs are pre-aligned!
            valid_overlap = False
            valid_mappings = []
            for mi, mapping in enumerate(sym_mappings):
                f1_mcs_pos = []
                m1_mcs_pos = []
                for i,j in mapping:
                    f1_at = f1.GetConformer().GetAtomPosition(j)
                    f1_c = [f1_at.x, f1_at.y, f1_at.z]
                    f1_mcs_pos.append(f1_c)
    
                    m1_at = m1.GetConformer().GetAtomPosition(i)
                    m1_c = [m1_at.x, m1_at.y, m1_at.z]
                    m1_mcs_pos.append(m1_c)
                
                f1_mcs_pos = np.array(f1_mcs_pos)
                m1_mcs_pos = np.array(m1_mcs_pos)
                
                rmsd = np.sqrt(((f1_mcs_pos - m1_mcs_pos)**2).sum(-1).mean())

                if rmsd < max_rmsd:
                    valid_overlap = True
                    valid_mappings.append(mapping)
            
            # Do not consider fragments with poor physical overlap
            if valid_overlap == False:
                continue
            else:
                #print(f'\tMCS physically overlaps for {f1.GetProp("_Name")}. Proceeding...')
                if f1_name not in mcs_cov_data[m1_name]:
                    mcs_cov_data[m1_name][f1_name] = {'valid_mappings': [],
                                                      'mcs_smarts': None,
                                                      'f1_size': f1_size,
                                                      'm1_size': m1_size}

                mcs_cov_data[m1_name][f1_name]['valid_mappings'] += valid_mappings
                mcs_cov_data[m1_name][f1_name]['mcs_smarts'] = res.smartsString

    # Count the number of unique low-rmsd atoms in m1
    #out_data = [f'mol_name\tlow_rmsd_mcs_coverage\tn_low_rmsd_mcs_atoms\tmol_size']
    out_data = []
    for m1_name in mcs_cov_data:
        unique_m1_ats = []
        for f1_name in mcs_cov_data[m1_name]:
            unique_f1_ats = []
            m1_size = mcs_cov_data[m1_name][f1_name]["m1_size"]
            smarts = mcs_cov_data[m1_name][f1_name]["mcs_smarts"]

            #print('\t\t', f1_name, smarts)

            for mapping in mcs_cov_data[m1_name][f1_name]['valid_mappings']:
                for m_at, f_at in mapping:
                    if m_at not in unique_m1_ats:
                        unique_m1_ats.append(m_at)

            #print(m1_name, f1_name)
            #print(f'\tf1 coverage: {len(unique_f1_ats)}/{mcs_cov_data[m1_name][f1_name]["f1_size"]}')
        print(f'\t{m1_name} low-RMSD MCS coverage: {len(unique_m1_ats)}/{m1_size}')
        accurate_mcs_coverage = len(unique_m1_ats)/m1_size
        out_data.append(f'{m1_name}\t{accurate_mcs_coverage}\t{len(unique_m1_ats)}\t{m1_size}')
                
    
    return mcs_cov_data, out_data

# Compile data for fragment screenign results
def collect_frag_structures(frag_paths):
    frag_data = {}
    frag_mols = []
    for path in frag_paths:
        pdb = path.split('/')[-1]

        if pdb not in frag_data:
            frag_data[pdb] = {'mols': [], 'rec' : None, 'sdfs': []}
        
        sdfs = glob.glob(f'{path}/{pdb}*ligand.sdf')
        rec = glob.glob(f'{path}/{pdb}_aligned.pdb')[0]
        


        if len(rec) == 0:
            print(f'WARNING: No receptor found in {path}')
        else:
            frag_data[pdb]['rec'] = rec

        for sdf in sdfs:
            mol = Chem.MolFromMolFile(sdf)
            if mol is not None:
                mol_smi = Chem.MolToSmiles(mol)
                mol.SetProp('_Name', pdb)
                mol.SetProp('smi', mol_smi)
                mol.SetProp('path', sdf)
                #frag_mols.append(mol)
                frag_data[pdb]['mols'].append(mol)
                frag_data[pdb]['sdfs'].append(sdf)
                #print(pdb, mol_smi, mol, sdf, rec)

                frag_mols.append(mol)

    return frag_data, frag_mols

# Store structure information for models generated with OF3
def read_of3_structures(of3_dir):
    of3_data = {}
    
    of3_dir = os.path.abspath(of3_dir)
    case_results = glob.glob(f'{of3_dir}/*/')

    for cr in case_results: 
        case_n = cr.split('/')[-2]
        #print(case_n, cr)

        if case_n not in of3_data:
            of3_data[case_n] = {}

        seeds = os.listdir(f'{cr}/')

        for s in seeds:
            if s not in of3_data[case_n]:
                of3_data[case_n][s] = {}
            
            # Assumes 5 models, numbered 1-5
            for i in range(1,6):
                model_path = f'{cr}/{s}/{case_n}_{s}_sample_{i}_model.cif'
                #print(model_path, os.path.exists(model_path))
                if os.path.exists(model_path):
                    model_ligs = glob.glob(f'{cr}/{s}/{case_n}_{s}_sample_{i}_*LIG*lig.sdf')
                    of3_data[case_n][s][i] = {'cif': model_path, 'sdfs': model_ligs}

    return of3_data

# Align each model to see if the proteins structure is ok
# Check if the predicted ligands overlap with the fragment ensemble
# If both are true, then the model can be advanced to MCS calculation
def check_frag_alignment(of3_seed_data, frag_ensemble, ref_rec_pdb, max_rmsd=3.0, tmpdir='tmp/'):
    valid_models = []
    invalid_models = []
    err_log = []

    cmd.reinitialize()
    cmd.load(ref_rec_pdb, 'ref_rec')
    cmd.load(frag_ensemble, 'frag_ensemble')

    cmd.remove('elem H') # No H in references

    for model in of3_seed_data:
        m_cif = of3_seed_data[model]['cif']
        m_ligs = of3_seed_data[model]['sdfs']
        
        cmd.load(m_cif, 'mdl_rec')
        rmsd = cmd.align('mdl_rec', 'ref_rec')
        rmsd = rmsd[0]
        
        #print(rmsd, max_rmsd)
        if rmsd > max_rmsd: 
            #print(f'\tFAILED Alignment for Model {model} (RMSD = {rmsd})')
            err_log.append(f'\tCONF_FAIL Alignment for Model {model} (RMSD = {rmsd})')
            cmd.delete('mdl_rec')
            continue
        
        #print(m_ligs)
        # Check if the ligand(s) superimpose
        for i, lig in enumerate(m_ligs):
            cmd.load(lig, f'mdl_lig-{i}')
            cmd.matrix_copy('mdl_rec', f'mdl_lig-{i}') # Transpose the ligand

            n_ov = cmd.count_atoms(f'mdl_lig-{i} within 1.0 of frag_ensemble')

            lig_n = os.path.basename(lig)
            out_sdf = f'{tmpdir}/{lig_n}'
            if n_ov > 0:
                cmd.save(out_sdf, f'mdl_lig-{i}')
                valid_models.append(out_sdf)
            else:
                invalid_models.append(out_sdf)
        
        cmd.delete('mdl_*')

    return valid_models, invalid_models, err_log

# For a mol object, get the 3D coordinates in list
def get_mol_coords(mol):
    conf = mol.GetConformer()
    n_ats = mol.GetNumAtoms()
    coord_arr = np.zeros((n_ats,3))
    
    for at_id in range(0,n_ats):
        at_pos = conf.GetAtomPosition(at_id)
        
        coord_arr[at_id][0] = at_pos.x
        coord_arr[at_id][1] = at_pos.y
        coord_arr[at_id][2] = at_pos.z

    return coord_arr

# Define chemical features that can be used for each molecule
# Fit RDKit featues to match E-FTMap atom types
def detect_pharmacophore_atoms(mols):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    
    pharm_data = {}
    #i = 0
    #for mol in tqdm.tqdm(mols):
    for i, mol in enumerate(mols):
        mol_coords = get_mol_coords(mol)

        # Get atom indices of pharmacophore features
        feats = factory.GetFeaturesForMol(mol)
        for j, feat in enumerate(feats):
            feat_atoms = feat.GetAtomIds()
            feat_type = feat.GetFamily()

            if feat_type not in pharm_data:
                pharm_data[feat_type] = []
            
            for at_id in feat_atoms:
                pharm_data[feat_type].append(mol_coords[at_id])
    
    return pharm_data

# Calculate the fraction of pharmacophore features for each mol
# in {aligned_mols} that overlaps with a pharmacophore feature
# observed in the fragment ensemble
def get_color_overlap(frag_pharm_coord_data, aligned_mols, overlap_dist=1.0):

    mol_scores = []
    mol_score_data = {}
    for m1 in aligned_mols:
        m1_name = os.path.basename(m1.GetProp('path'))

        m1_pharm_coords = detect_pharmacophore_atoms([m1])
        mol_score_data[m1_name] = {}

        #print(m1_name, m1.GetNumHeavyAtoms())
        tot_true = 0
        tot_feats = 0
        ov_data = {}
        for feat in m1_pharm_coords:
            ov_data[feat] = []
            
            try:
                assert frag_pharm_coord_data[feat]
            except:
                continue
            
            for m1_coord in m1_pharm_coords[feat]:
                is_ov = False

                for f1_coord in frag_pharm_coord_data[feat]:
                    dist = np.sqrt(((f1_coord - m1_coord)**2).sum(-1).mean())
                    
                    if dist <= overlap_dist:
                        is_ov = True
                
                #print('\t\t', m1_coord, is_ov)

                ov_data[feat].append(is_ov)

            n_true = ov_data[feat].count(True)
            tot_true += n_true
            tot_feats += len(ov_data[feat])

            color_ov = n_true/len(ov_data[feat])
            mol_score_data[m1_name][feat] = color_ov

            #print('\t\t',feat, n_true, color_ov)

        
        #if len(ov_data) == 0:
        #    mol_scores.append(None)
        #else:
        #    mol_scores.append(ov_data)
        
        try:
            tot_score = tot_true/tot_feats
        except:
            tot_score = None

        mol_score_data[m1_name]['total'] = tot_score
        #print(tot_score)
        #print(ov_data)

        #print(m1_name, mol_score_data[m1_name])


    return mol_score_data

def main():
    os.makedirs(args.outdir, exist_ok=True)

    # Collect fragments as baseline ligands
    #if args.fragment_list != None:
    #    frag_paths = get_frag_paths(args.fragment_list)
    #    _, frag_mols = collect_frag_structures(frag_paths)

    #if args.fragment_sdf != None:

    # Read fragment data
    suppl = Chem.SDMolSupplier(args.fragment_sdf)
    frag_mols = []
    for m in suppl:
        if m is not None:
            frag_mols.append(m)
    
    # Save a fragment ensemble mol file
    fragment_ensemble = f'{args.tmpdir}/fragment_ensemble.mol'
    if os.path.exists(fragment_ensemble) == False:
        cmd.reinitialize()
        cmd.load(args.fragment_sdf, 'frags')
        cmd.split_states('frags')
        cmd.delete('frags')
        cmd.save(fragment_ensemble)
    

    # Compile OF3 predictions and ligand mols
    of3_struct_data = read_of3_structures(args.of3_results_dir)
    
    frag_pharm_pos_data = detect_pharmacophore_atoms(frag_mols)
    #print(frag_pharm_data)

    n_cases = 3 # Debug
    n_case = 0 # Debug
    all_mcs_cov_data = {}
    err_out = []
    all_out_data = [f'mol_name\tlow_rmsd_mcs_coverage\tn_low_rmsd_mcs_atoms\tmol_size\tcolor_overlap']
    for case in tqdm.tqdm(of3_struct_data):
        #if n_case > n_cases: # Debug
        #    with open('tsv_frag_coverage.tsv', 'w') as fo: # Debug
        #        fo.write('\n'.join(all_out_data)) # Debug
        #    with open(f'json_frag_coverage_info.json', 'w') as fo: # Debug
        #        json.dump(all_mcs_cov_data, fo, indent=4) # Debug
        #    return # Debug
        #n_case += 1 # Debug

        if case not in all_mcs_cov_data:
            all_mcs_cov_data[case] = {}

        for seed in of3_struct_data[case]:
            aligned_models = glob.glob(f'{args.tmpdir}/{case}_{seed}*.sdf') # In case you rerun it/things crash
            if len(aligned_models) == 0:
                aligned_models, invalid_models, errs = check_frag_alignment(of3_struct_data[case][seed], fragment_ensemble, args.ref_rec, tmpdir=args.tmpdir)
                
                if len(errs) > 0:
                    for l in errs:
                        l += f' ({case} {seed})'
                        err_out.append(l)

                # Depreciated code to align to parent fragalysis file
                #ref_rec = f'{args.fragalysis_dir}/{case}/{case}.pdb'
                #aligned_models, invalid_models = check_frag_alignment(of3_struct_data[case][seed], fragment_ensemble, ref_rec, tmpdir=args.tmpdir)
                #print(case, seed, len(aligned_models), len(invalid_models))
            
            # Calculate MCS RMSDs for valid models
            aligned_mols = []
            for sdf in aligned_models:
                sample = os.path.basename(sdf).split(f'_{seed}_sample_')[1][0]
                #print(sdf, sample)
                mol = Chem.MolFromMolFile(sdf)
                if mol is not None:
                    mol_smi = Chem.MolToSmiles(mol)
                    mol.SetProp('path', sdf)
                    mol.SetProp('_Name', f'{case}.{seed}.{sample}')
                    mol.SetProp('smi', mol_smi)
                    aligned_mols.append(mol)
                    
                    #print('\t', sdf)
                    #sucos, color_score, shape_score = calc_sucos(sdf, args.frag_ensemble, write=False, return_all=True)

                    #print('\t', case, seed, sucos, color_score, shape_score)
            
            # Calculate color feature overlaps
            color_score_data = get_color_overlap(frag_pharm_pos_data, aligned_mols)

            # Calculate MCS RMSD metrics for each aligned cofolded molecule
            mcs_cov_data, out_data = get_mcs_cov(frag_mols, aligned_mols)

            # Append color features to the output
            outlines = []
            for l in out_data:
                m_name = l.split('\t')[0]
                
                l += f'\t{color_score_data[m_name]["total"]}'

                outlines.append(l)


            all_mcs_cov_data[case][seed] = mcs_cov_data


            #all_out_data += out_data
            all_out_data += outlines

    with open(f'{args.outdir}/tsv_frag_coverage.tsv', 'w') as fo:
        fo.write('\n'.join(all_out_data))
    
    with open(f'{args.outdir}/json_frag_coverage_info.json', 'w') as fo:
        json.dump(all_mcs_cov_data, fo, indent=4)
    
    with open(f'{args.outdir}/error_log.err', 'w') as fo:
        fo.write('\n'.join(err_out))

    # Delete the aligned OF3 ligand files
    #os.rmdir(args.tmpdir)
    shutil.rmtree(args.tmpdir)

if __name__=='__main__':
    main()
