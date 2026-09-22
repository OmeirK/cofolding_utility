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
import multiprocessing as mp
from pymol import cmd, stored
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import rdFMCS, ChemicalFeatures
#from calc_sucos_mod import main as calc_sucos

parser = argparse.ArgumentParser()

parser.add_argument('--ref_rec', '-r', help='Path to a reference .pdb file for a fragment screening hits. All OF3 predictions will be aligned to this structure.')
parser.add_argument('--fragment_sdf', '-fsdf', help='Path to an sdf with the subset of fragment structures to use for mcs_cov calculations. NOTE: all structures should be prealigned to --ref_rec.', default=None, required=True)
parser.add_argument('--of3_results_dir', '-of3_r', help='Path to the directory containing OF3 predictions.')
parser.add_argument('--outdir', '-o', help='Path to directory to store mcs_scoring_outputs (default = mcs-rmsd_score/', default='mcs-rmsd_score/')
parser.add_argument('--tmpdir', '-tmp', help='Path to directory to store temporary files (default = tmpmols/', default='tmpmols/')
parser.add_argument('--cpu_count', '-cpu', help='(Optional) Specify how many CPUs to use for parallelization. Leave blank to use all available CPUs', default=None)
parser.add_argument('--max_mcs_rmsd', '-rmsd', help='(Optional) Specify a maximum RMSD cutoff for MCS coverage calculations, in Angstroms. (Dfeault = 1.0)', default=1.0, type=float)

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
    # Calculate MCS coverage scores, and save output metrics
    mcs_score_data = {}
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
        #print(f'\t{m1_name} low-RMSD MCS coverage: {len(unique_m1_ats)}/{m1_size}')
        accurate_mcs_coverage = len(unique_m1_ats)/m1_size

        mcs_score_data[m1_name] = {'mcs_coverage': accurate_mcs_coverage,
                                   'n_low_rmsd_mcs_atoms': len(unique_m1_ats),
                                   'mol_size': m1_size
                                  }
    
    return mcs_cov_data, mcs_score_data

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


# Align each model to see if the proteins structure is ok
# Check if the predicted ligands overlap with the fragment ensemble
# If both are true, then the model can be advanced to MCS calculation
def check_frag_alignment(m_cif, m_ligs, frag_ensemble, ref_rec_pdb, max_rmsd=3.0, tmpdir='tmp/'):
    valid_models = []
    invalid_models = []
    err_log = []

    cmd.reinitialize()
    cmd.load(ref_rec_pdb, 'ref_rec')
    cmd.load(frag_ensemble, 'frag_ensemble')

    cmd.remove('elem H') # No H in references

    #for model in of3_seed_data:
    #    m_cif = of3_seed_data[model]['cif']
    #    m_ligs = of3_seed_data[model]['sdfs']
        
    cmd.load(m_cif, 'mdl_rec')
    rmsd = cmd.align('mdl_rec', 'ref_rec')
    rmsd = rmsd[0]
    
    #print(rmsd, max_rmsd)
    if rmsd > max_rmsd: 
        print(f'\tFAILED Alignment for Model {m_cif} (RMSD = {rmsd})')
        err_log.append(f'\tRECEPTOR_ALIGN_FAIL {m_cif} (RMSD = {rmsd})')
        cmd.delete('mdl_rec')
        return valid_models, invalid_models, err_log
    
    #print(m_ligs)
    # Check if the ligand(s) superimpose
    for i, lig in enumerate(m_ligs):
        cmd.load(lig, f'mdl_lig-{i}')
        cmd.matrix_copy('mdl_rec', f'mdl_lig-{i}') # Transpose the ligand

        n_ov = cmd.count_atoms(f'mdl_lig-{i} within 1.0 of frag_ensemble')

        lig_n = os.path.basename(lig)
        out_sdf = f'{tmpdir}/{lig_n}'
        #print(out_sdf, n_ov)
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

def extract_sdfs_from_cif(cif_file, out_path, fragalysis_dir=None):

    m_name = os.path.basename(cif_file).strip('.cif') # Remove .cif ext
    case = m_name.split('_')[0]
    cmd.reinitialize()
    cmd.load(cif_file)

    stored.lig_data = []
    cmd.iterate('hetatm', 'stored.lig_data.append("_".join([resn.split("_")[0], resi, chain]))')
    lig_data = list(set(stored.lig_data))

    lig_sdf_l = []
    for info in lig_data:
        lign, ligi, lc = info.split('_')

        if os.path.exists(f'{out_path}/{m_name}_{lc}-lig.sdf'):
            os.remove(f'{out_path}/{m_name}_{lc}-lig.sdf')
        
        lig_sdf = f'{out_path}/{m_name}_{lign}-{ligi}-{lc}-lig.sdf'
        cmd.save(lig_sdf, f'chain {lc}')


        # Check if there are any "aromatic" bond types in the molecule
        # If so, replace them with the kekulized form. (OST will fail otherwise)
        # Try to kekulize with fragalysis ligand first
        # Otherwise, try RDKit keulize
        mol = Chem.MolFromMolFile(lig_sdf)
        is_aromatic = False

        try:
            for bond in mol.GetBonds():
                if bond.GetIsAromatic():
                    is_aromatic = True
                    break
        except:
            pass

        if (is_aromatic) or (mol is None):
            if fragalysis_dir != None:
                # Try to assign bond orders from a fragalysis ligand template
                ref_lig = f'{fragalysis_dir}/{case}/{case}_ligand.sdf'
                ref_mol = Chem.MolFromMolFile(ref_lig)
                ref_smi = Chem.MolToSmiles(ref_mol, kekuleSmiles=True)
                template = AllChem.MolFromSmiles(ref_smi)
        
                cmd.save(f'{out_path}/tmp_lig.pdb', f'chain {lc}')
                docked_pdb = Chem.MolFromPDBFile(f'{out_path}/tmp_lig.pdb')
                try:
                    new_mol = AllChem.AssignBondOrdersFromTemplate(template, docked_pdb)
                except:
                    print(f'\tERR_AssignBondOrders failed: {lig_sdf}')
                    new_mol == None

                os.remove(f'{result_dir}/tmp_lig.pdb')
            
                if new_mol is not None:
                    print(f'\tAssigned bond order from template {ref_lig}')
                    print(f'\t\tSuccessfully fixed {lig_sdf}')
                    Chem.MolToMolFile(new_mol, lig_sdf)
                else:
                    # Try RDKit kekulize as a last resort 
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    #print('\tKekulize:', lig_sdf, mol) #Debug
                    Chem.MolToMolFile(mol, lig_sdf)
            else:
                # Try RDKit kekulize as a last resort, or if no fragalysis
                # directory is provided
                Chem.Kekulize(mol, clearAromaticFlags=True)
                #print('\tKekulize:', lig_sdf, mol) #Debug
                Chem.MolToMolFile(mol, lig_sdf)

        lig_sdf_l.append(lig_sdf)


        #

    return lig_sdf_l

# Get average iptm between a ligand and non-ligand chains
def get_model_iptm(conf_json, lig_sdf_l):
    with open(conf_json) as f:
        conf_data = json.load(f)
    
    ch_to_sdf_map = {}
    lig_chains = []
    for lig in lig_sdf_l:
        lig_inf = lig.split('_')[-1]
        lig_ch = lig_inf.split('-')[2]
        lig_chains.append(lig_ch)

        ch_to_sdf_map[lig_ch] = os.path.basename(lig)

    iptm_data = {}
    for ch_pair in conf_data["chain_pair_iptm"]:
        ch_pair_str = ch_pair[1:-1]
        ch_pair_l1 = ch_pair_str.split(',')
        
        iptm = conf_data["chain_pair_iptm"][ch_pair]
        
        valid_ligs = []
        for ch1 in ch_pair_l1:
            ch = ch1.strip()
            if ch in lig_chains:
                valid_ligs.append(ch)

        if len(valid_ligs) == 1:
            lc = valid_ligs[0]
            if lc not in iptm_data:
                iptm_data[lc] = []

            iptm_data[lc].append(iptm)

    out_data = {}
    for lc in iptm_data:
        sdf_n = ch_to_sdf_map[lc]
        out_data[sdf_n] = np.average(iptm_data[lc])


    return out_data  




def mp_func(mp_inp):
    #(case, fragment_ensemble, frag_mols, frag_pharm_pos_data
    case_name = mp_inp[0]
    fragment_ensemble = mp_inp[1]
    frag_pharm_pos_data = mp_inp[2]

    # Read fragment data
    suppl = Chem.SDMolSupplier(args.fragment_sdf)
    frag_mols = []
    for m in suppl:
        if m is not None:
            frag_mols.append(m)
    
    all_mcs_cov_data = {}
    err_out = []
    all_out_data = []
    for case in [case_name]:
        
        if case not in all_mcs_cov_data:
            all_mcs_cov_data[case] = {}

        for seed in os.listdir(f'{args.of3_results_dir}/{case}'):
            #print('\t', seed)
            for sample in range(1,6):
                model_path = f'{args.of3_results_dir}/{case}/{seed}/{case}_{seed}_sample_{sample}_model.cif'
                confidence_json = f'{args.of3_results_dir}/{case}/{seed}/{case}_{seed}_sample_{sample}_confidences_aggregated.json'
                if os.path.exists(model_path):
                    #print(f'\t{model_path}')
                    model_ligs = extract_sdfs_from_cif(model_path, args.tmpdir, fragalysis_dir=None)
                    
                    iptm_data = get_model_iptm(confidence_json, model_ligs)

                    aligned_models, invalid_models, errs = check_frag_alignment(model_path, model_ligs, fragment_ensemble, args.ref_rec, tmpdir=args.tmpdir)
                    if len(errs) > 0:
                        for l in errs:
                            l += f' ({case} {seed})'
                            err_out.append(l)
                    
                    #print(f'Invalid:', invalid_models)
                    # Load and annotate aligned ligands models
                    aligned_mols = []
                    for msdf in aligned_models:
                        ligid = os.path.basename(msdf).split('_')[-1]
                        ligid = ligid.split('.')[0]

                        m_mol = Chem.MolFromMolFile(msdf)
                        mol_smi = Chem.MolToSmiles(m_mol)
                        m_mol.SetProp('path', msdf)
                        m_mol.SetProp('_Name', f'{case}.{seed}.{sample}.{ligid}')
                        m_mol.SetProp('smi', mol_smi)
                        aligned_mols.append(m_mol)

                    # Calculate color feature overlaps
                    color_score_data = get_color_overlap(frag_pharm_pos_data, aligned_mols)

                    # Calculate MCS RMSD metrics for each aligned cofolded molecule
                    mcs_cov_data, mcs_score_data = get_mcs_cov(frag_mols, aligned_mols, max_rmsd=args.max_mcs_rmsd)
                    #out_data.append(f'{m1_name}\t{accurate_mcs_coverage}\t{len(unique_m1_ats)}\t{m1_size}')

                    # Append color features to the output
                    outlines = []
                    for m_name in mcs_score_data:
                        ligid = m_name.split('_')[-1]
                        ligid = ligid.split('.')[0]
                        
                        mcs_coverage = mcs_score_data[m_name]["mcs_coverage"]
                        color_overlap = color_score_data[m_name]["total"]

                        mcs_color_avg = np.average([mcs_coverage, color_overlap])
                        mcs_color_prod = mcs_coverage*color_overlap

                        outstr = f'{case}\t{seed}\t{sample}\t{ligid}\t{iptm_data[m_name]}\t{mcs_coverage}\t{color_overlap}\t{mcs_color_avg}\t{mcs_color_prod}\t{mcs_score_data[m_name]["mol_size"]}'
                        outlines.append(outstr)


                    #for l in out_data:
                        #m_name = l.split('\t')[0]
                        #l += f'\t{color_score_data[m_name]["total"]}\t{iptm_data[m_name]}'
                        #print(m_name, iptm_data[m_name])

                        #outlines.append(l)
                    
                    all_mcs_cov_data[case][seed] = mcs_cov_data


                    all_out_data += outlines
                    
                    # Deleteligand sdf files
                    for msdf in model_ligs:
                        os.remove(msdf)
                        
    return all_out_data, all_mcs_cov_data, err_out

def main():
    os.makedirs(args.outdir, exist_ok=True)

    suppl = Chem.SDMolSupplier(args.fragment_sdf)
    frag_mols = []
    for m in suppl:
        if m is not None:
            frag_mols.append(m)
    
    # Get pharmacophore atom data for fragment ensemble
    frag_pharm_pos_data = detect_pharmacophore_atoms(frag_mols)

    # Save a fragment ensemble mol file
    fragment_ensemble = f'{args.tmpdir}/fragment_ensemble.mol'
    if os.path.exists(fragment_ensemble) == False:
        cmd.reinitialize()
        cmd.load(args.fragment_sdf, 'frags')
        cmd.split_states('frags')
        cmd.delete('frags')
        cmd.save(fragment_ensemble)
    

    mp_inps = []
    for case in os.listdir(args.of3_results_dir):
        if (os.path.isdir(f'{args.of3_results_dir}/{case}') == False)  or (case == 'logs'):
            continue

        mp_inps.append((case, fragment_ensemble, frag_pharm_pos_data))

    # target   seed  sample   lig_id   pair_iptm   mcs_overlap color_overlap   mcs_color_avg  mcs_color_prod mol_size
    #all_out_data = [f'mol_name\tlow_rmsd_mcs_coverage\tn_low_rmsd_mcs_atoms\tmol_size\tcolor_overlap\tpair_iptm']
    all_out_data = [f'target\tseed\tsample\tlig_id\tpair_iptm\tmcs_overlap\tcolor_overlap\tmcs_color_avg\tmcs_color_prod\tmol_size']
    all_mcs_cov_data = {}
    err_out_all = []

    
    # Quick multiprocessing implementation
    print(f'Get MCS for {len(mp_inps)} inputs')
    if args.cpu_count == None:
        with mp.Pool(mp.cpu_count()) as pool:
            combined_results = list(tqdm.tqdm(pool.imap(mp_func, mp_inps, chunksize=1)))
    else:
        with mp.Pool(int(args.cpu_count)) as pool: 
            combined_results = list(tqdm.tqdm(pool.imap(mp_func, mp_inps, chunksize=1)))
        
        #print(combined_results[0][1])
        #print(combined_results[1][1])
        #print(combined_results[0][1])
        #print(combined_results[0][2])
        #print(len(combined_results[0][1]), len(combined_results[1][1]))
        
    for i in range(len(combined_results)):
        all_out_data += combined_results[i][0]
        all_mcs_cov_data.update(combined_results[i][1])
        err_out_all += combined_results[i][2]
        
    with open(f'{args.outdir}/tsv_frag_coverage.tsv', 'w') as fo:
        fo.write('\n'.join(all_out_data))
    
    with open(f'{args.outdir}/json_frag_coverage_info.json', 'w') as fo:
        json.dump(all_mcs_cov_data, fo, indent=4)
    
    with open(f'{args.outdir}/error_log.err', 'w') as fo:
        fo.write('\n'.join(err_out_all))

    shutil.rmtree(args.tmpdir)


if __name__=='__main__':
    main()
