import os
import glob
import json
import argparse
from pymol import cmd, stored
from rdkit import Chem
from rdkit.Chem import AllChem

parser = argparse.ArgumentParser()
parser.add_argument('--of3_results', '-r', help='Directory where results in OF3 format are stored, and updated ligand .sdf files will be saved')
parser.add_argument('--fragalysis_dir', '-fd', help='Directory where aligned/ fragalysis results are stored. This path is used to fix bond orders from model ligands if they are unavailable', default=None)

seeds = [1370180479, 1449838082, 1832854922, 1880307061, 2012026466]
args = parser.parse_args()

def main():
    case_l = []
    
    for ff in os.listdir(args.of3_results):
        if os.path.isdir(os.path.join(args.of3_results,ff)):
            case_l.append(ff)
            #print(ff)



    for case in case_l:
            
        for s in seeds:
            result_dir = f'{args.of3_results}/{case}/seed_{s}/'
            print(case, os.path.exists(result_dir), result_dir)

            models = glob.glob(f'{result_dir}/*_model.cif')

            for m in models:
                m_name = os.path.basename(m).strip('.cif') # Remove .cif ext
                cmd.reinitialize()
                cmd.load(m)

                stored.lig_data = []
                #cmd.iterate('hetatm', 'stored.lig_data.append("_".join([resn, resi, chain]))')
                cmd.iterate('hetatm', 'stored.lig_data.append("_".join([resn.split("_")[0], resi, chain]))') # Edit to fix AF3 error
                
                lig_data = list(set(stored.lig_data))
                
                print(lig_data)

                #for lc in lig_ch_l:
                for info in lig_data:
                    lign, ligi, lc = info.split('_')

                    if os.path.exists(f'{result_dir}/{m_name}_{lc}-lig.sdf'):
                        os.remove(f'{result_dir}/{m_name}_{lc}-lig.sdf')

                    lig_sdf = f'{result_dir}/{m_name}_{lign}-{ligi}-{lc}-lig.sdf'

                    print('\t', m, lign, lc, cmd.count_atoms(f'chain {lc}')) # Debug
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
                        if args.fragalysis_dir != None:
                            # Try to assign bond orders from a fragalysis ligand template
                            ref_lig = f'{args.fragalysis_dir}/{case}/{case}_ligand.sdf'
                            ref_mol = Chem.MolFromMolFile(ref_lig)
                            ref_smi = Chem.MolToSmiles(ref_mol, kekuleSmiles=True)
                            template = AllChem.MolFromSmiles(ref_smi)
                    
                            cmd.save(f'{result_dir}/tmp_lig.pdb', f'chain {lc}')
                            docked_pdb = Chem.MolFromPDBFile(f'{result_dir}/tmp_lig.pdb')
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
                                print('\tKekulize:', lig_sdf, mol) #Debug
                                Chem.MolToMolFile(mol, lig_sdf)
                        else:
                            # Try RDKit kekulize as a last resort, or if no fragalysis
                            # directory is provided
                            Chem.Kekulize(mol, clearAromaticFlags=True)
                            print('\tKekulize:', lig_sdf, mol) #Debug
                            Chem.MolToMolFile(mol, lig_sdf)


                rec_pdb = f'{result_dir}/{m_name}_rec.pdb'
                cmd.save(rec_pdb, 'polymer.protein')
                

if __name__=='__main__':
    main()
