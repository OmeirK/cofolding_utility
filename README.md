# Overlap Scoring for OF3 models

### Postprocess the OF3 results
Extract ligands from the OF3 model cifs and ensure that the protonation is consistent witht he ground-truth structures in fragalysis.

```
python3 util01_Py_extract_of3_ligand_sdfs.py -r=example_results/ -fd=sample_fragalysis_data/
```

### Calculate overlap score for OF3 results
Align OF3 models to a reference structure, and calcualte MCS+color overlap wiht respect to the ensemble of fragment/template ligands that bind to the target.

```
python3 util03_Py_fragment_rmsd_eval.py -r=ref_rec_A71EV2A-x0450a.pdb -fsdf=sample_fragalysis_data/fragment_ligands.sdf -of3_r=example_results/ -o=mcs-rmsd_score_output/
```
