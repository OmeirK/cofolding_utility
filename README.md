# Overlap Scoring for OF3 models

### Create the conda env:
```
mamba env create -f environment.yaml 
mamba activate mcs_scoring
```

### Calculate MCS+Color overlap score for OF3 results
Align OF3 models to a reference structure, and calcualte MCS+color overlap wiht respect to the ensemble of fragment/template ligands that bind to the target.

```
python3 util03_Py_fragment_rmsd_eval.py -r=ref_rec_A71EV2A-x0450a.pdb -fsdf=sample_fragalysis_data/fragment_ligands.sdf -of3_r=example_results/ -o=mcs-rmsd_score_output/
```

Optionally, provide the number of CPUs you want to use for the MCS calculation with the `--cpu_count` flag. This will parallelize the calculation, and speed up the process if you can use multiple CPUs.

Output scores can be viewed in `mcs-rmsd_score_output/tsv_frag_coverage.tsv`
