# Giant panda estrous-stimulation multi-omics analysis code

This private repository contains the key analysis scripts used for the study of fecal volatile and semi-volatile metabolomic responses of male giant pandas exposed to estrous female stimulation, with an accompanying comparative-genomics figure script.

## Included workflows

- `run_two_factor_metabolomics.py`: feature-wise two-factor models for stimulation condition, natural mating ability, and their interaction.
- `run_paired_interaction_analysis.py`: paired within-individual response and sensitivity analyses.
- `reproduce_us_ai_vs_st_plsda.py`: PLS-DA and permutation-based assessment for the AI group.
- `reproduce_us_nm_vs_st_plsda.py`: PLS-DA and permutation-based assessment for the NM group.
- `plot_panda_reproduction_evidence.py`: evidence-summary figure for the comparative-genomics analysis of reproduction-related genes.

## Data access and use

The GC-MS raw data, processed metabolite feature table, and study metadata are deposited in MetaboLights under accession `MTBLS15202`. The study is private during peer review and is scheduled for public release on 30 July 2027. This repository deliberately contains no raw data, processed abundance matrices, sample-identifying information, intermediate outputs, reviewer credentials, or publication drafts.

The scripts retain their analysis settings and input-column expectations. Before reuse, place the relevant tables in the input locations indicated within each script, or adapt the paths to a local project layout. Results may depend on the supplied metadata, package versions, random seeds, and the frozen input tables.

## Software environment

Python 3.12.13 was used with `pandas`, `NumPy`, `SciPy`, `statsmodels`, and `Matplotlib`. Comparative-genomics figure preparation additionally uses the same Python stack.

## Status

This repository is private while the manuscript is under review. It is intended for editorial and reviewer access and will be made public with the associated study materials after publication or the scheduled data release.
