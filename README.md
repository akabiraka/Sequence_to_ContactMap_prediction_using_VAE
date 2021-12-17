# Protein Sequence to Contact-map Prediction useing Varietional-Autoencoder.

## Overview of directories
1. "codes/" contains all codes to run this project.
2. "contact_maps/, distance_matrices/, fastas/" contain ground truth contact maps, distance matrices and fasta file for each project id.
3. "features/" contains 1-hot encoding computed from the protein sequence.
4. "output_images/" contains loss images, roc_curve images, ground truth and predicted contact maps images.
5. "pdbs/" contains downloaded protein information from protein data bank (PDB) using Bio Python.

## How to run this project?
First browse into "codes/" directory. 

```
1. cd project_path/codes
2. python ready_dataset
3. python full_run
```
To run in GMU-argo-clusters, use the following command:
```
1. Go into argo
2. Copy the project directory into /scratch/your_username/
3. cd /scratch/your_username/project_path/codes
4. module add cuda10.1
5. python ready_dataset
6. sbatch job.sh
```

"python ready_dataset" does three things:
1. Download protein data for each pdb_id from protein data bank and compute ground truth distance-matrix and contact-map for come specific threshold.
2. Each protein sequence is converted into 1-hot encoding to save the time while training.
3. Divide the dataset into train, test and validation set of some specific percentage.
