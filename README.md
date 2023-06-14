# Decongestion by Representation

Code for *"Decongestion by Representation: Learning
to Improve Economic Welfare in Marketplaces"*. 

First, make sure you have files as follows:
```
├── real_data
│   ├── create_movielens_data.py
│   ├── utils.py
│   ├── exp_run_sbatch.py
│   ├── py-sbatch.sh
│   ├── full_framework.py
│   ├── analyze_results.ipynb
│   ├── example_full_framework.ipynb
│   ├── example_full_framework.html
│   ├── example_data
│   │   ├── movielens__items_seed0__d12.pkl
├── synthetic
│   ├── synthetic_experiments.ipynb
├── env.yml
└── README.md
```

Create conda environment: <br>
`conda env create -f env.yml`

## Synthetic
Code is in a single notebook `synthetic_experiments.ipynb` located in `synthetic`.

## Real data
For real data (*Movielens*) experiments (all experiments):
1. Locate yourself at `real_data`
2. Create data - `python create_movielens_data.py`
3. Run all experiments (parallel run via sbatch) - `python exp_run_sbatch.py`   
4. Use `analyze_results.ipynb` to extract the final plots

For running a single experiment, replace step 3 with running `python full_framework.py <args>` with the relevant arguments (see arguments list by `python full_framework.py -h`). 

Alternatively, a running example for a single experiment of the full framework can be found in `example_full_framework.ipynb` (or in a static HTML version: `example_full_framework.html`).