# Linear and Nonlinear Signals
This repo contains the code to reproduce the results of the manuscript "The Effects of Nonlinear Signal on Expression-Based Prediction Performance".
In short, we compare linear and nonlinear models in multiple prediction tasks, and find that their predictive ability is roughly equivalent.
Further, this similarity is despite the fact that predictive nonlinear signal exists in the data for each of the tasks.

![model comparison figure](https://raw.githubusercontent.com/greenelab/linear_signal/master/figures/full_signal_combined.svg)

## Installation

### Python dependencies
The Python dependencies for this project are managed via [Conda](https://docs.conda.io/en/latest/miniconda.html).
To install them and activate the environment, use the following commands in bash:

``` bash
conda env create --file environment.yml
conda activate linear_models
```

### R setup
The R dependencies for this project are managed via [Renv](https://rstudio.github.io/renv/articles/renv.html).
To set up Renv for the repository, use the commands below within R while working in the `linear_signals` repo:

``` R
install.packages('renv')
renv::init()
renv::restore()
```

### Reproducing results
The pipeline to download all the data, and produce all the results shown in the manuscript is managed by Snakemake.
To reproduce all results files and figures, run
``` bash
snakemake -j <NUM_CORES>
```

Successfully running the full pipeline takes a few months on a single machine. 
If you want to speed up the process and see similar results, you can run the pipeline without hyperparameter optimization with

``` bash
snakemake -s no_hyperopt_snakefile -j <NUM_CORES>
```

If you are going to be running the pipeline in a cluster environment, it may be helpful to read through the file `slurm_snakefile`. 
[This blog post](https://bluegenes.github.io/snakemake-via-slurm/) might also be helpful.


### Intermediate steps
When running the full pipeline via snakemake, the data required will be automatically downloaded (excluding the sex prediction labels mentioned in the section below). 
If you'd like to skip the data download (and in doing so save yourself about a week of downloading and processing things), you can unpack [this Zenodo archive](TODO) to the `data/` dir.

Likewise, if you'd like to download the results files, they can be found [here](TODO).
If you only need the saved models, they can be found [here](TODO).

### Sex prediction setup
Before running scripts involving sex prediction, you need to download the Flynn et al. labels from [this link](https://figshare.com/s/985621c1705043421962) and put the results in the `saged/data` directory.
Because of the settings on the figshare repo it isn't possible to incorporate that part of the data download into the Snakefile, otherwise I would.


### Neptune setup 
If you want to log training results, you will need to sign up for a free neptune account [here](https://neptune.ai/).
1. The neptune module is already installed as part of the saged conda environment, but you'll need to grab an API token from the website.
2. Create a neptune project for storing your logs.
3. Store the token in secrets.yml in the format `neptune_api_token: "<your_token>"`, and update the `neptune_config` file to use your info.

## Directory Layout
|File/dir|Description|
|--------|-----------|
|Snakefile | Contains the rules Snakemake uses to run the full project |
|environment.yml | Lists the python dependencies and their versions in a format readable by Conda |
|neptune.yml | Lists information for Neptune logging |
|secrets.yml | Stores neputne API token (see Neptune setup section) |
|||
|data/| Stores the raw and intermediate data files used for training models |\
|dataset_configs/| Stores config information telling Dataset objects how to construct themselves |
|figures/| Contains images visualizing the results of the various analyses |
|logs/| Holds serialized versions of trained models |
|model_configs/| Stores config information for models such as default hyperparameters |
|notebook/| Stores notebooks used for visualizing results |
|results/| Records the accuracies of the models on various tasks |
|src/| The source code used to run the analyses |
|test/| Tests for the source code (runnable with pytest) |

