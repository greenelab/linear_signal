# SAGED

![Status Badge](https://github.com/greenelab/saged/workflows/PythonTests/badge.svg)

## Setup
Before running scripts involving sex prediction, you need to download the Flynn et al. labels from [this link](https://figshare.com/s/985621c1705043421962) and put the results in the `saged/data` directory.
Because of the settings on the figshare repo it isn't possible to incorporate that part of the data download into the Snakefile, otherwise I would.





### Neptune setup instructions
If you want to log training results, you will need to sign up for a free neptune account [here](https://neptune.ai/).
1. The neptune module is already installed as part of the saged conda environment, but you'll need to grab an API token from the website.
2. Create a neptune project for storing your logs.
3. Store the token in a file in the repo, then update the `config.yml` file to store your neptune username, the path to the secrets file, and your username/project name.

### Config file
An example config file exists at `saged/test/data/test_config.yml`
