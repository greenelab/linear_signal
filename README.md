# SAGED

![Status Badge](https://github.com/greenelab/saged/workflows/PythonTests/badge.svg)


This repository contains analyses designed to determine whether the semi-supervised learning
and representation learning are helpful in training models on gene expresion data.

It is currently a work in progress, so trust but verify when using the code.

Analysis plan:
- Plot dataset by labeled and unlabeled samples. Look at whether batch effect correction tools get studies to overlap
- After batch effect correction can we still differentiate between PBMCs and whole blood?
- If we simulate unlabeled samples via VAE on the labeled samples, does semi-supervision work?
This should test whether our models work correctly and whether the semi-supervision assumptions hold under controlled circumstances.
- Across all diseases, can we predict the disease better by using a semi-supervised method?
Compare semi-supervised and traditional supervised learning on neural net, SVM, and LR.
- For a single disease, probably sepsis, how does decreasing the amount of data change how helpful semi-supervised learning is?
Does this change when decreasing the amount of data uniformly or one study at a time?
- How do we know that semi-supervised learning is what’s leading to an increase in performance?
Do random projections work as well as PCA? Does a random tissue work as well as blood?
- How do batch effects affect us?
- - Do things get better after using ComBAT?
- - What about PEER?
- - Mutual nearest neighbors methods like INSCT might fit well here, but would struggle without labels I think?
- Figure: To what extent does each source of variation affect the results?
Similar to batch effect figure, but control for study, platform, both, and neither

### Neptune setup instructions
If you want to log training results, you will need to sign up for a free neptune account [here](https://neptune.ai/).
1. The neptune module is already installed as part of the saged conda environment, but you'll need to grab an API token from the website.
2. Create a neptune project for storing your logs.
3. Store the token in a file in the repo, then update the `config.yml` file to store your neptune username, the path to the secrets file, and your username/project name.
