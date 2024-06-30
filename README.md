# Feature Sensitivity and Model Discrimination in Preclinical Breast Cancer Photoacoustic Imaging

![Static Badge](https://img.shields.io/badge/build-passing-lime)
![Static Badge](https://img.shields.io/badge/logo-gitlab-blue?logo=gitlab)

This repository contains Ivo Petrov's apporach to the project **27 Feature Sensitivity and Model Discrimination in Preclinical Breast Cancer Photoacoustic Imaging**. The project aims to reproduce and extend the work done in the paper <a href="https://www.nature.com/articles/s41598-022-19084-w">Feasibility and sensitivity study of radiomic features in photoacoustic imaging of patient-derived xenografts</a>. We detail a framework that is used to identify viable radiomic features which are sensitive to the underlying cancer subtype. 2 types of breast cancer are considered, namely **basal** and **luminal B**. We then derive the features' importance in detecting said cancer type.

## Table of contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Features](#features)
4. [Build status](#build-status)
5. [Credits](#credits)


## Requirements

The user should preferrably have a version of Docker installed in order to ensure correct setup of environments. If that is not possible, the user is recommended to have Conda installed, in order to set up the requirements. If Conda is also not available, make sure that the packages described in `environment.yml` are available and installed.

## Setup

We provide two different set up mechanisms using either Docker or Conda. The former is recommended, as it ensures that the environment used is identical to the one used in the development in the project.
### Using Conda
If you decide to simply set up a Conda environment you will have to take care of setting up both the environment and the dataset. To create and activate a new environment, please use:

```conda env create -f environment.yml -n <env_name>```

```conda activate <env_name>```

If planning to use autodocumentation, please also run:

```pip install sphinx_rtd_theme```

In this case, downloading the data will not be done automatically, so feel free to use the following set of scripts:

```
mkdir ./data ; mkdir ./data/raw; cd ./data

wget https://github.com/loressa/Photoacoustic_radiomics_xenografts

cp ./Photoacoustic_radiomics_xenografts/All_* ./

unzip ./Photoacoustic_radiomics_xenografts/ModelsUncorrected.zip -d ./data/raw

rm -rf ./Photoacoustic_radiomics_xenografts/
```
### Using Docker
Using Docker is much simpler, as it only requires setting up a container. To do so, fist build the image using:

```docker build -t ivp24_project .```

Following this, you can run the container using:

```docker run --name <container_name> -ti ivp24_project```

If you require using your git/SSH credentials in any way, a safe way to do so can be done by mounting the relevant ssh folder as so:

```docker run -v <local_ssh_dir>:/root/.ssh --name <container_name> -ti ivp24_project```

The container can now be attached to VSCode or any suitale IDE or the code can be ran from the command line.

If you need to copy over any files to your local system, please use:

```docker cp <container_name>:/<source_dir> <dest_dir>```

## Result reproduction

We include all relevant commands we used for replicating the original work, as well as for any additional ablation studies covered in the report.

### Frequently used arguments
Below we list the meaning of commonly used arguments in the following scripts.

- *data_path*: The location of the data.
- *output_path*: Where to store the outputs.
- *config_path*: The location of the classifier configuration files.
- *validation*: An option on whether to perform the script in a cross-validated manner.
- *scores*: Where the individual feature scores are found.
- *method*: Which method to use for the initial feature reduction. Supports 'kw' (Kruskal-Wallist), 'ks' (Kolmogorov-Smirnov), 'fs' (Forward selection)

### Sensitivity analysis
For the replication part of this section, run the following command:

```python -m scripts.sensitivity_analysis <data_path> -o <out_path> --remove_luminal```

**NOTE**: If following the Docker setup, the correct data path will be the path of the *raw* data (usually `./data/raw`). This holds for any command of this section.

For the ablation study on unbalanced ANOVA please run:

```python -m scripts.sensitivity_analysis <data_path> -o <out_path>```

Finally, for the k-fold sensitivity analysis, the following command should be run:

```python -m scripts.k_fold_sensitivity_analysis <data_path> -o <out_path>``

### Feature normalization

We will generally use the preprocessed data that is submitted as part of the prior work, however, for the ablation study on the prior knowledge for VOI normalization, run the following command:

```python -m scripts.normalize_features <data_path> -o <out_path>```

Here the `data_path` argument should point to the folder with the *raw* data, as above. The `out_path` parameter is not mandatory, and will default to storing the normalized features in `/data/voi_norm`. Be warned that if you set `out_path` to `/data`, you will **rewrite the original data**. In such a case follow the guid for downloading the data. In the following sections, you can perform the ablation studies on the VOI normalization by changing the input folder.

### Scoring single features

For the purposes of this project, we only performed this step as described in the original work. The script to be run is as follows:

```python -m scripts.single_feature_scores <data_path> -i <out_path> --cfg <config_path> ```

### Feature reduction

For performing the feature reduction using the method described in the original work, use the following command:

```python -m scripts.reduce_features <data_path> -o <out_path> --scores <scores_path>```

The scores path can be determined from the output of the above section. For better consistency, it is better to store the results in a data-related folder for future use, i.e. `./data/reduced_features`. The method name will be attached at the end of the directory specified.

You can vary the correlation threshold by varying the `corr_thr` parameter, i.e.

```python -m scripts.reduce_features <data_path> -o <out_path> --scores <scores_path> --corr_thr 0.7```

For the ablation studies on the reduction method, change the `method` parameter, i.e.

```python -m scripts.reduce_features <data_path> -o <out_path> --scores <scores_path> --method ks```

for using the Kolmogorov-Smirnov test or

```python -m scripts.reduce_features <data_path> -o <out_path> --scores <scores_path> --method fs```

for using forward selection.

### Hyperparameter optimization

Before proceeding to determining the feature importance, the next step's ablation study requires that we have a set of optimal hyperparameters for the relevant models. We can do so using:

```python -m scripts.optimize_hyperparams <data_path> -o <out_file> --grids <grids_path>```

### Feature importance
Finally, after performing the feature reduction, we can apply the feature importance step using:

```python -m scripts.feature_importance <data_path> -o <out_path> --cfg <config_path>```

,where `config_path` can take the default configurations or the optimised ones. We can enable cross-validation through the `--validation` tag:

```python -m scripts.feature_importance <data_path> -o <out_path> --cfg <config_path> --validation``

## Features

As supplementary additions to the repository, we include Sphinx as an autodocumentation tool. To generate the html documentation, please run:

```sphinx-build -M html ./docs/source ./docs/build -a -E```

This will then create the necessary files in `docs/build/html`.

If necessary to move the build out of the Docker container, run the command:

```docker cp <container_name>:/ivp24/docs/build/html <dest_dir>```

And open the `index.html` file in the browser.

## Build status
Currently, the build is complete and the repository can be used to its full capacity.

## Credits
I would like to credit my supervisor Dr Lorena Escudero Sánchez for all the help in proposing the project, and guiding me along the way. This work is primarily based on [her previous work](https://www.nature.com/articles/s41598-022-19084-w), to which she provided useful [code as well as the radiomic feature data](https://github.com/loressa/Photoacoustic_radiomics_xenografts/tree/main). The code was used only for inspiration, as well as configuring the classifier hyperparameters, and was not directly rewritten from the repository.

Furthermore, for autodocumentation I used the Read the Docs theme, which can be found [here](https://sphinx-rtd-theme.readthedocs.io/en/stable/).

The `.pre-commit-config.yaml` configuration file content has been adapted from the Research Computing lecture notes.
