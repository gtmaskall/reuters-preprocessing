Lab 1 - Preprocessing Data
==========================

## Table of Contents
1. [Overview](#overview)
2. [Module Description](#description)
3. [Usage](#usage)
4. [Development](#development)
5. [Change Log](#change log)

## Overview
The purpose of this module is to preprocess a set of SGML documents representing a Reuters article database into a dataset of feature vectors and class labels. The datasets will be employed in future assignments for automated categorization, similarity search, and building document graphs.

## Description
This python module contains the following files and directories:

* preprocess.py - main module for preprocessing the Reuters article database
* feature1.py - sub-module that generates feature vector dataset #1
* feature2.py - sub-module that generates feature vector dataset #2
* feature3.py - sub-module that generates feature vector dataset #3
* tfidf.py - module for term frequency-inverse document frequency
* data/
    * reut2-xxx.sgm - formatted articles (replace xxx from {000,...,021})

Running `preprocess.py` will generate the following files

* dataset1.csv
* dataset2.csv
* dataset3.csv

The feature vectors in the datasets were generated using the following methodologies

* TF-IDF of title & body words to select the top 1000 words as features
* Filtering nouns & verbs from the term lists, and repeating the previous process

For a more detailed report of the methodology used to sanitize and construct these refined datasets and feature vectors, read the file in this project titled `Report1.md` using the following command

```
> less Report1.md
```

## Usage
To run the code, first ensure the `preprocess.py` file has execute privileges:

```
> chmod o+x preprocess.py
```

Next, ensure the `tfidf.py`, `feature1.py`, `feature2.py`, and `feature3.py` files are in the same directory as `preprocess.py`. Also,
ensure that the `data/` directory containing the `reut2-xxx.sgm` files is present. To begin preprocessing the data, run:

```
> python preprocess.py
```

or

```
> ./preprocess.py
```

The preprocessing might take some time to complete.

Once `preprocess.py` finishes execution, three datasets are generated by the code labeled `dataset1.csv`, `dataset2.csv`, and `dataset3.csv` in the project directory (same folder as `preprocess.py`). To view these datasets, run:

```
> less datasetX.csv
```

where `X` is replaced with 1, 2, or 3 depending on the dataset.

## Development
* This module was developed using python 2.7.10 using the NLTK and BeautifulSoup4 modules.

### Contributors
* Ankai Lou (lou.56@osu.edu)

## Change Log
2015-09-10 - Version 1.0.0:

* Initial code import
* Added functionality to generate parse tree
* Added functionality to generate document objects
* Added functionality to tokenize, stem, and filter words
* Added functionality to generate lexicons for title & body words
* Prepare documents for feature selection & dataset generation
