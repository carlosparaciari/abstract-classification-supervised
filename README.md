# Abstract classification (supervised learning)

### Goal

This project aims to understand whether the *keywords* assigned to a paper by the authors when submitting to a journal (in this case the journals in the Nature family) can be used to assign labels to the paper's abstract, and consequently use the large databases that journals provide as a training set for supervised learning.

### Methods

In the following, we use a small portion of the Nature corpus of paper metadata, specifically those Physics papers dealing with *quantum field theories*. We chose this set since there seems to be a reasonable number of papers falling inside this category.

We use the keywords assigned to each paper as labels. However, since the authors, not the editors, select the keywords upon submission, there is not a universal set of keywords. As a result, some keywords appear quite often, but otherwise, there is a large amount of very specific keywords. We use the fact that keywords are assigned hierarchically (for instance *physics - quantum field theory - AdS-CFT correspondence*) to select a specific keyword as the label for each paper, by constructing a tree of keywords, where the root is a keyword enclosing the highest number of papers, and the children are more specialized keywords.

We then clean the title and abstract of each paper using standard NLP techniques, and we fit different supervised models to the dataset, specifically,

- Ensemble learners
    - Random Forest
    - Gradient Boosting
- Support Vector Machine
- Logistic Regression

### Results

We find that using one of the keywords assigned to a paper as the label for classification is possible, and training supervised models over the obtained dataset provides interesting results.

Specifically, we were able to create a dataset of ~ 5,000 training samples (abstract with label belonging to 9 different classes) by starting from a larger set (~20,000) of papers with multiple keywords. While the process was wasteful, it took significantly less than manually classifying papers (and does not require expert knowledge).

We then mapped the abstracts into vectorized form, using the TF-IDF vectorization procedure. We split the dataset into training and test sets (70/30 splitting). The models we trained have a test accuracy ranging between 73% and 77%. In particular, the ensemble learner (Random Forest and Gradient Boosting) seems to be outperformed by the linear methods (Linear SVM and Logistic Regression) in this setting. We confirm this by comparing the models using the Mc Nemar test.

### Remarks

Several aspects might be worth exploring. Concerning the use of keywords as labels for classification,

- One might prefer to keep the problem as *multi-label classification*, that is, possibly more natural for the present setting.
- Instead of using the tree structure as we did, one could use the *apriori algorithm* to select a relevant subset of keywords and give a single name for each group of keywords. The problem with the apriori algorithm is that it will return overlapping sets.
- One might not use the keywords as labels, and instead simply *cluster* the abstracts (thus moving to an unsupervised setting). Clustering papers according to their abstract is also helpful to better understand what kind of (labeled) papers we should use to train a classifier that will then be deployed for classifying a specific set of papers. We perform a study in this direction over the papers in the *hep-th* category of the arXiv (see this [link](https://github.com/carlosparaciari/abstract-clustering) for the result of this study).

For what concern the vectorization and the classifiers used in this notebook,

- One could use different kinds of embeddings than the TF-IDF vectorization we used in this project. In a related project building on the dataset, we produced here (see the [link](https://github.com/carlosparaciari/abstract-classification-embedding)), we train a word2vec model on the abstracts we collect here, and we use this embedding with different Neural Network architectures to classify the abstracts. The test accuracy obtained exceeds 80%.

### Sections of the notebook

- Database creation
- Label selection
- Abstract and title characterization
- Cleaning title and abstracts
- Preprocessing
- Training models
    - Random Forests
    - Gradient Boosting
    - Support Vector Machine
    - Logistic Regression with $l_1$ regularization.
- Model evaluation
