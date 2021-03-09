# Abstract classification (supervised learning)

### Goal

The aim of this project is to understand whether the *keywords* assigned to a paper by the authors when submitting to a journal (in this case the journals in the Nature family) can be used to assign labels to the paper's abstract, and consequently use the large databases that journals provide as a training set for supervised learning.

### Methods

In the following, we use a small portion of the Nature corpus of paper metadata, specifically those Physics papers dealing with *quantum field theories*. We chose this set since there seems to be a reasonable number of papers falling inside this category.

We use the keywords assigned to each paper as labels. However, since the authors, not the editors, select the keywords uppon submission, there is not a universal set of keywords. As a result, there are some keywords that apper quite often, but otherwise there is a large amount of very specific keywords. We use the fact that keywords are assigned in a hierarchical fashion (for instance *physics - quantum field thoery - ads-cft correspondence*) to select a specific keyword as label for each paper, by constructing a tree of keywords, where the root is a keyword enclosing the highest number of papers, and the children are more specialized keywords.

We then clean the title and abstract of each paper using standard NLP techniques, and we fit different supervised models to the dataset, specifically,

- Ensamble learners
    - Random Forest
    - Gradient Boosting
- Support Vector Machine
- Logistic Regression

### Results

We find that using one of the keywords assigned to a paper as label for classification is possible, and training supervised models over the obtained dataset provides interesting results.

Specifically, we were able to create a dataset of ~ 5,000 training samples (abstract with label belonging to 9 different classes) by starting from a larger set (~20,000) of papers with multiple keywords. While the process was clearly wasteful, it took significantly less than manually classifying papers (and does not require expert knowledge).

We then mapped the abstracts into vectorised form, usig the tf-idf vectorisation procedure. We split the dataset into training and test sets (70/30 splitting). The models we trained have a test accuracy ranging between 73% and 77%. In particular, the ensamble learner (Random Forest and Gradient Boosting) seems to be outperformed by the linear methods (Linear SVM and Logistic Regression) in this setting. We confirm this by comparing the models using the Mc Nemar test.

### Remarks

There are several aspects that might be worth exploring. Conserning the use of keywords as labels for classification,

- One might prefer to keep the problem as *multi-label classification*, that is possibly more natural for the present setting.
- Instead of using the tree structure as we did, one could use the *apriori algorithm* to select relevant subset of keywords, and give a single name for each group of keywords. The problem with the apriori algorithm is that it will return overlapping sets.
- One might not use the keywords as label, and instead simply *cluster* the abstracts (thus moving to an unsupervised setting).

For what concern the vectorization used in this notebook, and the classifiers,

- One could use different kind of embeddings than the tf-idf vectorization we used in this project. In a related project building on the dataset we produced here (see the [link]()), we train a word2vec model on the abstracts we collect here, and we use this embedding with different Neural Network architectures to classify the abstracts. The test accuracy obtained exceeds 80%.

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