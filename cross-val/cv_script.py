import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# load the data
X_train = load_npz('X_train.npz')
y_train = np.load('y_train.npy',allow_pickle=True)

# The model we wish to cross-validate
model = GradientBoostingClassifier(random_state=2764)

# The range of parameters we want to check
model_params = {}

model_params['learning_rate'] = [1e-2,5e-2,1e-1,5e-1]
model_params['n_estimators'] = [600,800,1200,1600]
model_params['max_depth'] = [2,3,4,5,6]

# Setting cross-validation up
cross_validation = GridSearchCV(model,
                                model_params,
                                n_jobs=-1, # run the cv in parallel
                                cv=5, # number of sets the training is divided into
                                refit=False, # we will refit the model ourselves
                                verbose=2
                               )

# Run CV for the model
cross_validation.fit(X_train,y_train)

# Save CV results in dataframe
results_CV = pd.DataFrame(cross_validation.cv_results_)
results_CV.to_pickle('./cross_val_results.pkl')