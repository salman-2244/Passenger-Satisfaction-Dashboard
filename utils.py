import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self._classes = None
        self._feature_names_in = None

    def fit(self, X, y, **fit_params):
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
        else:
            raise ValueError("Input X must be a pandas DataFrame with feature names.")

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)

    @property
    def classes_(self):
        if self._classes is None:
            raise ValueError("The classes_ attribute is not set. Please fit the model first.")
        return self._classes

    @classes_.setter
    def classes_(self, value):
        self._classes = value

    @property
    def feature_names_in_(self):
        if self._feature_names_in is None:
            raise ValueError("The feature names are not set. Please fit the model with a DataFrame.")
        return self._feature_names_in
