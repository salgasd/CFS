import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from scipy.stats import spearmanr
from sklearn.base import TransformerMixin, BaseEstimator

from dataclasses import dataclass, field
from typing import Any
from queue import PriorityQueue


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


def calc_merit(data, subset, target, fc_method='pointbiserialr', ff_method='pearson'):
    k = len(subset)
    # Ср. корреляция фич с таргетом
    if fc_method == 'pointbiserialr':
        rcf_all = [abs(pointbiserialr(target, data[col]).correlation) for col in subset]
    elif fc_method == 'spearman':
        rcf_all = [abs(spearmanr(target, data[col]).correlation) for col in subset]
    rcf = np.sum(rcf_all)

    # Ср. корреляция между фичами
    corr = data[subset].corr(method=ff_method)
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().sum()

    return (k * rcf) / np.sqrt(k + k * (k-1) * rff)


class CFS(TransformerMixin, BaseEstimator):

    def __init__(self, fc_method, ff_method, max_backtrack=5, max_features=None) -> None:
        super().__init__()
        self.fc_method = fc_method
        self.ff_method = ff_method
        self.max_backtrack = max_backtrack
        self.max_features = max_features
    
    def fit(self, X: pd.DataFrame, y, **fit_params):
        features = X.columns.tolist()
        # Находим фичу с лучшей корреляцией с таргетом
        if self.fc_method == 'pointbiserialr':
            feats_cor = {col: abs(pointbiserialr(y, X[col]).correlation) for col in X}
        elif self.fc_method == 'spearman':
            feats_cor = {col: abs(spearmanr(y, X[col]).correlation) for col in X}

        best_feature = max(feats_cor, key=feats_cor.get)
        best_value = -feats_cor[best_feature]

        # Очередь
        q = PriorityQueue()
        q.put(PrioritizedItem(best_value, [best_feature]))

        visited = []
        n_backtrack = 0
        while not q.empty():
            temp = q.get()
            subset, priority = temp.item, temp.priority

            if best_value < priority:
                n_backtrack += 1
            else:
                best_value = priority
                self.best_subset = subset

            if self.max_features:
                if len(self.best_subset) >= self.max_features:
                    break

            if (n_backtrack >= self.max_backtrack):
                break

            for feature in features:
                temp_subset = subset + [feature]
                for node in visited:
                    if set(node) == set(temp_subset):
                        break
                else:
                    visited.append(temp_subset)
                    merit = -calc_merit(X, temp_subset, y, self.fc_method, self.ff_method)
                    q.put(PrioritizedItem(merit, temp_subset))
        return self

    def transform(self, X, y=None, **kwargs):
        df = X.copy().loc[:, self.best_subset]
        return df