# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:20:00 2017

@author: Vakhrushev-Ag
"""
import numpy as np
from numpy import random
import pandas as pd
import shelve
from sklearn.metrics import roc_auc_score

class MeanClassifier():
    """
    Класс выдает среднюю вероянтность по группе подбирая значение регуляризатора по сетке
    На выоходе дает предсказание out-of-fold
    Работает с DataFrame
    Параметры:
        alpha - заданное значение регуляризатора
        grid - стека значений регуляризаторов
        metric - что меряем
    """
    
    def __init__(self, alpha = 1, grid = None, metric = roc_auc_score):
        self.alpha = alpha
        self.grid = grid
        self.metric = metric
        
    def fit_predict(self, df, by, target, cv):
        if type(by) != list:
            by = [by]
        self.by = by
        if self.grid == None:
            grid = [self.alpha]
        else:
            grid = self.grid
        df = df[[target] + by].copy()
        index = df.index
        df.reset_index(inplace = True)   
        df['_folds'] = cv.test_folds
        df.set_index(by + ['_folds'], inplace = True)
        
        levels = list(range(len(by)))
        levels_all = levels + [len(levels)]
        
        self.prior = df[target].mean()
        prior_fld = df.groupby(level = -1)[target].mean()
        grp = df.groupby(level = levels_all)[target].agg(['sum', 'count']) #.reset_index(level = -1)

        grp_fld = []
        for i in range(len(cv)):
            fold_stats = grp.drop(i, level = -1, axis = 0).groupby(level = levels).sum().reset_index()
            fold_stats['_folds'] = i
            fold_stats.set_index(by + ['_folds'], inplace = True)
            grp_fld.append(fold_stats)
        grp_fld = pd.concat(grp_fld)

        df = df.join(grp_fld)
        df.reset_index(inplace = True)
        df.set_index(index.name, inplace = True)
        prior_fld = df['_folds'].map(prior_fld).values
        df = df.ix[index]
        Y = df[target].values
        sum_target = df['sum'].fillna(0).values
        count_target = df['count'].fillna(0).values
        
        self.scores = []
        best_score = -np.inf
        for i in grid: 
            try:
                X = (sum_target + prior_fld * i) / (count_target + i)
                score = self.metric(Y, X)
            except ValueError:
                score = -np.inf
            if score > best_score:
                self.alpha = i
                best_pred = X
                best_score = score
            self.scores.append(score)
        self.grp = grp.groupby(level = levels).sum()
        
        return best_pred
    
    def predict(self, df):
        res = pd.merge(df[self.by], self.grp, how = 'left', left_on = self.by, right_index = True)[['sum', 'count']].fillna(0)
        return    (res['sum'] + self.prior * self.alpha) / (res['count'] + self.alpha)         
        
    def save(self, filename):
        db = shelve.open(filename)
        for i in self.__dict__.keys():
            db[i] = self.__dict__[i]
        db.close()
        return self
        
    def load(self, filename):
        db = shelve.open(filename)
        for i in db.keys():
            self.__dict__[i] = db[i]
        db.close()
        return self