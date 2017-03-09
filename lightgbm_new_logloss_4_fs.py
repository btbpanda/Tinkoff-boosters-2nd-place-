# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:45:51 2017

@author: Vakhrushev-AG
"""

import pandas as pd
from pandas import Series, DataFrame

import numpy as np
from numpy import random
random.seed(999)

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.externals import joblib
import re
from itertools import combinations

from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from numba import jit, autojit
import lightgbm as lgb
from MeanClassifier import MeanClassifier


train = pd.read_csv('credit_train.csv', encoding = 'cp1251', sep = ';', index_col = 'client_id')
test = pd.read_csv('credit_test.csv', encoding = 'cp1251', sep = ';', index_col = 'client_id')
test['open_account_flg'] = np.nan

data = pd.concat([train, test])
data['score_shk'] = data['score_shk'].apply(lambda x: float('.' + x[2:]))
data['credit_sum'] = data['credit_sum'].apply(lambda x: float(x[:-3] + '.' + x[-2:]))
data['gender'] = data['gender'] == 'M'
data['tariff_id'] = data['tariff_id'].astype(str)
"""
cols = ['gender', 'age', 'marital_status', 'job_position', 'credit_sum',
       'credit_month', 'tariff_id', 'score_shk', 'education', 'living_region',
       'monthly_income', 'credit_count', 'overdue_credit_count',
       'open_account_flg', 'big_region']              
"""
# region prepr
data['_temp_reg_cnt'] = data['living_region'].map(data['living_region'].value_counts())
regions = data['living_region'].fillna('-1').unique()
regions = Series(regions, index = regions, name = 'regions')

stopwrds = set(['ОБЛ','ОБЛАСТЬ', 'РЕСП', 'РЕСПУБЛИКА', 'КРАЙ', 'Г', 'АО', 'АОБЛ', 'АВТОНОМНАЯ'])
def clean_region(x):
    x = re.sub("[.,]+", " ", str(x))
    wrds = x.split(' ')
    wrds_new = []
    for w in wrds:
        if not w in stopwrds:
            wrds_new.append(w)
    x = ''.join(wrds_new)
    return x
    
regions = regions.map(clean_region)

# exeptions
regions['ЧУКОТСКИЙ АO'] = 'ЧУКОТСКИЙ'
regions['ЧУВАШСКАЯ РЕСПУБЛИКА - ЧУВАШИЯ'] = 'ЧУВАШСКАЯ'
regions['ЧУВАШИЯ ЧУВАШСКАЯ РЕСПУБЛИКА -'] = 'ЧУВАШСКАЯ'
regions['ЧЧУВАШСКАЯ - ЧУВАШИЯ РЕСП'] = 'ЧУВАШСКАЯ'
regions['РЕСП ЧУВАШСКАЯ - ЧУВАШИЯ'] = 'ЧУВАШСКАЯ'
regions['ЧУВАШСКАЯ - ЧУВАШИЯ РЕСП'] = 'ЧУВАШСКАЯ'
regions['РЕСПУБЛИКАТАТАРСТАН'] = 'ТАТАРСТАН'
regions['ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ'] = '-1'
regions['ПЕРМСКАЯ ОБЛ'] = 'ПЕРМСКИЙ'
regions['ОРЁЛ'] = 'ОРЛОВСКАЯ'
regions['Г.ОДИНЦОВО МОСКОВСКАЯ ОБЛ'] = 'МОСКОВСКАЯ'
regions['МЫТИЩИНСКИЙ Р-Н'] = 'МОСКОВСКАЯ'
regions['МОСКОВСКИЙ П'] = 'МОСКОВСКАЯ'
regions['КАМЧАТСКАЯ ОБЛАСТЬ'] = 'КАМЧАТСКИЙ'
regions['КАМЧАТС??ИЙ КРАЙ'] = 'КАМЧАТСКИЙ'
regions['ДАЛЬНИЙ ВОСТОК'] = '-1'
regions['ДАЛЬНИЙВОСТОК'] = '-1'
regions['ГУСЬ-ХРУСТАЛЬНЫЙ Р-Н'] = 'ВЛАДИМИРСКАЯ'
regions['ЧИТИНСКАЯ ОБЛ'] = 'ЗАБАЙКАЛЬСКИЙ'
regions['ГОРЬКОВСКАЯ ОБЛ'] = 'НИЖЕГОРОДСКАЯ'
regions['ЭВЕНКИЙСКИЙ АО'] = 'КРАСНОЯРСКИЙ'
regions['ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУГ - ЮГРА'] = 'ХАНТЫ-МАНСИЙСКИЙ'
regions['АО ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУГ - Ю'] = 'ХАНТЫ-МАНСИЙСКИЙ'
regions['АО ХАНТЫ-МАНСИЙСКИЙ-ЮГРА'] = 'ХАНТЫ-МАНСИЙСКИЙ'
regions['СЕВ. ОСЕТИЯ - АЛАНИЯ'] = 'СЕВЕРНАЯОСЕТИЯ-АЛАНИЯ'
regions['РЕСП. САХА (ЯКУТИЯ)'] ='САХА/ЯКУТИЯ/'
regions['РЕСПУБЛИКА САХА'] ='САХА/ЯКУТИЯ/'
regions['ДАЛЬНИЙВОСТОК'] = '-1'
regions['САХА'] = 'САХА/ЯКУТИЯ/'
regions['98'] = 'САНКТ-ПЕТЕРБУРГ'
regions['74'] = 'ЧЕЛЯБИНСКАЯ'
regions['РОССИЯ'] = '-1'
regions['МОСКВОСКАЯ'] = 'МОСКОВСКАЯ'
regions['МОСКВОСКАЯ ОБЛ'] = 'МОСКОВСКАЯ'
regions['ЧЕЛЯБИНСК'] = 'ЧЕЛЯБИНСКАЯ'
regions['Г. ЧЕЛЯБИНСК'] = 'ЧЕЛЯБИНСКАЯ'
regions['БРЯНСКИЙ'] = 'БРЯНСКАЯ'

big_region = {'-1': '-1'}
for big_reg, reg_list in [('ДАЛЬНЕВОЧТОЧНЫЙ', ['АМУРСКАЯ','ЕВРЕЙСКАЯ','КАМЧАТСКИЙ','МАГАДАНСКАЯ','ПРИМОРСКИЙ','САХА/ЯКУТИЯ/','САХАЛИНСКАЯ','ХАБАРОВСКИЙ','ЧУКОТСКИЙ']),
                          ('ПРИВОЛЖСКИЙ', ['БАШКОРТОСТАН','КИРОВСКАЯ','МАРИЙЭЛ','МОРДОВИЯ','НИЖЕГОРОДСКАЯ','ОРЕНБУРГСКАЯ','ПЕНЗЕНСКАЯ','ПЕРМСКИЙ','САМАРСКАЯ','САРАТОВСКАЯ','ТАТАРСТАН','УДМУРТСКАЯ','УЛЬЯНОВСКАЯ','ЧУВАШСКАЯ']),
                          ('СЕВКАВ', ['ДАГЕСТАН','ИНГУШЕТИЯ','КАБАРДИНО-БАЛКАРСКАЯ','КАРАЧАЕВО-ЧЕРКЕССКАЯ','СЕВЕРНАЯОСЕТИЯ-АЛАНИЯ','СТАВРОПОЛЬСКИЙ','ЧЕЧЕНСКАЯ']), 
                          ('СЗ', ['АРХАНГЕЛЬСКАЯ','ВОЛОГОДСКАЯ','КАЛИНИНГРАДСКАЯ','КАРЕЛИЯ','КОМИ','ЛЕНИНГРАДСКАЯ','МУРМАНСКАЯ','НЕНЕЦКИЙ','НОВГОРОДСКАЯ','ПСКОВСКАЯ','САНКТ-ПЕТЕРБУРГ']), 
                          ('СИБИРЬ', ['АЛТАЙСКИЙ','АЛТАЙ','ЗАБАЙКАЛЬСКИЙ','БУРЯТИЯ','ИРКУТСКАЯ','КЕМЕРОВСКАЯ','КРАСНОЯРСКИЙ','НОВОСИБИРСКАЯ','ОМСКАЯ','ТОМСКАЯ','ТЫВА','ХАКАСИЯ']), 
                          ('УРАЛ', ['КУРГАНСКАЯ','СВЕРДЛОВСКАЯ','ТЮМЕНСКАЯ','ХАНТЫ-МАНСИЙСКИЙ','ЧЕЛЯБИНСКАЯ','ЯМАЛО-НЕНЕЦКИЙ']), 
                          ('ЦЕНТР', ['БЕЛГОРОДСКАЯ','БРЯНСКАЯ','ВЛАДИМИРСКАЯ','ВОРОНЕЖСКАЯ','ИВАНОВСКАЯ','КАЛУЖСКАЯ','КОСТРОМСКАЯ','КУРСКАЯ','ЛИПЕЦКАЯ','МОСКВА','МОСКОВСКАЯ','ОРЛОВСКАЯ','РЯЗАНСКАЯ','СМОЛЕНСКАЯ','ТАМБОВСКАЯ','ТВЕРСКАЯ','ТУЛЬСКАЯ','ЯРОСЛАВСКАЯ']), 
                          ('ЮГ', ['АДЫГЕЯ','АСТРАХАНСКАЯ','ВОЛГОГРАДСКАЯ','КАЛМЫКИЯ','КРАСНОДАРСКИЙ','РОСТОВСКАЯ'])
                            ]:
    for reg in reg_list:
        big_region[reg] = big_reg

data['living_region'] = data['living_region'].fillna('-1').map(regions)
data['big_region'] = data['living_region'].map(big_region)

data['reg_rate'] = data['_temp_reg_cnt'] / data['living_region'].map(data['living_region'].value_counts())
data['reg_rate'].fillna(0, inplace = True)
data['big_reg_rate'] = data['living_region'].map(data['living_region'].value_counts()) / data['big_region'].map(data['big_region'].value_counts())
data['big_reg_rate'].fillna(0, inplace = True)

data.drop('_temp_reg_cnt', axis = 1, inplace = True)

####################################################################################################
                    # FEATURE ENG
###################################################################################################


# some strange feats

def last_sign(x):
    if x == 0:
        return 1 / 8
    x = int(x * 100)
    try:
        n = int(np.log10(x)) 
    except OverflowError:
        return 1 / 8
    for i in range(n, -1, -1):
        if x % (10 ** i) == 0:
            return (n + 1 - i) / n
    else:
        return 1 / 8


data['sign_digits'] = data['monthly_income'].fillna(10000).apply(last_sign)
data['sign_credit'] = data['credit_sum'].apply(last_sign)
data['term_freq'] = data['credit_month'].map(data['credit_month'].value_counts()).rank(method = 'dense')
data['credit_freq'] = data['credit_sum'].map(data['credit_sum'].value_counts()).rank(method = 'dense')


# cols_list
cat_cols = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region', 'big_region']

num_cols = ['age', 'credit_sum', 'credit_month', 'score_shk', 'monthly_income', 
            'credit_count', 'overdue_credit_count'] 
            
for i in cat_cols:
    data[i] = pd.factorize(data[i])[0]            
 
         

# fill rare tariffs
data['tariff_id'][data['tariff_id'].map(train['tariff_id'].value_counts()) <= 11] = 'rare'
data['job_position'][data['job_position'].map(train['job_position'].value_counts()) <= 12] = 'rare'


# some feats
data['ovd_rate'] = (data['credit_count'] / data['overdue_credit_count'])
data['ovd_rate'][data['ovd_rate'] == np.inf] = 0
data['dummy_paym'] = data['credit_sum'] / data['credit_month']
data['dummy_pti'] = data['dummy_paym'] / data['monthly_income']
data['dummy_pti'][data['dummy_pti'] == np.inf] = 1
# some means

def means_mapping(var, by, func = 'mean'):
    if type(by) != list:
        by = [by]
    name = '{0}_by_{1}_mean'.format(var, by)
    grp = data.groupby(by)[[var]].agg(func)
    grp.columns = [name]
    return pd.merge(data[by], grp, left_on = by, right_index = True, how = 'left')[name]
    

data['age_by_job_ed_rate'] = data['age'] / means_mapping('age', ['job_position', 'education'])
data['inc_by_gen_reg_rate'] = data['monthly_income'] / means_mapping('monthly_income', ['gender', 'living_region'])
data['mth_by_tariff_rate'] = data['credit_month'] / means_mapping('credit_month','tariff_id')
data['score_by_tariff_rate'] = data['score_shk'] / means_mapping('score_shk','tariff_id')
data['ovd_by_tariff_mar_ed_rate'] = data['overdue_credit_count'] / means_mapping('overdue_credit_count',['marital_status', 'tariff_id', 'education'])

# NaNs

data['no_hist'] = data['credit_count'].isnull()
for i in ['ovd_by_tariff_mar_ed_rate', 'credit_count', 'overdue_credit_count']:
    if i in data.columns:
        data[i].fillna(-1, inplace = True)

for i in ['monthly_income', 'inc_by_gen_reg_rate', 'dummy_pti' ]:
    if i in data.columns:
        data[i].fillna(data[i].median(), inplace = True)
    
for i in ['ovd_rate', 'ovd_by_tariff_mar_ed_rate']:
    if i in data.columns:
        data[i].fillna(0, inplace = True)



train = data[:len(train)]
test  = data[len(train):]
folds = StratifiedKFold(train['open_account_flg'].values, 5, shuffle = True, random_state = 42)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'is_training_metric' : True,   
    'num_leaves': 250,
    'tree_learner' : 'serial',
    'num_threads': 4,
    'min_data_in_leaf': 7,
    'learning_rate': 0.001,
    'feature_fraction': 0.4,
    'feature_fraction_seed': 2,
    'bagging_fraction': 0.8,
    'bagging_seed': 3,
    'bagging_freq': 20,
    'min_sum_hessian_in_leaf' : 1,
    'max_depth': -1,
    'lambda_l1': 1,
    'lambda_l2': 10,
    'min_gain_to_split': 2,
    'verbose': 1,
    'metric_freq': 1,
    'max_bin': 1000
}
      
params_mse = params.copy()
params_mse['application'] = "regression"
params_mse['feature_fraction_seed'] = 10
params_mse['bagging_seed'] = 11
params_mse['lambda_l1'] = 0
params_mse['lambda_l2'] = 1
params_mse['min_gain_to_split'] = 0



     


train = data[:len(train)]
test  = data[len(train):]
Y = data['open_account_flg'].values[:len(train)]

it_scores = {}
added_inter = 0
for i in range(1, 4):
    for j in combinations(cat_cols, i):
        by = list(j)
        name = '_'.join(by) + '_it'
        model = MeanClassifier(grid = list(np.arange(20)))

        pred = model.fit_predict(train, by, 'open_account_flg', cv = folds)
        score = max(model.scores)
        
        
        if i == 1 or score > max([it_scores[x + '_it'] for x in by]):
            data[name] = np.hstack((pred, model.predict(test).values))
            added_inter += 1
        it_scores[name] = score
        print(name, score)  

data.drop('gender_it', axis = 1, inplace = True)

interactions_col = [x for x in it_scores.keys() if x in data.columns]

_X = data[interactions_col].values
_pred = np.empty_like(Y)
model = Lasso(.00001, selection = 'random', random_state = 42, max_iter = 1000000, positive = False)

# tuning penalty
# Lasso is faster than Logistic
for f0, f1 in folds:
    print('..training')
    model.fit(_X[f0], Y[f0])
    _pred[f1] = model.predict(_X[f1])
    
print(roc_auc_score(Y, _pred))    

model.fit(_X[:len(train)], Y)


data['meta_inter'] = np.hstack((_pred, model.predict(_X[len(train):])))
data.drop(np.array(interactions_col)[model.coef_ == 0], axis = 1, inplace = True)
print('Interactions added {0}'.format(np.array(interactions_col)[model.coef_ != 0]))

data.drop('open_account_flg', axis = 1, inplace = True)

X = data.values
X_test = X[len(train):]
X = X[:len(train)]


oof_pred = np.empty((X.shape[0], 2))
test_pred = np.empty((X_test.shape[0], len(folds), 2))

early_stopping_rounds = 1000
rounds = 1000000

for n, (f0, f1) in enumerate(folds):
    
    # train logloss as int with interactions
    params['feature_fraction_seed'] = 626
    params['bagging_seed'] = 765
    
    lgb_train = lgb.Dataset(X[f0], Y[f0], max_bin = 1000, 
                free_raw_data=False)
    lgb_eval = lgb.Dataset(X[f1], Y[f1], max_bin = 1000, reference = lgb_train,
                free_raw_data=False)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round = rounds,
                valid_sets = lgb_eval,  # eval training data
                early_stopping_rounds = early_stopping_rounds
                )
    predict = gbm.predict(np.vstack((X_test, X[f1])), num_iteration=gbm.best_iteration)
    test_pred[:, n, 0] = predict[:len(X_test)]
    oof_pred[f1, 0]    = predict[len(X_test):]
    
    # train mse as int with interactions
    params_mse['feature_fraction_seed'] = 146
    params_mse['bagging_seed'] = 543
      
    gbm = lgb.train(params_mse,
                lgb_train,
                num_boost_round = rounds,
                valid_sets = lgb_eval,  # eval training data
                early_stopping_rounds = early_stopping_rounds
                )
    predict = gbm.predict(np.vstack((X_test, X[f1])), num_iteration=gbm.best_iteration)
    test_pred[:, n, 1] = predict[:len(X_test)]
    oof_pred[f1, 1]    = predict[len(X_test):]  



columns = ['lgb_logloss_cat_fs', 'lgb_mse_cat_fs']
    


train_prediction = DataFrame(oof_pred, index = train.index, columns = columns)
train_prediction.to_csv('train_lightgbm_logloss_4_fs.csv')

test_prediction = DataFrame(test_pred.mean(axis = 1), index = test.index, columns = columns)
test_prediction.to_csv('test_lightgbm_logloss_4_fs.csv')
