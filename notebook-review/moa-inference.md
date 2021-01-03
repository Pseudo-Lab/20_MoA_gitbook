# MoA Inference 노트북 작성

## Setting

### Install and import libraries

```text
# TabNet
!pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.0-py3-none-any.whl pytorch-tabnet > /dev/null
# Iterative Stratification
!pip install /kaggle/input/iterative-stratification/iterative-stratification-master/ > /dev/null
```

```text
import os
import random
from shutil import copytree, ignore_patterns
import numpy as np
import pandas as pd
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.metrics import Metric

from scipy.stats import kurtosis
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, NMF
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import warnings
warnings.filterwarnings('ignore')
```

```text
# Util Functions
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

class DatasetWithoutLabel(Dataset):
    def __init__(self, X):
        super().__init__()
        self._X = X

    def __getitem__(self, item):
        return (self._X[item], )

    def __len__(self):
        return len(self._X)

def collate_fn_test(batch):
    X = [x[0] for x in batch]
    X = torch.FloatTensor(X)
    return X
```

```text
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
```

### 전처리 코드

> cate2num: category feature → numerical feature make\_folds: iterative stratified K Fold for multi-label classification task feature encoding: Rank Gauss, Feature statistics, PCA, Varaiance Threshold, LSVM feature selection

> Variance Threshold의 경우 데이터가 달라질 경우 feature selection이 달라지기때문에 weight를 불러 inference를 하는 경우 private testset에서 에러가 날 수 있음. 이러한 문제를 해결하기위해 mask를 따로 만들어 전처리함.

```text
def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    return df

def get_cols(test):
    g_cols = [c for c in test.columns if 'g-' in c]
    c_cols = [c for c in test.columns if 'c-' in c]
    num_g = len(g_cols)
    num_c = len(c_cols)
    return g_cols, c_cols, num_g, num_c

def make_folds(train_features, train_targets_scored, n_folds, seed):
    train = train_features.merge(train_targets_scored, on='sig_id')
    target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    cols = target_cols + ['cp_type']

    train_cp = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_cp.loc[:, 'fold'] = 0
    for n, (train_index, val_index) in enumerate(mskf.split(train_cp, train_cp[target_cols])):
        train_cp.loc[val_index, 'fold'] = int(n)
    train_cp['fold'] = train_cp['fold'].astype(int)

    train_ctl = train[train['cp_type'] == 'ctl_vehicle'].reset_index(drop=True)
    train_ctl.loc[:, 'fold'] = 100

    train = pd.concat([train_cp, train_ctl])
    train_features_with_fold = train_features.merge(train[['sig_id', 'fold']], on='sig_id')
    return train_features_with_fold

# <https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual?scriptVersionId=46264255>
def create_rankgauss(train, test, cols, QT_n_quantile_min, QT_n_quantile_max, seed):
    for col in (cols):
        kurt = max(kurtosis(train[col]), kurtosis(test[col]))
        QuantileTransformer_n_quantiles = n_quantile_for_kurt(kurt, calc_QT_par_kurt(QT_n_quantile_min,
                                                                                     QT_n_quantile_max))
        transformer = QuantileTransformer(n_quantiles=QuantileTransformer_n_quantiles, random_state=seed,
                                          output_distribution="normal")
        vec_len = len(train[col].values)
        vec_len_test = len(test[col].values)
        raw_vec = train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test[col] = \\
        transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
        return train, test

def calc_QT_par_kurt(QT_n_quantile_min=10, QT_n_quantile_max=200):
    # Calculation parameters of function: n_quantile(kurtosis) = k1*kurtosis + k0
    # For Train & Test datasets (GENES + CELLS features): minimum kurtosis = 1.53655, maximum kurtosis = 30.4929

    a = np.array([[1.53655, 1], [30.4929, 1]])
    b = np.array([QT_n_quantile_min, QT_n_quantile_max])

    return np.linalg.solve(a, b)

def n_quantile_for_kurt(kurt, calc_QT_par_kurt_transform):
    # Calculation parameters of function: n_quantile(kurtosis) = calc_QT_par_kurt_transform[0]*kurtosis + calc_QT_par_kurt_transform[1]
    return int(calc_QT_par_kurt_transform[0] * kurt + calc_QT_par_kurt_transform[1])

# Function to extract common stats features
def fe_stats(train, test, cols, kind='g'):
    for df in [train, test]:
        df[f'{kind}_sum'] = df[cols].sum(axis = 1)
        df[f'{kind}_mean'] = df[cols].mean(axis = 1)
        df[f'{kind}_std'] = df[cols].std(axis = 1)
        df[f'{kind}_kurt'] = df[cols].kurtosis(axis = 1)
        df[f'{kind}_skew'] = df[cols].skew(axis = 1)
    return train, test

def create_pca(train, test, cols, seed, kind='g', n_components=50, drop=False):
    train_ = train[cols].copy()
    test_ = test[cols].copy()
    data = pd.concat([train_, test_], axis = 0)
    pca = PCA(n_components=n_components, random_state=seed)
    data = pca.fit_transform(data)
    columns = [f'{kind}-pca-{i}' for i in range(n_components)]
    data = pd.DataFrame(data, columns = columns)
    train_ = data.iloc[:train.shape[0]]
    test_ = data.iloc[train.shape[0]:].reset_index(drop=True)
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop = True), train_], axis = 1)
    test = pd.concat([test.reset_index(drop = True), test_], axis = 1)
    return train, test

def fe_squared(train, test, cols):
    for df in [train, test]:
        for col in cols:
            df[f'{col}_squared'] = df[col] ** 2
    return train, test

# Function to scale our data
def scaling(train, test, cols):
    scaler = RobustScaler()
    scaler.fit(pd.concat([train[cols], test[cols]], axis = 0))
    train[cols] = scaler.transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    return train, test

def vt_selection(train, test, cols, VarianceThreshold_for_FS, kind='g', drop=True):
    train_ = train[cols].copy()
    test_ = test[cols].copy()
    data = train_.append(test_)
    
    if kind == 'g':
        mask = [True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, False, False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, False, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, False, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, True, True, True, False, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, False, True, False, False, True, False, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif kind == 'c':
        mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    data_transformed = data.loc[:, mask]

    train_features_transformed = data_transformed[ : train_.shape[0]]
    test_features_transformed = data_transformed[-test_.shape[0] : ]
    columns = [f'{kind}-vt-{i}' for i in range(train_features_transformed.shape[1])]
    train_features_transformed = pd.DataFrame(train_features_transformed)
    train_features_transformed.columns = columns
    test_features_transformed = pd.DataFrame(test_features_transformed)
    test_features_transformed.columns = columns
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop=True), train_features_transformed], axis = 1)
    test = pd.concat([test.reset_index(drop=True), test_features_transformed], axis = 1)
    return train, test

def lsvc_selection(train, test, cols, score_cols, kind='g', model='lsvc', drop=True):
    train_ = train[cols].copy()
    test_ = test[cols].copy()

    if model == 'lsvc':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_.values, [''.join(map(str, elem)) for elem in
                                                                               train[score_cols].values])
    else:
        raise NotImplementedError
    model = SelectFromModel(lsvc, prefit=True)

    train_ = model.transform(train_[cols].values)
    test_ = model.transform(test_[cols].values)
    columns = [f'{kind}-lsvc-{i}' for i in range(train_.shape[1])]
    train_ = pd.DataFrame(train_, columns=columns)
    test_ = pd.DataFrame(test_, columns=columns)
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop=True), train_], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_], axis=1)
    return train, test

def preprocess_fn(train_features, test_features, train_targets_scored, config):
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    train_features = make_folds(train_features, train_targets_scored, config.n_folds, config.seed[0])

    if config.rank_gauss:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = create_rankgauss(train_features, test_features, g_cols + c_cols,
                                                         config.QT_n_quantile_min, config.QT_n_quantile_max,
                                                         seed=config.seed[0])

    if config.feature_status:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = fe_stats(train_features, test_features, g_cols, kind='g')
        train_features, test_features = fe_stats(train_features, test_features, c_cols, kind='c')

    if config.pca:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = create_pca(train_features, test_features, g_cols,
                                                   seed=config.seed[0],
                                                   kind='g',
                                                   n_components=config.n_comp_GENES)
        train_features, test_features = create_pca(train_features, test_features, c_cols,
                                                   seed=config.seed[0],
                                                   kind='c',
                                                   n_components=config.n_comp_CELLS)

    if config.cell_squared:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = fe_squared(train_features, test_features, c_cols)

    if config.scaling:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = scaling(train_features, test_features, g_cols)
        train_features, test_features = scaling(train_features, test_features, c_cols)

    if config.vt_selection:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = vt_selection(train_features, test_features, g_cols,
                                                     config.VarianceThreshold_for_FS, kind='g', drop=True)
        train_features, test_features = vt_selection(train_features, test_features, c_cols,
                                                     config.VarianceThreshold_for_FS, kind='c', drop=True)

    if config.lsvc_selection:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = lsvc_selection(train_features, test_features, g_cols, score_cols, kind='g')
        train_features, test_features = lsvc_selection(train_features, test_features, c_cols, score_cols, kind='c')

    return train_features, test_features
```

## Inference

### Configurations

> 미리 설정한 config 값을 dataframe으로 관리하여 ensemble할 때 사용할 수 있음.

```text
class config:
    name = '2020_11_22_0029'
    description = 'Adam, nstep1'
    seed = [0, 1, 2]
    device = [1]
    num_scored = 206
    num_nonscored = 402

    # preprocess
    normalize = True
    remove_vehicle = True
    use_autoencoder = False

    feature_status = False
    rank_gauss = True
    pca = True
    cell_squared = False
    scaling = False
    vt_selection = True
    lsvc_selection = False

    n_comp_GENES = 463
    n_comp_CELLS = 60
    VarianceThreshold_for_FS = 0.84
    QT_n_quantile_min = 10
    QT_n_quantile_max = 200

    # Model
    n_d = 128
    n_a = 32
    n_independent = 1  
    n_shared = 1    
    cat_emb_dim = 4 
    n_steps = 1
    gamma = 1.0
    lambda_sparse = 0.

    # Training
    n_folds = 5
    n_pretrain_epochs = 100
    n_epochs = 100
    batch_size = 1024
    virtual_batch_size = 128
    momentum = 0.02

    lr = 2e-2
    weight_decay = 1e-5
    scheduler_step_size = 50
    scheduler_gamma = 0.9
    num_workers = 8
```

```text
commit_df = pd.DataFrame(columns = list(config.__dict__.keys()))
commit_df = commit_df.append(dict(config.__dict__), ignore_index=True)
```

### Predict testset

```text
def get_tabnet(
        input_dim, output_dim, n_d, n_a, n_steps, gamma,
        remove_vehicle, cat_emb_dim, n_independent, n_shared,
        virtual_batch_size, momentum, epsilon=1e-8, mask_type="sparsemax",
        ):
    if remove_vehicle:
        cat_idxs = [0, 1]
        cat_dims = [3, 2]   # Fixed
    else:
        cat_idxs = [0, 1, 2]
        cat_dims = [2, 3, 2]

    if cat_emb_dim == 0:
        cat_idxs = []
        cat_dims = []

    network = TabNet(
        input_dim,
        output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        virtual_batch_size=virtual_batch_size,
        momentum=momentum,
        mask_type=mask_type,
    )
    return network

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for feature in loader:
            if device:
                feature = feature.to(device)
            logit, _ = model(feature)
            pred = torch.sigmoid(logit)
            preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds
```

```text
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '../input/lish-moa'
log_dir = '../input/submission'
```

```text
test_cv_preds = []
for idx, config in commit_df.iterrows():
    # Load Data
    train_features = pd.read_csv(f'{data_dir}/train_features.csv')
    train_targets_scored = pd.read_csv(f'{data_dir}/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(f'{data_dir}/train_targets_nonscored.csv')
    test_features = pd.read_csv(f'{data_dir}/test_features.csv')
    submission = pd.read_csv(f'{data_dir}/sample_submission.csv')

    # Feature Engineering
    train_features, test_features = preprocess_fn(train_features, test_features, train_targets_scored, config)

    # Preprocess Data
    if config.remove_vehicle:
        train_features = train_features[train_features.cp_type != 'ctl_vehicle']
    train_features = train_features.merge(train_targets_scored, on='sig_id')
    train_features = train_features.merge(train_targets_nonscored, on='sig_id')
    train_features = cate2num(train_features)
    test_features = cate2num(test_features)

    if config.normalize:
        pass    # TODO; add normalization

    # Column Infos
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    nonscore_cols = [c for c in train_targets_nonscored.columns if c not in ['sig_id']]
    cate_cols = ['cp_time', 'cp_dose'] if config.remove_vehicle else ['cp_type', 'cp_time', 'cp_dose']
    g_cols = [c for c in test_features.columns if 'g-' in c]
    c_cols = [c for c in test_features.columns if 'c-' in c]
    num_g = len(g_cols)
    num_c = len(c_cols)
    num_cate = len(cate_cols)
    input_cols = g_cols + c_cols
    if config.cat_emb_dim > 0:
        input_cols = cate_cols + input_cols
    X_test = test_features.loc[:, input_cols].values
    
    for seed in config.seed:
        for fold in range(config.n_folds):            
            fold_train = train_features[train_features.fold != fold].reset_index(drop=True)
            fold_valid = train_features[train_features.fold == fold].reset_index(drop=True)

            X_train, y_train = fold_train.loc[:, input_cols].values, fold_train.loc[:, score_cols].values
            X_val, y_val = fold_valid.loc[:, input_cols].values, fold_valid.loc[:, score_cols].values

            model = get_tabnet(
                    input_dim=X_train.shape[-1],
                    output_dim=y_train.shape[-1],
                    n_d=config.n_d,
                    n_a=config.n_a,
                    n_steps=config.n_steps,
                    gamma=config.gamma,
                    remove_vehicle=config.remove_vehicle,
                    cat_emb_dim=config.cat_emb_dim,
                    n_independent=config.n_independent,
                    n_shared=config.n_shared,
                    virtual_batch_size=config.virtual_batch_size,
                    momentum=config.momentum,
                    mask_type="sparsemax",
                )
            in_ft = model.tabnet.final_mapping.in_features
            model.tabnet.final_mapping = nn.Linear(in_ft, y_train.shape[-1])
            path = os.path.join(log_dir, f'{config["name"]}_seed{seed}_fold{fold}.pt')
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

            test_loader = DataLoader(
                    DatasetWithoutLabel(X_test),
                    batch_size=config.batch_size,
                    collate_fn=collate_fn_test,
                    shuffle=False,
                )

            preds_test = predict(model, test_loader, device)
            test_cv_preds.append(preds_test)

test_preds_all = np.stack(test_cv_preds)
all_feat = [col for col in submission.columns if col not in ["sig_id"]]
# To obtain the same length of test_preds_all and submission
sig_id = test_features[test_features["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop=True)
tmp = pd.DataFrame(test_preds_all.mean(axis=0), columns=all_feat)
tmp["sig_id"] = sig_id
```

```text
sub = pd.merge(test_features[["sig_id"]], tmp, on="sig_id", how="left")
sub.fillna(0., inplace=True)
sub.to_csv("submission.csv", index = False)
```

![](../.gitbook/assets/image%20%281%29.png)



