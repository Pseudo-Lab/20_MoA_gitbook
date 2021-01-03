# TabNet Training



```text
import os

import torch

class config:
    #_workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    _workspace_path = '/content/drive/My Drive/2_kaggle/20_moa/model/lish-moa-dev-ohwi-tabnet/'

    name = '2020_11_20_0001'
    seed = [42, 43, 44]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/content/drive/My Drive/2_kaggle/20_moa/share/input'
    log_dir = os.path.join(_workspace_path, 'logs',  name)
    bkup_dir = os.path.join(_workspace_path, 'repo')
    sub_dir = os.path.join(_workspace_path, 'notebook', 'csv')  # 수동으로 만들어야?

    num_scored = 206
    num_nonscored = 402

    # preprocess
    normalize = True    # 논문은 따로 global normalization을 하지 않고 batch norm을 사용
    remove_vehicle = True
    use_autoencoder = False

    # from <https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual>
    # of version 27
    num_pca_g = 463
    num_pca_c = 60

    # Model
    n_d = 128
    n_a = 128
    n_independent = 1   # feature extractor layer-wise cell
    n_shared = 2        # feature extractor shared cell
    cat_emb_dim = 0     # 논문에 따로 내용이 없는 부분; 0 이면 사용 안함
    n_steps = 1
    gamma = 1.3
    lambda_sparse = 0 #0.01

    # Training
    n_folds = 5
    n_train_steps = 6900
    n_epochs = None      # 둘 중 하나로 선택
    batch_size = 512
    virtual_batch_size = 128
    momentum = 0.9
    lr = 2e-2
    weight_decay = 1e-5
    scheduler_step_size = 50
    scheduler_gamma = 0.95
    num_workers = 8
```

![](../.gitbook/assets/image%20%287%29.png)

#### n\_independent=4, n\_shared =4

![](../.gitbook/assets/image%20%2813%29.png)

#### label\_smoothing 0.001 → 0.003

