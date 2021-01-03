# nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp

[https://www.kaggle.com/hwigeon/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp](https://www.kaggle.com/hwigeon/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp)

* Cache 클래스는 살펴볼만함!

```text
from src.utils.cache import Cache
```

* CNN보다는 point-wise FFN이 낫지 않았을까?
  * model들의 output을 CNN을 사용하여 weight를 잘 섞음
* 2-stage
* weight optimization
  * Scipy minimize function w/Nelder-Mead

