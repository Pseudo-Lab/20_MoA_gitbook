# Public 46th / Private 34th Solution



## Method

![](../.gitbook/assets/image%20%286%29.png)

### pre-processing

* Add stat feature
* Add PCA feature
* Variance Threshold
  * Feature selector that removes all low-variance features.
* Rankgauss

  * Assign a spacing between -1 and 1 to the sorted features
  * Apply inverse error function → makes a gaussian distribution

![](../.gitbook/assets/image%20%2815%29.png)

### modeling

* Label Smoothing
* Transfer Learning by nonscored for NN, ResNet
* Shallow model
  * Short epoch, learning until limit before loss is NaN by NN
  * n\_steps=1, n\_shared=1 by TabNet
* Thresholdlng NN: input → linear → tanh → NN

### post-processing

* Ensemble. In particular, Tabnet and NN's ensemble is effective.
* 2 Stage Stacking by MLP, 1D-CNN, Weight Optimization

## What doesn't work

* pre-processing
* modeling
* post-processing

## Code Structure

1. Dataset Structure:

* Model Weights
* Inference codes for each models
* Python Packages

![](../.gitbook/assets/image%20%284%29.png)

2. Install python packages → Inference stage1 models → get the predictions of each models

![](../.gitbook/assets/image%20%2814%29.png)

![](../.gitbook/assets/image%20%2813%29.png)

3. Stacking \(MLP, 1D CNN, Weight Optimization\)

![](../.gitbook/assets/image%20%2812%29.png)

* **Not enough time**
  * Target Encoding to g-,c- bin's feature
  * XGBoost, CatBoost, CNN model for single model \(Stage 1\)
  * GCN model for stacking model \(Stage 2\)
  * Netflix Blending
  * PostPredict by LGBMWe noticed that there are columns that NN can't predict, but LGBM can \(e.g. cyclooxygenase\_inhibitor\). Therefore, we came up with the idea of repredicting only the columns that are good at LGBM. But not enough time.

## Takeover

* Clean Inference Code
  * Ensemble using various different models
  * Stacking

## \[Update\] Private 3rd Rank with Various Stacking

![](../.gitbook/assets/image%20%2811%29.png)

0.01608 → 0.01599

### 2D-CNN Stacking

![](../.gitbook/assets/image%20%285%29.png)

### GCN Stacking

![](../.gitbook/assets/image%20%2810%29.png)

* Adjacency Matrix: Matrix of ones / \(\# of classes\)^2
* Node: \(1, 5\)

