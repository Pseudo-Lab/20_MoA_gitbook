# 1st Place Winning Solution - Hungry for Gold

## 1st Place Winning Solution - Hungry for Gold

Assign: Chanran Kim Status: In Progress 타입: 노트북 리뷰, 디스커션 리뷰

#### References

* Discussion

  [Mechanisms of Action \(MoA\) Prediction](https://www.kaggle.com/c/lish-moa/discussion/201510)

* Notebook

  [Fork of Blending with 6 models\(5old-1new\)](https://www.kaggle.com/nischaydnk/fork-of-blending-with-6-models-5old-1new)

  [final\_best\_LB\_cleaned](https://www.kaggle.com/markpeng/final-best-lb-cleaned/notebook?scriptVersionId=48582693)

* 해당 팀원들이 나누어서 kernel을 작성함. 하지만 둘 다 인기는 없음... 이유는...?

## 1. Discussion

### 1. Overview

* 총 7개의 모델을 ensemble 하였다고 한다.
  * 3-stage NN stacking by non-scored and scored meta-features
  * 2-stage NN+TabNet stacking by non-scored meta-features
  * SimpleNN with old CV
  * SimpleNN with new CV
  * 2-heads ResNet
  * DeepInsight EfficientNet B3 NS
  * DeepInsight ResNeSt
* 다음 개요 다이어그램은 7개 모델을 블랜드하는 것을 보여줍니다.

Figure 1. Winning Weighted-Average Blend.

* 두 개의 최종 제출물은 가중 평균을 기반으로합니다. 우승한 블렌드는 best LB \(Table 1\)이며, best CV 블렌드 \(Table 3\)는 private LB에서 5위를 달성 할 수치입니다. 두 최종 제출물 모두 성능이 뛰어나고 강력하다는 것을 보여줍니다.

Table 1. Winning Blend with 7 Models \(Private LB: 0.01599\).

* 우승한 블렌드의 경우 블렌드 가중치 선택은 LB 점수와 단일 모델 간의 상관 관계라는 두 가지 요소를 기반으로했습니다.
* 더 나은 이해를 위해 대조군이 없는 각 모델 예측의 평균 상관 관계를 다른 단일 모델과 비교하여 계산했습니다 \(Table 2 참조\). 선택은 LB 점수를 최대화하기 위해 수동으로 수행되었습니다.
* 평균 상관 관계가 적은 모델에 더 높은 가중치가 부여되었습니다. 또한 일부 가중치는 이전 블렌드 제출 점수에 의해 결정되었습니다. 자세한 내용은 모델 다양성 섹션에서 확인할 수 있습니다.

Table 2. Mean Correlation Between the Submissions of Best Single Models.

* Best CV 블렌드에서 단일 모델 선택은 OOF \(Out-of-folds\) 예측을 기반으로합니다.
* Optuna의 TPE \(Tree-structured Parzen Estimator\) 샘플러와 Scipy의 SLSQP \(Sequential Least Squares Programming\) 방법을 사용하여 CV 최적화 가중치를 검색했습니다. Optuna \(3000 또는 5000 시도 미만\) 및 SLSQP의 결과 가중치는 거의 동일합니다.
* best CV의 로그 손실은 5개 모델에서 0.15107 \(private LB : 0.01601\)입니다. 흥미롭게도 검색 결과는 2-stage NN + TabNet 및 Simple NN 모델을 제거하여 성공적인 혼합에 기여했습니다.

Table 3. Best CV Blend with 5 Models \(CV: 0.015107, Private LB: 0.01601\).

* DeepInsight CNN의 추가는 \(다른 얕은 NN 모델과 다르게\) 다양성이 높기 때문에 최종 블렌드에서 중요한 역할을 했습니다. 호기심으로 우리는 CNN 모델을 사용하거나 사용하지 않은 블렌드의 로그 손실 점수를 비교했습니다 \(표 4 참조\).
* DeepInsight CNN 모델을 포함시키면서 비공개 LB 점수가 거의 0.00008 향상되었습니다!

Table 4. Winning Blends with/without DeepInsight CNNs.

### 2. Cross-Validation Strategies

* 대부분의 모델은 drug\_id 정보를 사용하여 @cdeotte가 공유하는 새로운 이중 계층화 CV를 사용하는 Simple NN 모델을 제외하고 MultilabelStratifiedKFold \(old CV\)를 기반으로합니다. 모델에서 CV 분할에 다른 시드를 사용했습니다. CV 전략의 선택은 CV 로그 손실 및 LB 로그 손실에 대한 정렬의 장점을 기반으로합니다. CV의 K는 5 또는 10입니다. 또한 분산을 줄이기 위해 여러 시드를 사용하여 모델을 훈련했습니다.
* 특히, new CV 모델의 로그 손실 점수가 훨씬 높기 때문에 최적화된 CV 가중치를 검색하기 위해 old CV와 new CV의 OOF 파일을 결합하는 것이 어렵다는 것을 알았습니다. 우리는 Best CV 블렌드에 대해 old CV 모델만 선택했으며, old CV에서 0.15107, private LB에서 0.01601 점을 기록했습니다.

### 3. 세부사항

* 우리팀에서 Nischay와 Kibuna는 DeepInsight CNN과 2- 헤드 ResNet 모델의 복제에 집중하는 동안 3-stage NN, 2-stage NN + TabNet 및 Simple NN 모델의 기능 엔지니어링 및 다단계 스태킹에 기여했습니다.
* **작고 불균형이 심한 다중 레이블 데이터셋에서 과적합의 위험을 극복하기 위해** 모델 학습 프로세스의 정규화 방법으로 **레이블 평활화\(label smoothing\)** 및 **가중치 감소\(weight decay\)**를 적용했습니다. 레이블 평활화는 매우 잘 작동했으며 모델이 예측에 대해 너무 확신하지 못하도록하여 과적합 가능성을 크게 줄였습니다.
* 모델을 시각적으로 더 잘 이해할 수 있도록 각 단일 모델에 대한 모델 아키텍처 다이어그램을 제공하여 내부 학습 프로세스와 NN 토폴로지를 설명했습니다. 이 섹션에서는 각 모델을 자세히 소개합니다.

### 4. Models

#### 1\) 3-Stage NN

* 이것은 최고의 단일 모델 \(CV : 0.01561, public LB : 0.01823, private LB : 0.01618\)이며 CV 및 LB 점수 모두에 대한 최종 혼합에 크게 추가되었습니다. Nischy와 Kibuna는 모델 예측을 메타 기능 및 다양한 엔지니어링 기능으로 다단계 스태킹이라는 아이디어를 기반으로 좋은 결과를 생성하는 데 큰 역할을했습니다. 제어 그룹\(Control group\) 행이 학습셋에서 제거됩니다.

Figure 2. 3-stage NN Model.

Figure 3. 3-FC Base NN Block Topology.

* 그림 3은 공개 노트북과 유사한 모델 기반 NN 블록의 자세한 토폴로지를 보여줍니다. 성능을 높이기 위해 훈련 설정을 일부 변경했습니다. 다단계 훈련에서는 **조정된 드롭 아웃 값**을 제외하고 동일한 아키텍처가 사용되었습니다.
* Stage 1
  * 먼저, **UMAP 및 요인 분석**을 사용하여 유전자 및 세포 기능에서 **extra feature을 생성**했습니다. **Quantile Transformer**는 원-핫 인코딩 기능을 제외한 모든 기능에 적용되었습니다. 이 단계에서 은닉 뉴런 크기가 2048 개인 NN 모델은 0 값 \(332 개 대상 포함\)을 제외한 점수가 매겨지지 않은 대상에 대해 15 에포크 동안 훈련됩니다. 채점되지 않은 예측은 다음 단계의 meta-features으로 재사용되었습니다.
* Stage 2
  * 점수가 매겨지지 않은\(unscored\) meta-features에 Quantile Transformer를 다시 적용하여 이전 단계의 원래 feature와 결합하고 점수가 매겨진 대상에 대해 2048개의 숨겨진 뉴런 크기로 25 epoch 동안 또 다른 NN을 학습했습니다. 마찬가지로 점수가 매겨진 예측은 다음 단계의 메타 기능으로 재사용되었습니다.
* Stage 3
  * 마지막 단계에서는 점수가 매겨진 meta-features에 Quantile Transformer를 적용하고 해당 meta-features 기반으로 1024 개의 숨겨진 뉴런으로 이루어진 NN을 25 epoch에 대해서 재학습 시켰으며, 라벨 스무딩 후 학습 데이터에서 target이 0.0005, 0.9995 구간으로 잘렸습니다.
* 각 단계에서 유사한 설정이 사용되었습니다
  * lr: 5e-3
  * batch\_size:  256
  * OneCycleLR \(decay:  1e-5 및 maximum lr: 1e-2\)
* 최종 모델은 0.01561의 CV 로그 손실을 생성했는데, 이는 모델 예측에 대한 클리핑으로 인해 2단계 모델보다 약간 높았지만 LB에서 정말 좋은 점수를 받았습니다.

## 2. Notebook

* 여러명이 작업한 것을 그냥 합쳐놓은 것으로 보인다.
* 파일 이름으로 구성이 나뉘어져 있는데,
  * 101-Preprocess.ipynb
  * 203-101-nonscored-pred-2layers.ipynb
  * 503-203-tabnet-with-nonscored-feature-

## 3. Review

* 정형 데이터에서는 Stacking이 매우 중요!
* Feature Engineering에서 UMAP
* non-scored feature의 활용을 stacking을 통해서
* CV 차이가 3% 이상?

