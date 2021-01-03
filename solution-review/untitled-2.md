# 3rd Place Public - We Should Have Trusted CV - 118th Private

## Various Models

* TabNet - LB 1830 \(private 1622\)
* DAE \(denoising autoencoder\) - LB 1830 \(private 1628\)
* EffNet \(deepinsight images\) - LB 1840 \(private 1618\)
* NVIDIA RAPIDS UMAP
* MX10 \(transfer learning\) - LB 1842 \(private 1628\)
* PyTorch NN - LB 1833 \(private 1619\)

## Post-processing \(11st → 118th\)

> 각 목표값에 대한 예측과 로그 손실을 개별적으로 그림으로 표시하면 모형이 특정 대상을 인식할 수 없다는 것을 알 수 있습니다. 이러한 어려운 목표값의 경우, 모델은 모든 drug에 대한 target을 training 데이터의 평균을 예측하기만 하면 됩니다. 아래 두 가지 예를 참조하십시오. x축은 drug 인덱스입니다. Y 축은 예측값입니다.

![](../.gitbook/assets/image%20%283%29.png)

![](../.gitbook/assets/image%20%282%29.png)

> Test 데이터의 분포가 Train 데이터와 다를 경우, Train Data에서 드문 Target이 Test Data에서 자주 발생할 수 있습니다. 그리고 Train Data에서 자주 발생하는 Target은 Test Data에서 드물게 발생할 수 있습니다. Test Data 평균에 대해 모르고 모델이 Target을 인식하지 못하면 모든 Target 평균에 더 가까운 값을 예측하는 것이 좋습니다. 따라서 인식되지 않는 "흔히 있는" Target 예측에 0.75를 곱하고 인식되지 않는 "희귀한" Target 예측에 4를 곱하여 예측을 후 처리했습니다 \(실제 승수는 Train Data 평균과 통계적 기대치에 따라 다르며, [여기 노트](https://www.kaggle.com/cdeotte/3rd-place-public-lb-1805/)에 나와 있습니다\).

```text
# CONVERT PROBABILITIES TO ODDS, APPLY MULTIPLIER, CONVERT BACK TO PROBABILITIES
def scale(x,k):
    x = x.copy()
    idx = np.where(x!=1)[0]
    y = k * x[idx] / (1-x[idx])
    x[idx] =  y/(1+y)
    return x

# DECREASE PREDICTIONS FOR UNCERTAIN OFTEN TARGETS IN TRAIN
FACTOR = 0.725 # 5:1 odds
ct = 0

for c in COLS:
    t_sum = train[c].sum()
    m1 = train[c].mean()
    m2 = sub[c].mean()
    ratio = m2/m1
    
    # LINEAR FORMULA
    m = (FACTOR + (70-t_sum/6)/(70-25)*0.11) / ratio
  
    if m<1:
        print(c)
        print('multiplier = %.3f'%m,'because t_sum = %i, ratio = %.2f'%(t_sum,ratio))
        sub[c] = scale(sub[c].values,m)
```

> acetylcholine\_receptor\_agonist multiplier = 0.738 because t\_sum = 190, ratio = 1.11 new ratio = 0.82, effective multiplier = 0.741

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d1a3357-43aa-4da1-8c4a-e3be7c8eeeb9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d1a3357-43aa-4da1-8c4a-e3be7c8eeeb9/Untitled.png)

### Post Process Improves Public LB 0.00011

> 위의 post-processing은 public LB를 0.00011까지 향상시킵니다. Public test dataset에는 특별히 조정되지 않습니다. 모델이 training dataset과 다른 분포를 갖는 dataset을 예측하고 있다면 사후 프로세스가 로그 손실을 개선할 것입니다. 안타깝게도 private test dataset은 trainaing과 유사하며 post-processing로 인해 private LB가 0.00008만큼 저하됩니다 \(테스트 데이터 세트의 분포는 [여기](https://www.kaggle.com/c/lish-moa/discussion/200832)에서 설명함\).

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0cad7a24-54fe-42ad-907c-075dd59e7b05/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0cad7a24-54fe-42ad-907c-075dd59e7b05/Untitled.png)

> public test datset에는 이러한 rare target의 train dataset보다 6배 많은 target이 있습니다.

#### Private Test is Same as Train

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7c084161-8ec4-41b5-b455-23b98f0371d4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7c084161-8ec4-41b5-b455-23b98f0371d4/Untitled.png)

