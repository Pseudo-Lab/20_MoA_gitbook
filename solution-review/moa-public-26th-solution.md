# \[MoA\] Public 26th solution



### Source

kernel : [https://www.kaggle.com/kazumitsusakurai/submission-for-moa](https://www.kaggle.com/kazumitsusakurai/submission-for-moa)

discussion : [https://www.kaggle.com/c/lish-moa/discussion/201565](https://www.kaggle.com/c/lish-moa/discussion/201565)

### Content

* 전처리

  * 오토인코더를 사용, TabNet만 PCA

* 오토인코더란?

  * 오토인코더\(Autoencoder\)는 아래의 그림과 같이 단순히 입력을 출력으로 복사하는 신경망이다. 어떻게 보면 간단한 신경망처럼 보이지만 네트워크에 여러가지 방법으로 제약을 줌으로써 어려운 신경망으로 만든다. 예를들어 아래 그림처럼 hidden layer의 뉴런 수를 input layer\(입력층\) 보다 작게해서 데이터를 압축\(차원을 축소\)한다거나, 입력 데이터에 노이즈\(noise\)를 추가한 후 원본 입력을 복원할 수 있도록 네트워크를 학습시키는 등 다양한 오토인코더가 있다. 이러한 제약들은 오토인코더가 단순히 입력을 바로 출력으로 복사하지 못하도록 방지하며, 데이터를 효율적으로 표현\(representation\)하는 방법을 학습하도록 제어한다.

* PCA란?

  * 주성분 분석은 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다. 이 때 서로 연관 가능성이 있는 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간의 표본으로 변환하기 위해 직교 변환을 사용한다.

* 블렌딩

  * 2 hidden layer nn
  * 3 hidden layer nn
  * 4 hidden layer nn
  * TabNet
  * DeepInsight 모델

* TabNet 모델

  * [https://arxiv.org/abs/1908.07442](https://arxiv.org/abs/1908.07442)

* DeepInsight 모델
  * [https://github.com/deepinsight](https://github.com/deepinsight)
* 추가작업

  * rank gauss
  * statistical features 추가 \(sum, mean, std, kurt, skew, median, etc..\)
  * smoothing loss



  **Take over**

  * 오토인코딩 코드
  * 블랜딩 모델 코드
  * deepinsight 모델

