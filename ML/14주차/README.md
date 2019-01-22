ML 스터디 14주차 : Confusion Matrix / Standardization, Normalization, Regularization / Data Preprocessing
=======================================================================================================

조민지
--------

***

정초이
---------

# Confusion Matrix

- 클래스 분류의 결과를 정리한 표

- 분류 모델이 정확한지 평가할 때 활용

- 머신러닝이나 통계적인 classification 문제에서 알고리즘의 성능을 visualization하는 table layout.

- 트레이닝 된 머신의 예상값과 실제 값을 비교하는 표

- 불균형한 Data set

  - ex) 14세 이하 10만명당 암 발병인원 14.8명. (암 발병 확률 : 0.014%)

    —> 굳이 머신러닝을 이용하지 않아도 확률이 너무 낮아서 '무조건 암이 아니다' 라고 해도

    정확도(Accuracy)가 99~100%이 나옴. —> 좋은 모델 X --> Confusion matrix 사용



## Positive & Negative

- 이항분류를 할 때 두가지 분류 중 한가지 분류에 더 관심이 많음
- ex) 화재 경보기 —> 화재 vs 일상 중 화재에 더 관심이 많음
- 이 때 **관심이 더 많은 쪽을 Positive, 그 반대를 Negative** 라고 함 



### TP, TN, FP, FN

![](./img/confusion/1.png)

> Actual condition : 실제 값, Predicted condition : 예상 값

- True Positive (진양성) : 예측과 실제 모두 P
- True Negative (진음성) : 예측과 실제 모두 N
- False Positive (위양성) : 실제로는 N인데 P로 예측된 것
- False Negative (위음성) : 실제로는 P인데 N으로 예측된 것



- 앞의 T & F : 예측이 맞았는지 틀렸는지
- 뒤의 P & N : 예측을 Positive로 했는지, Negative로 했는지



이 4가지 경우를 조합하여 여러가지 지표를 만들수 있음



## Error, Accuracy

- Accuracy (정확도) : 전체 중에서 올바르게 예측한 것이 몇개인가? (1에 가까울 수록 좋음)

![](./img/confusion/2.png)

- TP와 TN을 더하여, 전부의 합계로 나눔



## True Positive Rate, False Positive Rate

- True Positive Rate (진양성률) : 실제로 양성인 샘플에서, 양성이라고 판정된 샘플의 비율
  - ![](./img/confusion/3.png)
  - **검출률(Recall),** **감도(Sensitivity)**, 히트률(Hit Rate), 재현률 등이라고도 함.
  - 의미 : 전체 내가 맞추려는 것 중에서 내가 몇개를 맞췄는가? (1에 가까울 수록 좋음)



- False Positive Rate (위양성률) : 실제에는 음성인 샘플에서, 양성으로 판정된 샘플의 비율
  - **FP/(FP + TN)**
  - 오검출률, 오경보률 이라고도 함



## True Negative Rate (=Specificity)

- True Negative Rate (진음성률) : 실제로 음성인 샘플에서, 음성인 것으로 판정된 샘플의 비율

![](./img/confusion/5.png)

- **특이도 (Specificity)** 라고도 함



### Sensitivity & Specificity

이 두가지는 특히 중요한 척도라고 한다!
[여기](https://www.youtube.com/watch?v=U4_3fditnWg) 영상 참고



## Precision

- Precision (정밀도) : 양성으로 예측한 경우 중 진양성인 경우. 양성예측이 얼마나 정확한지!

![](./img/confusion/4.png)

- 정밀도에 대한 판단은 **양성의 경우를 계산하는 것**이 더 좋은 방법!

  - 양성예측의 경우, 예측 후 어떠한 행동이 뒤따르는 경우가 많음

    ex) 암 치료나 화재 대피 등.. 이 정밀도가 낮으면 불필요한 행동들을 해야할 수도 있음

  - 음성 예측의 경우, 확인이 어려움

    ex) 면접에서 안 뽑은 사람이 실력자인지 아닌지는 같이 일해보지 않으면 모름

- Positive Predictive Value (PPV) 라고도 함

- 의미 : 푼 문제 중에 맞춘 정답 개수가 몇개인가? (1에 가까울 수록 좋음)



## F1-score

- Precision과 Recall의 조화 평균을 이용하여 2개를 모두 고려해서 평가하는 방법
- 데이터 자체에 Positive 또는 Negative가 많을 경우에는 비율 자체가 편향되어 있기 때문에 조화평균을 이용
- ![](./img/confusion/6.png)
- 



## The Scoring matrics for multiclass classification

### AUC

- ROC 곡선의 아래쪽 부분
- ![](./img/confusion/7.png)
- 빨&노&파 —> ROC 커브
- ROC의 밑부분 영역이 클수록 모델의 성능이 우수한 것
  - 그림에서는 빨간 곡선이 가장 훌륭한 모델
- Reference Line : 0.5로 random 예측하는 것과 같음
  - 즉, 0.5 아래의 것은 아무 쓸모가 없음

### 분류할 주제가 여러가지일 때에는?

1. Classification Accuracy
2. Curve 아래의 ROC area 계산
   1. [튜토리얼](http://www.cs.bris.ac.uk/~flach/ICML04tutorial/)
   2. 1번보다 이게 낫다



***

남궁선
---------
## Data Preprocessing

### Missing data
|   A   |   B   |   C   |   D   |
|-------|-------|-------|-------|
|   1   |   2   |   3   |   4   |
|   5   |   6   |  NaN  |   8   |
|   10  |   11  |   12  |   NaN |

NaN : 손실된 데이터 (누락값)

***목표 : 손실된 데이터 문제를 해결해야한다.***
#### 1. 손실된 데이터가 있는 row(데이터) 또는 col(특성) 을 제거
1. 데이터가 손실된 row 들을 모두 제거한다.

    |   A   |   B   |   C   |   D   |
    |-------|-------|-------|-------|
    |   1   |   2   |   3   |   4   |

2. 데이터가 손실된 column 들을 모두 제거한다.

    |   A   |   B   |
    |-------|-------|
    |   1   |   2   |
    |   5   |   6   |
    |   10  |   11  |

-> 단점 : 도움이 될 수 있는 다른 데이터도 버릴 수 있다.

#### 2. 손실된 데이터에 영향을 받지 않는 학습 모델을 사용한다.
- Decision Tree 기반의 학습 모델(e.g. Random Forest) 등은 누락값에 영향을 받지 않아 데이터를 그대로 사용할 수 있다..
- 그러나 Linear Regression 이나 SGD 알고리즘은 누락값이 있을 때 사용할 수 없다.

#### 3. 대체값을 사용한다.
1. 누락값이 속한 row나 col의 평균값으로 치환한다 (산술평균, 조화평균, 기하평균 etc...)
    (예시 : 누락값이 속한 col의 산술평균으로 치환)

    |   A   |   B   |   C   |   D   |
    |-------|-------|-------|-------|
    |   1   |   2   |   3   |   4   |
    |   5   |   6   |   7.5 |   8   |
    |   10  |   11  |   12  |   6   |

2. 손실된 데이터를 전부 특정 값으로 치환한다.
    - ex) NaN을 전부 2로 치환한다.

    |   A   |   B   |   C   |   D   |
    |-------|-------|-------|-------|
    |   1   |   2   |   3   |   4   |
    |   5   |   6   |   2   |   8   |
    |   10  |   11  |   12  |   2   |

#### 4. ML 알고리즘을 사용하여 예측한다.
- Regression
- Decision trees
- Clustering algorithms (K-Mean\Median etc.)

### Categorical data
|   Color   |   Size   |   Price   |   Class label   |
|-------|-------|-------|-------|
|   green   |   M   |   10.1   |   class1   |
|   red   |   L   |   13.5   |   class2   |
|   blue  |   XL  |   15.3  |   class1   |

***목표 : 위 표에 있는 Color, Size, Class Label feature의 Catergorical data를 numerical 데이터로 바꾸어 주어야 한다.***

#### 1. Mapping ordinal features
XL : 3
L : 2
M : 1

|   Color   |   Size   |   Price   |   Class label   |
|-------|-------|-------|-------|
|   green   |   1   |   10.1   |   class1   |
|   red   |   2   |   13.5   |   class2   |
|   blue  |   3  |   15.3  |   class1   |

#### 2. Encoding class labels
Class1 : 0
Class2 : 1

|   Color   |   Size   |   Price   |   Class label   |
|-------|-------|-------|-------|
|   green   |   1   |   10.1   |   0   |
|   red   |   2   |   13.5   |   1   |
|   blue  |   3  |   15.3  |   0   |

#### 3. Performing one-hot encoding on nominal features
|Color_blue|Color_green|Color_red|   Size   |   Price   |   Class label   |
|-------|-------|--------|-------|-------|-------|
|0|1|0|   1   |   10.1   |   0   |
|0|0|1|   2   |   13.5   |   1   |
|1|0|0|   3  |   15.3  |   0   |

### Unbalanced dataset
|Positive|Negative|
|-----|-----|
|50|950|

***목표 : 한 쪽으로 과도하게 편향된 데이터셋의 문제를 해결한다.***

#### 1. Undersampling
- 다수 집단에서 랜덤하게 데이터를 버려서 소수집단과 갯수를 맞춘다.
- 소수 집단도 데이터셋이 충분하다면 유용하다.
- 단점 : 데이터가 손실된다.

    |Positive|Negative|
    |-----|-----|
    |50|50|
- 다양한 Undersampling 기법들
    - Tomek’s link method
    - Condensed Nearest Neighbour
    - One Sided Selection
    - Edited Nearest Neighbours
    - Neighbourhood Cleaning Rule  
#### 2. Oversampling
- 소수 집단에서 랜덤하게 데이터를 반복생성하여 다수 집단과 갯수를 맞춘다.
- 데이터셋의 크기가 작고, 데이터를 버리기 어려울 때 유용한다.
- 단점 : 데이터셋의 Noise가 증가한다.

    |Positive|Negative|
    |-----|-----|
    |950|950|
- 다양한 Oversampling 기법들
    - ADASYN
    - SMOTE  
#### 3. (Over + under) sampling
- SMOTEENN: SMOTE + ENN
- SMOTETomek: SMOTE + Tomek
#### Unbalanced dataset을 사용할 때의 모델 성능 평가방법
- 코헨의 카파(Cohen's kappa) : 실데이터와 예측된 분류 집합에 관한 강건한 척도
- F1 Score : 정확도와 재현률의 조화평균

### Multicollinearity
- 회귀분석에서 독립변수들 간에 강한 상관관계가 나타나는 문제
    e.g) X1 = aX2 + bX3, (X1, X2, X3 는 독립변수)

#### 1. 진단법
1. 결정계수 R^2값은 높아 회귀식의 설명력은 높지만 식안의 독립변수의 P-value값이 커서 개별 인자들이 유의하지 않는 경우가 있다. 이런 경우 독립변수들 간에 높은 상관관계가 있다고 의심된다.
2. 독립변수들간의 상관계수를 구한다.
3. 분산팽창요인(Variance Inflation Factor)를 구하여 이 값이 10을 넘는다면 보통 다중공선성의 문제가 있다.

#### 2. 해결법
1. 상관관계가 높은 독립변수중 하나 혹은 일부를 제거한다.
2. 변수를 변형시키거나 새로운 관측치를 이용한다.
3. 자료를 수집하는 현장의 상황을 보아 상관관계의 이유를 파악하여 해결한다.
4. PCA(Principle Component Analysis)를 이용한 diagnol matrix의 형태로 공선성을 없애준다. 

### Outlier
- 특정 범주보다 지나치게 떨어져있거나, 크거나 작은 값
<img src="./images/seon_1.png" width="50%">

#### 해결법
- Outlier 제거
- Quantile Binning
    - 연속형 숫자를 범주형으로 변환한다.
    - e.g) [1,2,3,4,5,6,7,8,100] -> [1,1,1,2,2,2,3,3,3]

### Data feature scaling
- Normalization
<img src="./images/seon_2.png" width="15%">
- Standardization
<img src="./images/seon_3.png" width="15%">
  
***
