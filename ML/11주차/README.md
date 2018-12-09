# ML 스터디 11주차 : k-NN Algorithm

## **남궁선**
## k-Nearest Neighbor Alogrithm

<img src="./images/1.png" width="50%">

### 1. 특징
- Classification, Regression에 모두 응용된다.
- 함수를 학습하지 않고 훈련 데이터 자체를 기억해두어 사용하는 [비모수](https://ko.wikipedia.org/wiki/%EB%B9%84%EB%AA%A8%EC%88%98_%ED%86%B5%EA%B3%84) 방식
- k-NN 분류에서 출력은 소속된 항목이다. 객체는 k개의 최근접 이웃 사이에서 가장 공통적인 항목에 할당되는 객체로 과반수 의결에 의해 분류된다
- k-NN 회귀에서 출력은 객체의 특성 값이다. 이 값은 k개의 최근접 이웃이 가진 값의 평균이다.
- 일관성있는 결과를 도출한다.
- 많은 계산량을 필요로한다.

### 2. k-NN 분류 알고리즘
1. 훈련 데이터들을 좌표공간에 매칭시킨다.
2. Hyperparameter k 의 갯수를 선택한다.
> 동률을 피하기 위하여 k는 홀수로 설정
3. 검증 데이터를 좌표공간에 표시한다.
4. 검증데이터로 부터 가장 가까운 거리의 k개의 훈련데이터를 구한다.  
5. k개의 훈련 데이터중 가장 많이 포함된 분류와 검증데이터의 정답을 비교한다.
6. 2~5를 반복하여 최적의 k를 찾는다.

### 3. 다양한 k-NN 기법들
- 거리 척도
    - 입력데이터에 적합한 방식을 선택한다.
    - Euclidean distance
    - Cosine similarity
    - Manhattan distance
    - Hamming distance
- 전처리
    - 특징 추출
        - 입력데이터로부터 특징을 추출한다. 
    - 차원 축소
        - 보통 10차원을 넘을 시에 차원축소를 진행한다.
    - 특징 추출과 차원 축소는 주성분 분석(PCA), 선형 판별 분석(LDA), 또는 정준 상관 분석(CCA) 기술을 전처리 과정으로 사용하면 한 과정으로 합칠 수 있다. 
- 데이터 축소
    - 대량의 데이터 집합에서 부정확하게 분류되고 있는 ***항목 이상치*** 들을 처리하는 과정
    - CNN(Compressed Nearest Neighbors) 알고리즘을 사용하여 제거한다.
    - 자세한 내용은 [링크](https://ko.wikipedia.org/wiki/K-%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)참조