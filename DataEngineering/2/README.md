# DE 스터디 2주차 : 배경지식

## **허진호**

### 주제 : NoSQL, CAP theory and NoSQL use case

### NoSQL이란

> "Not Only SQL"
>
> 기존의 관계형 DBMS가 갖고있는 특성 뿐만 아니라 다른 특성들을 부가적으로 지원하다는 것을 의미.



### NoSQL의 특징

1. 데이터의 스키마를 미리 정의할 필요가 없으며, 시간이 지나더라도 언제든지 바꿀 수 있음

2. 여러 데이터베이스 서버를 묶어서(클러스터링) 하나의 데이터베이스를 구성할 수 있음

3. 데이터와 트래픽이 증가함에 따라 기존의 수직적 확장에서 장비의 수를 늘리는 수평적 확장 방식이 가능

   > 수직적 확장?
   >
   > 기존의 H/W를 성능이 더 좋은 H/W로 바꾸거나 CPU, RAM, DISK등을 증설하는 방법

4. Shard key를 기준으로 하나의 테이블을 수평 분할하여 서로 다른 클러스터에 분산 저장하고 질의 가능

   > Shard key란?
   >
   > 클러스트의 샤드들 간에 컬렉션의 도큐먼트를 어떻게 분산할 것인가를 결정하는 것
   >
   > 참조) http://cinema4dr12.tistory.com/508 

5. 테이블 간 연결해서 조회할 수 있는 조인 기능이 없음

6. 관계형 데이터베이스에서는 지원하는 데이터 처리 완결성(ACID)이 보장되지 않음

   > ACID이란?
   >
   > 데이터베이스 트랜잭션이 안전하게 수행된다는 것을 보장하기 위한 성질
   >
   > 1. 원자성 - 트랜잭션이과 관련된 작업이 부분적으로 실행되다가 중단되지 않는 것을 보장
   > 2. 일관성 - 트랜잭션이 실행을 성공적으로 완료하면 언제나 일관성 있는 데이터베이스 상태를 유지하는 것
   > 3. 고립성 - 트랜잭션을 수행 시 다른 트랜잭션의 연산 작업이 끼어들지 않도록 보장하는 것
   > 4. 지속성 - 성공적으로 수행된 트랜잭션은 영원히 반영되어야 함을 의미한다. 





### NoSQL의 데이터 모델 종류

**1) Key-Value Stores**

![](https://cdn-images-1.medium.com/max/600/1*swUK-eLWsk-wudXSXRgyYQ.png)



Key/Value Stores는 고유한 Key에 하나의 Value를 가지고 있는 형태를 의미한다. 이런 단순한 구조때문에 GET이나 PUT 함수만을 지원한다. 이 데이터 모델의 장점은 단순함이다. 매우 간단한 추상화를 통해 데이터를 쉽게 분할하고 쿼리할 수 있으므로 시스템은 짧은 대기시간과 높은 처리량을 달성할 수 있다. 하지만 복잡한 쿼리 작업이 필요한 경우에는 강력하지 않다. Key/Value Stores의 예로는 Redis가 있다.





**2) Wide Column Stores**

![](https://cdn-images-1.medium.com/max/800/0*Pi2jgiFuXOjQuC5_.png)

![](https://cdn-images-1.medium.com/max/800/0*VjyLe9cZfVYDd6mI.png)



Wide column stores는 Key-Value Store가 가지는 단점들을 보완한 형태이다. Key-Value는 value 필드를 필터링 할 수 없고 전체 값을 반환하거나 전체를 업데이트 해야하는 단점을 가지고 있다.  위 그림같이 Key-Value Store에 값 부분에 열을 추가함으로써 클라이언트가 요구하거나 업데이트 해야할 부분을 지정할 수 있다. Wide Column store는 단일 항목 수준에서 열이 지정되어 있기 때문에 전체 스키마가 존재하지 않기 때문에 관계형 데이터베이스와 같지 않다.





**3) Document  Stores**

: ![](https://cdn-images-1.medium.com/max/600/1*gdxUo2ojiTX2JQIkA2hxcQ.png)



Document Stores는 값을 JSON 문서와 같은 반 구조화된 형식으로 제한하는 Key-Value store이다. Key-Value Store과의 차이점은 값이 구조화되어 있기 때문에 데이터 값 내에서 쿼리를 실행할 수 있다는 것이다. Document Stores에서는 전체 데이터를 가져올 필요 없이 데이터 값이 나타내는 것을 이해하므로 필요한 데이터를 바로 검색할 수 있다. 전체 문서를 id로 쉽게 가져올 수 있다. 또한 문서 부분만을 검색하는 쿼리도 실행할 수 있다.  Mongo DB가 이 종류에 해당됩니다.





**4) Graph Stores**

: 그래프 DB는 SQL 또는 다른 모든 NoSQL 데이터베이스와 매우 다르게 데이터를 그래프로 구성한다. 페이스북 친구 데이터는 전형적인 예이다. 그래프 데이터베이스는 노드와 엣지로 구성되며 둘 다 중요한 정보를 포함할 수 있다. 예를 들어, 두 개의 다른 노드가 사람을 나타낼 수 있고 모든 노드 프로파일 세부 정보(이메일, 주소, 사진 등)가 해당 노드와 함께 저장된다. 그들 사이의 엣지는 두 사람이 친구라는 것을 나타낼 수 있고, 그들의 우정의 기간과 같은 데이터를 저장할 수 있다.

[출처1]: https://medium.baqend.com/nosql-databases-a-survey-and-decision-guidance-ea7823a822d
[출처2]: https://medium.com/@adamberlinskyschine/wtf-is-nosql-f1338cec6053
[출처3]: https://medium.com/indexoutofrange/what-is-the-problem-with-key-value-databases-and-how-wide-column-stores-solve-it-5445efbae538





### CAP 이론이란

분산된 시스템이 가지는 세가지 특성을 동시에 충족시키는 것은 불가능하며, 이 중 두 가지만을 취할 수 있다는 것

**1. Consistency (일관성)** - 모든 노드가 같은 시간에 같은 데이터를 보여줘야 한다.

**2. Availability (가용성)** - 특정 노드가 장애가 나도 서비스가 가능해야 한다.

> 또는 데이터 저장소에 대한 모든 동작(read, write 등)은 항상 성공적으로 리턴되어야 한다.
>
>   “서비스가 가능하다”와 “성공적으로 리턴”이라는 표현이 애매하다. 
>
> CAP를 설명하는 문서들 중 “Fail!!”이라고 리턴을 하는 경우도 “성공적인 리턴”이라고 설명하는 것을 보았다.
>
> 출처: <http://hamait.tistory.com/197> [HAMA 블로그]  

**3. Partitions Tolerance (분리 내구성)** - 일부 메시지를 손실하더라도 시스템은 정상 동작을 해야 한다.

[출처]: http://wiki.nex32.net/%EC%9A%A9%EC%96%B4/cap%EC%A0%95%EB%A6%AC





### CAP Theorem 오해와 진실

![](http://eincs.com/images/2013/06/truth-of-cap-theorem-diagram.png)

Partition Tolerance는 분할 내구성 보다는 분할 용인이라고 번역하는 것이 맞다. P의 정의는 네트워크가 임의의 메시지 손실을 할 수 있는 것을 허용하느냐이다. P를 포기하려면 절대로 장애가 나지 않는 네트워크를 구성해야 하지만 그런 것은 세상에 존재하지 않는다. 따라서 P는 언제나 선택되어야 하며 결국 Availability와 Consistency중 하나를 선택해야하는 것이다. 또한 CAP Theorem은 분산시스템이 전제조건이므로 RDBMS를 CAP에 적용하는 것은 맞지 않다. 그러므로 RDBMS를 CA라 하는 것은 맞지 않다.

[출처]: http://eincs.com/2013/07/misleading-and-truth-of-cap-theorem/



### PACELC Theorem

![](http://happinessoncode.com/images/cap-theorem-and-pacelc/pacelc.png)

CAP 이론의 이러한 단점들을 보완하기 위해 나온 이론이 바로 PACELC 이론이다. CAP 이론이 네트워크 파티션 상황에서 일관성-가용성 축을 이용하여 시스템의 특성을 설명한다면, PACELC 이론은 거기에 정상 상황이라는 새로운 축을 더한다. PACELC는 P(네트워크 파티션)상황에서 A(가용성)과 C(일관성)의 상충 관계와 E(else, 정상)상황에서 L(지연 시간)과 C(일관성)의 상충 관계를 설명한다.



### NoSQL use case?

**1) MongoDB**

MongoDB는 오픈소스 크로스 플랫폼에 도큐먼트 지향형 NoSQL 데이터베이스이다. 명확한 스키마 정의가 없는 경우 MongoDB를 선택하는 것이 좋고 실시간 분석을 위해 확장성과 캐슁이 필요한 경우 MongoDB를 선택하는 것이 좋다. 그러나 트랜잭션 데이터(계정 시스템 등)에는 적합하지 않다. MongoDB는 모바일 앱, 컨텐츠 관리, 실시간 분석, IoT 애플리케이션 등에 자주 사용된다.

 

**2) DynamoDB**

DynamoDB는 AWS 스택을 이미 사용하고 있고 NoSQL 데이터베이스가 필요한 경우 가장 먼저 적용을 고려해보아야 할 데이터베이스이다. 



**3) Redis**

Redis는 인메모리에서 돌아가는 NoSQL 데이터베이스이다. 인메모리DB라  빠른 속도가 강점이다. 카카오뿐아니라 트위터, 페이스북, 인스타그램, 네이버 등 유명 인터넷 업체들이 사용자들의 대규모 메시지를 실시간으로 처리하기 위해 Redis를 사용 중이다.  Redis의 특징 중 하나는 '싱글 쓰레드'라는 점이다. 싱글 쓰레드는 1번에 1개의 명령어만 실행할 수 있다. 한 서비스에서 요청된 명령어에 대한 작업이 끝나기 전까진 다른 서비스에서 요청하는 명령을 못 받아들인다. 모든 키를 보여주거나 플러싱하는 명령어는 테스트 환경이나 소량의 데이터를 관리하는 시스템에서 모니터링하는 용도로만 써야 한다. 실행 대상을 전수처리하기 때문에 점차 데이터를 쌓아가는 환경에서는 운영에 차질을 빚을 정도로 속도가 느려진다. 일반적으로 텍스트 위주의 데이터를 처리하는데 적합하다.

[출처]: http://www.zdnet.co.kr/news/news_view.asp?artice_id=20131119174125



**4) ElasticSearch**

루씬을 기반으로 한 텍스트 검색 엔진 라이브러리이다.  사전 매핑 없이 JSON 문서 형식으로 입력하면 별도의 이벤트가 없어도 바로 색인을 시작한다. 이렇게 저장된 데이터는 별도의 재시작/갱신 없이도 바로 검색에 사용될 수 있다. 이는 곧 색인 작업이 완료됨과 동시에 검색이 가능하다는걸 의미한다. 이러한 특징들 덕분에 SOLR(솔라) 와 비교하여 실시간 검색 엔진 구현에 좀 더 적합하다.

[출처]: http://louie0.tistory.com/131



