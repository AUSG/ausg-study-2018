# ML 스터디 1주차 : Python Deep dive

## **남궁선**

### ***역사***
- 제작자 : Guido van Rossum
- Python 1.x : 1991년 발표
- Python 2.x : 2000년 10월 6일 배포
- Python 3.x : 2008년 12월 3일 배포

### ***특징***
- Platform independent : plaform 독립적
- Interpreter : C Interpreter 사용(표준)</br>
-> 현대에는 컴파일하여 바이너리 코드로 저장한 뒤 재실행시 바이너리 코드를 실행하여 속도를 향상시킨다.
- OO : 객체지향
- Dynamic typing : 실행 시 자료형 검사
- 다양한 유니코드 문자열 지원
- 들여쓰기를 사용하여 블록을 구분한다(공백 4칸).
- C/C++ 에 비하여 상대적으로 속도가 떨어지나, 범용으로는 큰 문제가 없다.
- Garbage Collector 가 내부에서 동작한다.
- Window, Mac OS, Unix, Linux, Palm OS 지원


### ***철학***
- Beautiful is better than ugly
- Explicit is better than implicit
- Simple is better than complex
- Complex is better than complicated
- Readability counts

### ***자료형***
1. 기본 자료형
    - Integer :int
    - Floating point : float
    - Complex number : complex
    - String : str
    - Unicode string    
    - Boolean : bool
    - Function
    - Bytes
    - Class

2. 집합형 자료형
    - List : list
    - Tuple : tuple
    - Dictionary : dict
    - Set : set, frozenset

### ***기술***
1. 유니코드
    ```
    유니코드 문자열의 한 글자가 플랫폼 및 빌드 옵션에 따라 무조건 1 or 2 or 4 바이트를 차지한다. 문자열 객체에서 가장 많은 공간(바이트)을 차지하는 문자를 기준으로 각 문자가 차지할 공간을 정한다.
    ```
2. CPython(표준)
    ```
    C로 구현된 파이썬. 파서, 인터프리터가 C로 구현되어있다.
    ```

3. Multi Threading
    ```
    파이썬은 멀티스레딩을 지원하기 위하여 GIL(Global Interpreter Lock), 즉 전역 인터프리터 락을 도입하여 사용하게 되었다. 따라서, python 스레드 10개를 만들어도 실제 Pthread/윈도우 스레드가 10개가 만들어지긴 하는데, GIL때문에 개중 동시에 하나밖에 안돌아가는 기이한 구조를 갖고 있다. 물론, 내부적으로 IO작업이 있을 시 바로 다른 스레드로 문맥 교환을 해주고, 바이트 코트를 100번 실행한 다음에는 인터프리터 차원에서 다른 스레드로 교체 해주므로 동시 작업 비슷한 효과가 난다. 이것은 구현이 매우 쉬워지고 빠른 개발을 할 수 있다는 장점이 있으나, 다중 코어 CPU가 보편화된 2006년 이후에는 다중 코어를 제대로 활용하지 못하는 구조적인 문제 때문에 성능에서 밀린다는 평가를 받게 되었다. 만일 특정 프로그램에 순진하게 CPU 코어를 2개 이상 동원하려고 할 경우, 뮤텍스(MutEx), 즉 한 스레드에 여러 개의 CPU가 연산을 행하여 내부 정보를 오염 시키는 것을 방지하는 역할을 맡는 GIL이 병목 현상을 일으켜 코어 하나를 쓸 때보다 오히려 성능이 크게 저하된다.

    이런 문제점 때문에 파이썬에서 병렬 처리가 필요할 때는 다중 스레드가 아닌 다중 프로세스로 GIL을 우회하는 방식을 사용한다. 2008년 이후에 multiprocessing이라는 모듈을 제공하는데 이 모듈은 자식 프로세스를 만드는 방향으로 다중 코어 사용 시 성능의 향상을 추구하고있다.
    ```
--------------
### 정초이

### 조민지