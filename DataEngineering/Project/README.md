[네이버 뉴스 크롤링]
* 크롤링은 Python으로 진행
    * bs4 실습
        * https://www.crummy.com/software/BeautifulSoup/bs4/doc/
        * https://www.dataquest.io/blog/web-scraping-tutorial-python/
    * Scrapy 실습
        * https://doc.scrapy.org/en/latest/intro/tutorial.html
    * Scrapy + bs4/ 혹은 둘중에 한가지만 사용
* MongoDB에 적재
    * MongoDB는 단일 노드 Cluster/ 각자 EC2에 올림
        * Title
        * Data
        * Main Text
        * # like
        * # comment 
    * BSON으로 적재
        * https://www.mongodb.com/json-and-bson
* crawling은 local 혹은 EC2에 사용
    * scheduling은 Luigi/Apache Airflow/ Crontab 이용(알아보고 쓰고 싶은 scheduling tool 쓸것)
        * 매일 00:00시에 하루 뉴스 크롤링
        * https://luigi.readthedocs.io/en/stable/central_scheduler.html
* 기사 일단 크롤링 하고 나오는 데이터 사용해서 분석까지....? 
    * 트렌트라던지....
