# SeoultechDS_Technical-Porforlio
Technical porfolio for Data Science Master degree

## 1. 데이터 사이언스 공통
- /Reproducibility challenge
해당 폴더 내 README.md에 설명 작성하였습니다.

 <br/>
  
## 2. 데이터 수집 및 정제
- /Zapier crawling
- Zapier 플랫폼 기반의 App, Zap 데이터 수집
- Zapier 내 App Category별로 속하는 모든 App의 정보 수집
  - "AppName": App 이름, "AppInfo": App 설명 텍스트, "Url": urls
- App이 사용된 모든 Zap 정보 수집
  - "Zap Name": zap 이름, "URL": zap url, "Using Apps": 사용한 Apps
- Zap의 세부 구성 요소 수집
  - "Zap Name": zap 이름, 'URL": zap url, "Using Apps": 사용한 Apps, "Component숫자": Zap의 구성 요소 정보
  - Component숫자 칼럼은 각 Zap을 구성하는 App 의 개수만큼 존재하며, 해당 App의 
    - App 카테고리
    - App 이름
    - Trigger and Action 카테고리 (Trigger나 Action 중 하나)
    - Trigger and Action이 수행하는 일
    - Trigger and Action의 이름
    - Trigger and Action이 수행하는 일에 대한 설명
  - 이 수집됨

- 사용 라이브러리
  - `selenium`
  - `pandas`
  - `numpy`
  - `os`
  - `glob`
  - `tqdm`
  - `time`
  - `re`
  ### zapier_apps_crawling1.ipynb 에서 수집할 카테고리를 선택하여 수집, 저장한 다음 해당 파일을 사용하여 Zap, detail을 수집하는 방식으로 수행하면 됨

 <br/>
 
## 3. 데이터 활용 및 분석
- /ML application

