# IGA-2020-Bigdata-Competition
IGA 2020 Bigdata Competition (CTR Prediction) 최우수상 

Team 오하이오

1. 실행환경
OS: Windows 10 64bit
Ram: 32G
CPU: i5-9500F
Language: Python 3.7
JDK: JDK-13.0.2
Visual Studio: 2017 Community (15.9)
Cmake: Cmake-3.16.3
Git: Git-2.25.0

2. 필요 라이브러리
pickle - 0.7.5
pandas - 0.25.1
numpy - 1.16.5
h2o - 3.28.0.3
hashlib - undefined
math - undefined
xlearn - 0.4.4

라이브러리 특이사항

1 - h2o:
h2o는 h2o.ai 에서 배포하는 머신러닝 라이브러리로, Java를 기반으로 하기 때문에 
실행을 위해선 JDK(Java Development Kit)이 설치되어 있어야 합니다. 이후 pip install h2o를 통해 설치 가능합니다.
자세한 내용은 http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html 을 참조하시기 바랍니다.

2 - xlearn:
xlearn은 FFM(Field-aware Factorization Machines)를 지원하는 라이브러리로, C++을 기반으로 하기 때문에
실행을 위해선 Visual Studio 2017 Communiry(version 준수), Cmake, Git 등의 개발환경이 필요합니다. 
pip install을 통해서 다운받을 수 없고 수동으로 C++ 소스코드를 build하고 whl 파일로 python에 설치해야합니다.
자세한 설치 방법 및 내용은 https://xlearn-doc.readthedocs.io/en/latest/install/install_windows.html 
그리고 https://xlearn-doc.readthedocs.io/en/latest/index.html 을 참조하시기 바랍니다.

3. 실행방법 및 시간
필요 라이브러리들을 모두 설치한 후, audience_profile.csv,train.csv,test.csv 파일을 폴더에 넣어줍니다.
이후 cmd를 통해 해당 폴더로 이동해서 preprocess.py, model.py, predict.py 순으로 실행하시면 됩니다.
평균 소요 시간은 각각 1시간 45분, 10분, 10분 입니다.
모든 계산이 끝난 뒤 생성되는 predict.csv가 최종 제출 파일입니다.

문서 끝.
