# NVIDIA DLI KDT Tensorflow

## TensorFlow를 활용한 인공신경망 구현

- [커리큘럼](https://observant-fang-c48.notion.site/NVIDIA-DLI-KDT-Tensorflow-2979f3df6c6980f3896ade0afb1c0fbe)
  - [실습파일 복사본](https://drive.google.com/drive/u/0/folders/110lCqf8GE72fvk9CyK47DXWyEMW68rNt)
    - 반드시 런타임 환경을 CPU로 맞추고 실행해볼 것(안 그럼 GPU 사용량만큼 돈 나감)



- 참고자료
  - [교재](https://observant-fang-c48.notion.site/NVIDIA-DLI-KDT-Tensorflow-2979f3df6c6980f3896ade0afb1c0fbe)
    - 오근철 강사님이 제공한 개발환경 설정 및 교보재 자료가 공유된 Notion 페이지
  - [허깅페이스 공식](https://huggingface.co/): AI 모델 공유 커뮤니티
    - [허깅페이스 런](https://huggingface.co/learn): 간단한 튜토리얼 제공
    - [깃허브](https://github.com/huggingface)
    - 기타 공부하지 좋은 자료 → 구글에서 '**위키독스 허깅페이스**' 검색
  - 데이터셋 참고 사이트
    - [케글](https://www.kaggle.com/): AI 모델 경진대회 사이트. 
      - 대회에 제출된 수상자 팀의 코드와 사용된 데이터셋도 공유되고 있다.
      1. 초급
         - Competions > getting started > 초보자용만 검색됨
         - 다른 사람들의 코드를 보며 학습 가능
    - [AI 허브](https://www.aihub.or.kr/)
      - 한국어 데이터 제공
      - 비정형 데이터 많음
  - [A Neural Network Playground - TensorFlow](https://playground.tensorflow.org/)
    - 딥러닝 모델의 뉴럴 네트워크를 마음대로 조작해 학습시키고 시뮬레이션 결과를 확인해볼 수 있는 곳
    - 시뮬레이션에 필요한 몇 가지 유형의 데이터셋이 제공되며 데이터셋 특징도 세부 조정이 가능하다.
  - [LLM Visualizaton](https://bbycroft.net/llm)
    - 출시후 공개된 LLM을 종류별로 시각화해 보여주는 곳
  - [ROC curve](https://angeloyeo.github.io/2020/08/05/ROC.html)
    - ROC curve: 다중분류에서 TP 비율과 FP 비율간의 관계를 그린 곡선
    - 모델의 성능이 좋을 수록 ROC curve가 직각에 가깝다.
      - ROC curve 적분값(AUC)이 정사각형 면적 1에 가까울수록 모델의 성능이 좋다. 
      - 성능이 안 좋으면 AUC는 직각삼각형 면적 0.5에 가깝다.
    - 객체 분류의 경우, PR curve를 더 많이 쓴다.
      - ROC curve는 왼쪽 위로 커브가 그려지는 반면에 PR curve는 오른쪽 위로 커브가 그려짐
- 튜토리얼 사이트
  - [PyTorch 튜토리얼(한국어)](https://tutorials.pytorch.kr/)
- 기술동향
  - [PyTorch 코리아 커뮤니티 > 읽을거리&정보공유 - 게시판](https://discuss.pytorch.kr/c/news/14)
- 공부하기 좋은 책(위키독스)
  - [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)
  - [D2L](https://d2l.ai/) 
    - 무료
    - 대학에서도 교재로 많이 활용
    - 실습 깃허브 코드 / 유튜브 동영상 쉽게 검색됨
    - 완독 시 대학원 들어갈 정도 수준의 지식 확보
  - [Deep Learning - Stanford CS231N](https://www.youtube.com/playlist?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)
    - 스탠포드 대학에서 진행하는 유튜브 딥러닝 강의 영상 목록
    - 연사들의 명성과 수준이 아주 높음
- 도구
  - [Weights & Biases: The AI Developer Platform](https://wandb.ai/site/ko/)
    - AI 모델 훈련 시 연속된 시도마다의 로그를 저장하고 추적하도록 지원(개인 무료)