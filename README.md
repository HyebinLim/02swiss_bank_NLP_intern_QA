# 🏦 Swiss Bank AI/NLP scientist Q&A App

스위스 취업 준비 경험을 담은 블로그 PDF를 기반으로,
질문에 답변해주는 Q&A 챗봇입니다🤖

## 주요 기능
- 스위스 취리히의 은행에서 AI/NLP data scientist로 일한 경험을 바탕으로
- 질문시 답변을 하거나, 궁금한 정보를 요약해 드립니다.
- 답변의 출처 페이지를 표시하여 직접 데이터소스를 확인해보실 수 있습니다.

## 실행 방법

### 로컬 실행
```bash
streamlit run streamlit_app.py
```

### 웹에서 사용하기
1. **Streamlit Cloud 배포** (추천)
   - [share.streamlit.io](https://share.streamlit.io)에서 배포
   - GitHub 저장소 연결 후 자동 배포
   - 웹 링크로 바로 접근 가능

2. **로컬 실행**
   - 위 명령어로 실행 후 `http://localhost:8501` 접속

streamlit web에서 OpenAI API 키 입력 후 질문!

## 환경 변수
- OpenAI API 키 필요: 실행 시 입력하시면 됩니다.

## 참고
- PDF 파일은 저작권에 유의해 주세요.
- 질문/답변 품질은 모델 및 임베딩 품질에 따라 달라질 수 있습니다.