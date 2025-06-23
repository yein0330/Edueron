# Edueron

**Edueron**은 PDF 기반 시험지를 자동으로 분석하여 문제를 추출하고, 개념 요약, 예상 문제 생성, 자동 채점 등을 지원하는 지능형 학습 도우미입니다.

> “시험지를 이해하고, 요약하고, 문제도 만들어주는 AI 도우미를 꿈꿉니다.”

---

## 🧠 주요 기능

- PDF에서 지문, 문항, 선택지 자동 추출
- 문항 유형 분석 (어휘, 문법, 문장 배열, 추론 등)
- 지문 기반 개념 요약 및 핵심어 추출
- 예상 문제 생성 (개발 예정)
- 손글씨 답안 채점 기능 (개발 예정)
- Gradio 기반 웹 인터페이스 제공

---

## 🛠 기술 스택

- Python 3.x
- [PyMuPDF](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber) – PDF 텍스트/레이아웃 추출
- [OpenCV](https://opencv.org/), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Gradio](https://www.gradio.app/) – 인터페이스
- Trillion-7B / BGE-M3 (LLM 모델 연동)

---

## 📁 폴더 구조

Edeuron/
├── data/ # 샘플 PDF 및 입력 파일
├── docs/ # 문서 예시, 결과 이미지
├── src/
│ ├── inference/ # LLM 추론 관련 코드
│ ├── pdf_extractor.py # PDF 구조 분석 및 텍스트 추출
│ ├── embedding_model.py# 임베딩 및 유사도 계산
│ └── Edeuron_demo.py # Gradio 인터페이스 실행 파일


---

## 🚀 실행 방법

1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

인터페이스 실행

```bash
python src/Edeuron_demo.py
```

Gradio 웹페이지 접속 후 PDF 업로드 및 분석

---
## 📌 개선 예정 사항
 PDF 2단 문단 구조 자동 재정렬

 문항 유형별 분리 및 정제 알고리즘 고도화

 수험생 손글씨 인식 기반 자동 채점 기능

 LLM을 활용한 정답 해설 및 피드백 생성

---

## 👤 개발자
이예인 (Yein Lee)
GitHub: @yein0330
