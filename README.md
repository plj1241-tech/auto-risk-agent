# 자동차 부품업계 리스크 분석 에이전트

DART 재무데이터와 거시경제지표를 결합해 21개 자동차 부품기업의 재무 리스크를 조회·비교하고, SHAP 설명과 관측 범위 내 시나리오 민감도를 제공하는 분석 애플리케이션입니다.

## 구현 상태

- Phase 1: 데이터 파이프라인 완료
- Phase 2: 패널회귀·XGBoost·SHAP 완료
- Phase 3: 위험분석 도구 4종과 Gemini 함수호출 완료
- Phase 4: 5단 애널리스트 리포트와 FastAPI 완료
- Phase 5: Streamlit 대시보드·PDF·배포 설정 완료, Cloud 게시 대기

분기 손익은 DART 보고값의 구조에 맞게 Q1~Q3 해당 분기값을 유지하고 Q4만 연간값에서 앞선 분기를 차감하도록 교정했습니다. 기존 492행 파일은 원 보고값을 역산해 복원했으며 변환 테스트를 포함합니다.

## 주요 기능

- 21개 기업·2019~2024년 리스크지표 조회
- 기업별 리스크지표와 거시변수 추이
- 동종업계 중앙값·건전성 순위 비교
- Z-score SHAP 모델 설명
- Z-score SHAP beeswarm 시각화
- 과거 관측 범위 내 거시변수 민감도
- Gemini 2.5 Flash 기반 자연어 질의
- 5단 구조 Markdown·PDF 리포트 다운로드
- FastAPI `/analyze`, `/report` API

## 모델 해석 범위

현재 모델은 기업별 재무 상태의 지속성에 크게 의존합니다. 동일 표본에서 직전 연도·분기 값을 그대로 사용하는 기준선이 대부분의 XGBoost 결과보다 높았고, 교정 분기 유동비율만 기준선을 소폭 상회했습니다. 시나리오 결과는 실제 미래값이나 거시변수의 인과효과가 아니라 과거 관측 범위 안에서 입력을 변경한 모델 민감도입니다. 신규기업 예측, 범위 밖 외삽, 투자·신용 의사결정에는 사용하지 않습니다.

검증 수치는 `data/outputs/model_validation_v2.csv`와 `data/outputs/q_model_validation.csv`에 저장됩니다. `data/outputs/v2_08_shap_beeswarm_z_score.png`에서 기업 ID의 기여가 거시변수보다 큰 것도 확인할 수 있습니다.

5장 발표안과 데모 순서는 `presentation_slides.md`에 정리되어 있습니다. Marp 호환 Markdown이라 VS Code Marp 확장 등으로 PDF·PPTX로 내보낼 수 있습니다.

## 로컬 실행

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

브라우저에서 `http://127.0.0.1:8501`을 엽니다.

FastAPI는 별도로 실행할 수 있습니다.

```powershell
python -m uvicorn app.main:app --reload
```

- API 문서: `http://127.0.0.1:8000/docs`
- 상태 확인: `GET /health`
- 자연어 분석: `POST /analyze`
- 리포트 생성: `POST /report`

## 환경변수

로컬 `.env`:

```env
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash
DART_API_KEY=your-key
ECOS_API_KEY=your-key
FRED_API_KEY=your-key
```

`.env`와 `.streamlit/secrets.toml`은 커밋하지 않습니다.

## Streamlit Cloud 배포

1. 이 프로젝트를 GitHub 저장소에 푸시합니다.
2. Streamlit Community Cloud에서 저장소를 연결합니다.
3. Main file path를 `app/streamlit_app.py`로 지정합니다.
4. App secrets에 아래 값을 등록합니다.

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.5-flash"
```

5. `data/processed/panel_annual_v2.parquet`와 `models/risk_predictor_v2.pkl`이 저장소에 포함되었는지 확인합니다.

## 검증

```powershell
python -m unittest discover -s tests -v
```

현재 자동화 테스트는 위험조회, 동종비교, 시나리오 가드레일, SHAP, 5단 리포트, PDF 생성, 분기 손익 교정의 정확성과 재실행 안전성을 검증합니다.

## 구조

```text
app/
  main.py                 FastAPI
  streamlit_app.py        Streamlit 대시보드
src/agent/
  risk_tools.py           Phase 3 도구 4종
  gemini_client.py        Gemini 함수호출
  report_generator.py     Markdown 리포트
  pdf_report.py           PDF 리포트
src/models/risk_model.py  모델·패널 접근 계층
src/features/repair_quarterly_flows.py  기존 분기 손익 교정
```
