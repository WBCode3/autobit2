# autobit2

Freqtrade (Binance USDT-M Futures) 전략 연구 저장소. **연구/백테스트 목적이며 투자 조언이 아닙니다. 실전 사용 시 손실 책임은 본인에게 있습니다.**

## 구성
- `strategies/` : 최종 후보 전략
  - `R9_Dyn90_SL18_StrHi_MR2.py` : 4h 돌파 + 유휴 슬롯 평균회귀 (공격적, 5Y MDD 55~75%)
  - `R8_Dyn85_SL18.py` : 4h 순수 돌파 (단순형)
  - `G4_M2_B25_m44.py` : 인과적 39페어 동적 유니버스
  - `P3_1h_Breakout_LowCorr.py` : 1h 저상관 돌파 (횡보장 보완용)
- `configs/backtest_top10.example.json` : 백테스트용 예제 설정 (키/비밀번호는 전부 placeholder)
- `scripts/` : 슬리피지(POV) 재현 및 구간 백테스트 스크립트 (경로는 본인 환경에 맞게 수정)
- `HANDOFF.md`, `BOT_USAGE.md`, `docker-compose.yml` : 진행 상황 / 운용 가이드 / 실행 예시

## 주의
- **API 키, 비밀번호, 토큰, 서버 IP는 절대 커밋하지 마세요.** `user_data/config.json`은 `.gitignore`로 제외돼 있습니다.
- 백테스트 수치는 과거 데이터 기준이며 슬리피지/체결 한계로 실전과 다를 수 있습니다. 고배수 세팅은 청산 위험이 큽니다.
- 시세 데이터(`user_data/data`)는 포함하지 않습니다: `freqtrade download-data`로 직접 받으세요.
