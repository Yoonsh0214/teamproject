# 데이터 받는 법

이 프로젝트는 **두 시장(한국 · 인도)**의 중고차 데이터를 같은 방법론으로 비교합니다.
아래 두 파일을 이 `data/` 폴더에 넣으세요.

| 파일명(권장) | 시장 | 출처 |
|---|---|---|
| `korea_cars.csv` | 한국 | DACON 자동차 가격 예측 해커톤 |
| `india_cars.csv` | 인도 | Kaggle CarDekho `Car details v3.csv` |

> 파일명을 다르게 저장했다면 `used_car_price_prediction.py` 상단 `DATASETS` 설정의 `path`만 고치면 됩니다.

---

## ① 한국 — DACON 자동차 가격 예측 (236114)
- 링크: https://dacon.io/competitions/official/236114/data
- 규모: train 약 57,920행 / 가격 단위 **백만원**
- 주요 컬럼: 생산년도, 모델출시년도, 브랜드, 차량모델명, 주행거리(km), 배기량(cc), 연료(가솔린/경유/CNG/하이브리드/LPG), 판매도시·구역, 가격(타겟)

### 절차
1. 위 링크 접속 → **DACON 회원가입/로그인**(무료)
2. 대회 페이지에서 **규칙 동의** 후 데이터 다운로드 (`train.csv`)
3. 받은 `train.csv` 를 이 폴더에 **`korea_cars.csv`** 로 저장

---

## ② 인도 — Kaggle CarDekho Vehicle dataset
- 링크: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- 규모: 약 8,100행 / 가격 단위 **INR(루피)**
- 주요 컬럼: year, km_driven, fuel, seller_type, transmission, owner, mileage("23.4 kmpl"), engine("1248 CC"), max_power, torque, seats, selling_price(타겟)

### 절차
1. 위 링크 접속 → **Kaggle 로그인**(무료) → **Download**
2. 압축 안의 **`Car details v3.csv`** 를 이 폴더에 **`india_cars.csv`** 로 저장

---

## 참고
- 두 데이터는 **통화·시장이 달라 가격을 직접 합치면 안 됩니다.** 각각 따로 모델을 학습해
  "같은 방법론을 두 시장에 적용했을 때의 결과"를 비교하는 구성입니다.
- 컬럼명이 바뀌면 `preprocess_korea()` / `preprocess_india()` 의 컬럼명을 데이터에 맞게 수정하세요.
