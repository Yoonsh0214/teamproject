"""
중고차 가격 예측 — 스켈레톤 코드 (두 시장 비교판)
AI+X: Deep Learning 최종 프로젝트 (Option A)

두 시장(한국 DACON · 인도 CarDekho)의 차량 스펙으로 중고차 가격을 예측하고,
선형회귀 · 랜덤포레스트 · 그래디언트 부스팅 · MLP 신경망을 비교한다.
"같은 방법론을 두 시장에 적용"하는 구성. (통화가 달라 R²로 시장 간 비교)

이 파일은 "스켈레톤"입니다. 함수 골격·핵심 라인·`# TODO` 가 있습니다.
data/README.md 를 보고 데이터 2개를 받아 TODO를 채운 뒤 실행하세요.

    python used_car_price_prediction.py
"""

# ============================================================
# 0. imports & 설정
# ============================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 신경망: Keras 사용. 설치가 부담되면 sklearn 버전으로 대체 가능.
#   from sklearn.neural_network import MLPRegressor
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

FIG_DIR = "figures"
BASE_YEAR = 2026
TEST_SIZE = 0.2
RANDOM_STATE = 42

plt.rcParams["font.family"] = "AppleGothic"   # 맥 한글 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 시장별 설정. preprocess 함수는 아래에 정의(파이썬은 위→아래로 읽으므로 함수 참조는 dict로 묶음)
# currency: 그래프 축 라벨용. 통화가 달라 RMSE는 시장 간 비교 금지(R²만 비교).
DATASETS = {
    "한국(DACON)": {"path": "data/korea_cars.csv", "currency": "백만원", "prep": "korea"},
    "인도(CarDekho)": {"path": "data/india_cars.csv", "currency": "INR", "prep": "india"},
}


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data(path):
    """CSV → DataFrame."""
    df = pd.read_csv(path)
    print(f"[load] {path}  shape={df.shape}")
    print(df.head())
    return df


# ============================================================
# 2. EDA
# ============================================================
def run_eda(df, name):
    """요약통계·결측치 확인. (그래프는 6번에서)"""
    print(f"\n[EDA] {name} 기술통계")
    print(df.describe(include="all"))
    print(f"[EDA] {name} 결측치\n", df.isna().sum())
    # TODO: 타겟 분포 왜도 확인, 범주형 value_counts 확인


# ============================================================
# 3-a. 전처리 — 인도(CarDekho)
# ============================================================
def _parse_number(series):
    """'23.4 kmpl', '1248 CC', '74 bhp' → 앞쪽 숫자만 float (없으면 NaN)."""
    def extract(x):
        if pd.isna(x):
            return np.nan
        m = re.search(r"[-+]?\d*\.?\d+", str(x))
        return float(m.group()) if m else np.nan
    return series.apply(extract)


def preprocess_india(df):
    """인도 데이터 → (X, y). y는 log1p(가격)."""
    df = df.copy()
    df["mileage"] = _parse_number(df["mileage"])
    df["engine"] = _parse_number(df["engine"])
    df["max_power"] = _parse_number(df["max_power"])
    df = df.drop(columns=["torque", "name"], errors="ignore")

    df["car_age"] = BASE_YEAR - df["year"]
    df = df.drop(columns=["year"])

    owner_map = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3,
                 "Fourth & Above Owner": 4, "Test Drive Car": 0}
    df["owner"] = df["owner"].map(owner_map)

    # TODO: 결측치 처리 (수치형 median)
    #   num = df.select_dtypes("number").columns
    #   df[num] = df[num].fillna(df[num].median())

    y = np.log1p(df["selling_price"])
    X = df.drop(columns=["selling_price"])
    X = pd.get_dummies(X, columns=["fuel", "seller_type", "transmission"], drop_first=True)
    return X, y


# ============================================================
# 3-b. 전처리 — 한국(DACON)
#   ※ 실제 컬럼명은 train.csv 헤더로 반드시 확인 후 맞출 것!
# ============================================================
def preprocess_korea(df):
    """한국 데이터 → (X, y). y는 log1p(가격)."""
    df = df.copy()

    # TODO: 아래 컬럼명을 실제 헤더에 맞게 수정 (예시는 가정값)
    COL_YEAR = "생산년도"
    COL_PRICE = "가격"
    COL_FUEL = "연료"          # 단일 범주일 경우. 이미 이진(원핫)이면 이 줄/아래 get_dummies 생략
    DROP_HIGH_CARD = ["브랜드", "차량모델명", "판매도시", "판매구역", "ID", "모델출시년도"]

    df["car_age"] = BASE_YEAR - df[COL_YEAR]
    df = df.drop(columns=[COL_YEAR], errors="ignore")

    # 고차원 범주(브랜드/모델/지역)는 1차 분석에서 제외 (차원 폭증 방지)
    df = df.drop(columns=DROP_HIGH_CARD, errors="ignore")

    # TODO: 결측치 처리 (수치형 median)

    y = np.log1p(df[COL_PRICE])
    X = df.drop(columns=[COL_PRICE])

    # 연료가 단일 범주 컬럼이면 원핫. 이미 이진 컬럼들로 쪼개져 있으면 이 블록 생략.
    if COL_FUEL in X.columns:
        X = pd.get_dummies(X, columns=[COL_FUEL], drop_first=True)
    # TODO: 남은 비수치 컬럼이 있으면 처리 (X = X.select_dtypes("number") 로 강제 가능)
    return X, y


PREP_FUNCS = {"india": preprocess_india, "korea": preprocess_korea}


# ============================================================
# 3-c. 공통 후처리 — 분할 + 스케일링
# ============================================================
def finalize(X, y):
    """train/test 분할 + 스케일링. 트리는 원본, 선형/신경망은 스케일본 사용."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, X.columns.tolist()


# ============================================================
# 4. 모델 학습 (두 시장 공통 재사용)
# ============================================================
def build_mlp(input_dim):
    if not HAS_KERAS:
        return None
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_models(X_tr, X_tr_s, y_tr):
    """4개 모델 학습 → (models dict, mlp history)."""
    models = {}
    lin = LinearRegression().fit(X_tr_s, y_tr)            # 스케일본
    models["LinearRegression"] = lin
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    models["RandomForest"] = rf
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(X_tr, y_tr)
    models["GradientBoosting"] = gb

    history = None
    mlp = build_mlp(X_tr_s.shape[1])
    if mlp is not None:
        history = mlp.fit(X_tr_s, y_tr, validation_split=0.2,
                          epochs=100, batch_size=32, verbose=0)
        models["MLP"] = mlp
    else:
        print("[train] Keras 미설치 → MLP 생략 (sklearn MLPRegressor로 대체 가능)")
    return models, history


# ============================================================
# 5. 평가
# ============================================================
def _predict(model, X_tree, X_scaled):
    """모델 종류에 맞는 입력으로 예측(로그 스케일)."""
    name = type(model).__name__
    if name == "LinearRegression" or "Sequential" in name:
        return np.ravel(model.predict(X_scaled))
    return np.ravel(model.predict(X_tree))


def evaluate(models, X_te, X_te_s, y_te):
    """모델별 R²/RMSE/MAE → DataFrame (원단위로 환산해 계산)."""
    y_true = np.expm1(y_te)
    rows = []
    for name, model in models.items():
        y_pred = np.expm1(_predict(model, X_te, X_te_s))
        rows.append({
            "model": name,
            "R2": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
        })
    return pd.DataFrame(rows).set_index("model")


# ============================================================
# 6. 시각화
# ============================================================
def plot_market_extras(name, df, models, history, feature_names):
    """시장별 보조 그래프(분포·중요도·학습곡선)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = "korea" if "한국" in name else "india"
    # TODO: 가격 분포(로그 전/후), 예측 vs 실제 산점도
    # feature importance (트리)
    if "RandomForest" in models:
        imp = pd.Series(models["RandomForest"].feature_importances_, index=feature_names)
        imp.sort_values().tail(10).plot(kind="barh")
        plt.title(f"{name} 변수 중요도(RF)"); plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/imp_{tag}.png"); plt.close()
    # MLP 학습곡선
    if history is not None:
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title(f"{name} MLP 학습곡선"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/loss_{tag}.png"); plt.close()


def plot_r2_comparison(all_results):
    """두 시장의 모델별 R²를 한 그래프에 비교(핵심 1장)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    r2 = pd.DataFrame({name: res["R2"] for name, res in all_results.items()})
    r2.plot(kind="bar")
    plt.title("시장·모델별 R² 비교"); plt.ylabel("R²"); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/r2_compare.png"); plt.close()
    print("\n[plot] figures/r2_compare.png 저장")


# ============================================================
# 7. main — 두 시장 루프 → 통합 비교
# ============================================================
def run_one(name, cfg):
    """한 시장 전체 파이프라인 → 결과 DataFrame."""
    print(f"\n{'='*50}\n[{name}] 시작\n{'='*50}")
    df = load_data(cfg["path"])
    run_eda(df, name)
    X, y = PREP_FUNCS[cfg["prep"]](df)
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, feats = finalize(X, y)
    models, history = train_models(X_tr, X_tr_s, y_tr)
    result = evaluate(models, X_te, X_te_s, y_te)
    print(f"\n[{name}] 성능\n", result.round(3))
    plot_market_extras(name, df, models, history, feats)
    return result


def main():
    all_results = {}
    for name, cfg in DATASETS.items():
        if not os.path.exists(cfg["path"]):
            print(f"[skip] {cfg['path']} 없음 → data/README.md 참고해 데이터 저장")
            continue
        all_results[name] = run_one(name, cfg)

    if len(all_results) >= 1:
        plot_r2_comparison(all_results)
        print("\n[통합] 시장별 R² 요약")
        print(pd.DataFrame({n: r["R2"] for n, r in all_results.items()}).round(3))
    print("\n완료. figures/ 그래프와 위 표를 블로그에 사용하세요.")


if __name__ == "__main__":
    main()
