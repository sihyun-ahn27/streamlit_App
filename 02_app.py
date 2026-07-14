from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


REQUIRED_COLUMNS = [
    "상품명",
    "카테고리",
    "가격",
    "총내용량",
    "단위",
    "총열량",
    "당류",
    "나트륨",
    "단백질",
]

OPTIONAL_COLUMNS = ["데이터출처", "비고"]

ALIASES = {
    "product": "상품명",
    "product_name": "상품명",
    "name": "상품명",
    "category": "카테고리",
    "price": "가격",
    "amount": "총내용량",
    "weight": "총내용량",
    "quantity": "총내용량",
    "unit": "단위",
    "calories": "총열량",
    "kcal": "총열량",
    "sugar": "당류",
    "sodium": "나트륨",
    "protein": "단백질",
    "source": "데이터출처",
    "note": "비고",
}

NUMERIC_COLUMNS = ["가격", "총내용량", "총열량", "당류", "나트륨", "단백질"]

DERIVED_COLUMNS = [
    "원/100kcal",
    "kcal/1000원",
    "원/100단위",
    "100단위당열량",
    "당류/100kcal",
    "나트륨/100kcal",
    "단백질/100kcal",
]

DISPLAY_ORDER = [
    "상품명",
    "카테고리",
    "가격",
    "총열량",
    "총내용량",
    "단위",
    "당류",
    "나트륨",
    "단백질",
    *DERIVED_COLUMNS,
    "데이터출처",
    "비고",
]


@dataclass
class CleaningSummary:
    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    invalid_rows_removed: int
    missing_by_column: Dict[str, int]


@dataclass
class KNNResult:
    recommendations: pd.DataFrame
    standardized_reference: pd.DataFrame
    feature_summary: pd.DataFrame
    used_features: List[str]
    candidate_count: int


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("원", "", regex=False)
        .str.replace("kcal", "", regex=False)
        .str.replace("mg", "", regex=False)
        .str.replace("g", "", regex=False)
        .str.replace("mL", "", regex=False)
        .str.replace("ml", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningSummary, List[str]]:
    """Validate, clean, and derive analysis columns.

    The function deliberately does not impute missing nutrition values because doing so
    could create false precision. Individual analyses drop rows only when a feature is
    actually required.
    """
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data.rename(columns={c: ALIASES.get(c.lower(), c) for c in data.columns})

    missing_required = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_required:
        return data, CleaningSummary(len(data), 0, 0, 0, {}), missing_required

    for c in OPTIONAL_COLUMNS:
        if c not in data.columns:
            data[c] = ""

    input_rows = len(data)
    data["상품명"] = data["상품명"].astype(str).str.strip()
    data["카테고리"] = data["카테고리"].astype(str).str.strip()
    data["단위"] = data["단위"].astype(str).str.strip().replace({"ml": "mL", "ML": "mL"})

    for c in NUMERIC_COLUMNS:
        data[c] = _to_numeric(data[c])

    missing_by_column = {c: int(data[c].isna().sum()) for c in NUMERIC_COLUMNS}

    before_dedup = len(data)
    data = data.drop_duplicates(subset=["상품명", "카테고리"], keep="last")
    duplicate_rows_removed = before_dedup - len(data)

    valid_mask = (
        data["상품명"].ne("")
        & data["카테고리"].ne("")
        & data["가격"].gt(0)
        & data["총내용량"].gt(0)
        & data["총열량"].gt(0)
    )
    invalid_rows_removed = int((~valid_mask).sum())
    data = data.loc[valid_mask].copy()

    # Nonnegative nutritional values only.
    for c in ["당류", "나트륨", "단백질"]:
        data.loc[data[c] < 0, c] = np.nan

    data["원/100kcal"] = data["가격"] / data["총열량"] * 100
    data["kcal/1000원"] = data["총열량"] / data["가격"] * 1000
    data["원/100단위"] = data["가격"] / data["총내용량"] * 100
    data["100단위당열량"] = data["총열량"] / data["총내용량"] * 100
    data["당류/100kcal"] = data["당류"] / data["총열량"] * 100
    data["나트륨/100kcal"] = data["나트륨"] / data["총열량"] * 100
    data["단백질/100kcal"] = data["단백질"] / data["총열량"] * 100

    data = data.reset_index(drop=True)
    summary = CleaningSummary(
        input_rows=input_rows,
        output_rows=len(data),
        duplicate_rows_removed=duplicate_rows_removed,
        invalid_rows_removed=invalid_rows_removed,
        missing_by_column=missing_by_column,
    )
    return data, summary, []


def similar_products(
    df: pd.DataFrame,
    reference_name: str,
    basis: str,
    tolerance_mode: str,
    tolerance_value: float,
    same_category: bool = True,
    include_reference: bool = True,
) -> pd.DataFrame:
    if reference_name not in set(df["상품명"]):
        raise ValueError("기준 상품을 찾을 수 없습니다.")
    if basis not in {"총열량", "가격"}:
        raise ValueError("basis는 '총열량' 또는 '가격'이어야 합니다.")

    ref = df.loc[df["상품명"] == reference_name].iloc[0]
    candidates = df.copy()
    if same_category:
        candidates = candidates[candidates["카테고리"] == ref["카테고리"]]

    diff = (candidates[basis] - ref[basis]).abs()
    if tolerance_mode == "절대값":
        mask = diff <= tolerance_value
    elif tolerance_mode == "상대비율":
        denominator = max(float(ref[basis]), 1e-9)
        mask = diff / denominator * 100 <= tolerance_value
    else:
        raise ValueError("tolerance_mode는 '절대값' 또는 '상대비율'이어야 합니다.")

    result = candidates.loc[mask].copy()
    if not include_reference:
        result = result[result["상품명"] != reference_name]

    result["열량차이"] = result["총열량"] - ref["총열량"]
    result["가격차이"] = result["가격"] - ref["가격"]
    result["열량차이율(%)"] = result["열량차이"].abs() / ref["총열량"] * 100
    result["가격차이율(%)"] = result["가격차이"].abs() / ref["가격"] * 100
    result["기준상품"] = np.where(result["상품명"] == reference_name, "기준", "비교")

    primary_diff = "열량차이율(%)" if basis == "총열량" else "가격차이율(%)"
    result = result.sort_values([primary_diff, "원/100kcal", "가격"], ascending=True)
    return result.reset_index(drop=True)


def mode_feature_weights(mode: str) -> Dict[str, float]:
    presets = {
        "가성비": {
            "가격": 2.0,
            "총열량": 1.8,
            "총내용량": 1.2,
            "원/100kcal": 2.2,
            "원/100단위": 1.4,
        },
        "저열량": {
            "가격": 1.4,
            "총열량": 2.5,
            "당류": 1.7,
            "나트륨": 1.0,
        },
        "영양균형": {
            "가격": 1.0,
            "당류": 2.0,
            "나트륨": 1.7,
            "단백질": 2.2,
            "총열량": 0.8,
        },
    }
    if mode not in presets:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")
    return presets[mode].copy()


def weighted_knn_recommend(
    df: pd.DataFrame,
    reference_name: str,
    feature_weights: Mapping[str, float],
    k: int = 5,
    same_category: bool = True,
) -> KNNResult:
    if reference_name not in set(df["상품명"]):
        raise ValueError("기준 상품을 찾을 수 없습니다.")

    weights = {f: float(w) for f, w in feature_weights.items() if float(w) > 0}
    if not weights:
        raise ValueError("가중치가 0보다 큰 변수를 하나 이상 선택해야 합니다.")

    reference = df.loc[df["상품명"] == reference_name].iloc[0]
    candidates = df.copy()
    if same_category:
        candidates = candidates[candidates["카테고리"] == reference["카테고리"]]

    features = [f for f in weights if f in candidates.columns]
    candidates = candidates.dropna(subset=features).copy()
    if reference_name not in set(candidates["상품명"]):
        missing = [f for f in features if pd.isna(reference[f])]
        raise ValueError("기준 상품에 선택 변수의 결측값이 있습니다: " + ", ".join(missing))
    if len(candidates) < 2:
        raise ValueError("추천할 비교 상품이 부족합니다. 카테고리 제한을 해제하거나 데이터를 늘려 주세요.")

    scaler = StandardScaler()
    X_standard = scaler.fit_transform(candidates[features])
    sqrt_weights = np.sqrt(np.array([weights[f] for f in features], dtype=float))
    X_weighted = X_standard * sqrt_weights

    ref_pos = int(np.flatnonzero(candidates["상품명"].to_numpy() == reference_name)[0])
    n_neighbors = min(k + 1, len(candidates))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(X_weighted)
    distances, indices = model.kneighbors(X_weighted[[ref_pos]])

    rows = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == ref_pos:
            continue
        row = candidates.iloc[idx].copy()
        row["KNN거리"] = float(distance)
        row["유사도점수"] = float(100 / (1 + distance))
        rows.append(row)
        if len(rows) >= k:
            break

    recommendations = pd.DataFrame(rows).reset_index(drop=True)
    if not recommendations.empty:
        recommendations["가격차이"] = recommendations["가격"] - reference["가격"]
        recommendations["열량차이"] = recommendations["총열량"] - reference["총열량"]
        recommendations["당류차이"] = recommendations["당류"] - reference["당류"]
        recommendations["나트륨차이"] = recommendations["나트륨"] - reference["나트륨"]
        recommendations["단백질차이"] = recommendations["단백질"] - reference["단백질"]

    reference_z = pd.DataFrame(
        {
            "변수": features,
            "원자료": [float(reference[f]) for f in features],
            "평균": scaler.mean_,
            "표준편차": scaler.scale_,
            "표준점수(z)": X_standard[ref_pos],
            "가중치": [weights[f] for f in features],
            "가중표준점수": X_weighted[ref_pos],
        }
    )

    feature_summary = pd.DataFrame(
        {
            "변수": features,
            "최솟값": [float(candidates[f].min()) for f in features],
            "최댓값": [float(candidates[f].max()) for f in features],
            "평균": scaler.mean_,
            "표준편차": scaler.scale_,
            "가중치": [weights[f] for f in features],
        }
    )

    return KNNResult(
        recommendations=recommendations,
        standardized_reference=reference_z,
        feature_summary=feature_summary,
        used_features=features,
        candidate_count=len(candidates),
    )


def pareto_front_mask(df: pd.DataFrame, objectives: Mapping[str, str]) -> pd.Series:
    """Return True for non-dominated rows.

    objectives maps column names to 'min' or 'max'. Rows with missing values in an
    objective are marked False.
    """
    cols = list(objectives)
    valid = df[cols].notna().all(axis=1)
    work = df.loc[valid, cols].astype(float).copy()
    for c, direction in objectives.items():
        if direction == "max":
            work[c] = -work[c]
        elif direction != "min":
            raise ValueError("objective direction must be 'min' or 'max'")

    values = work.to_numpy()
    efficient = np.ones(len(work), dtype=bool)
    for i, point in enumerate(values):
        # j dominates i if j is no worse on all dimensions and strictly better on one.
        no_worse = np.all(values <= point, axis=1)
        strictly_better = np.any(values < point, axis=1)
        dominated = np.any(no_worse & strictly_better)
        efficient[i] = not dominated

    mask = pd.Series(False, index=df.index)
    mask.loc[work.index] = efficient
    return mask


def pareto_objectives(mode: str) -> Dict[str, str]:
    mapping = {
        "가성비": {"가격": "min", "총열량": "max", "총내용량": "max"},
        "저열량": {"가격": "min", "총열량": "min", "당류": "min"},
        "영양균형": {
            "가격": "min",
            "당류": "min",
            "나트륨": "min",
            "단백질": "max",
        },
    }
    if mode not in mapping:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")
    return mapping[mode].copy()


def run_hedonic_regression(
    df: pd.DataFrame,
    category: str | None = None,
):
    """Fit an exploratory hedonic price regression with HC3 robust covariance.

    Returns (model, analyzed_frame). Residuals are descriptive unexplained price
    differences, not proof of brand premium or undervaluation.
    """
    data = df.copy()
    if category and category != "전체":
        data = data[data["카테고리"] == category].copy()

    cols = ["상품명", "카테고리", "가격", "총열량", "총내용량", "당류", "나트륨", "단백질"]
    data = data[cols].dropna().copy()
    if len(data) < 15:
        raise ValueError("회귀분석에는 결측값이 없는 상품이 최소 15개 필요합니다.")

    reg = data.rename(
        columns={
            "상품명": "product",
            "카테고리": "category",
            "가격": "price",
            "총열량": "calories",
            "총내용량": "amount",
            "당류": "sugar",
            "나트륨": "sodium",
            "단백질": "protein",
        }
    )

    if category and category != "전체":
        formula = "price ~ calories + amount + sugar + sodium + protein"
    else:
        formula = "price ~ calories + amount + sugar + sodium + protein + C(category)"

    model = smf.ols(formula, data=reg).fit(cov_type="HC3")
    reg["predicted_price"] = model.predict(reg)
    reg["residual"] = reg["price"] - reg["predicted_price"]
    reg["abs_residual"] = reg["residual"].abs()
    return model, reg


def coefficient_table(model) -> pd.DataFrame:
    conf = model.conf_int()
    table = pd.DataFrame(
        {
            "변수": model.params.index,
            "계수": model.params.values,
            "강건표준오차": model.bse.values,
            "p값": model.pvalues.values,
            "95%하한": conf[0].values,
            "95%상한": conf[1].values,
        }
    )
    return table


def display_columns(df: pd.DataFrame, extra: Sequence[str] | None = None) -> List[str]:
    cols = [c for c in DISPLAY_ORDER if c in df.columns]
    if extra:
        cols += [c for c in extra if c in df.columns and c not in cols]
    return cols


# =========================================================
# Streamlit UI
# =========================================================
import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



# =========================================================
# 1. 기본 설정과 디자인
# =========================================================
st.set_page_config(
    page_title="SnackLens | 매점 상품 비교",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#F47A1F"
DARK_ORANGE = "#D85F0D"
PALE_ORANGE = "#FFF3E8"
CREAM = "#FFF9F4"
TEXT = "#1F2328"
MUTED = "#6E737B"
BORDER = "#EEE7E0"
GREEN = "#2E8B57"
RED = "#D94B3D"
GRAY = "#B8B2AC"

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample_data.csv"

FALLBACK_SAMPLE_CSV = '상품명,카테고리,가격,총내용량,단위,총열량,당류,나트륨,단백질,데이터출처,비고\n초코송이(사용자 예시),과자,1150,36.0,g,190,15.0,85,3.0,사용자 제시 예시,실제 조사값으로 교체 필요\n칸쵸(사용자 예시),과자,2000,54.0,g,190,13.0,110,3.0,사용자 제시 예시,실제 조사값으로 교체 필요\n구운감자쌀과자(사용자 예시),과자,3000,48.0,g,190,5.0,210,2.0,사용자 제시 예시,실제 조사값으로 교체 필요\n오렌지 크래커,과자,2200,48.0,g,265,4.0,101,2.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n고소한 감자칩,과자,1200,47.0,g,208,17.4,248,2.9,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n미니 프레첼,과자,2250,87.0,g,475,9.9,372,4.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n통밀 비스킷,과자,2250,93.0,g,384,14.2,212,8.3,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n치즈 콘스낵,과자,1300,51.0,g,275,9.0,264,3.0,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n허니 라이스칩,과자,2900,80.0,g,391,23.7,356,4.5,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n바삭 옥수수칩,과자,1600,77.0,g,381,15.3,152,3.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 웨이퍼,과자,2050,35.0,g,146,5.6,110,2.6,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n담백한 쌀과자,과자,1300,68.0,g,281,20.4,433,2.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n레몬 탄산음료,음료,2200,292.0,mL,188,48.5,237,8.2,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n복숭아 아이스티,음료,2500,356.0,mL,87,33.4,278,12.8,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 우유음료,음료,1900,263.0,mL,154,32.5,193,0.3,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n사과 주스,음료,1700,252.0,mL,117,9.5,86,9.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n요거트 드링크,음료,4250,496.0,mL,417,34.2,95,16.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n보리 음료,음료,2100,345.0,mL,227,58.6,309,2.7,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n제로 탄산음료,음료,3300,351.0,mL,203,20.7,208,11.7,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n크림 단팥빵,빵,2150,124.0,g,462,16.5,504,15.7,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n통밀 샌드빵,빵,1700,73.0,g,218,8.3,387,8.6,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n치즈 소시지빵,빵,1850,102.0,g,353,24.4,539,9.9,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n미니 카스텔라,빵,2900,107.0,g,377,16.5,310,6.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 머핀,빵,2200,99.0,g,395,24.0,585,10.0,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n고구마 페이스트리,빵,1800,87.0,g,237,10.4,484,6.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n우유 식빵롤,빵,2900,86.0,g,269,15.8,512,9.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n바닐라 바,아이스크림,1100,77.0,g,196,25.9,104,3.0,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 콘,아이스크림,1500,140.0,g,285,44.4,191,4.9,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n딸기 컵,아이스크림,1550,78.0,g,116,14.0,84,4.5,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n요거트 아이스,아이스크림,2700,92.0,g,210,28.0,159,3.6,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n우유 샌드,아이스크림,1900,138.0,g,359,27.2,230,3.7,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n과일 셔벗,아이스크림,2150,70.0,g,188,18.8,107,3.3,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n플레인 요거트,유제품,2300,206.0,g,226,39.5,147,8.2,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n딸기 요거트,유제품,2300,209.0,g,219,35.7,249,18.2,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n저지방 우유,유제품,1500,108.0,g,86,12.2,94,4.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 우유,유제품,1700,177.0,g,219,14.6,71,9.3,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n치즈 스틱,유제품,2200,210.0,g,331,33.6,90,10.4,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n참치 삼각김밥,간편식,2700,171.0,g,360,13.1,846,25.6,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n불고기 삼각김밥,간편식,2650,90.0,g,219,5.4,859,4.9,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n치킨 샌드위치,간편식,2650,215.0,g,579,7.9,1948,29.0,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n에그 샌드위치,간편식,4700,177.0,g,442,2.1,1817,23.2,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n미니 김밥,간편식,4300,238.0,g,599,23.8,1240,36.9,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n구운 주먹밥,간편식,2350,196.0,g,379,10.3,1140,32.8,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n견과 에너지바,에너지바,1700,48.0,g,212,7.7,86,10.7,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n초코 단백질바,에너지바,2200,51.0,g,242,12.6,148,6.0,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n곡물 시리얼바,에너지바,1650,38.0,g,169,10.1,67,10.1,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n과일 오트바,에너지바,1900,51.0,g,228,5.9,119,9.8,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n저당 프로틴바,에너지바,1600,41.0,g,179,3.7,41,9.6,가상 샘플 데이터,학교 매점 실측 자료로 교체 필요\n'


def load_sample_data() -> pd.DataFrame:
    """Load sample_data.csv, or use the embedded sample when the file is missing."""
    if SAMPLE_PATH.exists():
        return load_sample_data()
    return pd.read_csv(io.StringIO(FALLBACK_SAMPLE_CSV))


def apply_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --orange: {ORANGE};
            --dark-orange: {DARK_ORANGE};
            --pale-orange: {PALE_ORANGE};
            --cream: {CREAM};
            --text: {TEXT};
            --muted: {MUTED};
            --border: {BORDER};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, Pretendard, "Noto Sans KR", Arial, sans-serif;
            color: var(--text);
        }}

        .stApp {{ background: #FFFFFF; }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFF8F1 0%, #FFFFFF 100%);
            border-right: 1px solid var(--border);
        }}

        .block-container {{
            padding-top: 1.45rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        h1, h2, h3 {{ letter-spacing: -0.035em; }}

        .hero {{
            padding: 2.25rem 2.45rem;
            border: 1px solid var(--border);
            border-radius: 26px;
            background:
                radial-gradient(circle at 90% 12%, rgba(244,122,31,.16), transparent 25%),
                linear-gradient(135deg, #FFFFFF 0%, #FFF8F1 100%);
            box-shadow: 0 14px 40px rgba(78, 48, 20, .06);
            margin-bottom: 1.15rem;
        }}

        .eyebrow {{
            display: inline-flex;
            color: var(--dark-orange);
            background: var(--pale-orange);
            border: 1px solid #FFD7B6;
            padding: .38rem .72rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: .82rem;
            margin-bottom: .9rem;
        }}

        .hero-title {{
            font-size: clamp(2.15rem, 4vw, 4rem);
            line-height: 1.03;
            font-weight: 900;
            letter-spacing: -0.058em;
            margin: 0 0 .9rem 0;
            color: #171717;
        }}

        .hero-title span {{ color: var(--orange); }}

        .hero-subtitle {{
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.75;
            max-width: 920px;
            margin: 0;
        }}

        .section-kicker {{
            color: var(--orange);
            font-weight: 850;
            font-size: .78rem;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }}

        .section-title {{
            font-size: 1.62rem;
            font-weight: 880;
            margin: 0 0 .35rem 0;
            color: #191919;
        }}

        .section-desc {{
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: .9rem;
        }}

        .metric-card {{
            min-height: 122px;
            padding: 1.12rem 1.18rem;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: #FFFFFF;
            box-shadow: 0 8px 26px rgba(65, 39, 15, .045);
        }}

        .metric-label {{
            color: var(--muted);
            font-size: .85rem;
            font-weight: 680;
            margin-bottom: .48rem;
        }}

        .metric-value {{
            font-size: 1.62rem;
            font-weight: 900;
            color: #171717;
            letter-spacing: -0.04em;
        }}

        .metric-note {{
            margin-top: .35rem;
            color: var(--muted);
            font-size: .8rem;
            line-height: 1.45;
        }}

        .soft-card, .orange-card {{
            padding: 1.18rem 1.25rem;
            border-radius: 18px;
            height: 100%;
        }}

        .soft-card {{ border: 1px solid var(--border); background: #FFFFFF; }}
        .orange-card {{ border: 1px solid #FFD6B4; background: var(--pale-orange); }}

        .insight {{
            border-left: 4px solid var(--orange);
            background: #FFF9F4;
            padding: 1rem 1.15rem;
            border-radius: 0 14px 14px 0;
            line-height: 1.72;
            color: #343434;
        }}

        .tag {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .38rem .72rem;
            font-weight: 800;
            font-size: .82rem;
            color: var(--dark-orange);
            background: var(--pale-orange);
            border: 1px solid #FFD8B8;
        }}

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            border-radius: 12px;
            border: 1px solid var(--orange);
            background: var(--orange);
            color: white;
            font-weight: 780;
            min-height: 42px;
        }}

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--dark-orange);
            background: var(--dark-orange);
            color: white;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: .45rem; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 12px 12px 0 0; padding: .55rem .9rem; }}
        .stTabs [aria-selected="true"] {{ color: var(--orange); }}

        hr {{ border-color: var(--border); }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div style="margin:1.15rem 0 .75rem 0;">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=58, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, Noto Sans KR, sans-serif", color=TEXT),
        title_font=dict(size=17, color=TEXT),
        hoverlabel=dict(bgcolor="#FFFFFF"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#F1ECE7", zerolinecolor="#E8DFD7")
    fig.update_yaxes(gridcolor="#F1ECE7", zerolinecolor="#E8DFD7")
    return fig


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rounded = [
        "가격",
        "총열량",
        "총내용량",
        "당류",
        "나트륨",
        "단백질",
        "원/100kcal",
        "kcal/1000원",
        "원/100단위",
        "100단위당열량",
        "당류/100kcal",
        "나트륨/100kcal",
        "단백질/100kcal",
        "열량차이",
        "가격차이",
        "열량차이율(%)",
        "가격차이율(%)",
        "KNN거리",
        "유사도점수",
    ]
    for c in rounded:
        if c in result.columns:
            digits = 0 if c in {"가격", "총열량", "나트륨", "가격차이", "열량차이"} else 2
            result[c] = result[c].round(digits)
    return result


def reference_radar(reference: pd.Series, top: pd.Series, features: list[str], source_df: pd.DataFrame) -> go.Figure:
    labels = {
        "가격": "가격",
        "총열량": "열량",
        "총내용량": "내용량",
        "당류": "당류",
        "나트륨": "나트륨",
        "단백질": "단백질",
        "원/100kcal": "원/100kcal",
        "원/100단위": "원/100단위",
    }
    usable = [f for f in features if f in labels][:6]
    if len(usable) < 3:
        usable = ["가격", "총열량", "총내용량"]

    mins = source_df[usable].min()
    maxs = source_df[usable].max()
    denom = (maxs - mins).replace(0, 1)
    ref_values = ((reference[usable] - mins) / denom * 100).tolist()
    top_values = ((top[usable] - mins) / denom * 100).tolist()
    theta = [labels[f] for f in usable]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ref_values + [ref_values[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=str(reference["상품명"]),
            line=dict(color=ORANGE, width=3),
            fillcolor="rgba(244,122,31,.17)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=top_values + [top_values[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=str(top["상품명"]),
            line=dict(color="#60656B", width=2),
            fillcolor="rgba(96,101,107,.08)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER)),
        title="선택 상품과 1순위 유사 상품",
    )
    return style_fig(fig, 470)


# =========================================================
# 2. 데이터 불러오기와 세션 상태
# =========================================================
apply_css()

if "added_products" not in st.session_state:
    st.session_state.added_products = []

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:.55rem .1rem .95rem .1rem;">
            <div style="font-size:1.55rem;font-weight:920;letter-spacing:-.05em;">
                Snack<span style="color:{ORANGE};">Lens</span>
            </div>
            <div style="color:{MUTED};font-size:.82rem;margin-top:.25rem;">
                school store decision support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "메뉴",
        ["홈", "조건 비교", "KNN 추천", "파레토 분석", "심화 분석", "데이터 관리"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### 데이터 선택")
    data_mode = st.radio("데이터", ["샘플 데이터", "CSV 업로드"], label_visibility="collapsed")
    uploaded = None
    if data_mode == "CSV 업로드":
        uploaded = st.file_uploader("매점 상품 CSV", type=["csv"])

    template = load_sample_data().head(5)
    st.download_button(
        "CSV 템플릿 받기",
        data=csv_bytes(template),
        file_name="snacklens_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("가격·열량·내용량은 제품 전체 기준으로 입력하세요. 1회 제공량과 혼동하지 않도록 주의합니다.")

if uploaded is not None:
    try:
        raw_df = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(f"CSV를 읽지 못했습니다: {exc}")
        st.stop()
else:
    raw_df = load_sample_data()
    if data_mode == "CSV 업로드":
        st.info("CSV를 업로드하기 전까지 샘플 데이터로 기능을 보여줍니다.")

if st.session_state.added_products:
    raw_df = pd.concat([raw_df, pd.DataFrame(st.session_state.added_products)], ignore_index=True)

df, cleaning_summary, missing_required = prepare_data(raw_df)
if missing_required:
    st.error("필수 열이 없습니다: " + ", ".join(missing_required))
    st.stop()

if len(df) < 10:
    st.warning("비교와 추천의 안정성을 위해 상품 30개 이상을 권장합니다.")

product_names = sorted(df["상품명"].unique().tolist())
categories = sorted(df["카테고리"].unique().tolist())


# =========================================================
# 3. 페이지
# =========================================================
if page == "홈":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">분산된 매점 정보를 한눈에</div>
            <div class="hero-title">가격표를 넘어,<br><span>목적에 맞는 비교</span>를 합니다.</div>
            <p class="hero-subtitle">
                SnackLens는 가격·열량·내용량·영양성분을 같은 표 안에서 비교하고,
                표준화된 K-최근접 이웃 알고리즘으로 유사 상품을 찾습니다.
                가장 많은 열량이나 가장 싼 상품을 정답으로 강요하지 않고,
                학생이 자신의 소비 목적을 고른 뒤 필요한 정보를 투명하게 확인하도록 돕습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("등록 상품", f"{len(df):,}개", "분석 가능한 상품 수")
    with c2:
        metric_card("카테고리", f"{df['카테고리'].nunique()}개", "같은 유형 비교 가능")
    with c3:
        metric_card("평균 가격", f"{df['가격'].mean():,.0f}원", "샘플 또는 업로드 자료 기준")
    with c4:
        metric_card("원/100kcal 범위", f"{df['원/100kcal'].min():,.0f}~{df['원/100kcal'].max():,.0f}", "단위가격 격차")

    section_header(
        "Core Design",
        "하나의 정답 대신 세 가지 소비 목적",
        "합리성은 소비자가 무엇을 중요하게 생각하는지에 따라 달라집니다.",
    )
    cols = st.columns(3)
    cards = [
        ("가성비", "에너지·양 대비 가격을 비교하되, 영양적 우수성과 동일시하지 않습니다."),
        ("저열량", "비슷한 가격대에서 총열량과 당류가 낮은 상품을 탐색합니다."),
        ("영양균형", "가격과 함께 당류·나트륨·단백질의 균형을 비교합니다."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="tag">{title}</div>
                    <div style="margin-top:.75rem;color:{MUTED};line-height:1.68;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header(
        "Workflow",
        "데이터과학의 전 과정을 앱으로 연결",
    )
    flow = st.columns(4)
    steps = [
        ("01", "수집", "영양성분표와 가격표를 직접 조사"),
        ("02", "정제", "총제품 기준과 단위를 통일"),
        ("03", "모델링", "표준화 후 가중 KNN 거리 계산"),
        ("04", "의사결정", "조건 비교·파레토·회귀 결과 제공"),
    ]
    for col, (num, title, desc) in zip(flow, steps):
        with col:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div style="font-size:.8rem;color:{ORANGE};font-weight:900;">{num}</div>
                    <div style="font-size:1.08rem;font-weight:880;margin:.32rem 0;">{title}</div>
                    <div style="color:{MUTED};font-size:.9rem;line-height:1.55;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="insight" style="margin-top:1rem;">
            <b>해석 원칙:</b> 원/100kcal이 낮다는 것은 에너지 단위당 가격이 낮다는 뜻일 뿐,
            건강·맛·포만감·만족도가 더 높다는 뜻은 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "조건 비교":
    section_header(
        "Rule-based Comparison",
        "비슷한 조건의 상품을 표로 비교합니다",
        "절대 범위와 상대 범위를 명시적으로 선택하여 ‘비슷함’의 기준을 투명하게 만듭니다.",
    )

    top1, top2, top3 = st.columns([1.5, 1, 1])
    with top1:
        reference_name = st.selectbox("기준 상품", product_names)
    with top2:
        basis_label = st.selectbox("비교 기준", ["열량", "가격"])
    with top3:
        same_category = st.checkbox("같은 카테고리만", value=True)

    basis = "총열량" if basis_label == "열량" else "가격"
    ref = df.loc[df["상품명"] == reference_name].iloc[0]

    mode_col, tol_col, sort_col = st.columns(3)
    with mode_col:
        tolerance_mode = st.radio("범위 방식", ["절대값", "상대비율"], horizontal=True)
    with tol_col:
        if tolerance_mode == "절대값":
            if basis == "총열량":
                tolerance_value = st.slider("허용 열량 차이(kcal)", 0, 150, 20, 5)
            else:
                tolerance_value = st.slider("허용 가격 차이(원)", 0, 2000, 300, 50)
        else:
            tolerance_value = st.slider("허용 차이율(%)", 0, 50, 10, 1)
    with sort_col:
        sort_label = st.selectbox(
            "정렬 기준",
            ["기준과 가까운 순", "원/100kcal 낮은 순", "가격 낮은 순", "열량 낮은 순"],
        )

    result = similar_products(
        df,
        reference_name,
        basis,
        tolerance_mode,
        float(tolerance_value),
        same_category=same_category,
        include_reference=True,
    )

    sort_map = {
        "기준과 가까운 순": "열량차이율(%)" if basis == "총열량" else "가격차이율(%)",
        "원/100kcal 낮은 순": "원/100kcal",
        "가격 낮은 순": "가격",
        "열량 낮은 순": "총열량",
    }
    result = result.sort_values(sort_map[sort_label]).reset_index(drop=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("기준 상품 가격", f"{ref['가격']:,.0f}원", reference_name)
    with m2:
        metric_card("기준 상품 열량", f"{ref['총열량']:,.0f}kcal", f"{ref['총내용량']:g}{ref['단위']}")
    with m3:
        metric_card("원/100kcal", f"{ref['원/100kcal']:,.0f}원", "에너지 단위가격")
    with m4:
        metric_card("조건 충족 상품", f"{len(result):,}개", "기준 상품 포함")

    table_cols = [
        "기준상품",
        "상품명",
        "카테고리",
        "가격",
        "총열량",
        "총내용량",
        "단위",
        "원/100kcal",
        "kcal/1000원",
        "원/100단위",
        "당류",
        "나트륨",
        "단백질",
        "열량차이",
        "가격차이",
        "열량차이율(%)",
        "가격차이율(%)",
    ]
    if result.empty:
        st.warning("설정한 범위 안에 상품이 없습니다. 범위를 넓혀 보세요.")
    else:
        st.dataframe(format_table(result[table_cols]), use_container_width=True, hide_index=True, height=410)

        plot = result.copy()
        plot["표시"] = np.where(plot["상품명"] == reference_name, "선택 상품", "비교 상품")
        fig = px.scatter(
            plot,
            x="총열량",
            y="가격",
            color="표시",
            size="총내용량",
            hover_name="상품명",
            hover_data=["카테고리", "원/100kcal", "당류", "나트륨", "단백질"],
            color_discrete_map={"선택 상품": ORANGE, "비교 상품": "#9D9A96"},
            title="가격-열량 비교 지도",
        )
        fig = style_fig(fig, 470)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="insight">
            <b>설계 결정:</b> 절대 범위는 이해하기 쉽고, 상대 범위는 상품 규모에 따라 비교 폭이 조정됩니다.
            예를 들어 10kcal 차이는 100kcal 상품에서는 10%지만 500kcal 상품에서는 2%입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "KNN 추천":
    section_header(
        "K-Nearest Neighbors",
        "표준화된 거리로 여러 조건이 비슷한 상품을 찾습니다",
        "가격·열량·영양성분의 단위가 다르므로 평균 0, 표준편차 1의 표준점수로 바꾼 뒤 가중 거리를 계산합니다.",
    )

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        reference_name = st.selectbox("기준 상품", product_names, key="knn_reference")
    with c2:
        mode = st.selectbox("소비 목적", ["가성비", "저열량", "영양균형", "직접 설정"])
    with c3:
        k = st.slider("추천 상품 수 K", 2, min(10, max(2, len(df) - 1)), 5)

    same_category = st.checkbox("같은 카테고리 안에서 추천", value=True, key="knn_category")

    all_features = ["가격", "총열량", "총내용량", "당류", "나트륨", "단백질", "원/100kcal", "원/100단위"]
    if mode == "직접 설정":
        selected_features = st.multiselect(
            "거리 계산에 사용할 변수",
            all_features,
            default=["가격", "총열량", "당류", "나트륨", "단백질"],
        )
        feature_weights: Dict[str, float] = {}
        if selected_features:
            weight_cols = st.columns(min(4, len(selected_features)))
            for i, feature in enumerate(selected_features):
                with weight_cols[i % len(weight_cols)]:
                    feature_weights[feature] = st.slider(
                        f"{feature} 가중치",
                        0.0,
                        3.0,
                        1.0,
                        0.1,
                        key=f"w_{feature}",
                    )
        else:
            feature_weights = {}
    else:
        feature_weights = mode_feature_weights(mode)
        st.markdown(
            "사용 변수: "
            + " · ".join([f"**{name}**({weight:g})" for name, weight in feature_weights.items()])
        )

    try:
        knn = weighted_knn_recommend(
            df,
            reference_name,
            feature_weights,
            k=k,
            same_category=same_category,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    ref = df.loc[df["상품명"] == reference_name].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("후보 데이터", f"{knn.candidate_count}개", "결측값·카테고리 필터 후")
    with m2:
        metric_card("사용 변수", f"{len(knn.used_features)}개", "표준화 후 거리 계산")
    with m3:
        metric_card("K", f"{k}", "가까운 상품 수")
    with m4:
        top_score = knn.recommendations["유사도점수"].iloc[0] if not knn.recommendations.empty else 0
        metric_card("최고 유사도", f"{top_score:.1f}점", "확률이 아닌 상대 점수")

    tab1, tab2, tab3 = st.tabs(["추천 결과", "표준화 확인", "KNN 원리"])
    with tab1:
        rec_cols = [
            "상품명",
            "카테고리",
            "가격",
            "총열량",
            "총내용량",
            "단위",
            "당류",
            "나트륨",
            "단백질",
            "원/100kcal",
            "원/100단위",
            "가격차이",
            "열량차이",
            "KNN거리",
            "유사도점수",
        ]
        st.dataframe(
            format_table(knn.recommendations[rec_cols]),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        if not knn.recommendations.empty:
            top = knn.recommendations.iloc[0]
            fig = reference_radar(ref, top, knn.used_features, df)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 기준 상품의 표준점수")
        st.dataframe(format_table(knn.standardized_reference), use_container_width=True, hide_index=True)
        st.markdown("#### 후보 데이터의 변수 범위")
        st.dataframe(format_table(knn.feature_summary), use_container_width=True, hide_index=True)
        st.caption("표준화는 각 변수의 단위를 없애지만, 어떤 변수를 더 중요하게 볼지는 가중치가 결정합니다.")

    with tab3:
        st.latex(r"z=\frac{x-\mu}{\sigma}")
        st.latex(r"d(i,j)=\sqrt{\sum_m w_m(z_{im}-z_{jm})^2}")
        st.markdown(
            """
            1. 각 변수를 평균과 표준편차로 표준화합니다.  
            2. 선택 목적에 따라 변수별 가중치를 적용합니다.  
            3. 가중 유클리드 거리가 가장 작은 K개 상품을 추천합니다.  
            4. 유사도점수는 `100/(1+거리)`로 만든 설명용 상대 점수이며 선택 확률이 아닙니다.
            """
        )

    st.markdown(
        """
        <div class="insight">
            <b>확률과 통계 연결:</b> 가격은 천 원 단위, 나트륨은 mg 단위이므로 원자료를 그대로 사용하면 가격이 거리를 지배합니다.
            표준점수는 각 값이 평균에서 표준편차 몇 배만큼 떨어져 있는지를 나타내어 서로 다른 단위의 변수를 공통 척도로 비교하게 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "파레토 분석":
    section_header(
        "Pareto Efficiency",
        "여러 기준에서 동시에 뒤처지는 상품을 구분합니다",
        "다른 상품보다 비싸면서 목적에 맞는 특성도 나쁘다면 지배된 상품으로 볼 수 있습니다. 파레토 상품은 정답이 아니라 효율적 후보 집합입니다.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("파레토 기준", ["가성비", "저열량", "영양균형"])
    with c2:
        category = st.selectbox("카테고리", ["전체"] + categories)
    with c3:
        selected_name = st.selectbox("강조 상품", product_names)

    subset = df.copy() if category == "전체" else df[df["카테고리"] == category].copy()
    objectives = pareto_objectives(mode)
    mask = pareto_front_mask(subset, objectives)
    subset["파레토여부"] = np.where(mask, "파레토 후보", "지배됨")

    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("분석 상품", f"{len(subset)}개", category)
    with p2:
        metric_card("파레토 후보", f"{int(mask.sum())}개", "다른 상품에 완전히 지배되지 않음")
    with p3:
        ratio = mask.mean() * 100 if len(mask) else 0
        metric_card("후보 비율", f"{ratio:.1f}%", "선택 기준에 따라 변함")

    if mode == "가성비":
        x, y, size = "총열량", "가격", "총내용량"
        title = "가성비 파레토: 낮은 가격·높은 열량·높은 내용량"
    elif mode == "저열량":
        x, y, size = "총열량", "가격", "당류"
        title = "저열량 파레토: 낮은 가격·낮은 열량·낮은 당류"
    else:
        x, y, size = "당류", "가격", "단백질"
        title = "영양균형 파레토: 낮은 가격·당류·나트륨, 높은 단백질"

    subset["강조"] = np.where(subset["상품명"] == selected_name, "선택 상품", subset["파레토여부"])
    color_map = {"선택 상품": ORANGE, "파레토 후보": "#50545A", "지배됨": "#D8D4D0"}
    fig = px.scatter(
        subset,
        x=x,
        y=y,
        size=size,
        color="강조",
        hover_name="상품명",
        hover_data=["카테고리", "총열량", "가격", "당류", "나트륨", "단백질", "원/100kcal"],
        color_discrete_map=color_map,
        title=title,
    )
    fig = style_fig(fig, 520)
    st.plotly_chart(fig, use_container_width=True)

    pareto_table = subset[subset["파레토여부"] == "파레토 후보"].copy()
    st.markdown("#### 파레토 후보 표")
    st.dataframe(
        format_table(pareto_table[display_columns(pareto_table, ["파레토여부"])]),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    st.markdown(
        """
        <div class="insight">
            파레토 후보끼리도 최종 선택은 달라질 수 있습니다. 어떤 학생은 가격을, 다른 학생은 당류나 단백질을 더 중요하게 보기 때문입니다.
            따라서 파레토 분석은 선택지를 줄이는 도구이지 하나의 상품을 강제하는 순위표가 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "심화 분석":
    section_header(
        "Advanced Analytics",
        "단위가격 분포와 헤도닉 가격 회귀를 탐색합니다",
        "가격이 열량·중량·영양성분·카테고리로 얼마나 설명되는지 분석하고, 설명되지 않은 차이를 잔차로 확인합니다.",
    )

    tab1, tab2 = st.tabs(["단위가격 분석", "헤도닉 회귀"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            metric = st.selectbox("분석 지표", ["원/100kcal", "kcal/1000원", "원/100단위", "100단위당열량"])
        with c2:
            cat_filter = st.selectbox("카테고리 필터", ["전체"] + categories, key="unit_category")
        unit_df = df if cat_filter == "전체" else df[df["카테고리"] == cat_filter]

        left, right = st.columns([1.15, 1])
        with left:
            fig = px.histogram(
                unit_df,
                x=metric,
                color_discrete_sequence=[ORANGE],
                nbins=22,
                marginal="box",
                title=f"{metric} 분포",
            )
            fig = style_fig(fig, 450)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            ranked = unit_df.sort_values(metric).head(12)
            fig = px.bar(
                ranked.sort_values(metric, ascending=False),
                x=metric,
                y="상품명",
                orientation="h",
                color_discrete_sequence=[ORANGE],
                title=f"{metric} 하위 12개 상품",
                text_auto=".3s",
            )
            fig = style_fig(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

        st.caption("원/100kcal이 낮다고 건강성이 높다는 뜻은 아닙니다. 단위가격은 여러 비교 지표 중 하나입니다.")

    with tab2:
        category = st.selectbox("회귀 범위", ["전체"] + categories, key="reg_category")
        try:
            model, reg = run_hedonic_regression(df, category)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                metric_card("분석 표본", f"{int(model.nobs)}개", "결측값 없는 상품")
            with r2:
                metric_card("R²", f"{model.rsquared:.3f}", "표본 내 설명력")
            with r3:
                metric_card("조정 R²", f"{model.rsquared_adj:.3f}", "변수 수를 고려")
            with r4:
                metric_card("평균 절대 잔차", f"{reg['residual'].abs().mean():,.0f}원", "실제-예측 가격 차이")

            coef = coefficient_table(model)
            st.markdown("#### 회귀계수")
            st.dataframe(format_table(coef), use_container_width=True, hide_index=True, height=330)

            fig = px.scatter(
                reg,
                x="predicted_price",
                y="price",
                hover_name="product",
                color="residual",
                color_continuous_scale=["#59636E", "#FFFFFF", ORANGE],
                title="실제 가격과 회귀 예측 가격",
                labels={"predicted_price": "예측 가격", "price": "실제 가격", "residual": "잔차"},
            )
            lo = min(reg["predicted_price"].min(), reg["price"].min())
            hi = max(reg["predicted_price"].max(), reg["price"].max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#999", dash="dash"))
            fig = style_fig(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

            reg_view = reg.rename(
                columns={
                    "product": "상품명",
                    "category": "카테고리",
                    "price": "실제가격",
                    "predicted_price": "예측가격",
                    "residual": "잔차",
                }
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 예상보다 가격이 높은 상품")
                st.dataframe(
                    format_table(reg_view.sort_values("잔차", ascending=False)[["상품명", "카테고리", "실제가격", "예측가격", "잔차"]].head(10)),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_b:
                st.markdown("#### 예상보다 가격이 낮은 상품")
                st.dataframe(
                    format_table(reg_view.sort_values("잔차")[["상품명", "카테고리", "실제가격", "예측가격", "잔차"]].head(10)),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                """
                <div class="insight">
                    <b>해석 주의:</b> 양의 잔차를 곧바로 ‘브랜드 프리미엄’, 음의 잔차를 ‘저평가’라고 확정할 수 없습니다.
                    맛·브랜드 인지도·포장·유통비처럼 수집하지 않은 요인이 잔차에 함께 들어갑니다.
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "데이터 관리":
    section_header(
        "Data Management",
        "데이터 수집과 정제가 탐구의 출발점입니다",
        "제품 전체 기준의 열량과 중량을 기록하고, 사용자 제보는 세션에만 임시 추가한 뒤 CSV로 내려받을 수 있습니다.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("입력 행", f"{cleaning_summary.input_rows}개", "업로드·샘플·제보 합계")
    with m2:
        metric_card("정제 후", f"{cleaning_summary.output_rows}개", "필수값 검증 후")
    with m3:
        metric_card("중복 제거", f"{cleaning_summary.duplicate_rows_removed}개", "상품명+카테고리 기준")
    with m4:
        metric_card("유효하지 않은 행", f"{cleaning_summary.invalid_rows_removed}개", "가격·내용량·열량 0 이하 등")

    tab1, tab2, tab3 = st.tabs(["데이터 표", "상품 제보", "조사 지침"])

    with tab1:
        st.dataframe(
            format_table(df[display_columns(df)]),
            use_container_width=True,
            hide_index=True,
            height=520,
        )
        st.download_button(
            "정제·파생지표 포함 CSV 받기",
            data=csv_bytes(df),
            file_name="snacklens_cleaned_products.csv",
            mime="text/csv",
        )

    with tab2:
        st.caption("제보 데이터는 현재 브라우저 세션에만 저장됩니다. 앱을 새로 시작하면 사라질 수 있으므로 CSV를 내려받아 합쳐 주세요.")
        with st.form("add_product_form", clear_on_submit=True):
            a, b, c = st.columns(3)
            with a:
                product = st.text_input("상품명")
                category = st.selectbox("카테고리", categories + ["기타"])
                price = st.number_input("가격(원)", min_value=1, value=1500, step=100)
            with b:
                amount = st.number_input("총내용량", min_value=0.1, value=60.0, step=1.0)
                unit = st.selectbox("단위", ["g", "mL"])
                calories = st.number_input("총열량(kcal)", min_value=1.0, value=200.0, step=5.0)
            with c:
                sugar = st.number_input("당류(g)", min_value=0.0, value=10.0, step=0.5)
                sodium = st.number_input("나트륨(mg)", min_value=0.0, value=150.0, step=10.0)
                protein = st.number_input("단백질(g)", min_value=0.0, value=3.0, step=0.5)
            note = st.text_input("비고(선택)")
            submitted = st.form_submit_button("세션 데이터에 추가", use_container_width=True)

        if submitted:
            if not product.strip():
                st.error("상품명을 입력하세요.")
            else:
                st.session_state.added_products.append(
                    {
                        "상품명": product.strip(),
                        "카테고리": category,
                        "가격": price,
                        "총내용량": amount,
                        "단위": unit,
                        "총열량": calories,
                        "당류": sugar,
                        "나트륨": sodium,
                        "단백질": protein,
                        "데이터출처": "사용자 제보",
                        "비고": note,
                    }
                )
                st.success("상품을 추가했습니다. 페이지가 다시 실행됩니다.")
                st.rerun()

    with tab3:
        st.markdown(
            """
            **현장 조사 권장 절차**

            1. 매점 가격표와 제품 영양성분표를 함께 촬영한다.  
            2. ‘1회 제공량’이 아니라 **제품 전체의 총열량**인지 확인한다.  
            3. 총내용량은 g 또는 mL로 기록하고 단위를 별도 열에 보존한다.  
            4. 당류·나트륨·단백질은 가능하면 제품 전체 기준으로 환산한다.  
            5. 품절·가격 변경 날짜를 비고에 기록하여 데이터의 시점을 남긴다.  
            6. 같은 제품이 중복되면 가장 최근 조사값을 사용한다.
            """
        )
        missing_table = pd.DataFrame(
            {"열": cleaning_summary.missing_by_column.keys(), "결측값 수": cleaning_summary.missing_by_column.values()}
        )
        st.markdown("#### 원자료 결측값")
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight">
            실제 제출본에서는 샘플 데이터를 그대로 사용하지 말고 학교 매점의 실측 자료로 교체해야 합니다.
            앱의 분석 결과는 데이터의 정확도와 최신성에 직접 의존합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
