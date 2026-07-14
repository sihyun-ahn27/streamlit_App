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
