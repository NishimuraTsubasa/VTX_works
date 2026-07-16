from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "input"


def generate(seed: int = 42, n_stocks: int = 240, n_months: int = 30) -> None:
    rng = np.random.default_rng(seed)
    countries = np.array(["US", "Japan", "UK", "Germany"])
    currencies = {"US": "USD", "Japan": "JPY", "UK": "GBP", "Germany": "EUR"}
    sectors = np.array(["Technology", "Financials", "Industrials", "Consumer", "Healthcare"])
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    stock_country = rng.choice(countries, size=n_stocks, p=[0.35, 0.25, 0.2, 0.2])
    stock_sector = rng.choice(sectors, size=n_stocks)
    stock_ids = [f"ZZ{j:010d}" for j in range(n_stocks)]
    base_mcap = np.exp(rng.normal(23.5, 1.1, size=n_stocks))

    factors = {name: np.zeros((n_months, n_stocks)) for name in ["FA0101", "FA0102", "FA1001", "FA2001", "FA3001", "FA4001"]}
    for name in factors:
        factors[name][0] = rng.normal(size=n_stocks)
        phi = {"FA0101": 0.88, "FA0102": 0.82, "FA1001": 0.55, "FA2001": 0.80, "FA3001": 0.70, "FA4001": 0.75}[name]
        for t in range(1, n_months):
            common = rng.normal(scale=0.18)
            factors[name][t] = phi * factors[name][t - 1] + np.sqrt(1 - phi**2) * rng.normal(size=n_stocks) + common

    rows = []
    country_effect = {c: rng.normal(scale=0.006, size=n_months) for c in countries}
    sector_effect = {s: rng.normal(scale=0.004, size=n_months) for s in sectors}
    returns = np.zeros((n_months, n_stocks))
    returns[0] = rng.normal(0.008, 0.05, size=n_stocks)
    for t in range(1, n_months):
        f1 = factors["FA0101"][t - 1]
        f2 = factors["FA0102"][t - 1]
        f3 = factors["FA1001"][t - 1]
        f4 = factors["FA2001"][t - 1]
        f5 = factors["FA3001"][t - 1]
        f6 = factors["FA4001"][t - 1]
        alpha = (
            0.0045 * f1
            + 0.0025 * (f2**2 - 1.0)
            + 0.0040 * np.maximum(f3 - 0.15, 0.0)
            - 0.0038 * f4
            + 0.0010 * f5
            + 0.0030 * f6
        )
        structural = np.array([country_effect[c][t] + sector_effect[s][t] for c, s in zip(stock_country, stock_sector)])
        returns[t] = 0.006 + alpha + structural + rng.normal(scale=0.045, size=n_stocks)

    for t, date in enumerate(dates):
        mcap = base_mcap * np.exp(0.003 * t + rng.normal(scale=0.08, size=n_stocks))
        for j, isin in enumerate(stock_ids):
            row = {
                "date": date,
                "ISIN": isin,
                "stock_return": returns[t, j],
                "market_cap": mcap[j],
                "currency": currencies[stock_country[j]],
                "sector": stock_sector[j],
                "country": stock_country[j],
            }
            for name in factors:
                value = factors[name][t, j]
                # 約3%のランダム欠損
                row[name] = np.nan if rng.random() < 0.03 else value
            rows.append(row)
    data = pd.DataFrame(rows)

    factor_master = pd.DataFrame([
        ["FA0101", "Value", 1, 1, 1.0],
        ["FA0102", "Value", 1, 1, 1.0],
        ["FA1001", "Momentum", 1, 1, 1.0],
        ["FA2001", "Quality", 1, -1, 1.0],
        ["FA3001", "Growth", 1, 1, 1.0],
        ["FA4001", "Low_Risk", 1, 1, 1.0],
    ], columns=["Factor_Code", "Factor_Group", "Enabled", "Direction", "Base_Weight"])
    group_settings = pd.DataFrame([
        ["Value", 1, "ic_adjusted"],
        ["Momentum", 1, "equal_weight"],
        ["Quality", 1, "equal_weight"],
        ["Growth", 1, "equal_weight"],
        ["Low_Risk", 1, "equal_weight"],
    ], columns=["Factor_Group", "Enabled", "Aggregation_Method"])
    feature_control = pd.DataFrame([
        ["group", "Value", 1, "selected", 1],
        ["group", "Momentum", 1, "selected", 1],
    ], columns=["Scope_Type", "Scope_Value", "Enabled", "Generation_Mode", "Include_Raw"])
    derived_rules = pd.DataFrame([
        ["VAL_DIFF_1", "group", "Value", "difference", 1, 0, 1, 1, 1, 1, 1, "inherit", 1],
        ["VAL_MADEV_6", "group", "Value", "rolling_mean_deviation", 0, 6, 4, 1, 1, 1, 1, "inherit", 1],
        ["MOM_DIFF_1", "group", "Momentum", "difference", 1, 0, 1, 1, 1, 1, 1, "inherit", 1],
    ], columns=["Rule_ID", "Scope_Type", "Scope_Value", "Feature_Type", "Difference_Periods", "Window_Periods", "Min_Periods", "Source_Lag_Periods", "Exclude_Source_From_Baseline", "Enabled", "Selected", "Direction_Mode", "Custom_Direction"])
    factor_overrides = pd.DataFrame(columns=["Factor_Code", "Transform", "Winsorize", "Neutralize", "Rank_Normalize", "Min_Coverage"])
    group_overrides = pd.DataFrame(columns=["Factor_Group", "Lookback_Periods", "Min_Periods", "Max_Weight", "Weight_Smoothing", "Fallback_Method", "PCA_Anchor_Factor"])

    OUT.mkdir(parents=True, exist_ok=True)
    factors_path = OUT / "factors_and_returns.xlsx"
    with pd.ExcelWriter(factors_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        pd.DataFrame({
            "Item": ["Purpose", "Row Grain", "Primary Key", "Return Alignment", "Missing Factor Values"],
            "Definition": [
                "各時点・各銘柄の属性、リターン、FAコードを入力",
                "1行 = 1時点 x 1銘柄",
                "date + ISIN",
                "サンプルは当月リターン。Configで1期先へシフト",
                "空欄/NaN。0で補完しない",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame([
            ["date", "Date", "ファクター観測時点"],
            ["ISIN", "Text", "銘柄キー"],
            ["stock_return", "Float", "入力行の日付に対応する当月リターン"],
            ["market_cap", "Float", "予測時点の時価総額"],
            ["currency", "Text", "ISO通貨コード"],
            ["sector", "Text", "セクター"],
            ["country", "Text", "国・市場区分"],
            ["FAxxxx", "Float", "ファクター生値"],
        ], columns=["Column", "Data_Type", "Definition"]).to_excel(writer, sheet_name="Column_Dictionary", index=False)
        data.to_excel(writer, sheet_name="data", index=False)
        wb = writer.book
        header = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_row(0, 22, header)
            ws.set_column(0, 0, 16)
            ws.set_column(1, 20, 15)

    master_path = OUT / "factor_master.xlsx"
    with pd.ExcelWriter(master_path, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Sheet": ["Factor_Master", "Group_Settings", "Feature_Engineering_Control", "Derived_Feature_Rules", "Factor_Overrides", "Group_Overrides"],
            "Edit": ["必須", "必須", "派生特徴量利用時", "派生特徴量利用時", "例外時のみ", "例外時のみ"],
            "Purpose": ["FAコードのグループ・方向", "グループ統合方法", "派生生成対象", "差分・移動平均乖離ルール", "FA固有例外", "グループ固有例外"],
        }).to_excel(writer, sheet_name="README", index=False)
        factor_master.to_excel(writer, sheet_name="Factor_Master", index=False)
        group_settings.to_excel(writer, sheet_name="Group_Settings", index=False)
        feature_control.to_excel(writer, sheet_name="Feature_Engineering_Control", index=False)
        derived_rules.to_excel(writer, sheet_name="Derived_Feature_Rules", index=False)
        factor_overrides.to_excel(writer, sheet_name="Factor_Overrides", index=False)
        group_overrides.to_excel(writer, sheet_name="Group_Overrides", index=False)
        wb = writer.book
        header = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_row(0, 22, header)
            ws.set_column(0, 20, 20)

    print(f"Generated: {factors_path}")
    print(f"Generated: {master_path}")


if __name__ == "__main__":
    generate()
