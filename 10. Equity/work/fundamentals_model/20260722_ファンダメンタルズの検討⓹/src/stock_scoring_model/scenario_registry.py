from __future__ import annotations

SCENARIO_DESCRIPTIONS = {
    "N00_Direct_RawScore_EW": "中立化済みRaw Factor Scoreを全FA直接等ウェイト",
    "N01_Hierarchical_FactorCount_EW": "階層表示だがFA数比例ウェイトでN00を再現",
    "N02_Hierarchical_Group_EW": "グループ内・グループ間を等ウェイト",
    "N03_FactorReturn_Correlation": "Q5-Q1 Factor Return相関によるグループ内ウェイト",
    "N04_FactorReturn_Correlation_ShrunkEW": "相関ウェイトをEqual Weightへ縮小した本線Aggregate Score",
    "N05_L3_OLS_MainEffects": "国別OLS、Aggregate FactorScore主効果のみ",
    "N06_L3_Ridge_MainEffects": "国別Ridge、Aggregate FactorScore主効果のみ",
    "N07_L3_Ridge_SelectedInteractions": "国別Ridge、Factor主効果＋選択Sector×Factor交差項",
}
