from __future__ import annotations

SCENARIO_DESCRIPTIONS = {
    "S00_Current_Direct_EW": "0-1順位へ変換した全FAを直接等ウェイト",
    "S01_Missing_Adjusted_EW": "欠損FAを除外し利用可能FAで再正規化",
    "S02_Winsorized_Direct_EW": "外れ値処理後のFAを直接等ウェイト",
    "S03_Neutralized_Direct_EW": "国・セクター・サイズ中立化後のFAを直接等ウェイト",
    "S04_Hierarchical_Equal_Weight": "FAグループ内・グループ間の階層等ウェイト",
    "S05_Correlation_Adjusted_IC": "相関調整ICウェイトでグループ集約",
    "S06_Selected_Factor_Models": "グローバル単一FAのOOF SubScoreをFactorScoreへ集約",
    "S07_Layer3_Final_Model": "国別・地域・部分プーリングを選択する第3層最終モデル",
}
