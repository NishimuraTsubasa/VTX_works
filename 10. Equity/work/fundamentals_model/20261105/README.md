# Factor Tilt Research Lab — 数式・戦略・実装 README

**対象コード:** `factor_tilt_research.py` v1.0.0  
**資料版:** 1.0 / 2026-11-05  
**対応スライド:** `factor_tilt_strategies.pptx`  
**コード SHA-256:** `6c38b5d687ea38059b2486422da3443816059ef6bbf0d2cef47dfb320b852f06`

このREADMEは、添付されたコードの**現行実装を説明する技術仕様書**です。実行コードは変更していません。コードにないモデル・制約を実装済みとして扱わず、数式の記号と実際の変数・関数を対応させています。数式はGitHub／VS Code等のLaTeX対応Markdownビューアで閲覧してください。

> **最初に読む結論**  
> 本コードは、翌1か月のファクターリターンを予測する12モデルと、その予測を配分へ変換する3ルールを分離した比較基盤です。`all`では12モデル×3ルールの36組合せに3ベンチマークを加えた39系列を出力します。**39種類の独立した投資仮説ではなく、同じになる系列もあります。** `core`は10モデル・33系列です。Kagkadis et al.（2024）の完全再現ではありません。

---

## 目次

1. [この実装で比較するもの](#01-overview)
2. [実行方法と入力契約](#02-input)
3. [記号・時点・配列形状](#03-notation)
4. [共通前処理と予測後処理](#04-preprocessing)
5. [M01–M04：基準と個別回帰](#05-local)
6. [M05–M08：情報共有と非線形化](#06-pooled)
7. [M09–M12：PCA／PLSと状態空間](#07-pc)
8. [ネストした時系列検証](#08-validation)
9. [配分の共通定義と共分散](#09-allocation)
10. [S01–S03：シグナルから配分へ](#10-strategies)
11. [B01–B03：ベンチマーク](#11-benchmarks)
12. [損益・費用・保有比率ドリフト](#12-backtest)
13. [3因子の計算例](#13-example)
14. [シグナル・戦略評価指標](#14-metrics)
15. [不確実性と事後診断](#15-inference)
16. [比較実験の推奨設計と限界](#16-research)
17. [出力・API・他のChatGPTへの引継ぎ](#17-handoff)
18. [設定・候補・コード位置・検証記録](#18-audit)
19. [原論文との差分と参考資料](#19-references)

<a id="01-overview"></a>
## 1. この実装で比較するもの

研究上の問いを三段階に分けます。

- **予測:** ポートフォリオ特性から、翌月のファクターリターンを過去平均よりよく予測できるか。
- **タイミング:** 平均的に強いファクターを好むだけでなく、各ファクターの通常の期待値からの変化を捉えられるか。
- **戦略:** 同じ予測に、配分上限・リスク・売買抑制を適用しても、比較対象に対する追加リターンが残るか。

| ID | コード上のモデル名 | モデルの役割 | core |
|---|---|---|---|
| M01 | `mean` | 学習窓の過去平均 | ○ |
| M02 | `local_ridge` | ファクター別の線形回帰＋縮小 | ○ |
| M03 | `local_pls_ridge` | 因子内の教師あり圧縮＋Ridge | — |
| M04 | `local_ew_ridge` | 最近の観測を重くする個別Ridge | ○ |
| M05 | `pooled_ridge` | 全因子で標準化単位の傾きを共有 | ○ |
| M06 | `partial_pool` | 共有係数＋因子固有差 | ○ |
| M07 | `pooled_interactions` | 2次項・交互作用を共有 | ○ |
| M08 | `pooled_histgb` | 浅い木のブースティングを共有 | — |
| M09 | `pc_pls_ridge` | 相関PCA＋PC別PLS＋Ridge | ○ |
| M10 | `pc_pls_ew` | 同じ圧縮構造＋指数減衰Ridge | ○ |
| M11 | `pc_pls_static_kalman` | Q=0の静的対照 | ○ |
| M12 | `pc_pls_tvp` | Qを選択する時変係数回帰 | ○ |

配分ルールは `timing_rank`、`timing_mv`、`level_mv`。ベンチマークは `baseline_equal`、`baseline_inverse_vol`、`baseline_initial_fixed` です。文字列でモデルを選ぶ際、`mean`を指定しなくても自動追加されます。

**実装位置:** `model_registry`（L598–L631） / `resolve_models`（L634–L651）

<a id="02-input"></a>
## 2. 実行方法と入力契約

### 2.1 最短の実行手順

```bash
python -m pip install numpy pandas scipy scikit-learn threadpoolctl
python factor_tilt_research.py self-test
python factor_tilt_research.py list-models
python factor_tilt_research.py demo --models all --out demo_results
python factor_tilt_research.py template --out input_template
python factor_tilt_research.py run --returns returns.csv --features features.csv --models all --out results
```

上記は1行ごとのコマンドです。PowerShellでも行継続文字なしで実行できます。デモは合成データの動作確認であり、実際の38ファクターの運用成績ではありません。出力先が空でなければ停止します。上書きするときだけ `--overwrite` を明示してください。

全モデル実行は必須ではありません。比較の出発点は例えば以下です。

```bash
python factor_tilt_research.py run --returns returns.csv --features features.csv --models mean,local_ridge,pooled_ridge,partial_pool,pc_pls_ridge,pc_pls_static_kalman,pc_pls_tvp --out comparison_01
```

### 2.2 リターンと特徴量

`returns.csv` は横持ちです。`date`は**実現リターンの対象月末**です。

```csv
date,VALUE,QUALITY,MOMENTUM
2013-06-30,0.012,-0.004,0.006
2013-07-31,-0.003,0.008,0.010
```

`features.csv` は月×因子の縦持ちです。以下は列の形だけを示す例であり、この2行だけでは実行できません。

```csv
date,factor,available_at,x_value_spread,x_profitability
2013-06-30,VALUE,2013-06-28 15:00:00,1.25,0.08
2013-06-30,QUALITY,2013-06-28 15:00:00,0.72,0.16
```

| 項目 | 契約 |
|---|---|
| `date` | 時刻なし・タイムゾーンなしの月末日。財務の決算日ではなく、意思決定用スナップショット日 |
| `available_at` | その行に含まれる全特徴量が利用可能となった日時。同じ基準時刻にそろえる |
| 特徴量 | 既定では`x_`列。`--feature-cols`で明示選択も可能 |
| リターン単位 | 小数。1%は`0.01`。因子方向とリスク単位は投入前にそろえる |
| ユニバース | 全月×全因子の固定・バランスパネル。因子数は2本以上 |
| 欠損 | 特徴量のNaNは可。リターンのNaN/inf、特徴量のinfは不可 |
| 月抜け | 不可。行を詰めて「翌月」を別の期間へ置き換えない |
| PIT確認 | `available_at <= dateの終日`を検査。公表原本・改訂履歴を独立監査する機能ではない |

`--allow-unverified-pit`は公表日時の欠損を明示的に許すだけです。既知の未来日時の行を許すオプションではありません。月末の引け後情報を月末価格で取引できるという保証もありません。

`snapshots_from_events()`は、因子単位の**特徴量ベクトル全体を更新するイベント**を月末へas-of結合します。特徴量ごとの部分更新イベントをそのまま渡さず、外部アダプタで完全なスナップショットにしてください。古い情報の将来値によるbackfillは行いません。

### 2.3 入力側が用意するもの／本コードが行わないもの

銘柄段階の財務データ集約、実際のロング／ショート構成銘柄、業種調整、因子の発見・公表時点、個別銘柄の上場廃止、取引可能時刻、改訂履歴、原資産のコストは外部側の責任範囲です。`available_at`を付けただけで、それらの先読みがなくなるわけではありません。

**実装位置:** `build_dataset`（L298–L370） / `snapshots_from_events`（L387–L416）

<a id="03-notation"></a>
## 3. 記号・時点・配列形状

| 記号 | 意味 | コード対応 |
|---|---|---|
| T | 入力の月数 | `len(data.dates)` |
| N | ファクター数（今回の想定38） | `len(data.factors)` |
| P | 1因子あたりの特徴量数 | `len(data.feature_names)` |
| r_{k,t} | 前月末からt月末までの実現リターン | `data.returns[t,k]` |
| x_{k,t} | t月末に利用可能なP次元特性 | `data.features[t,k,:]` |
| n | 当該fitで使う教師ペア数 | `len(x)` |
| K | 保持するリターンPC数 | `params['pcs']` |
| H | PC別または因子別のPLS成分数 | `params['pls_components']` |
| μ̂_{k,t+1\|t} | t時点の翌月予測 | `prediction` |
| δ̂_{k,t} | 過去平均からの予測差 | `timing_prediction` |
| w_t | ファクタースリーブへの非負配分 | `weight` |
| b | 等ウェイト基準、1/N | `base` |
| a_t | 基準からのアクティブ配分 w_t−b | `active_weight` |

時点tの学習集合は、拡大型なら

$$
\mathcal D_t=\{(\mathbf X_s,\mathbf r_{s+1}):s=0,\ldots,t-1\}.
$$

`x = features[start:t]`、`y = returns[start+1:t+1]`、予測用`xnew = features[t]`です。最新の教師ラベルはr_tで、予測対象はr_{t+1}です。ローリング窓では`start=max(0,t-train_window)`になります。

**min_train=60は「60個の教師ペア」**です。最初の意思決定には61か月分のリターン行が存在します。2013-06開始なら最初の意思決定は2018-06末、最初の評価リターンは2018-07です。T=161なら、OOSは161−60−1=100か月です。

```text
学習特性       X_0, X_1, ... , X_(t-1)
対応ラベル     r_1, r_2, ... , r_t       ← t時点で既知
当月特性                          X_t   → 予測 r_(t+1)
外側評価                                      ↑ 後から実現
```

`walk_forward()`は実現値の存在する期間のみ評価し、最終行の特徴量で未実現の翌月を予測するライブ実行は行いません。ライブ予測のAPI例は第17節を参照してください。

<a id="04-preprocessing"></a>
## 4. 共通前処理と予測後処理

### 4.1 特徴量：因子ごとの時間方向に処理する

因子k・特徴量pごとに、**学習窓内だけ**で中央値m、下側分位点l、上側分位点uを求めます。既定のwinsor分位点は1%／99%です。

$$
x^{\mathrm{fill}}_{s,k,p}=
\begin{cases}
x_{s,k,p}&\text{観測済み},\\
m_{k,p}&\text{NaN},
\end{cases}
\qquad
x^c_{s,k,p}=\min(\max(x^{\mathrm{fill}}_{s,k,p},l_{k,p}),u_{k,p}).
$$

$$
z_{s,k,p}=\frac{x^c_{s,k,p}-\bar x^c_{k,p}}{s^x_{k,p}},
\qquad
(s^x_{k,p})^2=\frac1n\sum_s(x^c_{s,k,p}-\bar x^c_{k,p})^2.
$$

非欠損率が20%未満、または処理後標準偏差が`1e-10`以下の列は、そのfit内で0にします。全欠損の中央値は0へ置換します。予測時にも学習済みの中央値・分位点・平均・標準偏差を使い、未来の値を使って列の採否を変更しません。

**横断標準化ではありません。** この処理は「他の因子より割安か」ではなく、「当該因子の通常水準と比べてどうか」を入力する設計です。共有モデルを使うには、同じ特徴量名が各因子で同じ経済的意味・符号を持つ必要があります。

### 4.2 教師リターンの標準化

$$
\bar r_k=\frac1n\sum_s y_{s,k},\qquad
s^y_k=\sqrt{\frac1n\sum_s(y_{s,k}-\bar r_k)^2},\qquad
\widetilde y_{s,k}=\frac{y_{s,k}-\bar r_k}{s^y_k}.
$$

`normalize_target()`は`ddof=0`。ほぼ一定の教師列は尺度1とします。各モデルの予測は最後に元のリターン単位へ戻します。

一方、候補評価・順位配分・予測クリップに使う`train_sigma`は、**ddof=1の標本標準偏差**に下限`1e-6`を置いたものです。両者を同じ変数とみなさないでください。

### 4.3 予測のクリップとタイミング成分

$$
\widehat\mu_k=\operatorname{clip}\left(\widehat\mu^{\rm raw}_k,
\bar r_k-c\widehat\sigma_k,\bar r_k+c\widehat\sigma_k\right),
\qquad c=3.
$$

$$
\widehat\delta_k=\widehat\mu_k-\bar r_k.
$$

`raw_prediction`と`prediction`を両方保存します。候補選択・標準評価・配分に使うのは**クリップ後**の`prediction`です。クリップ回数は診断に残ります。過去平均はその学習窓に対応し、全期間平均ではありません。

**実装位置:** `PanelScaler`（L420–L456） / `normalize_target`（L459–L463） / `forecast_once`（L654–L754）

<a id="05-local"></a>
## 5. M01–M04：基準と個別回帰

### M01 `mean` — 過去平均

$$\widehat\mu_{k,t+1\mid t}=\bar r_{k,t},\qquad\widehat\delta_{k,t}=0.$$

特徴量を使いません。予測誤差評価の基準です。`mean__timing_rank`は等ウェイトと一致します。`mean__level_mv`は更新される過去平均から配分するため、一般に固定配分ではありません。

`mean__timing_mv`もゼロ予測差を入力しますが、`sleeve_total`でドリフトと売買抑制があると、等ウェイトに完全に戻さず取引を抑える場合があります。したがって、全モードで必ず等ウェイトと同じとは限りません。

### M02 `local_ridge` — 因子別Ridge

各因子について、標準化特性と標準化リターンを使って

$$
(\widehat a_k,\widehat{\boldsymbol\beta}_k)
=\arg\min_{a,\boldsymbol\beta}
\frac1n\sum_s(\widetilde y_{s,k}-a-\mathbf z_{s,k}'\boldsymbol\beta)^2
+\lambda\|\boldsymbol\beta\|_2^2
$$

を解きます。切片は罰しません。

$$
\widehat\mu_k=\bar r_k+s^y_k(\widehat a_k+\mathbf z_{\rm new,k}'\widehat{\boldsymbol\beta}_k).
$$

候補は`lam=1.0,0.1`。**候補パラメータはモデル全体で一つ選び、因子別に異なるλを選ぶ実装ではありません。**

### 共通Ridge関数の正規化

非負の観測重みvを正規化してω_s=v_s/Σvとし、加重中心化したZ_c、y_cを用いると

$$
\widehat{\boldsymbol\beta}
=(Z_c'\Omega Z_c+\lambda I)^{-1}Z_c'\Omega y_c,
\qquad \Omega=\operatorname{diag}(\omega_s).
$$

これは重みvを渡すscikit-learnのRidgeで`alpha = lam * sum(v)`とする尺度に対応します。`lam`をそのまま`Ridge(alpha=lam)`に置き換えると、目的関数の縮小の強さが変わります。

### M03 `local_pls_ridge` — 因子内PLS＋Ridge

因子kの学習行列Z_kと教師$\widetilde y_k$から、PLSの投影行列R_kを推定します。

$$
T_k=(Z_k-\bar Z_k)R_k,
\qquad
U_{s,k,h}=\frac{T_{s,k,h}-\bar T_{k,h}}{s^T_{k,h}}.
$$

UをM02の説明変数に置き換えます。PLSは教師との関係を使う圧縮です。`PLSRegression(..., scale=False)`で中心化は行いますが、API内の自動尺度調整は使いません。投影後のscoreを本コードでddof=0により標準化します。[R2]

成分数候補はH=1,2、λ=1固定です。PLSの直接の`predict()`は使わず、**PLSのscoreを別のRidgeへ渡します。** 最大成分数は説明変数の有効列数とn−1でも制限され、ほぼ定数のscoreは除かれます。入力が全定数または教師が定数なら、ゼロscoreへフォールバックします。

PLSの投影は一般に、逐次抽出した重みWと負荷量P_Lを用いて

$$R=W(P_L'W)^{+}$$

と表されます（+は一般化逆行列）。単に最初の生の重みベクトルを全成分へ適用する方法とは異なります。[R2, R7]

### M04 `local_ew_ridge` — 個別の指数減衰Ridge

学習ペアを古い順にs=0,…,n−1とし、

$$v_s=2^{-(n-1-s)/H_{1/2}}$$

を共通Ridgeへ渡します。半減期候補は36か月／12か月、λ=1です。

**前処理・教師の基準平均と尺度は、指数減衰ではなく通常の学習窓統計です。** 回帰の中心化と誤差重みが加重になります。直近を重視する効果と、係数を状態として推定する効果を比較するための対照モデルです。

**実装位置:** `ridge_predict`（L466–L481） / `pls_scores`（L484–L506） / `forecast_once`（L654–L754）

<a id="06-pooled"></a>
## 6. M05–M08：情報共有と非線形化

### M05 `pooled_ridge` — 標準化単位の共通傾き

学習パネルをnN行×P列に並べ、

$$
\min_{a,\boldsymbol\beta}
\frac1{nN}\sum_{k,s}(\widetilde y_{s,k}-a-\mathbf z_{s,k}'\boldsymbol\beta)^2
+\lambda\|\boldsymbol\beta\|_2^2
$$

を解きます。各因子内でZとyは中心化されているため、共通切片は数値誤差を除き0です。元単位では因子別の$\bar r_k,s^y_k$を戻します。

元単位の傾きは概ね$s^y_k\beta_p/s^x_{k,p}$となるので、**全因子で元単位の係数が同じという意味ではありません。** 学習行数は増えますが、同じ月の因子間相関がなくなるわけではありません。

候補はλ=1,0.1。因子IDダミーや因子名を特徴量として入力する実装ではありません。

### M06 `partial_pool` — 共有係数＋因子固有差

$$
\widetilde y_{s,k}=\mathbf z_{s,k}'(\boldsymbol\beta+\mathbf u_k)+\varepsilon_{s,k}.
$$

実装の目的関数は

$$
\mathcal L=
\frac1N\sum_{k=1}^N\frac1n\sum_s
\left[\widetilde y_{s,k}-\mathbf z_{s,k}'(\boldsymbol\beta+\mathbf u_k)\right]^2
+\lambda_b\|\boldsymbol\beta\|_2^2
+\frac{\lambda_u}{N}\sum_k\|\mathbf u_k\|_2^2.
$$

候補は$(\lambda_b,\lambda_u)=(0.1,10),(0.1,1)$。λ_uが大きいほど固有差を抑え、共有モデルへ近づきます。これは**決定論的な罰則付き回帰**で、事後分布をサンプリングする階層ベイズモデルではありません。

ブロック消去の計算は次の通りです。

$$
G_k=Z_k'Z_k/n,\quad h_k=Z_k'\widetilde y_k/n,\quad A_k=G_k+\lambda_uI.
$$

$$
\widehat\beta=
\left[\frac1N\sum_k(G_k-G_kA_k^{-1}G_k)+\lambda_b I\right]^{-1}
\left[\frac1N\sum_k(h_k-G_kA_k^{-1}h_k)\right].
$$

$$\widehat u_k=A_k^{-1}(h_k-G_k\widehat\beta).$$

巨大な因子ダミー行列を明示的に作らず、P次元の線形方程式を中心に計算します。デフォルトの診断出力は共有係数と固有差のRMSのみで、係数全体の保存は未実装です。

### M07 `pooled_interactions` — 2次項・交互作用

$$
\phi(\mathbf z)=
[z_1,\ldots,z_P,z_1^2,z_1z_2,\ldots,z_P^2]'.
$$

列数は

$$D=P+P(P+1)/2=P(P+3)/2.$$

P=6なら27列です。512列を超える設定は停止します。展開後も因子別・学習窓内で中心化／標準化し、nN行にして共有Ridgeを推定します。候補はλ=10,1。

「割安度×収益性改善」などの相互作用を表現できますが、どの組合せが有効かを経済理論から固定しているわけではありません。全2次項を作る実装です。因子横断の相関を無視して独立標本数を数えないでください。

### M08 `pooled_histgb` — 共有型Histogram Gradient Boosting

$$
\widehat f_M(\mathbf z)=c+\eta\sum_{m=1}^{M}h_m(\mathbf z),
\qquad
\widehat\mu_k=\bar r_k+s^y_k\widehat f_M(\mathbf z_{{\rm new},k}).
$$

nN行の標準化パネルを使い、全因子で一つの関数fを共有します。既定の`loss='squared_error'`は内部でhalf squared errorを使います。木の葉の値にはL2正則化が適用されます。[R3]

| パラメータ | 実装値 |
|---|---|
| `max_leaf_nodes` | 候補4または7 |
| `max_iter` | 60 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_samples_leaf` | max(20,3N)：N=38なら114 |
| `l2_regularization` | 10 |
| `early_stopping` | False：内部のランダム検証分割を使わない |
| `random_state` | `config.seed` |

特徴量の中央値補完は共通前処理で済ませるため、木のネイティブ欠損処理を積極的に使う設計ではありません。上記以外のライブラリ既定値は使用バージョンに依存し、`manifest.json`へバージョンを記録します。

**実装位置:** `partial_pool_predict`（L563–L595） / `interaction_map`（L509–L518） / `forecast_once`（L654–L754）

<a id="07-pc"></a>
## 7. M09–M12：PCA／PLSと状態空間

### 7.1 全4モデルに共通する圧縮構造

因子別に標準化した教師リターン行列をỸ（n×N）とします。`PCA(..., svd_solver='full')`により、上位K成分の基底V（N×K）とPCリターンF（n×K）を作ります。

$$F=\widetilde YV.$$

厳密にはPCA内の標本平均も差し引きますが、Ỹは既に列中心化されています。**単位分散化後のPCAなので、原リターンの共分散PCAではなく相関構造のPCA**です。RPPCAは実装されていません。

説明変数は、標準化特性の全因子×全特徴量を並べた

$$\mathbf d_s=\operatorname{vec}(Z_s)\in\mathbb R^{NP}$$

です。各PC jの教師F_jに対して、同じD（n×NP）から別々のPLS写像R_jを推定します。

$$U_j=\operatorname{standardize}\big((D-\bar D)R_j\big).$$

**重要:** 本コードは$V_j'X_s$によるPCポートフォリオ特性を先に作りません。全NP列からPLS圧縮する比較モデルです。この差は、原論文の完全再現と呼べない理由の一つです。[R1]

各PCの予測$\widehat f_j$を集め、

$$
\widehat{\mathbf r}^{\rm raw}
=\bar{\mathbf r}+D_yV\widehat{\mathbf f},
\qquad D_y=\operatorname{diag}(s^y_1,\ldots,s^y_N)
$$

として元のN因子へ戻します。実装は`pca.inverse_transform()`を使うため、PCA内平均の加算も含みます。保持しなかったPC方向の標準化予測は0、つまりその方向は過去平均へ戻す扱いです。Kは候補指定に加えN、n−1で制限されます。

### M09 `pc_pls_ridge`

各PC jに対して

$$\widehat f_j=\widehat a_j+\mathbf u_{{\rm new},j}'\widehat\theta_j$$

をRidgeで推定します。K=3、H=2固定、λ=1,0.1が候補です。PC数やPLS数をJSONのResearchConfigで指定する機能はありません。変更は`model_registry()`の候補定義で行います。

### M10 `pc_pls_ew`

M09と同じPCA／PLS写像の後、PC別回帰だけを指数減衰Ridgeに置き換えます。K=3、H=2、λ=1、半減期36／12か月です。

PCA、PLS、入力の標準化は等重みのままです。前処理まで指数減衰化するモデルではありません。

### M11 `pc_pls_static_kalman`

M09と同じ写像を使い、PCごとの回帰を以下のカルマン更新で解きます。K=3、H=2、λ=1、q=0固定です。**同じλ・同じ学習窓・同じ写像ならM09と同じ点予測**を返す対照実験です。M09が内側検証でλ=0.1を選んだ月に、両モデルが一致する必要はありません。

### M12 `pc_pls_tvp`

K=3、H=2、λ=1固定で、q∈{0,0.0001,0.001}から選びます。同じqを各PCへ適用します。PCごとに別々のqを選ぶ実装ではありません。q=0が選ばれれば、当該時点の点予測はM11と一致します。

### 7.2 カルマン回帰：状態・単位・更新式

各PCのPLS scoreをA、PC教師をvとし、関数内部でさらに学習窓により中心化・ddof=0標準化します。

$$a_s=(u_s-\bar u)/s_u,\quad v_s=(f_s-\bar f)/s_f.$$

状態モデルは

$$v_s=a_s'\beta_s+\epsilon_s,\quad \epsilon_s\sim N(0,1),$$

$$\beta_s=\beta_{s-1}+\eta_s,\quad \eta_s\sim N(0,qI),$$

$$\beta_0=0,\qquad P_0=I/(n\lambda).$$

ここでβは標準化された単位の傾きです。**切片は状態ではなく学習窓の平均として戻します。観測分散1は固定であり、残差分散の最尤推定ではありません。** qは1か月あたりの標準化係数の変動分散です。q=0.0001なら係数イノベーションの標準偏差は0.01です。

古い学習ペアから順に、次を計算します。

$$P_s^-=P_{s-1}+qI,$$
$$S_s=1+a_s'P_s^-a_s,\qquad K_s=P_s^-a_s/S_s,$$
$$e_s=v_s-a_s'\widehat\beta_{s-1},\qquad
\widehat\beta_s=\widehat\beta_{s-1}+K_se_s.$$

Joseph形式の共分散更新は

$$P_s=(I-K_sa_s')P_s^-(I-K_sa_s')'+K_sK_s'.$$

数値誤差対策としてP_sを対称化します。次期の状態平均はランダムウォークのため$\widehat\beta_n$であり、PC予測は

$$\widehat f_{\rm new}=\bar f+s_f\,a_{\rm new}'\widehat\beta_n.$$

将来のqは予測分散へ影響しますが、平均を直接加算するものではありません。本コードは校正済み予測区間を返しません。

### 7.3 q=0とRidgeの等価性

q=0、観測分散1の場合、n件観測後の平均は

$$\widehat\beta_n=(n\lambda I+A'A)^{-1}A'v
=(A'A/n+\lambda I)^{-1}A'v/n.$$

これは

$$\min_\beta\frac1n\|v-A\beta\|_2^2+\lambda\|\beta\|_2^2$$

の解です。scoreが同じ尺度で、λ、中心化、教師が同じなら静的Ridgeと一致します。コードには単体の等価性と、PCA／PLS全体を経由した等価性の両テストがあります。

### 7.4 q>0は「毎月のRidgeを繰り返す」ことではない

q>0では係数経路の変化を許します。初期事前と状態変化に対応する罰則の概念形は

$$n\lambda\|\beta_0\|^2+\sum_s(v_s-a_s'\beta_s)^2+
\frac1q\sum_s\|\beta_s-\beta_{s-1}\|^2.$$

λは初期事前の精度、qは経路の滑らかさを支配します。**毎期β_sをゼロへ戻す平均回帰や、毎期同じL2罰則を追加する実装ではありません。** 「縮小付きTVP」はこの制約付き比較モデルの呼称で、階層ベイズ分散縮小やstochastic volatilityは含みません。

### 7.5 座標と時点の扱い

各外側予測でPCA／PLSを再推定し、その新しい座標で、学習窓の先頭からカルマンフィルタを再実行します。前月の座標のβやPを流用しません。

PLSはその時点の学習ラベル全体から推定されます。そのため、フィルタ内部に現れる学習窓途中の状態は「当時リアルタイムに利用可能だった推定」とは限りません。評価するのは**学習窓末の1歩先予測だけ**です。`beta_last_standardized`の時間変化を、そのまま経済的な係数変化と解釈することも避けてください。

**実装位置:** `kalman_regression_predict`（L521–L560） / `forecast_once`（L654–L754）

<a id="08-validation"></a>
## 8. ネストした時系列検証

### 8.1 外側と内側

外側は毎月再学習するwalk-forward。ハイパーパラメータは既定12か月ごとに再選択し、それ以外の月も選択済みパラメータでモデル・前処理は再推定します。

内側originの集合は、当該学習窓のペア数nに対して

$$v_0=\max(\text{inner\_min\_train},n-\text{inner\_validation\_months}),$$

$$\mathcal V=\{v_0,v_0+d,v_0+2d,\ldots\}\cap[0,n-1],
\quad d=\text{inner\_stride}.$$

各vで`x[:v],y[:v]`を学習し、`x[v]`から`y[v]`を予測します。候補hのスコアは

$$CV(h)=\frac1{|\mathcal V|}\sum_{v\in\mathcal V}\frac1N
\sum_k\left(\frac{y_{v,k}-\widehat\mu^{(h,v)}_k}{\widehat\sigma^{(v)}_k}\right)^2.$$

内側でも標準化・PCA・PLSをfitし直します。全因子を同じ日付で分けます。パラメータは全因子に共通の候補として選びます。

### 8.2 既定設定が意味すること

`min_train=60`、`inner_min_train=36`、`inner_validation_months=12`、`inner_stride=3`なので、最初の内側originは48,51,54,57です。**直近12か月の枠を使っても、既定では4評価時点しかありません。** モデル選択のばらつきを懸念する場合は、候補を固定する、`inner_stride=1`にする、検証窓を変えるなどを事前に定めて比較してください。長い窓が必ず優れるという主張ではありません。

単一候補のモデルはCVを実行せず、スコアをNaNとして固定候補であることを示します。複数候補では最小スコアを採用します。`cv_tolerance>0`なら、最良値の(1+tolerance)以内に入る**registryで先頭の候補**を選びます。これは1標準誤差ルールや有意差検定ではありません。

候補失敗は記録して除外し、全候補が失敗すれば停止します。外側予測が失敗したときに黙って過去平均へ置き換える処理もありません。

**外側OOSを見てモデル・特徴量を選んだ時点で、その期間は最終的な独立holdoutではありません。** データを後から延長しても過去予測が変わらないテストはありますが、研究者の試行錯誤による過学習を防止する機能ではありません。

**実装位置:** `select_candidate`（L758–L807） / `walk_forward`（L810–L847）

<a id="09-allocation"></a>
## 9. 配分の共通定義と共分散

### 9.1 非負のスリーブ配分

$$b=\mathbf1/N,\quad a=w-b,\quad \mathbf1'w=1,\quad 0\le w_k\le c_N.$$

既定の上限は

$$c_N=\min(1,\max(0.10,2/N)).$$

N=38なら1因子あたり10%です。`factor_max_weight`で変更できます。N×上限<1なら実行不能として停止します。

**LSファクターにもw≥0で配分します。** ファクターを逆向きにショートする配分は実装されていません。スリーブ合計1は、原資産のlong-only・市場中立・gross exposure=1を保証しません。

### 9.2 共分散

意思決定時点までの直近`covariance_window=60`か月の原リターンを使い、Ledoit–Wolf縮小共分散を推定します。

$$\widehat\Sigma_{LW}=(1-\zeta)S+\zeta\frac{\operatorname{tr}(S)}N I.$$

ζはライブラリのLedoit–Wolf推定値です。その後、数値安定化として

$$\widehat\Sigma=\tfrac12(\widehat\Sigma_{LW}+\widehat\Sigma_{LW}')+\epsilon I,$$

$$\epsilon=\max\{10^{-8}\operatorname{tr}(\widehat\Sigma_{LW})/N,10^{-12}\}$$

を用います。単位は月次です。[R4]

### 9.3 capped simplexへの射影

$$\Pi_{\mathcal C}(v)=\arg\min_{w\in\mathcal C}\|w-v\|_2^2,$$

$$\mathcal C=\{w:\mathbf1'w=1,0\le w_k\le c_N\}.$$

解は$w_k=\operatorname{clip}(v_k-\tau,0,c_N)$、Σw_k=1を満たすτを二分探索します。負の重みを切って単純に再正規化する方法とは異なります。

**実装位置:** `project_capped_simplex`（L850–L867） / `estimate_covariance`（L870–L876）

<a id="10-strategies"></a>
## 10. S01–S03：シグナルから配分へ

### S01 `timing_rank` — タイミング予測のリスク調整順位

まず

$$s_k=\widehat\delta_k/\max(\widehat\sigma_k,10^{-6})$$

を作り、昇順の平均順位を$\operatorname{rank}(s_k)$とします。

$$q_k=\operatorname{rank}(s_k)-(N+1)/2.$$

全スコア差が1e−12未満ならq=0にします。予備ティルトは

$$d_k=B\frac{q_k}{\max(\sum_j|q_j|,1)},\qquad B=0.25.$$

$$w^{(0)}=\Pi_{\mathcal C}(b+d).$$

基準からの予測年率ボラティリティ

$$TE=\sqrt{12(w^{(0)}-b)'\widehat\Sigma(w^{(0)}-b)}$$

が$\tau_{TE}=0.05$を超えると、

$$w=b+(w^{(0)}-b)\frac{\tau_{TE}}{TE}$$

へ縮めます。超えなければそのままです。

**Bはターンオーバー上限ではありません。** 射影前の等ウェイトからのL1ティルト量です。`prediction_shrinkage`はこのルールに適用されず、目的関数の売買抑制もありません。順位の小さな入れ替わりが売買につながるため、費用控除後の比較が必要です。

### S02 `timing_mv` — タイミング差を使う制約付き最適化

予測信号は

$$g=\rho\widehat\delta,\qquad\rho=0.50.$$

次の目的関数を最小化します。

$$
\min_{w,u}
-g'(w-b)
+\frac\gamma2(w-b)'\widehat\Sigma(w-b)
+\frac\kappa2\|w-b\|_2^2
+c_{opt}\sum_k u_k,
$$

$$\mathbf1'w=1,\quad0\le w_k\le c_N,$$

$$u_k\ge w_k-w_k^{pre},\qquad u_k\ge-(w_k-w_k^{pre}),\qquad u_k\ge0,$$

$$12(w-b)'\widehat\Sigma(w-b)\le\tau_{TE}^2.$$

既定はγ=20、κ=0.01、$c_{opt}=10/10000=0.001$、$\tau_{TE}=0.05$。w_preは取引前の配分です。

これは**等ウェイト対比のアクティブリスクを罰する最適化**で、総リスク$w'\Sigma w$を直接最小化する通常の最小分散配分ではありません。`tilt_l1_budget`もこの最適化には使いません。

コードは数値安定化のため目的関数・勾配に1000を掛けますが、最適解は変わりません。SLSQP、`ftol=1e-9`、`maxiter=300`を用い、失敗・制約違反では停止します。[R5]

### S03 `level_mv` — 予測水準を使う同じ最適化

S02から信号だけを

$$g=\rho\widehat\mu$$

へ置き換えます。制約・共分散・罰則は同じです。

**S02との違いは平均的な因子プレミアムを信号に含めること。** 成績差がタイミング能力によるものか、恒常的に特定因子へ寄せた効果かを、平均ウェイト診断と合わせて確認します。合計配分制約があるため、全因子へ同じ定数を加えても信号の順位付けや線形効用差は変わりません。

### 10.4 制約が全戦略で共通とは限らない

| 条件 | timing_rank | timing_mv / level_mv | equal | inverse_vol | initial_fixed |
|---|---|---|---|---|---|
| 合計1・非負・上限 | ○ | ○ | ○ | ○ | ○ |
| 5%の予測active vol上限 | 毎月縮小 | 毎月制約 | active=0 | **未適用** | **初回のみ** |
| 売買抑制の目的関数 | なし | あり | なし | なし | 初回のみ |
| シグナルの50%縮小 | 未適用 | 適用 | — | — | 初回適用 |
| L1ティルト予算0.25 | 射影前のみ | 未適用 | — | — | 未適用 |

上限やリスク条件が同一と誤認して、平均／ボラティリティ比だけで順位を結論にしないでください。

**実装位置:** `target_weights`（L931–L960） / `solve_active_allocation`（L879–L928）

<a id="11-benchmarks"></a>
## 11. B01–B03：ベンチマーク

### B01 `baseline_equal`

$$w_{k,t}=1/N.$$

全戦略のリターン差・IR・HAC t・ブートストラップの基準です。初期構築コストを課金します。`factor_pnl`なら以後の配分変更はありませんが、`sleeve_total`ではリターンにより保有比率がドリフトするため、等ウェイトへ戻す取引が発生します。

### B02 `baseline_inverse_vol`

$$v_k=\frac{1/\sqrt{\widehat\Sigma_{kk}}}{\sum_j1/\sqrt{\widehat\Sigma_{jj}}},\qquad w=\Pi_{\mathcal C}(v).$$

毎月の共分散対角だけを使います。相関を含むリスク寄与均等化（ERC）ではありません。アクティブリスク上限5%も適用していません。

### B03 `baseline_initial_fixed`

最初の外側意思決定時点t_0で、過去平均をS03と同様に信号とし、

$$w^*=\operatorname{ActiveOptimizer}(\rho\bar r_{t_0},\widehat\Sigma_{t_0},0)$$

を一度計算します。その後の全月で目標ウェイトを$w_t=w^*$に固定します。

**buy-and-holdではなく、固定「目標」配分へのリバランスです。** sleeve_totalでは毎月のドリフトを戻す売買が発生します。初回に5%のリスク上限を満たしても、将来の共分散が変わったときに再制約する処理はありません。

**実装位置:** `simulate_portfolios`（L1078–L1190）

<a id="12-backtest"></a>
## 12. 損益・費用・保有比率ドリフト

### 12.1 `factor_pnl` — LSファクターP&Lの研究用混合

$$R^g_{t+1}=w_t' r_{t+1},\qquad
V_t=\sum_k|w_{k,t}-w_{k,t}^{pre}|,$$

$$c_t=\nu V_t,\qquad R^n_{t+1}=R^g_{t+1}-c_t,$$

$$w_{t+1}^{pre}=w_t.$$

$\nu=\text{cost\_bps}/10000$。既定10bpsです。初回はw_pre=0なので、Σw=1なら初期コストは10bpsです。

**コストに0.5を掛けません。** `half_l1_turnover`は表示用に0.5Vを保存しますが、片道notional費用の計算対象はVです。LSファクター内部の売買、原資産での相殺、借株・調達・税金・流動性は含みません。ファクターP&Lの累積・CAGRは、この混合を月次資本リターンとして扱う研究用プロキシです。

### 12.2 `sleeve_total` — 投資可能なスリーブの総収益率

費用支払後NAVに対する目標ウェイトをw、支払前NAVに対する保有比率をw_preとし、

$$c_t=\nu\sum_k|(1-c_t)w_{k,t}-w_{k,t}^{pre}|$$

を二分探索で解きます。

$$R^n_{t+1}=(1-c_t)(1+w_t'r_{t+1})-1.$$

次月の取引前ウェイトは

$$w_{k,t+1}^{pre}=\frac{w_{k,t}(1+r_{k,t+1})}{1+w_t'r_{t+1}}.$$

費用が全NAVから比例的に支払われるので、このドリフト比率では(1−c_t)が相殺されます。初回w_pre=0なら

$$c_0=\frac\nu{1+\nu}$$

です。単純に$\nu$を引くfactor_pnlとはわずかに異なります。スリーブ内部費用は入力総収益率に含まれる想定で、二重控除に注意してください。

### 12.3 最適化の費用と実現費用は別パラメータ

`turnover_penalty_bps`は配分を決める際のL1ペナルティ、`cost_bps`はバックテスト損益から控除する費用です。既定では両者10bpsですが、自動連動しません。特にsleeve_totalの最適化は$\|w-w^{pre}\|_1$の近似罰則で、上の自己資金式を最適化内部で厳密に扱っていません。

`cost_sensitivity.csv`は同一の目標ウェイトを保持した費用変更です。費用別に再最適化した戦略を比較するには、`cost_bps`と`turnover_penalty_bps`を変更して別実験を実行してください。

**実装位置:** `funded_rebalance_cost`（L963–L981） / `simulate_portfolios`（L1078–L1190）

<a id="13-example"></a>
## 13. 3因子の計算例：同じ予測からティルトと費用を読む

以下は**説明用の仮定値**で、38因子の実証成績ではありません。S01・factor_pnlを使います。

| 因子 | 過去平均 | 翌月予測 | 予測差 | 学習σ | 予測差/σ |
|---|---:|---:|---:|---:|---:|
| A | 0.60% | 0.40% | −0.20% | 1.00% | −0.20 |
| B | 0.30% | 0.30% | 0.00% | 2.00% | 0.00 |
| C | 0.10% | 0.50% | +0.40% | 2.00% | +0.20 |

順位の中心化は(−1,0,1)。B=0.25なので予備ティルトは(−0.125,0,+0.125)。等ウェイトへ加えると

$$w=(0.208333,0.333333,0.458333).$$

3因子で既定上限は2/3なので射影による変更はありません。説明用の共分散を$\Sigma=\operatorname{diag}(0.01^2,0.02^2,0.02^2)$と置くと、予測active volは約0.968%/年で、5%未満です。

取引前が等ウェイトなら売買notionalは0.25、片道費用10bpsなら

$$c=0.001\times0.25=0.00025=2.5\mathrm{bps}.$$

翌月実現リターンが(1.0%,−0.5%,2.0%)なら、grossは約0.9583%、netは約0.9333%です。**これは初期構築ではなく、既に等ウェイトを保有している月の例**です。

<a id="14-metrics"></a>
## 14. シグナル・戦略評価指標

### 14.1 予測の評価

各外側月・因子について、実現r、予測μ̂、その時点の平均r̄、学習σ̂を用います。

$$\mathrm{RMSE}=\sqrt{\operatorname{mean}[(r-\widehat\mu)^2]},\qquad
\mathrm{MAE}=\operatorname{mean}|r-\widehat\mu|,$$

$$R^2_{OOS}=1-\frac{\sum(r-\widehat\mu)^2}{\sum(r-\bar r)^2},$$

$$\mathrm{scaled\ MSE}=\operatorname{mean}\left[((r-\widehat\mu)/\widehat\sigma)^2\right].$$

全体のR²は全月×全因子の二乗誤差を集計した比率であり、因子別R²の単純平均ではありません。分母がほぼ0ならNaNです。

- `sign_accuracy`: sign(r)とsign(μ̂)の一致率。
- `timing_direction_accuracy`: sign(r−r̄)とsign(μ̂−r̄)の一致率。
- `timing_correlation`: r−r̄とμ̂−r̄のPearson相関。全体集計では月×因子をまとめるため、因子間の尺度差にも注意。
- `rank_ic_level`: 各月の因子横断で、μ̂とrのSpearman順位相関。
- `rank_ic_timing`: 各月の因子横断で、μ̂−r̄とr−r̄のSpearman順位相関。

平均モデルのタイミング信号は定数0なので、タイミングIC・相関がNaNになるのは自然です。これは欠損データの不備ではありません。方向一致率では0もそのまま符号として比較するため、平均モデルの値は通常小さくなります。

### 14.2 戦略の評価

M個の月次netリターンRを用いて

$$\mu_{ann}=12\bar R,\quad\sigma_{ann}=\sqrt{12}\,s_R\;(ddof=1),$$

$$\mathrm{mean\_vol\_ratio\_rf0}=\mu_{ann}/\sigma_{ann},$$

$$\mathrm{CAGR}=\exp\left(\frac{12}{M}\sum_t\log(1+R_t)\right)-1.$$

無リスク金利系列を入力しないため、rf0比率は**実際の超過収益Sharpeと同一ではありません**。原資産レバレッジが異なるLS因子P&Lの比較では、資本単位の確認も必要です。

$$W_0=1,\quad W_t=\prod_{s\le t}(1+R_s),\quad
\mathrm{MDD}=\min_t\left(W_t/\max_{u\le t}W_u-1\right).$$

初期NAV=1を最大ドローダウン計算に含めます。月次netが−100%以下なら通常のCAGR計算を停止します。

等ウェイトとの差d_t=R_t−R_{eq,t}について

$$\mathrm{IR}=\frac{12\bar d}{\sqrt{12}\,s_d},\qquad
\mathrm{active\ vol}=\sqrt{12}\,s_d.$$

`annualized_cost_fraction`と`annualized_traded_notional`は月平均の12倍です。累積費用そのものではありません。

### 14.3 HAC t統計

中心化したz_t=d_t−d̄と、L=min(hac_lags,M−1)を用い、

$$\widehat\gamma_\ell=\frac1M\sum_{t=\ell+1}^{M}z_tz_{t-\ell},$$

$$\widehat\Omega=\widehat\gamma_0+2\sum_{\ell=1}^{L}
\left(1-\frac\ell{L+1}\right)\widehat\gamma_\ell,
\qquad t_{HAC}=\frac{\bar d}{\sqrt{\widehat\Omega/M}}.$$

既定L=3。多重検定調整済みではありません。小標本の精度や、研究者によるモデル選択を補正するものでもありません。

**実装位置:** `signal_metrics`（L998–L1032） / `performance_metrics`（L1051–L1075） / `hac_t_stat_mean`（L1035–L1048）

<a id="15-inference"></a>
## 15. 不確実性と事後診断

### 15.1 paired circular moving-block bootstrap

戦略ごとの等ウェイト差を月で整列し、全戦略に**同じ月ブロック**を適用します。既定のブロック長は6か月、再標本化回数500。各開始点を一様に抽出し、末尾を越えたら先頭へ循環します。抽出系列を元のOOS月数に切り詰め、12×平均差の2.5%／97.5%分位点を出力します。

これは**固定されたOOS損益経路を条件とする区間**です。モデルの再学習、前処理再推定、ハイパーパラメータ選択、特徴量探索、多重検定を再現しません。500回を精密な推論の普遍的な推奨値と扱わず、研究目的に応じて設計してください。

### 15.2 事後平均ウェイト診断

全OOSを使って$\bar w=M^{-1}\sum_t w_t$を計算し、

$$D=12\operatorname{mean}_t[(w_t-\bar w)'r_{t+1}]$$

を出力します。平均的な配分の違いと、配分変化のgross効果を分ける記述的な診断です。

**全OOS平均ウェイトは将来情報を使うため、投資可能なOOSベンチマークではありません。** grossだけの分解で、コスト差も含みません。独立した証拠やタイミング能力の因果的な証明として扱わないでください。

**実装位置:** `block_bootstrap_comparison`（L1193–L1218） / `mean_weight_diagnostic`（L1221–L1237）

<a id="16-research"></a>
## 16. 比較実験の推奨設計と限界

以下は実装済み成績ではなく、実データへ接続した後の実験案です。

| 問い | 比較 | そろえる条件 |
|---|---|---|
| 縮小した線形関係で十分か | mean / local_ridge | 同じデータ・OOS・費用 |
| 因子間共有に価値があるか | local_ridge / pooled_ridge / partial_pool | 特徴量の意味・標準化・候補探索量 |
| 教師あり圧縮に価値があるか | local_ridge / local_pls_ridge | 学習窓とλの扱い |
| 非線形化が必要か | pooled_ridge / pooled_interactions / pooled_histgb | 同じ外側月、探索履歴を記録 |
| 時変係数が必要か | M09のλ=1固定 / M11 / M12 | PCA=3、PLS=2、λ=1、同じ窓 |
| 直近重視だけで十分か | M09 / M10 / M12 | 前処理の等重みは共通 |
| 平均プレミアム選好か、タイミングか | level_mv / timing_mv | 予測・最適化パラメータ・費用を共通 |

M09は既定でλを選ぶ一方、M12はλ=1固定です。**既定値のままでは「時変性だけ」の厳密なアブレーションになりません。** λを固定した比較を追加する場合は、候補を一つに制限し、その変更を記録してください。

短いデータでは、nN行を独立したnN時点とみなさないこと、4点しかない既定の内側検証を認識すること、外側OOSを見た選択を最終評価と混同しないことが特に重要です。2013年以前の状態・レジームに対する一般化を、このデータだけで保証できません。

研究ログには、使用期間、特徴量の定義、因子の採用理由、公表時点、全候補、期間分割、費用・リスク制約、失敗した試行も保存してください。本コードの`manifest.json`だけで過去の試行を自動的に履歴管理するわけではありません。

### 現行版に含まれないもの

RPPCA、Lasso、Autoencoder、IPCA、銘柄パネル学習、レジームスイッチング、階層ベイズの事後分布、確率的ボラティリティ、多期間ラベルのpurge、原資産の売買最適化、無リスク金利・実資本レバレッジの整備、独立holdoutロック、多重検定補正、モデルオブジェクトの永続保存は未実装です。

<a id="17-handoff"></a>
## 17. 出力・API・他のChatGPTへの引継ぎ

### 17.1 出力一覧

| ファイル | 用途／注意 |
|---|---|
| `predictions.csv` | decision/target日、model/factor、raw/clip予測、過去平均、timing差、学習σ、実現値 |
| `model_selection.csv` | 候補params、内側誤差、採用フラグ、検証日、失敗理由 |
| `model_diagnostics.csv` | 学習期間、PCA説明率、最終Kalman状態等。保存内容はモデルごとに異なる |
| `signal_summary.csv` | モデル全体の予測指標 |
| `signal_by_factor.csv` | 因子ごとの予測指標 |
| `signal_monthly_ic.csv` | 月ごとの横断IC |
| `signal_correlations.csv` | timing予測のモデル間Pearson相関。月×因子を並べたもの |
| `strategy_returns.csv` | gross/net、費用、売買notional、予測総／active vol、最適化診断 |
| `weights.csv` | 各月各戦略の目標・取引前・activeウェイト |
| `strategy_summary.csv` | 戦略全期間の成績と等ウェイト差 |
| `cost_sensitivity.csv` | 同じポジションで費用を変えた成績 |
| `bootstrap_vs_equal.csv` | 条件付きブロック再標本化区間 |
| `expost_mean_weight_diagnostic.csv` | 非投資可能な事後平均ウェイト診断 |
| `manifest.json` | 設定、候補、期間、環境、コード・入力ハッシュ、限界 |
| `REPORT.md` | 自動生成される結果要約。本READMEとは別物 |

CSVはUTF-8 BOM付き。モデル診断の一部セルはJSON文字列です。DataFrameを再読込するときは`decision_date`,`target_date`を日時型へ戻してください。

### 17.2 Pythonから入力を接続する

```python
from pathlib import Path
import pandas as pd
from factor_tilt_research import build_dataset, ResearchConfig, run_experiment

returns = pd.read_csv('returns.csv')
features = pd.read_csv('features.csv')
data = build_dataset(returns, features)  # PIT日時を確認してから投入
config = ResearchConfig(min_train=60, bootstrap_samples=500)
result = run_experiment(data, config, models='all', out=Path('results_api'))
print(result.signal_summary)
```

### 17.3 予測を再学習せず配分を比較する

```python
from dataclasses import replace
import pandas as pd
from factor_tilt_research import load_csv, ResearchConfig, simulate_portfolios

data = load_csv('returns.csv', 'features.csv')
pred = pd.read_csv(
    'results/predictions.csv',
    parse_dates=['decision_date', 'target_date'],
)
# min_train/train_windowは予測生成時と同じ設定を使う。
# 実務ではresults/manifest.jsonから元設定を復元し、配分関係だけ変更する。
config = ResearchConfig(min_train=60)
new_config = replace(config, cost_bps=25.0, turnover_penalty_bps=25.0)
monthly, weights, summary, sensitivity = simulate_portfolios(
    data, pred, new_config, rules=('timing_rank', 'timing_mv', 'level_mv')
)
```

予測の日付と入力のOOSが一致しないと停止します。過去の予測を別のリターン定義・因子順序・特徴量セットへ黙って流用しないでください。元設定はmanifestから復元してください。

### 17.4 最新月から翌月を予測する（評価ではない）

```python
from factor_tilt_research import load_csv, ResearchConfig, forecast_once

data = load_csv('returns.csv', 'features.csv')
config = ResearchConfig()
# この例は過去データだけで事前に決めた固定paramsを使う。
params = {'pcs': 3, 'pls_components': 2, 'lam': 1.0, 'q': 0.0001}
forecast = forecast_once(
    'pc_pls_tvp', params,
    data.features[:-1], data.returns[1:], data.features[-1], config,
)
print(dict(zip(data.factors, forecast.prediction)))
```

この呼出しは翌月の予測値だけを出します。q=0.0001はコード例の仮設定で、採用を推奨するものではありません。実行前に十分な学習件数と現在月の公表可能性を確認し、必要なら`select_candidate()`で最新時点以前のデータだけを使って選択してください。将来リターンは不要ですが、自動発注・将来の評価結果出力は行いません。

### 17.5 他のChatGPTへ渡す作業指示

以下は引継ぎ時にそのまま使える指示です。

> `factor_tilt_research.py`と本README、PowerPointを確認し、既存38ファクターのデータへ接続してください。最初に入力の単位・因子方向・利用可能日時・改訂履歴を監査してください。コードのSECTIONに沿って分割し、予測生成と配分比較を分離してください。現行版は相関PCA＋全因子特徴量のflatten＋PC別PLSであり、原論文のPCウェイトによる特性集約やRPPCA/Lassoとは異なることを保持してください。前処理は因子別の時間標準化です。q=0の等価性、未来データ変更に対する過去予測の不変性、配分制約、費用式のテストを維持してください。新モデル追加時は候補数と検証設計を記録し、外側OOSを同じ実験のチューニングへ戻さないでください。未実装の拡張は、既存結果と区別したバージョンで作成してください。

### 17.6 分割先と拡張時の変更点

| SECTION | 推奨モジュール | 主な内容 |
|---|---|---|
| 00 | `config.py` | dataclass、入力・結果契約 |
| 01 | `data.py` | CSV、PIT日時検査、as-ofアダプタ |
| 02 | `preprocessing.py` | PanelScaler、Ridge、PLS、交互作用 |
| 03 | `models.py` | registry、forecast_once、部分共有、カルマン |
| 04 | `validation.py` | 内側選択、外側walk-forward |
| 05 | `allocation.py` | 共分散、simplex、最適化、費用関数 |
| 06 | `backtest.py` | 損益、指標、ブートストラップ、診断 |
| 07 | `reporting.py` | public API、manifest、CSV／Markdown |
| 08 | `demo.py`, `tests/` | 合成データ、22テスト |
| 09 | `cli.py` | argparse、main |

新モデルはregistryだけでなく`forecast_once()`のdispatchも更新します。新配分ルールは`target_weights()`に加え、`run_experiment()`の許可リストとCLI・文書を更新します。現在の候補λ／q／成分数はResearchConfigではなくregistryにあります。モデル保存や説明用の全係数書出しを加える場合、毎回変わるPCA／PLS座標との対応も保存してください。

<a id="18-audit"></a>
## 18. 設定・候補・コード位置・検証記録

### 18.1 ResearchConfigの既定値

| 設定 | 既定値 | 意味 |
|---|---|---|
| `min_train` | `60` | 最初の学習教師ペア数 |
| `train_window` | `None` | Noneなら拡大型。整数なら外側の最大学習ペア数 |
| `inner_min_train` | `36` | 内側の最低学習ペア数 |
| `inner_validation_months` | `12` | 内側検証の期間枠 |
| `inner_stride` | `3` | 内側評価originの間隔（月） |
| `tune_every` | `12` | ハイパーパラメータの再選択間隔（月） |
| `cv_tolerance` | `0.0` | 最良score近傍で先頭候補を選ぶ許容比率 |
| `winsor_quantile` | `0.01` | 下側／上側winsor分位点 |
| `min_feature_coverage` | `0.2` | 因子×特性の最低非欠損比率 |
| `prediction_clip_sigma` | `3.0` | 過去平均±この値×学習標本σで予測clip |
| `covariance_window` | `60` | 共分散の過去月数 |
| `factor_max_weight` | `None` | Noneならmin(1,max(0.10,2/N)) |
| `tilt_l1_budget` | `0.25` | 順位ルールの射影前L1ティルト量 |
| `active_risk_aversion` | `20.0` | 最適化のactive分散罰則γ |
| `anchor_penalty` | `0.01` | active配分L2罰則κ |
| `max_active_vol_annual` | `0.05` | 順位・最適化の年率active vol上限 |
| `prediction_shrinkage` | `0.5` | MV系の信号倍率ρ |
| `turnover_penalty_bps` | `10.0` | 最適化の売買抑制罰則。実現費用と別 |
| `cost_bps` | `10.0` | 損益控除する片道notional費用 |
| `cost_scenarios_bps` | `(0.0, 10.0, 25.0)` | 固定ポジションの費用感応度 |
| `return_kind` | `factor_pnl` | factor_pnlまたはsleeve_total |
| `bootstrap_samples` | `500` | 条件付き再標本化回数。0で無効 |
| `bootstrap_block` | `6` | ブロック月数 |
| `hac_lags` | `3` | HACのラグ数 |
| `seed` | `42` | 合成データ・木・ブートストラップ用 |
| `threads` | `1` | threadpoolctl制限 |

CLIで直接指定できるのは一部だけです。全設定は`--config config.json`で渡せます。JSONと同じ項目のCLIフラグがあればCLIが優先されます。未知のJSONキーは停止します。以下は例です。

```json
{
  "min_train": 60,
  "inner_min_train": 36,
  "inner_validation_months": 12,
  "inner_stride": 1,
  "tune_every": 12,
  "cost_bps": 10.0,
  "turnover_penalty_bps": 10.0,
  "return_kind": "factor_pnl"
}
```

### 18.2 registryの実際の候補

**`mean`**

```json
[
  {}
]
```

**`local_ridge`**

```json
[
  {
    "lam": 1.0
  },
  {
    "lam": 0.1
  }
]
```

**`local_pls_ridge`**

```json
[
  {
    "lam": 1.0,
    "pls_components": 1
  },
  {
    "lam": 1.0,
    "pls_components": 2
  }
]
```

**`local_ew_ridge`**

```json
[
  {
    "lam": 1.0,
    "half_life": 36.0
  },
  {
    "lam": 1.0,
    "half_life": 12.0
  }
]
```

**`pooled_ridge`**

```json
[
  {
    "lam": 1.0
  },
  {
    "lam": 0.1
  }
]
```

**`partial_pool`**

```json
[
  {
    "lam_shared": 0.1,
    "lam_deviation": 10.0
  },
  {
    "lam_shared": 0.1,
    "lam_deviation": 1.0
  }
]
```

**`pooled_interactions`**

```json
[
  {
    "lam": 10.0
  },
  {
    "lam": 1.0
  }
]
```

**`pooled_histgb`**

```json
[
  {
    "max_leaf_nodes": 4,
    "max_iter": 60
  },
  {
    "max_leaf_nodes": 7,
    "max_iter": 60
  }
]
```

**`pc_pls_ridge`**

```json
[
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0
  },
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 0.1
  }
]
```

**`pc_pls_ew`**

```json
[
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "half_life": 36.0
  },
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "half_life": 12.0
  }
]
```

**`pc_pls_static_kalman`**

```json
[
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "q": 0.0
  }
]
```

**`pc_pls_tvp`**

```json
[
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "q": 0.0
  },
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "q": 0.0001
  },
  {
    "pcs": 3,
    "pls_components": 2,
    "lam": 1.0,
    "q": 0.001
  }
]
```

### 18.3 コード索引

行番号は先頭のSHA-256に対応する実ファイルの物理行番号です。コード編集後は関数名を基準に追跡してください。

| 名前 | ソース行 |
|---|---|
| `ResearchConfig` | 145–212 |
| `ResearchData` | 216–227 |
| `ModelSpec` | 231–234 |
| `ForecastResult` | 238–243 |
| `ExperimentResult` | 247–261 |
| `jsonable` | 264–281 |
| `jdump` | 284–285 |
| `_month_ends` | 289–295 |
| `build_dataset` | 298–370 |
| `load_csv` | 373–384 |
| `snapshots_from_events` | 387–416 |
| `PanelScaler` | 420–456 |
| `normalize_target` | 459–463 |
| `ridge_predict` | 466–481 |
| `pls_scores` | 484–506 |
| `interaction_map` | 509–518 |
| `kalman_regression_predict` | 521–560 |
| `partial_pool_predict` | 563–595 |
| `model_registry` | 598–631 |
| `resolve_models` | 634–651 |
| `forecast_once` | 654–754 |
| `select_candidate` | 758–807 |
| `walk_forward` | 810–847 |
| `project_capped_simplex` | 850–867 |
| `estimate_covariance` | 870–876 |
| `solve_active_allocation` | 879–928 |
| `target_weights` | 931–960 |
| `funded_rebalance_cost` | 963–981 |
| `safe_corr` | 985–995 |
| `signal_metrics` | 998–1032 |
| `hac_t_stat_mean` | 1035–1048 |
| `performance_metrics` | 1051–1075 |
| `simulate_portfolios` | 1078–1190 |
| `block_bootstrap_comparison` | 1193–1218 |
| `mean_weight_diagnostic` | 1221–1237 |
| `_markdown_table` | 1240–1250 |
| `build_manifest` | 1253–1285 |
| `export_results` | 1288–1352 |
| `prepare_output_directory` | 1355–1360 |
| `run_experiment` | 1363–1391 |
| `make_synthetic` | 1395–1435 |
| `self_test` | 1438–1679 |
| `write_template` | 1683–1700 |
| `config_from_args` | 1703–1718 |
| `make_parser` | 1721–1755 |
| `main` | 1758–1800 |

### 18.4 この資料作成時の確認

添付の元コードを変更せず、`python factor_tilt_research.py self-test`を再実行しました。**22/22通過**。ログは同梱`verification.txt`に記録しています。ここでのテストは動作・数理・時点整合性の確認であり、実データの予測力の証明ではありません。

確認環境：Python 3.13.5 / numpy 2.3.5 / pandas 2.2.3 / scipy 1.17.0 / scikit-learn 1.8.0 / threadpoolctl 3.6.0。

重要テストは、q=0とRidgeの一致、部分共有と密行列解の一致、強い固有差罰則で共有回帰へ近づくこと、全12モデルの有限予測、prefix不変性、未来変更に対する過去予測不変性、初期費用、スリーブのドリフト、制約付き最適化、CSV・manifest出力です。

数式の照合には現行ソースを使い、Ridgeのλ尺度、標準偏差のddof、PCAの単位、qの尺度、コストの片道notionalを明記しました。資料内の説明用数値は実ファクターの結果ではありません。

<a id="19-references"></a>
## 19. 原論文との差分と参考資料

### 19.1 「参考にした研究」と「この実装」を区別する

Kagkadis, Nolte, Nolte & Vasilas, “Factor Timing with Portfolio Characteristics,” *The Review of Asset Pricing Studies*, 14(1), 84–118 (2024), DOI: `10.1093/rapstu/raad010`。[R1]

| 項目 | この実装 |
|---|---|
| 銘柄特性→因子特性 | 外部入力。構成銘柄からの集約は未実装 |
| 入力特性の標準化 | 因子別・時間方向。原論文の横断標準化をそのまま再現しない |
| リターン圧縮 | 標準化教師に対する相関PCA。RPPCAなし |
| PC予測用の特性 | 全N×P列をflattenしてPC別PLS。PCウェイトによる特性集約なし |
| 回帰 | Ridge、共有モデル、制約付きTVP等。Lassoなし |
| 配分 | 非負スリーブ＋上限＋active risk。原論文の全投資戦略を再現しない |
| 目的 | 短い月次履歴での拡張比較・引継ぎ基盤 |

上記の相違を無視して、原論文と成績だけを比較しないでください。

### 19.2 参照元

外部資料は背景・APIの確認用です。本READMEの実装上の仕様は、SHA-256を固定した同梱ソースを一次資料としています。閲覧確認日：2026-11-05。

- **[R1]** 原論文：Kagkadis et al.（2024）、DOI `10.1093/rapstu/raad010`。
- **[R2]** scikit-learn 1.8 `PLSRegression`：中心化、scale、score、rotationの定義。
- **[R3]** scikit-learn 1.8 `HistGradientBoostingRegressor`：損失、葉数、L2、early stopping。
- **[R4]** scikit-learn `LedoitWolf`／Covariance estimation：縮小共分散の定義。
- **[R5]** SciPy `minimize(method='SLSQP')`：制約付き最適化のAPI。
- **[R6]** statsmodels State Space Methods：フィルタと状態空間の背景。本コードはstatsmodelsを依存関係として使わず、カルマン更新を直接実装。
- **[R7]** scikit-learn `_pls.py`：rotation行列の実装確認。

```text
R1 https://academic.oup.com/raps/article/14/1/84/7191017
R2 https://scikit-learn.org/1.8/modules/generated/sklearn.cross_decomposition.PLSRegression.html
R3 https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
R4 https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
R5 https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
R6 https://www.statsmodels.org/stable/statespace.html
R7 https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cross_decomposition/_pls.py
```

---

**このREADMEとPowerPointは仕様説明です。実データでの優位性や、実取引への即時利用可能性を確認したレポートではありません。**
