# リリースノート

## v0.1.0 — 初版

薄層クロマトグラフィー (TLC) の Rf を、溶媒の組成から予測する小さなツール。Snyder の溶離力 ε° と混合則を使った線形階層モデル `logit(Rf) = a + c·[強溶媒の種類] + b·ε_mix + u_化合物` (化合物ごとのオフセットは ridge で縮小)。

### 機能
- `tlc_snyder.py`: `solvents` (溶媒の一覧) / `fit` (学習) / `eval` (交差検証) / `rf` (Rf の予測) / `suggest` (目標 Rf になる溶媒比の提案)。
- 比較用のモデル: 系ダミー + 極性溶媒分率 (`hier_model.py`)、Hansen / Hildebrand (`hansen_model.py`)、RF / Ridge / PLS と Leave-One-Compound-Out CV (`cv_baseline.py`)。
- `parametercalc.py`: RDKit による分子特徴量。
- `solvent_ref.csv`: 代表的な 6 溶媒 (Hexane, Toluene, CH2Cl2, CHCl3, AcOEt, MeOH) の物性表。

### 注意
- 同梱の `tlc.csv` は動作確認用のダミーデータ (12 化合物 / 36 行)。Rf は実測ではない。
- 学習データにない溶媒系への予測は外挿で、精度は未検証。データが少ないと CV の誤差は大きくばらつく。
- `hansen_model.py` / `cv_baseline.py` は、自分で用意する `solvent.csv` (Hansen パラメータ) も読む。
