# TLC Rf 予測

薄層クロマトグラフィー (TLC) の Rf を、溶媒の組成から予測する小さなツール。
Snyder の溶離力 ε° と混合則を使った線形階層モデル:

`logit(Rf) = a + c·[強溶媒の種類] + b·ε_mix + u_化合物`

化合物ごとのオフセット u は ridge で縮小する。測定のある化合物と未知の化合物で、正則化の強さを自動で切り替える。

## データについて
`tlc.csv` は動作確認用のダミーデータ (12 化合物 / 36 行)。市販化合物の SMILES を使っているが、Rf は実測ではなく、傾向だけ似せて作った架空の値。
自分の測定値を使うときは、`tlc.csv` を同じ形式で置き換える。

列: `Compound_ID,SMILES,Solvent_1,Solvent_2,Ratio_1,Ratio_2,Rf,State` (Solvent_1 が弱溶媒、Ratio は体積比)

`molecular_features.csv` は `parametercalc.py` が `tlc.csv` から作る (RDKit 必要)。

## 使い方
```
python tlc_snyder.py solvents                                       # 使える溶媒の一覧
python tlc_snyder.py fit                                            # tlc.csv から学習
python tlc_snyder.py eval                                           # 交差検証
python tlc_snyder.py rf      --id Anisole --solvent Hexane/AcOEt --ratio 3:1
python tlc_snyder.py suggest --id Anisole --solvent Hexane/AcOEt --target 0.3
```
`--solvent` は 弱溶媒/強溶媒 の形で指定する (強弱は ε° で自動判定)。

## ファイル
| ファイル | 内容 |
|---|---|
| `tlc_snyder.py` | 予測ツール本体 |
| `snyder_model.py` | Snyder ε° と混合則のモデル (α の感度分析つき) |
| `hier_model.py` | 系ダミー + 極性溶媒分率の線形モデル |
| `hansen_model.py` | Hansen / Hildebrand を溶媒変数にした比較 |
| `cv_baseline.py` | Leave-One-Compound-Out CV と RF / Ridge / PLS の比較 |
| `parametercalc.py` | RDKit による分子特徴量の計算 |
| `solvent_ref.csv` | 代表的な 6 溶媒の物性表 (Hexane, Toluene, CH2Cl2, CHCl3, AcOEt, MeOH) |

`hansen_model.py` / `cv_baseline.py` は `solvent.csv` (溶媒ごとの Hansen パラメータ) も読む。自分で用意する。

## 溶媒物性について
`solvent_ref.csv` には代表的な 6 溶媒だけを入れている。他の溶媒を使うときは、自分で文献から値を調べて 1 行ずつ追加する。

列: `Solvent,eps0_Al2O3,ETN,Hildebrand_dT,MolarVolume,dD_ref,dP_ref,dH_ref`
- `eps0_Al2O3` (Snyder 溶離力) と `MolarVolume` (cm3/mol) は必須。他の列は `hansen_model.py` などの比較用。
- 値の出典: Reichardt, *Solvents and Solvent Effects in Organic Chemistry* (ε°, ET^N)、*Polymer Handbook* Ch. 16 (Hildebrand, Hansen, 分子体積)。各文献の著作権は原著者・出版社に帰属する。

## 注意
- 学習データにない溶媒系への予測は外挿で、精度は未検証。
- データ量が少ない場合、CV の誤差は大きくばらつく。

## AI の利用について
このリポジトリのコードと README の作成には、AI (Claude / Claude Code) を支援ツールとして利用した。
内容の確認と公開の判断は作者が行っている。

## ライセンス
MIT License (`LICENSE` 参照)。ライセンスの対象はこのリポジトリのコードで、引用した文献の値や利用する外部ソフトウェアには及ばない。
