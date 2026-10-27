# TLC Rf 予測

薄層クロマトグラフィー (TLC) の Rf を、溶媒の組成から予測する小さなツール。
Snyder の溶離力 ε° と混合則を使った線形階層モデル:

`logit(Rf) = a + c·[強溶媒の種類] + b·ε_mix + u_化合物`

化合物ごとのオフセット u は ridge で縮小する。測定のある化合物と未知の化合物で、正則化の強さを自動で切り替える。

## データについて
`tlc.csv` は動作確認用のダミーデータ (12 化合物 / 36 行)。市販化合物の SMILES を使っているが、Rf は実測ではなく、傾向だけ似せて作った架空の値。
自分の測定値を使うときは、`tlc.csv` を同じ形式で置き換える。`tlc.csv` を更新すると、次の実行時に自動で学習し直される (`fit` を手で実行する必要はない)。

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

### TLC を 1〜2 枚流して補正する
学習データにない化合物では、まず SMILES が必須 (`--smiles "..."` か対話入力。空だとエラー。RDKit があれば構文も検証する)。続けて `rf` / `suggest` で、TLC を流した結果を聞かれる。`溶媒 比=Rf` の形で最大 2 枚入力すると、その化合物のオフセット u を実測から推定し、Snyder の溶媒パラメータ (ε_mix) による予測に反映する。Enter だけでスキップ (溶媒のみの予測)。
```
TLC を流した結果があれば入力してください (...)
  1 枚目: Hexane/AcOEt 5:1=0.15
  2 枚目: Hexane/AcOEt 2:1=0.4
```
別の溶媒系の結果でも使える。単一溶媒は `Acetone=0.6`。

### 数字だけで使う (対話モード)
引数なしで `python tlc_snyder.py` を実行すると対話モードになる。実験室で名前を打たなくて済むよう、数字で選べる。
- 化合物: 登録済みは番号。未知の化合物だけ SMILES を入力
- 溶媒: 番号 2 つ (弱い方 強い方) で混合、1 つで単一溶媒。例 `1 5` = Hexane/AcOEt。Enter で前回と同じ溶媒
- 比: `3 1` のように数字を空白区切りで入力
- TLC の実測: `5 1 0.15` (比1 比2 Rf)。予測する溶媒系と同じ系で流した結果として扱う

### 構造から予測する (公開データ TLCexp で事前学習)
未知の化合物は、SMILES から RDKit の記述子を作って予測することもできる。公開データ (TLCexp, Zenodo: https://zenodo.org/records/18467289) を入手して一度だけ学習する。ライセンスは配布元で確認すること。
```
pip install -r requirements.txt
python import_tlcexp.py rf_dataset_0203_total.csv    # data/tlcexp_clean.csv を作る
python tlc_snyder.py pretrain                        # 事前学習 (4 コアで約 30 秒。2 回目以降は記述子のキャッシュで数秒)
```
学習済みモデルがあると、未知の化合物は構造モデルで予測し、TLC の実測 (1〜2 枚) があればそのずれを補正する。なければ従来どおり溶媒のみの予測。
- 溶媒の効き (ε_mix の傾き) は、同じ化合物を溶媒比を変えて測ったデータから線形に推定する。公開データ全体で学習すると「Rf が 0.3〜0.5 になるように溶媒比を選んだ」という実験の都合が混ざり、溶媒比を変えても予測が動かなくなるため。
- 精度は過信しない。別の論文の化合物を予測する検証では MAE≈0.19 で、平均を答える予測 (0.17) と同程度だった。構造モデルは実測の補正の土台であり、単独では当てにならない。
- `data/` `models/` `user_data/` は `.gitignore` 済み。

## ファイル
| ファイル | 内容 |
|---|---|
| `tlc_snyder.py` | 予測ツール本体 |
| `snyder_model.py` | Snyder ε° と混合則のモデル (α の感度分析つき) |
| `hier_model.py` | 系ダミー + 極性溶媒分率の線形モデル |
| `hansen_model.py` | Hansen / Hildebrand を溶媒変数にした比較 |
| `cv_baseline.py` | Leave-One-Compound-Out CV と RF / Ridge / PLS の比較 |
| `parametercalc.py` | RDKit による分子特徴量の計算 |
| `solvent_ref.csv` | 溶媒の物性表 (17 溶媒: Hexane, Toluene, CH2Cl2, CHCl3, AcOEt, MeOH, Acetone, THF, MeCN, Et2O ほか) |

`hansen_model.py` / `cv_baseline.py` は `solvent.csv` (溶媒ごとの Hansen パラメータ) も読む。自分で用意する。

## 溶媒物性について
`solvent_ref.csv` は、手元の溶媒物性表 (xlsx) から作った。他の溶媒を使うときは、表に載っている値を 1 行ずつ追加する。

列: `Solvent,eps0_SiO2,eps0_approx,TLC_strength,Hildebrand_dT,MolarVolume,dD_ref,dP_ref,dH_ref`
- `eps0_SiO2` (Snyder 溶離力、シリカゲル基準。そのまま ε° として使う) と `MolarVolume` (cm3/mol) は必須。`eps0_approx=1` は元の表で概算値 (*) の溶媒。`TLC_strength` は表の別の溶媒強度の列 (現在は未使用)。他の列は `hansen_model.py` などの比較用。
- 値の出典: 手元の溶媒物性表 (ε° = Snyder 溶出力、Hildebrand、Hansen、密度)。`MolarVolume` は表の密度と分子量 (表にないので別途与えた) から `分子量 / 密度` で計算した。DMF は表に ε° がないので入れていない。

## 注意
- 学習データにない溶媒系への予測は外挿で、精度は未検証。
- データ量が少ない場合、CV の誤差は大きくばらつく。

## AI の利用について
このリポジトリのコードと README の作成には、AI (Claude / Claude Code) を支援ツールとして利用した。
内容の確認と公開の判断は作者が行っている。

## ライセンス
MIT License (`LICENSE` 参照)。ライセンスの対象はこのリポジトリのコードで、引用した文献の値や利用する外部ソフトウェアには及ばない。
