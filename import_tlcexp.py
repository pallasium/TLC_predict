"""TLCexp (Zenodo) の rf_dataset_0203_total.csv を、このツールの形式 (tlc.csv と同じ列) に変換する。

使い方: python import_tlcexp.py rf_dataset_0203_total.csv [出力 data/tlcexp_clean.csv]

- 3 成分の溶媒・Rf が (0, 1) の外の行は除く。
- 溶媒は solvent_ref.csv にあるものだけ使う。Petroleum ether / Isohexane / Heptane は Hexane として扱う (近似)。
  Water, Cyclohexane, MTBE, Triethylamine, 酸類などは表にないので、その行ごと除く。
- Compound_ID は RDKit の正規化 SMILES。State には元データの label (standard / non-standard) を入れる。
データのライセンスは、配布元 (Zenodo) のページで確認すること。リポジトリには入れない (data/ は .gitignore 済み)。
"""
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
MAP = {'Ethyl acetate': 'AcOEt', 'Petroleum ether': 'Hexane', 'Hexane': 'Hexane', 'Isohexane': 'Hexane', 'Heptane': 'Hexane',
       'Dichloromethane': 'CH2Cl2', 'Methanol': 'MeOH', 'Diethyl ether': 'Et2O', 'Pentane': 'Pentane', 'Acetone': 'Acetone',
       'i-PrOH': 'iPrOH', 'Chloroform': 'CHCl3', 'Toluene': 'Toluene', 'Acetonitrile': 'MeCN', 'Ethanol': 'EtOH', 'THF': 'THF'}


def convert(src, dst):
    df = pd.read_csv(src, encoding='utf-8-sig')
    n0 = len(df)
    df = df[df.Solvent_3.isna() & (df.Rf_value > 0) & (df.Rf_value < 1)]
    print(f'3 成分・Rf が範囲外の行を除く: {n0} -> {len(df)}')
    single = df.Solvent_2.isna()
    a, b = df.Solvent_1.map(MAP), df.Solvent_2.map(MAP)
    ok = a.notna() & (single | b.notna())
    df, a, b, single = df[ok], a[ok], b[ok].fillna(a[ok]), single[ok]
    print(f'溶媒がすべて solvent_ref.csv にある行: {len(df)}')
    r1 = np.where(single, 1.0, df.Fraction_1); r2 = np.where(single, 1.0, df.Fraction_2)
    same = (a == b).values                     # Hexane/Petroleum ether のように同じ溶媒同士は単一溶媒
    r1[same] = 1.0; r2[same] = 1.0
    can = {s: (Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) else None) for s in df.SMILES.unique()}
    cid = df.SMILES.map(can)
    out = pd.DataFrame({'Compound_ID': cid.values, 'SMILES': df.SMILES.values, 'Solvent_1': a.values, 'Solvent_2': b.values,
                        'Ratio_1': r1, 'Ratio_2': r2, 'Rf': df.Rf_value.values, 'State': df.label.values, 'Source': df.doi.values})
    out = out[out.Compound_ID.notna()]
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.to_csv(dst, index=False)
    print(f'保存: {dst}  {len(out)} 行 / {out.Compound_ID.nunique()} 化合物')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'tlcexp_clean.csv'))
