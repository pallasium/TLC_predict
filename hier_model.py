"""階層(線形)モデル: logit(Rf) = g(分子特徴量) + 溶媒系ごとの傾き * 極性変数
溶媒効果を物理的な1変数(極性溶媒の体積分率の対数など)に押し込み、分子特徴量は切片だけを説明させる。
LOCO-CVで比較。使い方: python hier_model.py [追加特徴量csv ...]
"""
import sys, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from cv_baseline import load

POLAR = {'AcOEt': 'AcOEt', 'MeOH': 'MeOH'}


def add_solvent_vars(df):
    # 極性溶媒(AcOEt/MeOH)の体積分率（比を体積比とみなす）
    pol1 = df['Solvent_1'].isin(POLAR)
    fpol = np.where(pol1, df['f1'], 1 - df['f1'])
    df['fpol'] = fpol
    df['logfpol'] = np.log(fpol)
    df['sys_meoh'] = df['is_meth']
    df['u_hex'] = np.where(df.is_meth == 0, df['logfpol'], 0.0)   # Hexane/AcOEt 系の傾き用
    df['u_chl'] = np.where(df.is_meth == 1, df['logfpol'], 0.0)   # CHCl3/MeOH 系の傾き用
    df['u_hex_lin'] = np.where(df.is_meth == 0, fpol, 0.0)
    df['u_chl_lin'] = np.where(df.is_meth == 1, fpol, 0.0)
    return df


def loco(df, feats, alphas=np.logspace(-2, 3, 30)):
    X, y, g = df[feats].fillna(0).values, df['logit_Rf'].values, df['Compound_ID'].values
    pred = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    p = 1 / (1 + np.exp(-pred))
    e = p - df['Rf'].values
    return np.abs(e).mean(), np.sqrt((e ** 2).mean()), p


if __name__ == '__main__':
    df = add_solvent_vars(load(sys.argv[1:]))
    print(f'n={len(df)} ベースライン(平均) MAE={np.abs(df.Rf - df.Rf.mean()).mean():.3f}')
    solv_sets = {
        '系ダミーのみ': ['sys_meoh'],
        '系+log(fpol)共通傾き': ['sys_meoh', 'logfpol'],
        '系+log(fpol)系別傾き': ['sys_meoh', 'u_hex', 'u_chl'],
        '系+fpol系別傾き': ['sys_meoh', 'u_hex_lin', 'u_chl_lin'],
    }
    mol_sets = {
        '(分子なし)': [],
        'TPSA': ['TPSA'],
        'TPSA+logP': ['TPSA', 'MolLogP'],
        'TPSA+logP+HBD': ['TPSA', 'MolLogP', 'NumHDonors'],
        'TPSA+logP+芳香環': ['TPSA', 'MolLogP', 'NumAromaticRings'],
    }
    xt = [c for c in df.columns if c.startswith('xtb_')]
    if xt:
        mol_sets['TPSA+logP+xtbGsolv差'] = ['TPSA', 'MolLogP', 'xtb_Gsolv_water_minus_hexane']
        mol_sets['xtb全部'] = xt
    rows = []
    for (sn, sf), (mn, mf) in itertools.product(solv_sets.items(), mol_sets.items()):
        mae, rmse, _ = loco(df, sf + mf)
        rows.append((mae, rmse, sn, mn))
    for mae, rmse, sn, mn in sorted(rows)[:15]:
        print(f'MAE={mae:.3f} RMSE={rmse:.3f}  溶媒:{sn:22s} 分子:{mn}')
