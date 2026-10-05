"""溶媒側を Hansen パラメータ(δD, δP, δH; 混合は体積分率の加重平均)で表した線形モデルの LOCO-CV。
 logit(Rf) = 切片(分子特徴量) + Σ 溶媒Hansen項 [+ 分子×溶媒の交互作用]
使い方: python hansen_model.py [追加特徴量csv ...]
"""
import sys, itertools
import numpy as np
import pandas as pd
from cv_baseline import load
from hier_model import add_solvent_vars, loco


def add_hansen_vars(df):
    df = add_solvent_vars(df)
    df['mix_dD'] = df['mix_dDispersion']; df['mix_dP'] = df['mix_dPolar']; df['mix_dH'] = df['mix_dHydrogenBonding']
    df['mix_dPH'] = np.sqrt(df.mix_dP ** 2 + df.mix_dH ** 2)             # 極性+水素結合の合成
    df['mix_dTot'] = np.sqrt(df.mix_dD ** 2 + df.mix_dP ** 2 + df.mix_dH ** 2)
    # 系別傾き (系ごとにδの効き方が違う場合)
    for c in ('dP', 'dH', 'dPH'):
        df[f'{c}_hex'] = np.where(df.is_meth == 0, df[f'mix_{c}'], 0.0)
        df[f'{c}_chl'] = np.where(df.is_meth == 1, df[f'mix_{c}'], 0.0)
    # 分子の極性 × 溶媒の水素結合/極性 (吸着と溶媒競合の釣り合い)
    df['TPSA_x_dH'] = df.TPSA / 100 * df.mix_dH
    df['TPSA_x_dP'] = df.TPSA / 100 * df.mix_dP
    df['HBD_x_dH'] = df.NumHDonors * df.mix_dH
    return df


if __name__ == '__main__':
    df = add_hansen_vars(load(sys.argv[1:]))
    print(f'n={len(df)} ベースライン(平均) MAE={np.abs(df.Rf - df.Rf.mean()).mean():.3f}')
    solv = {
        '[参照] 系+log(fpol)': ['sys_meoh', 'logfpol'],
        'δP': ['sys_meoh', 'mix_dP'],
        'δH': ['sys_meoh', 'mix_dH'],
        'δP+δH': ['sys_meoh', 'mix_dP', 'mix_dH'],
        'δD+δP+δH': ['sys_meoh', 'mix_dD', 'mix_dP', 'mix_dH'],
        '√(δP²+δH²)': ['sys_meoh', 'mix_dPH'],
        'δtotal': ['sys_meoh', 'mix_dTot'],
        'δP+δH (系ダミーなし)': ['mix_dP', 'mix_dH'],
        'δD+δP+δH (系ダミーなし)': ['mix_dD', 'mix_dP', 'mix_dH'],
        'δP系別+δH系別': ['sys_meoh', 'dP_hex', 'dP_chl', 'dH_hex', 'dH_chl'],
        'δPH系別': ['sys_meoh', 'dPH_hex', 'dPH_chl'],
        'ET30': ['sys_meoh', 'mix_ET30'],
    }
    mol = {
        '(分子なし)': [],
        'TPSA': ['TPSA'],
        'TPSA+logP': ['TPSA', 'MolLogP'],
        'TPSA×δH': ['TPSA', 'TPSA_x_dH'],
        'TPSA×δP': ['TPSA', 'TPSA_x_dP'],
        'TPSA+HBD×δH': ['TPSA', 'NumHDonors', 'HBD_x_dH'],
    }
    ads = [c for c in df.columns if c.startswith('xtb_Eads')]
    if ads:
        df['Eads_x_dH'] = df['xtb_Eads_min'] * df['mix_dH'] / 100
        mol['xtb Eads_min'] = ['xtb_Eads_min']
        mol['xtb Eads_min×δH'] = ['xtb_Eads_min', 'Eads_x_dH']
    rows = []
    for (sn, sf), (mn, mf) in itertools.product(solv.items(), mol.items()):
        mae, rmse, _ = loco(df, sf + mf)
        rows.append((mae, rmse, sn, mn))
    rows.sort()
    for mae, rmse, sn, mn in rows[:20]:
        print(f'MAE={mae:.3f} RMSE={rmse:.3f}  溶媒:{sn:24s} 分子:{mn}')
    print('--- 分子なし のみ(溶媒表現の比較) ---')
    for mae, rmse, sn, mn in sorted(r for r in rows if r[3] == '(分子なし)'):
        print(f'MAE={mae:.3f} RMSE={rmse:.3f}  溶媒:{sn}')
