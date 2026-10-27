"""Snyder の溶離力 ε° (solvent_ref.csv: シリカゲル基準) と混合則を使ったモデルの LOCO-CV。
 二成分系の溶媒強度 (Snyder-Soczewinski):
   ε_AB = ε_A + log10( N_B * 10^(α n_b (ε_B - ε_A)) + (1 - N_B) ) / (α n_b)
   A=弱溶媒(Hexane/CHCl3), B=強溶媒(AcOEt/MeOH), N_B=B のモル分率(体積比とモル体積から)
   n_b = 吸着サイトを占める B 分子の面積。分子体積から n_b = 6.0*(V/89.4)^(2/3) (ベンゼン=6.0 基準)で近似。
 α: 吸着剤の活性度(シリカ ~0.57。ここでは感度分析)。ε° は solvent_ref.csv のシリカ基準の値をそのまま使う (scale=1.0)。
使い方: python snyder_model.py [追加特徴量csv ...]
"""
import os, sys
import numpy as np
import pandas as pd

REF = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solvent_ref.csv'), index_col=0)
_EPS0, _VOL = REF['eps0_SiO2'], REF['MolarVolume']


def snyder_eps(df, alpha, scale=1.0):
    """溶媒の強弱は ε° の大小で自動判定 (弱=A, 強=B)。df は Solvent_1/2, Ratio_1/2 (体積比) を持つ"""
    e1 = df.Solvent_1.map(_EPS0).values; e2 = df.Solvent_2.map(_EPS0).values
    if np.isnan(e1).any() or np.isnan(e2).any():
        bad = sorted(set(df.Solvent_1[np.isnan(e1)]) | set(df.Solvent_2[np.isnan(e2)]))
        raise KeyError(f'solvent_ref.csv にない溶媒: {bad}')
    s1_strong = e1 >= e2
    weak = np.where(s1_strong, df.Solvent_2, df.Solvent_1); strong = np.where(s1_strong, df.Solvent_1, df.Solvent_2)
    vol_w = np.where(s1_strong, df.Ratio_2, df.Ratio_1); vol_s = np.where(s1_strong, df.Ratio_1, df.Ratio_2)
    Vw = pd.Series(weak).map(_VOL).values; Vs = pd.Series(strong).map(_VOL).values
    nw, ns = vol_w / Vw, vol_s / Vs
    NB = ns / (ns + nw)
    eA = pd.Series(weak).map(_EPS0).values * scale
    eB = pd.Series(strong).map(_EPS0).values * scale
    nb = 6.0 * (Vs / 89.4) ** (2 / 3)
    k = alpha * nb
    return eA + np.log10(NB * 10 ** (k * (eB - eA)) + (1 - NB)) / k, NB


if __name__ == '__main__':
    from cv_baseline import load          # sklearn を読み込むので、解析として実行するときだけ import する
    from hier_model import add_solvent_vars, loco
    df = add_solvent_vars(load(sys.argv[1:]))
    S = 'sys_meoh'
    print(f'n={len(df)} ベースライン(平均) MAE={np.abs(df.Rf - df.Rf.mean()).mean():.3f}   [参照 系+log(fpol)]'
          f' MAE={loco(df, [S, "logfpol"])[0]:.3f}')
    for a in (0.3, 0.57, 0.8, 1.2):
        df['eps_mix'], df['NB'] = snyder_eps(df, a)
        for name, f in {'ε_mix のみ': ['eps_mix'], 'ε_mix + 系ダミー': [S, 'eps_mix'],
                        'ε_mix + 系ダミー + log(fpol)': [S, 'eps_mix', 'logfpol']}.items():
            m, r, _ = loco(df, f)
            print(f'alpha={a:4.2f} MAE={m:.3f} RMSE={r:.3f}  {name}')
    # 純粋な溶媒強度 (混合則なしの線形平均 / ET_N / Hildebrand 平均)
    f1 = df.f1
    ev = lambda col: f1 * df.Solvent_1.map(REF[col]) + (1 - f1) * df.Solvent_2.map(REF[col])
    df['eps_lin'], df['dT_mix'] = ev('eps0_SiO2'), ev('Hildebrand_dT')
    for col in ('eps_lin', 'dT_mix'):
        for name, f in {col: [col], col + '+系': [S, col]}.items():
            m, r, _ = loco(df, f); print(f'線形平均 MAE={m:.3f} RMSE={r:.3f}  {name}')
