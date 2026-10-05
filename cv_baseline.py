"""化合物単位 Leave-One-Compound-Out CV で、特徴量セット/モデルの本当の汎化精度を比較する。
同じ化合物の別の溶媒比が学習側に入らないようにする（train3.py の検証は学習データ再予測で楽観的）。
使い方: python cv_baseline.py [追加特徴量csv ...]   (Compound_ID をキーに結合される)
"""
import sys
# Windows コンソール (cp932) で ≈ や ² などが表示できず UnicodeEncodeError になるのを防ぐ
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'): _s.reconfigure(errors='replace')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

SOL_PARAMS = ['ET30', 'dDispersion', 'dPolar', 'dHydrogenBonding', 'RelativePermittivity']


def load(extra_csvs=()):
    sol = pd.read_csv('solvent.csv', index_col=0)
    sol.index = sol.index.str.strip()
    df = pd.read_csv('tlc.csv').merge(pd.read_csv('molecular_features.csv'), on='Compound_ID')
    for f in extra_csvs:
        df = df.merge(pd.read_csv(f), on='Compound_ID', how='left')
    f1 = df['Ratio_1'] / (df['Ratio_1'] + df['Ratio_2'])
    df['f1'] = f1
    for p in SOL_PARAMS:
        df[f'mix_{p}'] = f1 * df['Solvent_1'].map(sol[p]) + (1 - f1) * df['Solvent_2'].map(sol[p])
    df['is_meth'] = ((df['Solvent_1'] == 'MeOH') | (df['Solvent_2'] == 'MeOH')).astype(int)
    df['logit_Rf'] = np.log(df['Rf'].clip(0.01, 0.99) / (1 - df['Rf'].clip(0.01, 0.99)))
    return df


def loco(df, feats, model, target='Rf'):
    X, y, g = df[feats].fillna(0).values, df[target].values, df['Compound_ID'].values
    pred = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = model()
        m.fit(X[tr], y[tr])
        pred[te] = np.ravel(m.predict(X[te]))
    if target == 'logit_Rf':
        pred = 1 / (1 + np.exp(-pred))
    err = pred - df['Rf'].values
    return np.abs(err).mean(), np.sqrt((err ** 2).mean()), pred


MODELS = {
    'RF(train3相当)': lambda: make_pipeline(StandardScaler(), RandomForestRegressor(
        n_estimators=500, bootstrap=False, random_state=0)),
    'RF(浅め)': lambda: make_pipeline(StandardScaler(), RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0)),
    'Ridge': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
    'PLS(2)': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2)),
}

if __name__ == '__main__':
    df = load(sys.argv[1:])
    mol = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'FractionCSP3']
    sets = {
        'train3相当(mol+溶媒)': mol + ['mix_dDispersion', 'mix_dPolar', 'mix_dHydrogenBonding',
                                      'mix_RelativePermittivity', 'is_meth'],
        'コンパクト(TPSA,logP,δP,δH,f1)': ['TPSA', 'MolLogP', 'mix_dPolar', 'mix_dHydrogenBonding', 'is_meth'],
    }
    extra_cols = [c for c in df.columns if c.startswith('xtb_')]
    if extra_cols:
        sets['コンパクト+xtb全部'] = sets['コンパクト(TPSA,logP,δP,δH,f1)'] + extra_cols
    print(f'n={len(df)}  化合物数={df.Compound_ID.nunique()}  ベースライン(平均予測) MAE='
          f'{np.abs(df.Rf - df.Rf.mean()).mean():.3f}')
    for sname, feats in sets.items():
        for mname, mk in MODELS.items():
            for tgt in ('Rf', 'logit_Rf'):
                mae, rmse, _ = loco(df, feats, mk, tgt)
                print(f'{sname:32s} {mname:16s} {tgt:9s} MAE={mae:.3f} RMSE={rmse:.3f}')
