"""構造 (RDKit 記述子) + 溶媒 (Snyder ε_mix など) から logit(Rf) を予測する事前学習モデル。tlc_snyder.py と tlc_ft.py が共有する。

公開データ (TLCexp) を import_tlcexp.py で変換して data/tlcexp_clean.csv に置き、`python tlc_snyder.py pretrain` で学習する。
学習済みモデルは models/pretrained.joblib (pickle 形式なので、他人から受け取ったものは開かないこと)、
記述子は models/desc_cache.pkl に保存される。
"""
import os, sys, time
import numpy as np
import pandas as pd
from snyder_model import snyder_eps, REF

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, 'data', 'tlcexp_clean.csv')
PRE = os.path.join(HERE, 'models', 'pretrained.joblib')
ALPHA = 0.57
ALC = ('MeOH', 'EtOH', 'iPrOH')
COLS = ['Compound_ID', 'SMILES', 'Solvent_1', 'Solvent_2', 'Ratio_1', 'Ratio_2', 'Rf']


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def die(msg):
    print('エラー: ' + msg); raise SystemExit(1)


# ---------- 特徴量 ----------
def rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
    except ImportError:
        die('RDKit が必要です (pip install rdkit)')
    RDLogger.DisableLog('rdApp.*')
    return Chem, Descriptors


def canon(smi):
    Chem, _ = rdkit(); m = Chem.MolFromSmiles(smi)
    if m is None: die(f'SMILES を解釈できません: {smi!r}')
    return Chem.MolToSmiles(m)


def n_cpu():
    """使えるコア数を自動検出 (コンテナなどで CPU が制限されているときは、その制限に従う)"""
    try: return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count() or 1


def _desc_one(smi):
    """1 分子の記述子 (ワーカープロセスで動く。トップレベルに置く必要がある)。塩は最大のフラグメント (対イオンを除いた本体)"""
    Chem, Descriptors = rdkit()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
    v = []
    for _, f in Descriptors.descList:
        try: v.append(f(m))
        except Exception: v.append(np.nan)
    return v


DESC_CACHE = os.path.join(HERE, 'models', 'desc_cache.pkl')
_DESC = None
def _cache_key():
    import rdkit
    return (rdkit.__version__, len(rdkit_desc_list()))


def rdkit_desc_list():
    return rdkit()[1].descList


def descriptors(smiles):
    """SMILES のリスト -> 記述子の行列。全コアで並列計算し、計算済みの分は models/desc_cache.pkl から読む (2 回目以降は速い)"""
    global _DESC
    import pickle
    if _DESC is None:
        _DESC = {}
        if os.path.exists(DESC_CACHE):
            try:
                c = pickle.load(open(DESC_CACHE, 'rb'))
                if c['key'] == _cache_key(): _DESC = c['data']
            except Exception:
                pass
    todo = sorted(set(smiles) - set(_DESC))
    if todo:
        n = n_cpu()
        if len(todo) >= 300 and n > 1:
            from concurrent.futures import ProcessPoolExecutor
            print(f'記述子を計算中: {len(todo)} 分子 (コア数 {n} で並列)')
            with ProcessPoolExecutor(n) as ex:
                vals = list(ex.map(_desc_one, todo, chunksize=max(1, len(todo) // (n * 8))))
        else:
            vals = [_desc_one(s) for s in todo]
        for s, v in zip(todo, vals):
            if v is None: die(f'SMILES を解釈できません: {s!r}')
            _DESC[s] = v
        if len(todo) >= 300:
            os.makedirs(os.path.dirname(DESC_CACHE), exist_ok=True)
            pickle.dump({'key': _cache_key(), 'data': _DESC}, open(DESC_CACHE, 'wb'))
    return np.array([_DESC[s] for s in smiles], dtype=float).clip(-1e6, 1e6)


def solv_feats(df):
    eps, NB = snyder_eps(df, ALPHA)
    A, B = df.Solvent_1, df.Solvent_2
    col = lambda c, s: s.map(REF[c]).values
    alc = (A.isin(ALC) | B.isin(ALC)).astype(float).values
    return np.column_stack([eps, NB, alc, col('eps0_SiO2', A), col('eps0_SiO2', B), col('dP_ref', A), col('dP_ref', B),
                            col('dH_ref', A), col('dH_ref', B)]), eps


def read_data(path, what):
    if not os.path.exists(path): die(f'{what}がありません: {path}')
    df = pd.read_csv(path)
    miss = [c for c in COLS if c not in df.columns]
    if miss: die(f'{path} に列がありません: {miss}  (必要な列: {COLS} と任意の State)')
    bad = ~(df.Solvent_1.isin(REF.index) & df.Solvent_2.isin(REF.index))
    if bad.any():
        print(f'注意: solvent_ref.csv にない溶媒の行を {bad.sum()} 行除きます: ' + ', '.join(sorted((set(df.Solvent_1[bad]) | set(df.Solvent_2[bad])) - set(REF.index))))
        df = df[~bad]
    df = df[(df.Rf > 0) & (df.Rf <= 1)].reset_index(drop=True)
    df['State'] = df['State'].fillna('NA').astype(str) if 'State' in df.columns else 'NA'
    return df


# ---------- 事前学習 ----------
# 溶媒の強さ (ε_mix) の効きは、同じ化合物を溶媒比を変えて測ったデータ (化合物内の変動) から線形に推定する。
# 公開データ全体で ε_mix と Rf の関係を学習すると、「Rf が 0.3〜0.5 になるように溶媒比を選んだ」という実験の都合が混ざり、
# 溶媒比を変えても予測が動かなくなる。そこで、構造モデル (勾配ブースティング) には溶媒の強さ (ε_mix, 混合比) を入れず、
# 化合物ごとの切片だけを学習させる:  logit(Rf) = c·[アルコール系] + b·ε_mix + g(構造, 溶媒の種類)
GBM_COLS = slice(2, None)          # solv_feats の [ε_mix, NB] を除いた列 + 記述子


def fit_slopes(F, y, cid, lam=0.1):
    """化合物内の変動 (縮小つき within 変換) から [アルコール系の補正 c, ε_mix の傾き b] を推定"""
    Z = np.column_stack([np.ones(len(y)), F[:, 2], F[:, 0]])
    g = pd.Series(cid); cnt = g.map(g.value_counts()).values; w = cnt / (cnt + lam)
    mean = lambda a: pd.DataFrame(a).groupby(g.values).transform('mean').values
    Zt = Z - w[:, None] * mean(Z); yt = y - w * mean(y[:, None])[:, 0]; Zt[:, 0] = 1
    th = np.linalg.lstsq(Zt, yt, rcond=None)[0]
    return float(th[1]), float(th[2])


def pretrain(data=PUB):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    df = read_data(data, '公開データ (先に import_tlcexp.py を実行)')
    t = time.time()
    F = solv_feats(df)[0]; y = logit(df.Rf.values)
    c, b = fit_slopes(F, y, df.Compound_ID.astype(str).values)
    X = np.hstack([F, descriptors(df.SMILES.tolist())])[:, GBM_COLS]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(X, y - c * F[:, 2] - b * F[:, 0])
    os.makedirs(os.path.dirname(PRE), exist_ok=True)
    joblib.dump({'gbm': gbm, 'c': c, 'b': b, 'n_rows': len(df)}, PRE)
    print(f'事前学習 {len(df)} 行 / {df.Compound_ID.nunique()} 化合物 ({time.time() - t:.0f}s)  ε_mix の傾き b={b:.2f}, アルコール系 c={c:+.2f} -> {PRE}')


def load_pretrained(required=True):
    """学習済みモデルを返す。required=False でモデルがない / 読めないときは None"""
    if not os.path.exists(PRE):
        if required: die('事前学習モデルがありません。先に `python tlc_snyder.py pretrain` を実行してください')
        return None
    try:
        import joblib
        m = joblib.load(PRE)
        if 'b' not in m: raise ValueError('古い形式です')
        return m
    except Exception as e:
        if required: die(f'事前学習モデルを読めません ({e})。`pretrain` をやり直してください')
        return None


def base_z(model, df):
    """事前学習モデルの予測 (logit) と ε_mix。df は SMILES と Solvent_1/2, Ratio_1/2 を持つ"""
    F, eps = solv_feats(df)
    X = np.hstack([F, descriptors(df.SMILES.tolist())])[:, GBM_COLS]
    return model['gbm'].predict(X) + model['c'] * F[:, 2] + model['b'] * F[:, 0], eps
