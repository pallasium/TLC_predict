"""Snyder の溶離力 ε_mix を使った TLC Rf 予測モデル(線形・階層)。

 logit(Rf) = a + c·[強溶媒=MeOH] + b·ε_mix + u_化合物      (u は縮小(ridge)した化合物ごとのオフセット)
  - ε_mix : Snyder-Soczewinski 混合則 (snyder_model.snyder_eps, α=0.57, シリカ=0.8×アルミナ)
  - u_化合物: 学習データに測定値がある化合物だけ学習される。未知化合物は u=0 (= 溶媒だけで予測)

分子特徴量(RDKit 記述子など)は LOCO-CV で改善しなかったため入れていない (README 参照)。

使い方:
  python tlc_snyder.py eval                                   # CV 評価
  python tlc_snyder.py fit                                    # 学習して tlc_snyder_model.json を保存
  python tlc_snyder.py rf      --id <ID> --solvent Hexane/AcOEt --ratio 3:1
  python tlc_snyder.py suggest --id <ID> --solvent Hexane/AcOEt --target 0.3
  python tlc_snyder.py suggest --smiles "N[C@@H](C)C(=O)OC" --solvent CHCl3/MeOH --target 0.3   # 未知化合物
  (--solvent は 溶媒A/溶媒B。強弱は ε° で自動判定。使える溶媒は solvent_ref.csv。学習データにない溶媒系は外挿で未検証)
"""
import argparse, json, os, re, sys
# Windows コンソール (cp932) で ≈ や ² などが表示できず UnicodeEncodeError になるのを防ぐ
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'): _s.reconfigure(errors='replace')
import numpy as np
import pandas as pd
from snyder_model import snyder_eps

MODEL = 'tlc_snyder_model.json'
ALPHA = 0.57
LAM_KNOWN, LAM_UNKNOWN = 0.1, 100.0
ALCOHOLS = ('MeOH', 'EtOH', 'iPrOH', 'nPrOH')
TRAINED_PAIRS = ({'Hexane', 'AcOEt'}, {'CHCl3', 'MeOH'})   # 学習データにある溶媒系
EPS0 = pd.read_csv('solvent_ref.csv', index_col=0)['eps0_Al2O3']


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def design(df, comps):
    """df: Solvent_1,Solvent_2,Ratio_1,Ratio_2,Compound_ID を持つ表 -> (X, 溶媒部分の列数)"""
    d = df.copy()
    d['is_meth'] = (d.Solvent_1.isin(ALCOHOLS) | d.Solvent_2.isin(ALCOHOLS)).astype(int)   # アルコール系 (学習データでは MeOH のみ)
    eps, _ = snyder_eps(d, ALPHA)
    X = np.column_stack([np.ones(len(d)), d.is_meth.values, eps])
    U = np.zeros((len(d), len(comps)))
    for j, c in enumerate(comps):
        U[:, j] = (d.Compound_ID.values == c)
    return X, U


def fit(df, lam):
    comps = sorted(df.Compound_ID.unique())
    X, U = design(df, comps)
    A = np.hstack([X, U])
    pen = np.diag([0, 0, 0] + [lam] * len(comps))
    beta = np.linalg.solve(A.T @ A + pen + 1e-9 * np.eye(A.shape[1]), A.T @ logit(df.Rf.values))
    return {'lam': lam, 'alpha': ALPHA, 'fixed': beta[:3].tolist(), 'offset': dict(zip(comps, beta[3:].tolist()))}


def predict_logit(m, df):
    """m は fit() の結果、または {'known': fit(λ小), 'unknown': fit(λ大)} の 2 組"""
    if 'known' in m:
        isk = df.Compound_ID.isin(m['known']['offset'].keys()).values
        out = np.empty(len(df))
        if isk.any(): out[isk] = predict_logit(m['known'], df[isk])
        if (~isk).any(): out[~isk] = predict_logit(m['unknown'], df[~isk])
        return out
    X, _ = design(df, [])
    off = df.Compound_ID.map(m['offset']).fillna(0.0).values
    return X @ np.array(m['fixed']) + off


def sigmoid(z): return 1 / (1 + np.exp(-z))


def load():
    return pd.read_csv('tlc.csv')


def cv(df, lam):
    # (A) LOCO: 未知化合物 (u=0)   (B) LOMO: 同じ化合物の他の測定値があるとき
    pa = np.zeros(len(df)); pb = np.zeros(len(df))
    for c in df.Compound_ID.unique():
        te = df.Compound_ID == c
        pa[te.values] = sigmoid(predict_logit(fit(df[~te], lam), df[te]))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        pb[i] = sigmoid(predict_logit(fit(df[tr], lam), df.iloc[[i]]))[0]
    return pa, pb


def cmd_eval(_):
    df = load()
    multi = df.groupby('Compound_ID').Rf.transform('size') > 1
    base = np.abs(df.Rf - df.Rf.mean()).mean()
    print(f'n={len(df)} 化合物={df.Compound_ID.nunique()} 複数測定の化合物の行={multi.sum()}  平均予測 MAE={base:.3f}')
    best = None
    for lam in (0.03, 0.1, 0.3, 1, 3, 10, 100):
        pa, pb = cv(df, lam)
        ea, eb = np.abs(pa - df.Rf), np.abs(pb - df.Rf)
        print(f'lam={lam:6.2f}  (A)未知化合物 MAE={ea.mean():.3f} RMSE={np.sqrt((ea**2).mean()):.3f}   '
              f'(B)既知化合物の別条件 MAE(全行)={eb.mean():.3f}  MAE(複数測定行のみ)={eb[multi].mean():.3f}')
        if best is None or eb[multi].mean() < best[0]: best = (eb[multi].mean(), lam)
    print(f'-> 既知化合物向け lam={LAM_KNOWN}, 未知化合物向け lam={LAM_UNKNOWN} の 2 組を運用 (下記)')
    pred = np.zeros(len(df)); kind = []
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        mm = {'known': fit(df[tr], LAM_KNOWN), 'unknown': fit(df[tr], LAM_UNKNOWN)}
        pred[i] = sigmoid(predict_logit(mm, df.iloc[[i]]))[0]
    e = np.abs(pred - df.Rf.values)
    print(f'運用時の誤差 (1 行ずつ除いて予測): 全行 MAE={e.mean():.3f} RMSE={np.sqrt((e**2).mean()):.3f}   '
          f'同じ化合物の他の測定がある行 MAE={e[multi.values].mean():.3f}   他の測定がない行 MAE={e[~multi.values].mean():.3f}   [平均予測 {base:.3f}]')
    print(f'  (注: 「他の測定がない行」は未知化合物の予測 = 溶媒だけの予測。ただし化合物単位ではなく行単位の除外なので (A) より少し楽観的)')


def cmd_fit(a):
    df = load()
    m = {'known': fit(df, LAM_KNOWN), 'unknown': fit(df, LAM_UNKNOWN)}
    json.dump(m, open(MODEL, 'w'), indent=1)
    for k in m: print(f'saved {MODEL} [{k}] lam={m[k]["lam"]} 固定項(切片, MeOH系, ε_mix)=', np.round(m[k]['fixed'], 3))


def split_solvent(text):
    """'A/B' -> (A, B)、'A' (単一溶媒) -> (A, A)"""
    parts = [x.strip() for x in text.split('/') if x.strip()]
    if len(parts) not in (1, 2):
        print(f'エラー: 溶媒は 「A/B」(混合) か 「A」(単一) の形式で指定してください (入力: {text!r})'); raise SystemExit(1)
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def parse_ratio(text):
    nums = [x for x in re.split(r'[:/,\s]+', text.strip()) if x]
    try:
        r = [float(x) for x in nums]
    except ValueError:
        r = []
    if len(r) != 2 or min(r) < 0 or sum(r) <= 0:
        print(f'エラー: 比は 「3:1」 のように 2 つの数で指定してください (入力: {text!r})'); raise SystemExit(1)
    return r[0], r[1]


def rows(a, ratios):
    s1, s2 = split_solvent(a.solvent)
    cid = a.id if a.id else '__new__'
    return pd.DataFrame({'Compound_ID': cid, 'Solvent_1': s1, 'Solvent_2': s2,
                         'Ratio_1': [r[0] for r in ratios], 'Ratio_2': [r[1] for r in ratios]})


def load_model():
    return json.load(open(MODEL))


def note(a, m):
    if a.id and a.id not in m['known']['offset']:
        print(f'注意: {a.id} は学習データにない -> 溶媒のみの予測 (化合物差は反映されません)')
    elif not a.id:
        print('注意: 未知化合物 -> 溶媒のみの予測 (化合物差は反映されません。実測 1 点で補正すると精度が上がる)')
    s1, s2 = split_solvent(a.solvent)
    bad = [s for s in (s1, s2) if s not in EPS0.index]
    if bad:
        print(f'エラー: solvent_ref.csv にない溶媒 {bad}'); cmd_solvents(); raise SystemExit(1)
    if {s1, s2} not in TRAINED_PAIRS:
        print(f'注意: {a.solvent} は学習データにない溶媒系{"(単一溶媒)" if s1 == s2 else ""}。溶媒の強さは Snyder の ε° と混合則から外挿しており、'
              f'精度は未検証 (系の補正は {"アルコール系 (MeOH と同じ補正)" if {s1, s2} & set(ALCOHOLS) else "補正なし (Hexane/AcOEt と同じ)"})')


def cmd_rf(a):
    m = load_model(); note(a, m)
    s1, s2 = split_solvent(a.solvent)
    r1, r2 = (1.0, 1.0) if s1 == s2 else parse_ratio(a.ratio)     # 単一溶媒は比が不要
    p = sigmoid(predict_logit(m, rows(a, [(r1, r2)])))[0]
    print(f'{a.id or "新規"}  {a.solvent}{"" if s1 == s2 else " " + a.ratio}  -> 予測 Rf = {p:.2f}')


def cmd_suggest(a):
    m = load_model(); note(a, m)
    s1, s2 = split_solvent(a.solvent)
    if s1 == s2:
        p = sigmoid(predict_logit(m, rows(a, [(1.0, 1.0)])))[0]
        print(f'{a.id or "新規"}  {s1} (単一溶媒) -> 予測 Rf = {p:.2f}   (単一溶媒は比を変えられないので、目標 Rf からの探索はできません。'
              f'混合溶媒 A/B で指定してください)')
        return
    strong1 = EPS0[s1] >= EPS0[s2]           # ε° が大きい方が強溶媒
    strong = s1 if strong1 else s2
    fs = np.arange(1, 100) / 100.0          # 強溶媒の体積分率
    df = rows(a, [(f, 1 - f) if strong1 else (1 - f, f) for f in fs])
    p = sigmoid(predict_logit(m, df))
    i = int(np.argmin(np.abs(p - a.target)))
    f = fs[i]
    r1, r2 = (f, 1 - f) if strong1 else (1 - f, f)
    print(f'{a.id or "新規"}  {a.solvent}  目標 Rf={a.target}  -> 強溶媒 {strong} の体積分率 {f:.2f} '
          f'(= {s1}:{s2} = {r1:.2f}:{r2:.2f} ≒ {max(r1, r2) / min(r1, r2):.1f}:1) で予測 Rf={p[i]:.2f}')
    if abs(p[i] - a.target) > 0.05:
        print(f'  ※ この溶媒系では目標 Rf に届きません (強溶媒 1%〜99% の予測 Rf は {p.min():.2f}〜{p.max():.2f})。'
              f'より強い溶媒を含む系に変えてください。')
    print('  参考: 強溶媒分率 -> 予測Rf: ' + ', '.join(f'{x:.2f}->{y:.2f}' for x, y in zip(fs[[4, 9, 19, 29, 49, 69]], p[[4, 9, 19, 29, 49, 69]])))
    print('  (CV での誤差の目安: 未知化合物 MAE≈0.12, 既知化合物の別条件は `eval` 参照)')


def cmd_solvents(_=None):
    ref = pd.read_csv('solvent_ref.csv', index_col=0).sort_values('eps0_Al2O3')
    print('使える溶媒 (ε° が大きいほど強い溶媒。ε°=Snyder 溶離力 アルミナ基準, V=分子体積 cm3/mol):')
    for n, r in ref.iterrows():
        tag = ' [学習データあり]' if n in ('Hexane', 'AcOEt', 'CHCl3', 'MeOH') else ''
        print(f'  {n:13s} ε°={r.eps0_Al2O3:.2f}  V={r.MolarVolume:6.1f}{tag}')
    print('溶媒系の指定例: Hexane/AcOEt  Toluene/AcOEt  CH2Cl2/MeOH   (学習データのある系は Hexane/AcOEt と CHCl3/MeOH)')


def interactive():
    """引数なしで起動したときの対話モード (VS Code の実行ボタンなど)"""
    if not os.path.exists(MODEL):
        print('モデルがないので学習します...'); cmd_fit(None)
    m = load_model()
    ids = sorted(m['known']['offset'])
    ref = pd.read_csv('solvent_ref.csv', index_col=0).sort_values('eps0_Al2O3')
    sol = [n + ('*' if n in ('Hexane', 'AcOEt', 'CHCl3', 'MeOH') else '') for n in ref.index]
    print('Enter だけで終了。')
    while True:
        print('\n登録済みの化合物ID: ' + '  '.join(ids) + '  | 未知の化合物は new')
        cid = input('化合物ID (未知の化合物は new): ').strip()
        if not cid: return
        print('指定できる溶媒 (弱→強, *=学習データあり): ' + '  '.join(sol))
        print('  混合は A/B (例 Hexane/AcOEt, CHCl3/MeOH, Toluene/AcOEt)、単一溶媒は名前だけ (例 Acetone)')
        solv = input('溶媒 (A/B または A): ').strip()
        mode = input('1=目標Rfから溶媒比を探す  2=溶媒比からRfを予測 [1]: ').strip() or '1'
        ns = argparse.Namespace(id=None if cid == 'new' else cid, smiles=None, solvent=solv)
        try:
            if mode == '2':
                if '/' in solv: ns.ratio = input('比 (例 3:1, 溶媒の書いた順): ').strip()
                else: ns.ratio = ''
                cmd_rf(ns)
            else:
                t = input('目標Rf [0.3]: ').strip(); ns.target = float(t) if t else 0.3; cmd_suggest(ns)
        except SystemExit:
            pass
        except Exception as e:
            print('入力エラー:', e)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        try:
            interactive()
        except (EOFError, KeyboardInterrupt):
            print()
        raise SystemExit
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('eval').set_defaults(f=cmd_eval)
    sp.add_parser('solvents').set_defaults(f=cmd_solvents)
    sp.add_parser('fit').set_defaults(f=cmd_fit)
    for n, f in (('rf', cmd_rf), ('suggest', cmd_suggest)):
        p = sp.add_parser(n); p.add_argument('--id'); p.add_argument('--smiles'); p.add_argument('--solvent', required=True)
        if n == 'rf': p.add_argument('--ratio', required=True)
        else: p.add_argument('--target', type=float, default=0.3)
        p.set_defaults(f=f)
    a = ap.parse_args(); a.f(a)
