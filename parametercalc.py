import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def generate_molecular_features(input_file, output_file):
    df = pd.read_csv(input_file)
    unique_compounds = df[['Compound_ID', 'SMILES']].dropna(subset=['SMILES']).drop_duplicates('Compound_ID')
    
    features = []
    for _, row in unique_compounds.iterrows():
        cpd_id = row['Compound_ID']
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None: continue
            
        feat_dict = {
            'Compound_ID': cpd_id,
            # --- 基本物性 ---
            'MolLogP': Descriptors.MolLogP(mol),
            'ExactMolWt': Descriptors.ExactMolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            # --- 論文で重視される水素結合・官能基指標 ---
            'NumHDonors': Descriptors.NumHDonors(mol),      # シリカへの供与
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # シリカからの受容
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol), # ヘテロ原子数
            # --- 形状・柔軟性 ---
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol), # 柔軟性
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),   # 芳香族性
            'FractionCSP3': Descriptors.FractionCSP3(mol),           # 立体度
            # --- 電子状態の代理指標 ---
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol), # 最大部分電荷（吸着点）
            'MolMR': Descriptors.MolMR(mol) # 分子屈折（分散力の指標）
        }
        features.append(feat_dict)
    
    pd.DataFrame(features).to_csv(output_file, index=False)
    print(f"論文準拠のパラメータ抽出完了: {output_file}")

if __name__ == "__main__":
    generate_molecular_features('tlc.csv', 'molecular_features.csv')