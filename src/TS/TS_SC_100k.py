import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib


df = pd.read_csv('/blue/lic/huangzihang/repos/elion/src/RL_feedback_loop/results_vina/LGBM_suzuki_vina_2.csv')
df['fp'] = df.SMILES.apply(uru.smi2numpy_fp)
# threshold = -8
threshold = -9
df['Affinity'] = (df['Affinity'] <= threshold).astype(int)

train, test = train_test_split(df)
cls = LGBMClassifier()
cls.fit(np.stack(train.fp),train.Affinity)
joblib.dump(cls, '/blue/lic/huangzihang/repos/elion/src/TS/LGBM_Classifier/LGBM_Suzuki_vina_2.pkl')

from ts_main_csv import read_input, run_ts, parse_input_dict

ts_input_dict = read_input('/blue/lic/huangzihang/repos/elion/src/TS/config/LGBM_Suzuki_100k.json')
score_df = run_ts(ts_input_dict)
