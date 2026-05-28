import sys
sys.path.append('../..')

from ts_main import read_input_yml, run_ts

ts_input_dict = read_input_yml('../../input_TS.yml')
score_df = run_ts(ts_input_dict)