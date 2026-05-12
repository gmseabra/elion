import sys
sys.path.append('/blue/lic/huangzihang/repos/elion/src/elion')

from ts_main import read_input_yml, run_ts

ts_input_dict = read_input_yml('/blue/lic/huangzihang/repos/elion/src/elion/input_TS.yml')
score_df = run_ts(ts_input_dict)