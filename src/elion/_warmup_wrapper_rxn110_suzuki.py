import sys, os, logging
print('[WRAPPER] starting', flush=True)
# Redirect ALL logging levels (including DEBUG) to stdout
# so elion's [evaluate] score lines appear in stdout not stderr.
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    force=True,
    format='%(levelname)s %(name)s: %(message)s'
)
sys.argv = ['elion.py', '-i', '/home/huangzihang/raytmp/elion_ts_k3tr2v1a/input_TS_rxn110_suzuki.yml']
os.environ['TS_WARMUP_CHECKPOINT'] = '/home/huangzihang/repos/user_data/Warmup_TS/suzuki_20260611_224623_warmup.json'

# Import and patch BEFORE any elion modules load
print('[WRAPPER] sys.path before loader: ' + str(sys.path[:6]), flush=True)
import warmup_checkpoint_loader
print('[WRAPPER] loader done, running elion.py', flush=True)

# Use exec instead of runpy to avoid module caching issues
with open('elion.py') as _f:
    _code = _f.read()
exec(compile(_code, 'elion.py', 'exec'), {'__name__': '__main__', '__file__': 'elion.py'})
