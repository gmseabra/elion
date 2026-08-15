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
<<<<<<< HEAD
sys.argv = ['elion.py', '-i', '/home/huangzihang/repos/elion/src/UI/.raytmp/elion_ts_k8zknpgx/input_TS_rxn208_snar.yml']
os.environ['TS_WARMUP_CHECKPOINT'] = '/home/huangzihang/repos/user_data/Warmup_TS/snar_20260804_205925_warmup.json'
=======
sys.argv = ['elion.py', '-i', '/home/huangzihang/repos/elion/src/UI/.raytmp/elion_ts_gxadx2w_/input_TS_rxn208_snar.yml']
os.environ['TS_WARMUP_CHECKPOINT'] = '/home/huangzihang/repos/user_data/Warmup_TS/snar_20260709_152242_warmup.json'
>>>>>>> 8137cead6ea75c769e35aab6c374c2b5fd3c50c5

# Import and patch BEFORE any elion modules load
print('[WRAPPER] sys.path before loader: ' + str(sys.path[:6]), flush=True)
import warmup_checkpoint_loader
print('[WRAPPER] loader done, running elion.py', flush=True)

# Use exec instead of runpy to avoid module caching issues
with open('elion.py') as _f:
    _code = _f.read()
exec(compile(_code, 'elion.py', 'exec'), {'__name__': '__main__', '__file__': 'elion.py'})
