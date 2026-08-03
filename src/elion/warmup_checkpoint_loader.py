import os, json, sys

print('[LOADER] warmup_checkpoint_loader.py executing', flush=True)
_ckpt_path = os.environ.get("TS_WARMUP_CHECKPOINT", "")
print(f'[LOADER] checkpoint path: {_ckpt_path!r}', flush=True)

if not _ckpt_path or not os.path.isfile(_ckpt_path):
    print('[LOADER] no checkpoint — warmup runs normally', flush=True)
else:
    try:
        with open(_ckpt_path) as _f:
            _ckpt = json.load(_f)

        _prior_mean = _ckpt["prior_mean"]
        _prior_std  = _ckpt["prior_std"]
        _known_var  = _prior_std ** 2
        _belief_by_name = {}
        for _comp_list in _ckpt.get("components", {}).values():
            for _r in _comp_list:
                _belief_by_name[_r["reagent_name"]] = _r

        print(f'[LOADER] loaded: prior_mean={_prior_mean:.4f} prior_std={_prior_std:.4f} '
              f'n_reagents={len(_belief_by_name)}', flush=True)

        # thompson_sampling.py lives in generators/TS/, which isn't on sys.path
        # yet (the loader runs before elion.py sets up its path). Add it.
        if 'thompson_sampling' not in sys.modules:
            _ts_dir_cands = [
                '/home/huangzihang/repos/elion/src/elion/generators/TS',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generators', 'TS'),
            ]
            for _d in _ts_dir_cands:
                if os.path.isfile(os.path.join(_d, 'thompson_sampling.py')):
                    if _d not in sys.path:
                        sys.path.insert(0, _d)
                        print(f'[LOADER] added to sys.path: {_d}', flush=True)
                    break

        import thompson_sampling as _ts_mod

        _orig_init = _ts_mod.ThompsonSampler.__init__

        def _patched_init(self, *args, **kwargs):
            # Call original __init__ first
            _orig_init(self, *args, **kwargs)
            # Then install warm_up override directly on this instance
            def _instance_warm_up(num_warmup_trials, *a, **kw):
                print('[LOADER] instance warm_up called — injecting checkpoint beliefs', flush=True)
                import traceback as _tb
                try:
                    restored = skipped = 0
                    for reagent_list in self.reagent_lists:
                        for reagent in reagent_list:
                            belief = _belief_by_name.get(reagent.reagent_name)
                            if belief:
                                reagent.current_phase  = "search"
                                reagent.current_mean   = belief["current_mean"]
                                reagent.current_std    = belief["current_std"]
                                reagent.known_var      = belief.get("known_var") or _known_var
                                reagent.num_scores     = belief["num_scores"]
                                reagent.initial_scores = []
                                restored += 1
                            else:
                                reagent.current_phase  = "search"
                                reagent.current_mean   = _prior_mean
                                reagent.current_std    = _prior_std
                                reagent.known_var      = _known_var
                                reagent.num_scores     = 0
                                reagent.initial_scores = []
                                skipped += 1
                    self._warmup_std = _prior_std
                    print(f'[LOADER] restored={restored} skipped={skipped}', flush=True)
                    # Return synthetic result — callers expect list of [score, smiles, name]
                    return [[_prior_mean, "checkpoint", "checkpoint"]]
                except Exception as _e:
                    print(f'[LOADER] ERROR: {_e}', flush=True)
                    _tb.print_exc()
                    # Fall back to real warmup
                    return _ts_mod.ThompsonSampler.warm_up(self, num_warmup_trials, *a, **kw)
            # Assign directly — instance attributes bypass the descriptor protocol
            # so types.MethodType is not needed (and was causing a double-self bug:
            # TS.py calls ts.warm_up(n) → bound method passes self_ but n was left
            # unbound → TypeError: missing 1 required positional argument 'n').
            self.warm_up = _instance_warm_up
            print('[LOADER] instance.warm_up override installed', flush=True)

        _ts_mod.ThompsonSampler.__init__ = _patched_init
        print('[LOADER] ThompsonSampler.__init__ patched — override will install on every new instance', flush=True)

    except Exception as _e:
        import traceback
        print(f'[LOADER] FATAL ERROR loading checkpoint: {_e}', flush=True)
        traceback.print_exc()
