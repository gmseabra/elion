#!/usr/bin/env python
"""
generators/TS.py
================
Thompson Sampling generator for Elion.

Plugs /generators/TS/ts_main.py into the Elion generator interface so that
    python elion.py -i input_TS.yml
runs the full TS warmup + search and returns scored molecules to Elion's
estimate_properties / check_and_adjust_thresholds pipeline.

The yml must have a 'generator.TS' block (see input_TS.yml for full example):

    generator:
      name: "TS"
      TS:
        reaction_smarts: "..."
        num_warmup_trials: 3
        num_ts_iterations: 5000
        ts_mode: "maximize"
        reagent_file_list: [...]
        results_filename: "..."   # optional
        batch_size: 1             # optional
        eval_batch_size: 256      # optional
        log_level: INFO           # optional

The reward_function block is read by Elion and passed to Estimators as usual.
TS does NOT build its own ElionEstimatorEvaluator — it receives the shared
Estimators object from elion.py so thresholds, reward_hook, and
check_and_adjust_thresholds all work exactly as they do for other generators.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

# ── make TS sub-package importable ───────────────────────────────────────────
_TS_DIR = str(Path(__file__).resolve().parent / 'TS')
if _TS_DIR not in sys.path:
    sys.path.insert(0, _TS_DIR)

from thompson_sampling import ThompsonSampler
from ts_logger import get_logger
from .AbstractGenerator import AbstractGenerator


class TS(AbstractGenerator):
    """Thompson Sampling generator — wraps ts_main.run_ts() as an Elion generator.

    elion.py calls:
        generator.generate_mols()          — not used in TS (search is in bias_generator)
        generator.bias_generator(ctrl_opts, estimator)  — runs warmup + search
    """

    def __init__(self, generator_properties: dict):
        self.name = generator_properties.get('name', 'TS')

        ts_cfg = generator_properties.get('TS', {})

        self.reaction_smarts   = ts_cfg['reaction_smarts']
        self.num_warmup_trials = int(ts_cfg.get('num_warmup_trials', 3))
        self.num_ts_iterations = int(ts_cfg.get('num_ts_iterations', 5000))
        self.ts_mode           = ts_cfg.get('ts_mode', 'maximize')
        self.reagent_file_list = ts_cfg['reagent_file_list']
        self.results_filename  = ts_cfg.get('results_filename', None)
        self.batch_size        = int(ts_cfg.get('batch_size', 1))
        self.eval_batch_size   = int(ts_cfg.get('eval_batch_size', 256))
        log_level_str          = ts_cfg.get('log_level', 'INFO').upper()
        self.log_level         = getattr(logging, log_level_str, logging.INFO)
        self.log_filename      = ts_cfg.get('log_filename', None)

        self._logger = get_logger(__name__, filename=self.log_filename)
        self._logger.setLevel(self.log_level)
        self._logger.info('TS generator initialised | mode=%s | iterations=%d',
                          self.ts_mode, self.num_ts_iterations)

    # ── Elion interface ───────────────────────────────────────────────────────

    def generate_mols(self) -> list:
        """Not used by TS — TS generates and scores in one pass inside bias_generator.
        Returns empty list so elion.py's generate_mols() path doesn't crash.
        """
        self._logger.warning('TS.generate_mols() called — TS generates inside bias_generator(). '
                             'Use run_type: bias_generator in your yml.')
        return []

    def bias_generator(self, ctrl_opts: dict, estimator) -> None:
        """Run the full Thompson Sampling warmup + search.

        Called by elion.py's bias_generator() function. Receives the shared
        Estimators object so that check_and_adjust_thresholds() fires at the
        right time (after every TS search cycle if desired).

        Args:
            ctrl_opts:  config['Control'] dict from elion.py
            estimator:  properties.Estimators instance (shared with elion.py)
        """
        import pandas as pd

        self._logger.info('='*60)
        self._logger.info('THOMPSON SAMPLING  mode=%s  iterations=%d',
                          self.ts_mode, self.num_ts_iterations)
        self._logger.info('='*60)

        # ── build a thin evaluator wrapper around the shared Estimators ──────
        # This lets ThompsonSampler call evaluator.evaluate(mol) and
        # evaluator.evaluate_batch(mols) without duplicating the Estimators
        # object or losing threshold state.
        evaluator = _EstimatorsEvaluatorAdapter(estimator, logger=self._logger)

        # ── build and configure ThompsonSampler ──────────────────────────────
        ts = ThompsonSampler(
            mode=self.ts_mode,
            db_name='eXplore',
            log_level=self.log_level,
        )
        ts.set_hide_progress(False)
        ts.set_evaluator(evaluator)
        ts.read_reagents_csv(
            reagent_file_list=self.reagent_file_list,
            num_to_select=None,
        )
        ts.set_reaction(self.reaction_smarts)

        # ── warmup ───────────────────────────────────────────────────────────
        ts.warm_up(
            num_warmup_trials=self.num_warmup_trials,
            eval_batch_size=self.eval_batch_size,
        )

        # ── search ───────────────────────────────────────────────────────────
        out_list = ts.search(
            num_cycles=self.num_ts_iterations,
            batch_size=self.batch_size,
        )

        # ── after search: adjust thresholds once on all found molecules ──────
        # This is the hook that makes reward_hook / allowed_threshold_jumps work
        # for standalone TS runs, mirroring what the RL loop does in TS.py.
        if out_list:
            scored_smiles = [row[1] for row in out_list]
            scored_mols   = [Chem.MolFromSmiles(s) for s in scored_smiles
                             if Chem.MolFromSmiles(s) is not None]
            if scored_mols:
                predictions = estimator.estimate_properties(scored_mols)
                estimator.check_and_adjust_thresholds(predictions)
                self._logger.info('Thresholds after TS search:')
                for prop, cls in estimator.properties.items():
                    if cls.optimize:
                        self._logger.info('  %-20s threshold=%.3f  converged=%s',
                                          prop, cls.threshold, cls.converged)

        # ── save results ──────────────────────────────────────────────────────
        out_df = pd.DataFrame(out_list, columns=['score', 'SMILES', 'Name'])
        if self.results_filename:
            out_df.to_csv(self.results_filename, index=False)
            self._logger.info('Results saved to %s', self.results_filename)

        ascending = self.ts_mode != 'maximize'
        print('\nTop 10 TS results:')
        print(out_df.sort_values('score', ascending=ascending)
                    .drop_duplicates(subset='SMILES').head(10).to_string(index=False))

        return out_df

    def generate_smis(self) -> list:
        """Not applicable for TS."""
        return []


# ── thin adapter: makes Estimators look like an Evaluator ────────────────────

class _EstimatorsEvaluatorAdapter:
    """Wraps a shared Estimators object to satisfy ThompsonSampler's evaluator
    interface (evaluate / evaluate_batch / counter) without constructing a
    second Estimators instance. This preserves threshold state across the run.
    """

    def __init__(self, estimator, logger=None):
        self._estimator = estimator
        self._count = 0
        self._logger = logger or logging.getLogger(__name__)

    @property
    def counter(self) -> int:
        return self._count

    def evaluate(self, mol) -> float:
        """Score a single molecule."""
        try:
            predictions = self._estimator.estimate_properties([mol])
            rewards     = self._estimator.estimate_rewards(predictions)
            self._count += 1
            return float(rewards['TOTAL'][0])
        except Exception as e:
            self._logger.error('[_EstimatorsEvaluatorAdapter.evaluate] %s', e)
            return float('nan')

    def evaluate_batch(self, mols: list) -> list[float]:
        """Score a batch of molecules in one Estimators call."""
        if not mols:
            return []
        try:
            predictions = self._estimator.estimate_properties(mols)
            rewards     = self._estimator.estimate_rewards(predictions)
            self._count += len(mols)
            return [float(v) for v in rewards['TOTAL']]
        except Exception as e:
            self._logger.error('[_EstimatorsEvaluatorAdapter.evaluate_batch] %s', e)
            return [float('nan')] * len(mols)