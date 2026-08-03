import logging
import random
from typing import List, Optional, Tuple

import functools
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from disallow_tracker import DisallowTracker
from reagent import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents_csv
from evaluators import DBEvaluator

# ── PROBE2 DEBUG FILE LOGGING ─────────────────────────────────────────────────
# Writes timing/progress to a file in the debug dir, independent of the browser
# (ts_worker.js filters unrecognized stdout lines). tail -f this to verify the
# stdout-volume fix holds past iter 4000. This logging is CHEAP (one short line
# every 25 iters) and writes to a SEPARATE file, so it does NOT feed the SSE
# pipe and cannot itself cause the backpressure we're fixing.
import os as _os_probe, sys as _sys_probe, time as _time_probe
_PROBE2_LOG_DIR = "/home/huangzihang/repos/elion/src/visualizer/debug"
_PROBE2_LOG_PATH = _os_probe.path.join(_PROBE2_LOG_DIR, "probe2.log")
_probe2_fh = None
def _probe2_open_log():
    global _probe2_fh, _PROBE2_LOG_PATH
    try:
        _os_probe.makedirs(_PROBE2_LOG_DIR, exist_ok=True)
        _probe2_fh = open(_PROBE2_LOG_PATH, "a", buffering=1)
    except Exception:
        try:
            _PROBE2_LOG_PATH = "/tmp/probe2.log"
            _probe2_fh = open(_PROBE2_LOG_PATH, "a", buffering=1)
        except Exception:
            _probe2_fh = None
    return _probe2_fh
def _probe2_log(msg):
    global _probe2_fh
    if _probe2_fh is None:
        _probe2_open_log()
    if _probe2_fh is not None:
        try:
            _probe2_fh.write(f"[{_time_probe.strftime('%H:%M:%S')}] {msg}\n")
            _probe2_fh.flush()
        except Exception:
            pass
_probe2_open_log()
_probe2_log(f"[PROBE2-IMPORT] thompson_sampling (LEAN FIX) imported from: {__file__} "
            f"| log -> {_PROBE2_LOG_PATH}")
# ──────────────────────────────────────────────────────────────────────────────


class ThompsonSampler:
    def __init__(self, mode="maximize", db_name="SYNPLE", log_filename: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        :param log_level: logging level controlling verbosity.
                          logging.DEBUG  — full Bayesian math, per-score updates, winner indices
                          logging.INFO   — per-iteration results, warmup stats (default, recommended for production)
                          logging.WARNING — only unexpected conditions
                          Set via json/yaml config key "log_level": "DEBUG" / "INFO" / "WARNING"
        """
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self.logger.setLevel(log_level)
        self._log_level = log_level
        self._disallow_tracker = None
        self.hide_progress = False
        self._mode = mode
        self.db_name = db_name
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress bars
        """
        self.hide_progress = hide_progress

    def read_reagents_csv(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents_csv(
            reagent_file_list, self.db_name, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float, list]:
        """Evaluate a set of reagents without immediately updating reagent scores.
        Score updates are deferred so the caller can batch them.

        :param choice_list: list of reagent ids
        :return: (product_smiles, product_name, score, selected_reagents)
                 selected_reagents is the list of Reagent objects; caller is
                 responsible for calling add_score / add_score_batch on them.
        """
        selected_reagents = []
        for idx, choice in enumerate(choice_list):
            selected_reagents.append(self.reagent_lists[idx][choice])

        prod = self.reaction.RunReactants([r.mol for r in selected_reagents])
        product_name = "_".join([r.reagent_name for r in selected_reagents])
        res = np.nan
        product_smiles = "FAIL"

        if prod:
            try:
                prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
                Chem.SanitizeMol(prod_mol)
                product_smiles = Chem.MolToSmiles(prod_mol)
                if isinstance(self.evaluator, DBEvaluator):
                    res = self.evaluator.evaluate(product_name)
                    res = float(res)
                else:
                    res = self.evaluator.evaluate(prod_mol)
                if np.isfinite(res):
                    self.logger.debug(
                        '[evaluate] score=%.6f for reagents: %s (deferred update)',
                        res, [r.reagent_name for r in selected_reagents])
            except Exception as e:
                self.logger.error('Error during evaluation: %s', e)
                self.logger.error('Reactant SMILES causing failure: %s',
                                  [r.smiles for r in selected_reagents])
                self.logger.error('Raw product tuple: %s', prod)

        return product_smiles, product_name, res, selected_reagents

    def evaluate_batch(self, choice_lists: List[List[int]]) -> List[Tuple[str, str, float, list]]:
        """Run reactions for a batch of choice_lists, then score all valid products
        in a single evaluator call — avoiding per-molecule model overhead (e.g. CHEMBERT
        spinning up DataLoader workers for every molecule individually).

        Falls back to per-molecule evaluate() if the evaluator does not implement
        evaluate_batch() (e.g. DBEvaluator, FPEvaluator).

        :param choice_lists: list of choice_list, each as passed to evaluate()
        :return: list of (product_smiles, product_name, score, selected_reagents)
                 in the same order as choice_lists; failed reactions get score=nan.
        """
        # ── Step 1: run all reactions, collect valid products ──────────────────
        results = []          # final output, same length as choice_lists
        valid_indices = []    # positions in choice_lists that produced a valid mol
        valid_mols = []       # RDKit mol for each valid product
        valid_names = []      # product name for each valid product
        valid_reagents = []   # list[Reagent] for each valid product

        for idx, choice_list in enumerate(choice_lists):
            selected_reagents = [self.reagent_lists[i][c]
                                 for i, c in enumerate(choice_list)]
            prod = self.reaction.RunReactants([r.mol for r in selected_reagents])
            product_name = "_".join([r.reagent_name for r in selected_reagents])

            mol_ok = False
            if prod:
                try:
                    prod_mol = prod[0][0]
                    Chem.SanitizeMol(prod_mol)
                    product_smiles = Chem.MolToSmiles(prod_mol)
                    mol_ok = True
                except Exception as e:
                    self.logger.error('Reaction sanitize error idx=%d: %s', idx, e)

            if mol_ok:
                valid_indices.append(idx)
                valid_mols.append(prod_mol)
                valid_names.append(product_name)
                valid_reagents.append(selected_reagents)
                results.append([product_smiles, product_name, np.nan, selected_reagents])
            else:
                results.append(["FAIL", product_name, np.nan,
                                 [self.reagent_lists[i][c] for i, c in enumerate(choice_list)]])

        if not valid_mols:
            return results

        # ── Step 2: score all valid products in ONE evaluator call ─────────────
        try:
            if isinstance(self.evaluator, DBEvaluator):
                # DB lookup has no batch advantage — fall through to per-mol
                raise AttributeError("DBEvaluator: use per-mol path")

            # Use evaluate_batch if available (ElionEstimatorEvaluator),
            # otherwise fall back to per-molecule
            if hasattr(self.evaluator, 'evaluate_batch'):
                scores = self.evaluator.evaluate_batch(valid_mols)
            else:
                scores = [self.evaluator.evaluate(m) for m in valid_mols]

            for list_idx, score in zip(valid_indices, scores):
                results[list_idx][2] = float(score) if np.isfinite(float(score)) else np.nan
                sel_r = results[list_idx][3]
                self.logger.debug(
                    '  [evaluate] score=%.6f being added to %d reagents: %s',
                    score, len(sel_r), [r.reagent_name for r in sel_r])
                # Log add_score/warmup immediately after evaluate — interleaved per molecule
                # _warmup_reagent_counts tracks buffered score count per reagent name
                # across all batches so far (updated here, before flush runs)
                if not hasattr(self, '_warmup_reagent_counts'):
                    self._warmup_reagent_counts = {}
                for r in sel_r:
                    self._warmup_reagent_counts[r.reagent_name] = (
                        self._warmup_reagent_counts.get(r.reagent_name, 0) + 1)
                    self.logger.debug(
                        '    [add_score/warmup] reagent=%s | buffered score=%.6f '
                        '(warmup count so far: %d)',
                        r.reagent_name, score,
                        self._warmup_reagent_counts[r.reagent_name])

        except AttributeError:
            # Fallback: per-molecule (DBEvaluator or missing evaluate_batch)
            for list_idx, (mol, name, sel_r) in zip(
                    valid_indices,
                    zip(valid_mols, valid_names, valid_reagents)):
                try:
                    if isinstance(self.evaluator, DBEvaluator):
                        res = float(self.evaluator.evaluate(name))
                    else:
                        res = self.evaluator.evaluate(mol)
                    results[list_idx][2] = res if np.isfinite(res) else np.nan
                except Exception as e:
                    self.logger.error('evaluate fallback error: %s', e)
        except Exception as e:
            self.logger.error('evaluate_batch error: %s', e)

        return [tuple(r) for r in results]

    def _flush_score_batch(self, pending: list) -> None:
        """Flush a list of (selected_reagents, score) pairs to reagents using batch updates.

        Scores are grouped per Reagent object, then applied in one add_score_batch call,
        eliminating the one-molecule-at-a-time overhead visible in the logs.

        :param pending: list of (selected_reagents: List[Reagent], score: float)
        """
        score_map: dict[int, tuple] = {}  # id(reagent) -> (reagent, [scores])
        for selected_reagents, score in pending:
            for reagent in selected_reagents:
                rid = id(reagent)
                if rid not in score_map:
                    score_map[rid] = (reagent, [])
                score_map[rid][1].append(score)

        self.logger.debug(
            '[_flush_score_batch] flushing scores for %d unique reagents across %d evaluations',
            len(score_map), len(pending))
        for reagent, scores in score_map.values():
            reagent.add_score_batch(scores)
            # Note: [add_score/warmup] lines are logged in evaluate_batch for interleaving

    def warm_up(self, num_warmup_trials=3, eval_batch_size: int = 256):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners.

        Key optimisation: reaction/partner selection (which must stay sequential so the
        disallow_tracker stays consistent) is separated from scoring. Choices are collected
        per-reagent with correct tracker updates, then evaluated in one evaluate_batch()
        call per reagent — so CHEMBERT runs once per reagent slot instead of once per mol.

        :param num_warmup_trials: number of random partner trials per reagent
        :param eval_batch_size: max molecules per evaluate_batch() call (tune to GPU memory)
        """
        # ── Checkpoint restore: skip warmup if TS_WARMUP_CHECKPOINT is set ────
        import os as _os, json as _json
        _ckpt_path = _os.environ.get("TS_WARMUP_CHECKPOINT", "")
        if _ckpt_path and _os.path.isfile(_ckpt_path):
            try:
                with open(_ckpt_path) as _f:
                    _ckpt = _json.load(_f)
                _prior_mean = _ckpt["prior_mean"]
                _prior_std  = _ckpt["prior_std"]
                _known_var  = _prior_std ** 2
                _beliefs = {}
                for _comp_list in _ckpt.get("components", {}).values():
                    for _r in _comp_list:
                        _beliefs[_r["reagent_name"]] = _r
                self.logger.info(
                    "[checkpoint] Loading warmup checkpoint: %s "
                    "(prior_mean=%.4f, prior_std=%.4f, %d reagents)",
                    _ckpt_path, _prior_mean, _prior_std, len(_beliefs))
                restored = skipped = 0
                for reagent_list in self.reagent_lists:
                    for reagent in reagent_list:
                        belief = _beliefs.get(reagent.reagent_name)
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
                self.logger.info(
                    "[checkpoint] Restored %d reagents from checkpoint, %d set to prior. "
                    "Skipping warmup phase.", restored, skipped)
                return [[_prior_mean, "checkpoint", "checkpoint"]]
            except Exception as _e:
                self.logger.warning(
                    "[checkpoint] Failed to load checkpoint %s: %s — running warmup normally",
                    _ckpt_path, _e)
        # ── End checkpoint restore ─────────────────────────────────────────────
        idx_list = list(range(len(self.reagent_lists)))
        reagent_count_list = [len(x) for x in self.reagent_lists]
        warmup_results = []
        pending_warmup: list[tuple[list, float]] = []
        self._warmup_reagent_counts = {}  # tracks buffered score count per reagent

        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            current_max = reagent_count_list[i]

            # ── Phase 1: selection (sequential — tracker must stay in sync) ────
            # disallow_tracker.update() is called immediately after each selection,
            # exactly as the original code did. Only evaluate() is deferred.
            reagent_choices: list[list[int]] = []  # valid choice_lists for this slot

            for j in tqdm(range(current_max),
                          desc=f"Warmup {i + 1} of {len(idx_list)} [selecting]",
                          disable=self.hide_progress):
                for k in range(num_warmup_trials):
                    current_list = [DisallowTracker.Empty] * len(idx_list)
                    current_list[i] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                    if j in disallow_mask:
                        continue
                    current_list[i] = j
                    for p in partner_list:
                        current_list[p] = DisallowTracker.To_Fill
                        disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                        selection_scores = np.random.uniform(size=reagent_count_list[p])
                        selection_scores[list(disallow_mask)] = np.nan
                        current_list[p] = np.nanargmax(selection_scores).item(0)
                    # update tracker immediately — same as original, keeps state correct
                    self._disallow_tracker.update(current_list)
                    reagent_choices.append(list(current_list))

            # ── Phase 2: evaluate in batches (parallelisable, model-friendly) ──
            n_batches = max(1, math.ceil(len(reagent_choices) / eval_batch_size))
            self.logger.info(
                'Warmup %d/%d: scoring %d combos in %d batches of up to %d',
                i + 1, len(idx_list), len(reagent_choices), n_batches, eval_batch_size)

            for batch_idx, batch_start in enumerate(tqdm(
                    range(0, len(reagent_choices), eval_batch_size),
                    desc=f"Warmup {i + 1} of {len(idx_list)} [scoring]",
                    disable=self.hide_progress)):
                batch = reagent_choices[batch_start: batch_start + eval_batch_size]
                self.logger.debug('Warmup %d/%d batch %d/%d (%d mols)',
                                  i + 1, len(idx_list), batch_idx + 1, n_batches, len(batch))
                batch_results = self.evaluate_batch(batch)
                for product_smiles, product_name, score, selected_reagents in batch_results:
                    if np.isfinite(score):
                        warmup_results.append([score, product_smiles, product_name])
                        pending_warmup.append((selected_reagents, score))

                # ── batch summary log (fires after every evaluate_batch call) ──
                batch_scores = [r[2] for r in batch_results if np.isfinite(r[2])]
                if batch_scores and warmup_results:
                    all_scores = [ws[0] for ws in warmup_results]
                    self.logger.debug(
                        '[evaluate_batch] prior_mean=%.6f | prior_std=%.6f | '
                        'num_warmup_scores=%d',
                        np.mean(all_scores), np.std(all_scores), len(all_scores))

        warmup_scores = [ws[0] for ws in warmup_results]
        self.logger.info(
            'warmup score stats: cnt=%d, mean=%.4f, std=%.4f, min=%.4f, max=%.4f',
            len(warmup_scores), np.mean(warmup_scores), np.std(warmup_scores),
            np.min(warmup_scores), np.max(warmup_scores))
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        self.logger.info('[warm_up] prior_mean=%.6f | prior_std=%.6f | num_warmup_scores=%d',
                         prior_mean, prior_std, len(warmup_scores))

        self.logger.debug('[warm_up] flushing %d warmup pairs to reagents in batch', len(pending_warmup))
        self._flush_score_batch(pending_warmup)

        for i in range(len(self.reagent_lists)):
            for j in range(len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                    self.logger.debug(
                        '[warm_up] component %d, reagent %d (%s): init mu=%.6f, std=%.6f, num_scores=%d',
                        i, j, reagent.reagent_name,
                        reagent.current_mean, reagent.current_std, reagent.num_scores)
                except ValueError:
                    self.logger.info('Skipping reagent %s: no successful evaluations during warmup',
                                     reagent.reagent_name)
                    self._disallow_tracker.retire_one_synthon(i, j)
        self.logger.info('Top score found during warmup: %.3f', max(warmup_scores))
        return warmup_results

    def search(self, num_cycles=25, batch_size: int = 1, log_batch_size: int = 1):
        """Run the search with batched score updates.

        :param num_cycles: total number of search iterations
        :param batch_size: number of molecules to evaluate before flushing scores
                           back to reagents. batch_size=1 matches the original
                           per-molecule behaviour. Larger values (e.g. 10-50) reduce
                           the per-score Python overhead at the cost of slightly delayed
                           Bayesian updates within each batch.
        :param log_batch_size: number of scored molecules to collect before printing
                               as a batch in the debug log. Controlled by
                               generator.batch_size in yml (passed from ts_main).
                               1 = log every molecule individually.
        :return: a list of [score, smiles, name] results
        """
        out_list = []
        rng = np.random.default_rng()
        pending_batch: list[tuple[list, float]] = []
        log_batch: list[tuple[int, float, str, str]] = []  # (iter, score, smiles, name)

        # PROBE2 timing (writes to the debug file only — see _probe2_log)
        _t_select = 0.0
        _t_score  = 0.0
        _t_flush  = 0.0
        _t_win    = _time_probe.perf_counter()
        _probe_last_i = 0
        _probe2_log(f"[PROBE2-ENTER] search() num_cycles={num_cycles} batch_size={batch_size} "
                    f"debug_on={self.logger.isEnabledFor(logging.DEBUG)} "
                    f"reagent_sizes={[len(r) for r in self.reagent_lists]}")

        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            _sel_t0 = _time_probe.perf_counter()
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)

            for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
                reagent_list = self.reagent_lists[cycle_id]
                selected_reagents[cycle_id] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                stds = np.array([r.current_std for r in reagent_list])
                mu = np.array([r.current_mean for r in reagent_list])

                self.logger.debug(
                    '--- cycle_id=%d | mu range: [%.6f, %.6f], std range: [%.6f, %.6f] | competitors=%d',
                    cycle_id, mu.min(), mu.max(), stds.min(), stds.max(),
                    len(reagent_list) - len(disallow_mask))

                choice_row = rng.normal(size=len(reagent_list)) * stds + mu
                if disallow_mask:
                    choice_row[np.array(list(disallow_mask))] = np.nan
                    self.logger.debug('disallow_mask: %s', disallow_mask)

                winner_idx = self.pick_function(choice_row)

                # Log winner for this cycle (parsed by the UI at ts_worker.js)
                self.logger.debug(
                    'winner | cycle_id=%d | reagent=%s | '
                    'sampled=%.6f | mu=%.6f | std=%.6f | num_scores=%d',
                    cycle_id,
                    reagent_list[winner_idx].reagent_name,
                    choice_row[winner_idx],
                    reagent_list[winner_idx].current_mean,
                    reagent_list[winner_idx].current_std,
                    reagent_list[winner_idx].num_scores)

                # NOTE: the per-cycle "top-5 competitors" debug block was removed.
                # It emitted ~5 lines/cycle (~10 lines/iter) that the dashboard
                # never parses. Under DEBUG that stdout volume backs up the output
                # pipe once the browser consumer lags (~iter 4000), blocking
                # elion's write() and stalling the run. Winner/post-update/mu-range
                # lines the UI DOES parse are kept.

                selected_reagents[cycle_id] = winner_idx

            _t_select += _time_probe.perf_counter() - _sel_t0

            _score_t0 = _time_probe.perf_counter()
            self._disallow_tracker.update(selected_reagents)
            smiles, name, score, sel_reagent_objs = self.evaluate(selected_reagents)
            _t_score += _time_probe.perf_counter() - _score_t0

            # (removed per-iter 'iter=N | score=' debug line — not parsed by the UI)

            if np.isfinite(score):
                out_list.append([score, smiles, name])
                pending_batch.append((sel_reagent_objs, score))
                log_batch.append((i, score, smiles, name))

            # Flush scores to reagents as a batch (Bayesian update frequency)
            if len(pending_batch) >= batch_size or i == num_cycles - 1:
                if pending_batch:
                    _flush_t0 = _time_probe.perf_counter()
                    self._flush_score_batch(pending_batch)
                    _t_flush += _time_probe.perf_counter() - _flush_t0
                    for sel_r_objs, _ in pending_batch:
                        for comp_idx, reagent_idx in enumerate(selected_reagents):
                            r = self.reagent_lists[comp_idx][reagent_idx]
                            self.logger.debug(
                                'post-update | comp=%d, reagent=%d (%s): mu=%.6f, std=%.6f, num_scores=%d',
                                comp_idx, reagent_idx, r.reagent_name,
                                r.current_mean, r.current_std, r.num_scores)
                    pending_batch = []

            # Batch molecule logging removed (was ~1 debug line per molecule,
            # not parsed by the UI — pure stdout volume that backs up the pipe).
            if len(log_batch) >= log_batch_size or i == num_cycles - 1:
                if log_batch:
                    log_batch = []

            if i % 100 == 0 and out_list:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info('Iteration: %d | max score: %.6f | smiles: %s | name: %s',
                                 i, top_score, top_smiles, top_name)

            # ── PROBE2 timing to the debug file (every 25 iters) ──────────────
            # If the volume fix worked, these it/s stay flat past iter 4000.
            if i > 0 and i % 25 == 0:
                _dt = _time_probe.perf_counter() - _t_win
                _n = i - _probe_last_i
                _itps = _n / _dt if _dt > 0 else 0
                _probe2_log(
                    f"[PROBE2] iter={i} | {_itps:.1f} it/s | per-iter ms: "
                    f"select={_t_select/_n*1000:.2f} "
                    f"score={_t_score/_n*1000:.2f} "
                    f"flush={_t_flush/_n*1000:.2f}")
                _t_select = _t_score = _t_flush = 0.0
                _t_win = _time_probe.perf_counter()
                _probe_last_i = i

        return out_list