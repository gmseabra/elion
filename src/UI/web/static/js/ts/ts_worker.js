// =============================================================================
// ts_worker.js — Web Worker for Thompson Sampling log parsing
// Runs off the main thread. Receives raw SSE lines, emits structured updates.
//
// Messages TO worker:   { type: 'line', data: '<raw log line>' }
//                       { type: 'reset' }
// Messages FROM worker: { type: 'reagent_update', id, score, wc, partner, best, mu, std, count }
//                       { type: 'wu_event', event: {...} }
//                       { type: 'ts_event', event: {...} }
//                       { type: 'phase', phase: 'Warmup 2' }
//                       { type: 'log', html: '...' }         // top-5 log lines only
//                       { type: 'status', text: '...' }
//                       { type: 'done' }
// =============================================================================

// ── Reagent state (Welford online stats) ─────────────────────────────────────
const reagents = {};   // id → { mu, std, count, best, bestPartner, _M2, _prevCount }
let   top5Ids  = new Set();
let   top5Min  = 0;

function getTop5() {
    const entries = Object.entries(reagents);
    if (!entries.length) return new Set();
    entries.sort((a, b) => b[1].best - a[1].best);
    const top = entries.slice(0, 5);
    top5Min = top.length === 5 ? top[4][1].best : 0;
    return new Set(top.map(([id]) => id));
}

let top5Dirty = false;
let updatesSinceTop5 = 0;

function maybeRefreshTop5() {
    updatesSinceTop5++;
    if (updatesSinceTop5 >= 200) { top5Dirty = true; updatesSinceTop5 = 0; }
}

function updateReagent(id, score, wc, partner) {
    const n = wc ?? 1;
    if (!reagents[id]) {
        if (Object.keys(reagents).length > 8000 && top5Ids.size >= 5 && score < top5Min * 0.75) return;
        reagents[id] = { mu: score, std: 0, count: n, best: score, bestPartner: partner, _M2: 0, _prevCount: n };
        if (score > top5Min || top5Ids.size < 5) { top5Dirty = true; }
        postReagentUpdate(id);
        return;
    }
    const r = reagents[id];
    if (n > r._prevCount) {
        for (let i = 0; i < n - r._prevCount; i++) {
            r.count++;
            const delta = score - r.mu;
            r.mu += delta / r.count;
            r._M2 += delta * (score - r.mu);
        }
        r._prevCount = n;
        r.std = r.count >= 2 ? Math.sqrt(r._M2 / (r.count - 1)) : 0;
    }
    if (score > r.best) {
        r.best = score;
        if (partner) r.bestPartner = partner;
        if (score > top5Min || top5Ids.size < 5) top5Dirty = true;
    }
    maybeRefreshTop5();
    postReagentUpdate(id);
}

function postReagentUpdate(id) {
    const r = reagents[id];
    if (!r) return;
    self.postMessage({
        type: 'reagent_update',
        id, mu: r.mu, std: r.std, count: r.count,
        best: r.best, bestPartner: r.bestPartner,
    });
}

// ── Parser state ──────────────────────────────────────────────────────────────
let pendingScore    = null;
let pendingReagents = [];
let wuSeenForPair   = {};
let evalPartnerMap  = {};
let currentPhase    = 'Warmup 1';

// TS state
let pendingIter    = undefined;
let pendingIterScore = null;
let pendingIterSmiles = null;
let pendingWinnerIdx = null;
let pendingWinnerMu  = null;
let pendingMasked    = null;
let pendingUpdates   = [];
let pendingUpdateReagent = null;

// ── Iteration tracking: winner pairs → ts_event + batch_stats ─────────────
let _iterWinnerBuf  = [];    // pending winner objs; flushed when both cycle=0,1 arrive
let _iterIndex      = 0;     // auto-incrementing iteration counter
let _totalScored    = 0;     // cumulative iterations scored
const _prevNumScores = {};   // name → last seen num_scores (for batch delta)
// Rolling score buffer — window sized to eval_batch_size (updated via set_batch_size msg)
let   _BATCH_MAX    = 20;   // default; overridden when main thread sends set_batch_size
const _batchScores  = [];
let   _batchSum     = 0;
let   _batchSumSq   = 0;

function _pushBatchScore(score) {
    _totalScored++;
    while (_batchScores.length >= _BATCH_MAX) {
        const old = _batchScores.shift();
        _batchSum -= old; _batchSumSq -= old * old;
    }
    _batchScores.push(score);
    _batchSum += score; _batchSumSq += score * score;
    const n = _batchScores.length;
    const mean = _batchSum / n;
    const vari = n > 1 ? (_batchSumSq - _batchSum * _batchSum / n) / (n - 1) : 0;
    self.postMessage({ type: 'batch_stats', iter: _iterIndex, mean,
        std: Math.sqrt(Math.max(0, vari)), count: _totalScored, n, score });
}

// ── Top-5 reagent bars computed in the worker (off main thread) ──────────────
// The main thread used to sort/filter the whole reagent table every render,
// which (with many reagents) stalled the UI and made the cursor lag. Here we do
// that work in the worker and emit a small, pre-sorted, ready-to-render list.
let _barsEmitPending = false;
let _barsLastEmit    = 0;
function _emitBarsReady() {
    // Throttle to ~10 Hz — the bars don't need to update faster than the eye sees,
    // and this caps postMessage traffic regardless of how fast iterations stream.
    if (_barsEmitPending) return;
    const now = (self.performance && self.performance.now) ? self.performance.now() : Date.now();
    const sinceLast = now - _barsLastEmit;
    const doEmit = () => {
        _barsEmitPending = false;
        _barsLastEmit = (self.performance && self.performance.now) ? self.performance.now() : Date.now();
        const table = self._reagentTable || {};
        const arr = Object.values(table);
        if (!arr.length) { self.postMessage({ type: 'bars_ready', bars: [] }); return; }

        // ── Stable top-5 by BEST molecule score (monotonic) ───────────────────
        // Best-score per reagent only ever increases, so ranking by it is stable:
        // once a reagent produces a strong molecule it stays ranked and the list
        // stops churning. μ (posterior mean) fluctuates every update, which is
        // why ranking by μ made the list flip constantly.
        //
        // Reagents with no best score yet are ranked below all scored ones,
        // ordered by μ among themselves (so the list fills before any molecule
        // has been scored, e.g. very early in a run).
        //
        // Hysteresis: a reagent currently SHOWN stays unless a challenger beats
        // its best score by more than MARGIN — stops the lower slots flipping
        // between reagents with near-identical best scores.
        const MARGIN = 0.15;
        const rank = (r) => (r.best != null ? r.best : -Infinity);

        const scored   = arr.filter(r => r.best != null).sort((a, b) => b.best - a.best);
        const unscored = arr.filter(r => r.best == null).sort((a, b) => b.mu - a.mu);
        const ordered  = [...scored, ...unscored];   // natural ranking

        const prevShown = self._lastBarIds || [];
        // The challenger pool = top-5 of the natural ranking
        const naturalTop5 = ordered.slice(0, 5).map(r => r.name);

        let finalIds;
        if (!prevShown.length) {
            finalIds = naturalTop5;
        } else {
            const incumbents = prevShown.filter(id => table[id]).slice(0, 5);
            // Start by keeping all current incumbents.
            let keep = incumbents.slice();
            // A challenger (not currently shown) may DISPLACE the weakest incumbent
            // if it beats that incumbent's best score by more than MARGIN. We apply
            // displacements one at a time, weakest-incumbent-first, so a clearly
            // superior new reagent gets in while marginal ones don't cause churn.
            const notShown = ordered.filter(r => !keep.includes(r.name));
            for (const challenger of notShown) {
                if (keep.length < 5) { keep.push(challenger.name); continue; }
                // Find the weakest kept reagent by best score
                let weakestIdx = 0, weakestBest = Infinity;
                keep.forEach((id, i) => {
                    const v = rank(table[id]);
                    if (v < weakestBest) { weakestBest = v; weakestIdx = i; }
                });
                if (rank(challenger) > weakestBest + MARGIN) {
                    keep[weakestIdx] = challenger.name;   // displace
                } else {
                    break;   // ordered desc by best; nothing further can beat margin
                }
            }
            // Top up if any incumbents vanished
            for (const id of naturalTop5) {
                if (keep.length >= 5) break;
                if (!keep.includes(id)) keep.push(id);
            }
            finalIds = keep;
        }

        // Build the kept reagent objects, then display ordered by best score
        // (then μ) so positions are deterministic.
        const kept = finalIds.map(id => table[id]).filter(Boolean);
        kept.sort((a, b) => {
            const ra = rank(a), rb = rank(b);
            if (rb !== ra) return rb - ra;
            return b.mu - a.mu;
        });
        const top5 = kept.slice(0, 5);
        self._lastBarIds = top5.map(r => r.name);

        const maxMu = Math.max(...top5.map(r => r.mu), 0.001);
        const bars = top5.map(r => {
            const muPct   = +(r.mu / maxMu * 100).toFixed(1);
            const stdPct  = +Math.min((r.std / maxMu) * 100, 18).toFixed(1);
            const stdLeft = +Math.max(0, muPct - stdPct / 2).toFixed(1);
            return {
                id: r.id, name: r.name,
                mu: r.mu, std: r.std, sc: r.sc,
                best: r.best, bestPartner: r.bestPartner, delta_mu: r.delta_mu,
                muPct, stdPct, stdLeft,
            };
        });
        self.postMessage({ type: 'bars_ready', bars });
    };
    if (sinceLast >= 100) {
        doEmit();
    } else {
        _barsEmitPending = true;
        setTimeout(doEmit, 100 - sinceLast);
    }
}

// Pending cycle stats from "--- cycle_id=N | mu range: ..." lines.
// The mu/std range line is ONE per cycle (covering all reagents), not per slot,
// so we hold a single cycle-wide range and apply it to both slots when the
// winners for the cycle are emitted.
let _pendingCycleStats = {};
let _pendingCycleRange = null;

// ── Parse one line ────────────────────────────────────────────────────────────
function parseLine(rawLine) {
    // Strip Python logging prefix: "YYYY-MM-DD:HH:MM:SS,mmm LEVEL /path:lineno MESSAGE"
    const logPrefixM = rawLine.match(/^\d{4}-\d{2}-\d{2}[:\s][\d:,]+\s+\w+\s+\S+:\d+\s+(.*)/);
    const line = (logPrefixM ? logPrefixM[1] : rawLine).trim();
    if (!line) return;

    // ── AUTHORITATIVE per-iteration stats from the backend ────────────────────
    // The backend (ts_routes.py) parses the score, computes rolling mean/std, and
    // emits "[TS:stats] iter=N mean=X std=Y score=Z". We trust this completely —
    // it's the SAME value persisted to the session JSON, so the live chart and the
    // post-reload chart are guaranteed identical. This replaces the worker's own
    // score guessing (which could fall back to Thompson samples or posterior means
    // when its [evaluate] regex missed, producing wrong live numbers).
    // ── Engine-measured timing ────────────────────────────────────────────────
    // thompson_sampling.search() emits this every 25 iterations from its own
    // perf_counter accumulators. It is GROUND TRUTH: it is what the engine did,
    // measured inside the engine, unaffected by the Viz-speed slider, SSE
    // latency or browser render time — all of which the display-derived speed
    // sampler in ts_ui.js is subject to.
    //
    // Forwarded as-is; ts_ui.js's _tsSpeedTick prefers it when present and falls
    // back to the sampled display rate when it is absent (older engine, or a
    // run that has not reached iteration 25 yet).
    const timeM = line.match(
        /^\[TS:timing\]\s+iter=(\d+)\s+itps=([\d.eE+\-]+)\s+select_ms=([\d.eE+\-]+)\s+score_ms=([\d.eE+\-]+)\s+flush_ms=([\d.eE+\-]+)/);
    if (timeM) {
        self.postMessage({ type: 'engine_timing', payload: {
            iter:      parseInt(timeM[1], 10),
            itps:      parseFloat(timeM[2]),
            select_ms: parseFloat(timeM[3]),
            score_ms:  parseFloat(timeM[4]),
            flush_ms:  parseFloat(timeM[5]),
            t:         Date.now(),
        }});
        return;
    }

    const statsM = line.match(/^\[TS:stats\]\s+iter=(\d+)\s+mean=([\d.eE+\-]+)\s+std=([\d.eE+\-]+)\s+score=([\d.eE+\-]+)/);
    if (statsM) {
        const iter  = parseInt(statsM[1], 10);
        const mean  = parseFloat(statsM[2]);
        const std   = parseFloat(statsM[3]);
        const score = parseFloat(statsM[4]);
        _iterIndex   = iter;          // keep worker's counter in sync with backend
        _totalScored = iter;
        self._lastBackendScore = score;   // reuse in the ts_log line for consistency

        // Assign this authoritative molecule score as the 'best' for the winners
        // of the iteration that just produced it (stashed by the winner-pair flush).
        // This is the SAME value the backend persists to the JSON top-5, so the
        // live ranking and the JSON ranking stay identical. Thompson samples are
        // never used here.
        if (self._pendingScoreReagents && self._pendingScoreReagents.length) {
            if (!self._reagentBestScore) self._reagentBestScore = {};
            self._pendingScoreReagents.forEach(rname => {
                const prev = self._reagentBestScore[rname];
                if (prev == null || score > prev) self._reagentBestScore[rname] = score;
                if (self._reagentTable && self._reagentTable[rname]) {
                    const tb = self._reagentTable[rname].best;
                    if (tb == null || score > tb) self._reagentTable[rname].best = score;
                }
            });
            self._pendingScoreReagents = null;
            _emitBarsReady();   // re-rank now that real best scores are in
        }

        self.postMessage({ type: 'batch_stats', iter, mean, std, score });
        return;
    }

    // tqdm phase
    const phaseM = line.match(/Warmup\s+(\d+)\s+of\s+\d+\s+\[(\w+)\]/i);
    if (phaseM) {
        currentPhase = `Warmup ${phaseM[1]}`;
        self.postMessage({ type: 'phase', phase: currentPhase });
        const text = line.replace(/\|[^|]*$/, '').trim();
        self.postMessage({ type: 'status', text });
        return;
    }

    // [_flush_score_batch]
    const flushM = line.match(/\[_flush_score_batch\]\s+flushing scores for (\d+) unique reagents/i);
    if (flushM) {
        evalPartnerMap = {};
        self.postMessage({ type: 'status', text: `Flushing ${flushM[1]} reagents — ${currentPhase}` });
        return;
    }

    // [evaluate] score=X → ['reagentA', 'reagentB']
    // Also matches older format: [evaluate] score=X being added to N reagents: [...]
    const evalM = line.match(/\[evaluate\]\s+score=([\d.eE+\-]+)\s*(?:→|being added to \d+ reagents:)\s*\[(.+?)\]/);
    if (evalM) {
        pendingScore    = parseFloat(evalM[1]);
        pendingReagents = evalM[2].replace(/['"]/g,'').split(',').map(s=>s.trim());
        wuSeenForPair   = {};
        const sc = pendingScore;
        const [rA, rB] = pendingReagents;
        if (rA && rB) {
            evalPartnerMap[rA] = { partner: rB, score: sc };
            evalPartnerMap[rB] = { partner: rA, score: sc };
        }
        // Show in TS log when in TS inference phase (level >= 1)
        if (currentPhase.startsWith('TS') && _LOG_LEVEL >= 1) {
            self.postMessage({ type: 'ts_log', html:
                `&nbsp;&nbsp;<span style="color:#334155">[evaluate]</span> ` +
                `score=<span style="color:#34d399;font-weight:500">${pendingScore.toFixed(6)}</span> ` +
                `<span style="color:#1e3a5f">→ ['${pendingReagents.join("', '")}']</span>` });
        }
        // Refresh top5 before deciding log
        // Always refresh top5 before log filter check
        top5Ids = getTop5(); top5Dirty = false;
        // Show all lines until top5 is established; then filter to top5 only
        const _showLog = top5Ids.size < 5 || pendingReagents.some(r => top5Ids.has(r));
        if (_showLog) {
            self.postMessage({ type: 'log', html:
                `<span style="color:#334155">  [evaluate]</span> ` +
                `<span style="color:#94a3b8">score=</span><span style="color:#c4b5fd;font-weight:500">${pendingScore.toFixed(6)}</span> ` +
                `<span style="color:#334155">→</span> ` +
                `<span style="color:#475569">['${pendingReagents.join("', '")}']</span>`
            });
        }
        return;
    }

    // [add_score/warmup]
    const wuM = line.match(/\[add_score\/warmup\]\s+reagent=(\S+)\s+\|\s+buffered score=([\d.eE+\-]+)\s+\(warmup count so far:\s*(\d+)\)/);
    if (wuM) {
        const id    = wuM[1];
        const score = parseFloat(wuM[2]);
        const wc    = parseInt(wuM[3]);
        const pm    = evalPartnerMap[id];
        const partner = pm?.partner || pendingReagents.find(r => r !== id) || null;
        updateReagent(id, score, wc, partner);
        wuSeenForPair[id] = { score, wc };

        if (top5Ids.size < 5 || top5Ids.has(id)) {
            self.postMessage({ type: 'log', html:
                `&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4c1d95">[add_score/warmup]</span> ` +
                `<span style="color:#34d399">${id}</span> ` +
                `buf=<span style="color:#c4b5fd">${score.toFixed(4)}</span> ` +
                `n=<span style="color:#a78bfa">${wc}</span>` +
                (partner ? ` <span style="color:#1e3a5f">⊕${partner}</span>` : '')
            });
        }

        // Flush wu event when both pair members seen
        const expected = pendingReagents;
        const seen     = Object.keys(wuSeenForPair);
        const allSeen  = expected.length >= 2 ? expected.every(r => seen.includes(r)) : seen.length >= 2;
        if (allSeen) {
            const [k1, k2] = expected.length >= 2 ? expected : seen;
            self.postMessage({ type: 'wu_event', event: {
                phase: currentPhase, score: pendingScore ?? score,
                r1: k1, wc1: wuSeenForPair[k1]?.wc,
                r2: k2, wc2: wuSeenForPair[k2]?.wc,
            }});
            wuSeenForPair = {};
            pendingScore  = null;
            pendingReagents = [];
        }
        return;
    }

    // --- cycle_id block --- signals TS inference phase
    const cycleM = line.match(/---\s*cycle_id\s*\(component being selected\):\s*(\d+)\s*---/i);
    if (cycleM) {
        if (!currentPhase.startsWith('TS')) {
            currentPhase = 'TS inference';
            self.postMessage({ type: 'phase', phase: 'TS inference' });
            self.postMessage({ type: 'switch_tab', tab: 'ts' });
        }
        if (_DEBUG_LEVEL >= 2) self.postMessage({ type: 'debug', msg: '[worker] cycle_id detected, phase='+currentPhase });
        return;
    }

    // === iteration N result ===
    const iterM = line.match(/===\s*iteration\s+(\d+)\s+result\s*===/i);
    if (iterM) {
        pendingIter = parseInt(iterM[1]); pendingUpdates = [];
        self._cycleReagents = [];
        if (_DEBUG_LEVEL >= 2) self.postMessage({ type: 'debug', msg: '[worker] === iteration '+pendingIter+' ===' });
        self.postMessage({ type: 'ts_log', html:
            `<span style="color:#1e3a5f">─── </span>` +
            `<span style="color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:0.05em">iter&thinsp;</span>` +
            `<span style="color:#475569;font-family:monospace;font-weight:600">${pendingIter}</span>` +
            `<span style="color:#1e3a5f"> ───</span>` });
        return;
    }

    // score: X | smiles: Y
    const resM = line.match(/^score:\s*([\d.eE+\-]+)\s*\|\s*smiles:\s*(\S+)/i);
    if (resM) {
        pendingIterScore = parseFloat(resM[1]); pendingIterSmiles = resM[2];
        self.postMessage({ type: 'ts_log', html:
            `&nbsp;&nbsp;<span style="color:#334155">score=</span><span style="color:#34d399;font-weight:500">${pendingIterScore.toFixed(4)}</span> ` +
            `<span style="color:#1e3a5f;font-size:9px">${pendingIterSmiles?.substring(0,40)||''}</span>` });
        return;
    }

    // winner_idx
    const winM = line.match(/winner_idx:\s*(\d+)\s*\|\s*winner mu:\s*([\d.eE+\-]+)/i);
    if (winM) { pendingWinnerIdx = parseInt(winM[1]); pendingWinnerMu = parseFloat(winM[2]); return; }

    // [add_score] before: (verbose only)
    if (_LOG_LEVEL >= 2 && currentPhase.startsWith('TS')) {
        const beforeM = line.match(/\[add_score\]\s+before:\s+mu=([\d.eE+\-]+),\s+std=([\d.eE+\-]+)/i);
        if (beforeM) {
            self.postMessage({ type: 'ts_log', html:
                `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e293b">before:</span> ` +
                `μ=<span style="color:#475569">${parseFloat(beforeM[1]).toFixed(4)}</span> ` +
                `σ=${parseFloat(beforeM[2]).toFixed(4)}` });
            return;
        }
        const updateM = line.match(/\[_update_(mean|std)\]\s+(.+)/i);
        if (updateM) {
            self.postMessage({ type: 'ts_log', html:
                `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e293b">[${updateM[1]}]</span> ` +
                `<span style="color:#334155">${updateM[2].substring(0,80)}</span>` });
            return;
        }
    }

    // [add_score] reagent=ID | observed_score=X
    const rsM = line.match(/\[add_score\]\s+reagent=(\S+)\s+\|\s+observed_score=([\d.eE+\-]+)/i);
    if (rsM) {
        pendingUpdateReagent = rsM[1];
        const obs = parseFloat(rsM[2]);
        if (currentPhase.startsWith('TS') && _LOG_LEVEL >= 1) {
            self.postMessage({ type: 'ts_log', html:
                `&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e3a5f">[add_score]</span> ` +
                `<span style="color:#34d399">${rsM[1]}</span> ` +
                `<span style="color:#475569">score=</span><span style="color:#c4b5fd">${obs.toFixed(6)}</span>` });
        }
        return;
    }
    // fallback: [add_score] reagent=ID | (old format without observed_score)
    const rsM2 = line.match(/\[add_score\]\s+reagent=(\S+)\s+\|/);
    if (rsM2) { pendingUpdateReagent = rsM2[1]; return; }

    // [add_score] after:
    const afM = line.match(/\[add_score\]\s+after:\s+mu=([\d.eE+\-]+),\s+std=([\d.eE+\-]+)\s+\(delta_mu=([+\-][\d.eE+\-]+),\s+delta_std=([+\-][\d.eE+\-]+),\s+num_scores=(\d+)\)/i);
    if (afM && pendingUpdateReagent) {
        const delta_mu = parseFloat(afM[3]);
        const mu_after = parseFloat(afM[1]);
        const std_after= parseFloat(afM[2]);
        const n        = parseInt(afM[5]);
        const dmu      = parseFloat(afM[3]);
        pendingUpdates.push({
            id: pendingUpdateReagent, name: pendingUpdateReagent,
            mu_before: mu_after - delta_mu, mu_after,
            std_after, sc: n,
        });
        if (currentPhase.startsWith('TS') && _LOG_LEVEL >= 1) {
            const col = dmu >= 0 ? '#34d399' : '#f87171';
            self.postMessage({ type: 'ts_log', html:
                `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e3a5f">→</span> ` +
                `μ=<span style="color:#67e8f9">${mu_after.toFixed(4)}</span> ` +
                `σ=${std_after.toFixed(4)} ` +
                `<span style="color:${col}">(Δμ=${dmu >= 0 ? '+' : ''}${dmu.toFixed(4)})</span> ` +
                `n=${n}` });
        }
        pendingUpdateReagent = null;
        if (pendingIter !== undefined && pendingUpdates.length >= 2) {
            self.postMessage({ type: 'ts_event', event: {
                iter: pendingIter, score: pendingIterScore, smiles: pendingIterSmiles,
                winner_idx: pendingWinnerIdx, winner_mu: pendingWinnerMu,
                masked: pendingMasked, updates: [...pendingUpdates],
            }});
            pendingIter    = undefined;
            pendingUpdates = [];
        }
        return;
    }

    // post-update — real elion format: "post-update 37375446 μ=6.3021 σ=0.7219 n=3"
    // (id is the building-block number; μ/σ are Unicode; n is the score count)
    const puM = line.match(/post-update\s+(\S+)\s+[\u03bcu\u00b5]=([\d.eE+\-]+)\s+[\u03c3s]=([\d.eE+\-]+)\s+n=(\d+)/);
    if (puM) {
        const rid  = puM[1];
        const rmu  = parseFloat(puM[2]);
        const rstd = parseFloat(puM[3]);
        const rsc  = parseInt(puM[4]);

        // ── Per-slot μ/σ range tracking ───────────────────────────────────────
        // elion does NOT emit a "mu range: [X,Y]" summary line, so derive the
        // pool μ/σ ranges from the per-reagent post-update lines. Reagents within
        // a cycle alternate slot 0, slot 1 (same pairing the _cycleReagents logic
        // below relies on), so the slot for THIS post-update is the current length
        // of _cycleReagents (0 → slot 0, 1 → slot 1).
        const slot = (self._cycleReagents && self._cycleReagents.length) ? 1 : 0;
        if (!self._slotRanges) self._slotRanges = {0: null, 1: null};
        const sr = self._slotRanges[slot];
        if (!sr) {
            self._slotRanges[slot] = { muMin: rmu, muMax: rmu, stdMin: rstd, stdMax: rstd };
        } else {
            if (rmu  < sr.muMin)  sr.muMin  = rmu;
            if (rmu  > sr.muMax)  sr.muMax  = rmu;
            if (rstd < sr.stdMin) sr.stdMin = rstd;
            if (rstd > sr.stdMax) sr.stdMax = rstd;
        }
        // Emit the updated range for this slot so the sidebar populates live.
        const _r = self._slotRanges[slot];
        self.postMessage({ type: 'cycle_stats', slot,
            competitors: '—',
            muMin: _r.muMin, muMax: _r.muMax,
            stdMin: _r.stdMin, stdMax: _r.stdMax,
        });

        // ── Reagent state tracking in the WORKER (off main thread) ────────────
        // Maintain the full reagent table here; the main thread never sorts or
        // filters — it just renders the pre-computed top-5 we send below.
        if (!self._reagentTable) self._reagentTable = {};
        const prev = self._reagentTable[rid];
        const prevMu = prev ? prev.mu : rmu;
        self._reagentTable[rid] = {
            id: rid, name: rid, mu: rmu, std: rstd, sc: rsc,
            delta_mu: rmu - prevMu,
            best: (self._reagentBestScore || {})[rid] ?? (prev ? prev.best : null),
            bestPartner: (self._cyclePartner && self._cyclePartner[rid]) ||
                         (prev ? prev.bestPartner : '—'),
        };

        // Link partners within a cycle
        if (!self._cycleReagents) self._cycleReagents = [];
        self._cycleReagents.push(rid);
        if (self._cycleReagents.length >= 2) {
            const [a, b] = self._cycleReagents;
            self._cycleReagents = [];
            if (!self._cyclePartner) self._cyclePartner = {};
            self._cyclePartner[a] = b;
            self._cyclePartner[b] = a;
            if (self._reagentTable[a]) self._reagentTable[a].bestPartner = b;
            if (self._reagentTable[b]) self._reagentTable[b].bestPartner = a;
        }

        // Compute and emit the sorted top-5 — ONE message, ready to render.
        _emitBarsReady();

        self.postMessage({ type: 'ts_log', html:
            `&nbsp;&nbsp;<span style="color:#0e7490">post-update</span> ` +
            `<span style="color:#34d399">${rid}</span> ` +
            `μ=<span style="color:#67e8f9">${rmu.toFixed(4)}</span> ` +
            `σ=${rstd.toFixed(4)} n=${rsc}` });

        // Push to pendingUpdates for the iteration event
        if (pendingIter !== undefined) {
            pendingUpdates.push({ id: rid, name: rid,
                mu_before: rmu, mu_after: rmu, std_after: rstd, sc: rsc });
            if (pendingUpdates.length >= 2) {
                self.postMessage({ type: 'ts_event', event: {
                    iter: pendingIter, score: pendingIterScore, smiles: pendingIterSmiles,
                    winner_idx: pendingWinnerIdx, winner_mu: pendingWinnerMu,
                    masked: pendingMasked, updates: [...pendingUpdates],
                }});
                pendingIter = undefined; pendingUpdates = [];
            }
        }
        return;
    }

    // winner | cycle_id=N | reagent=NAME | sampled=X | mu=Y | std=Z | num_scores=W
    const winnerLineM = line.match(/winner\s*\|\s*cycle_id=(\d+)\s*\|\s*reagent=(\S+)\s*\|\s*sampled=([\d.eE+\-]+)\s*\|\s*mu=([\d.eE+\-]+)\s*\|\s*std=([\d.eE+\-]+)\s*\|\s*num_scores=(\d+)/i);
    if (winnerLineM) {
        const cycle=parseInt(winnerLineM[1]), name=winnerLineM[2],
              sampled=parseFloat(winnerLineM[3]), mu=parseFloat(winnerLineM[4]),
              std=parseFloat(winnerLineM[5]), sc=parseInt(winnerLineM[6]);
        const prevSc=_prevNumScores[name]??0;
        _prevNumScores[name]=sc;
        // Track best sampled score per reagent (for bar display)
        if (!self._cycleWinnerSampled) self._cycleWinnerSampled = {};
        if (self._cycleWinnerSampled[name] == null || sampled > self._cycleWinnerSampled[name]) {
            self._cycleWinnerSampled[name] = sampled;
        }
        _iterWinnerBuf.push({cycle,name,sampled,mu,std,sc});

        // Populate the reagent table from winner lines as well. Some elion runs
        // surface reagent μ/σ primarily through "winner" lines rather than (or in
        // addition to) "post-update" lines; without this the bars stay empty.
        if (!self._reagentTable) self._reagentTable = {};
        {
            const prev = self._reagentTable[name];
            const prevMu = prev ? prev.mu : mu;
            self._reagentTable[name] = {
                id: name, name: name, mu: mu, std: std, sc: sc,
                delta_mu: mu - prevMu,
                best: (self._reagentBestScore || {})[name] ?? (prev ? prev.best : null),
                bestPartner: (self._cyclePartner && self._cyclePartner[name]) ||
                             (prev ? prev.bestPartner : '—'),
            };
        }
        _emitBarsReady();

        self.postMessage({ type: 'ts_log', html:
            `<span style="color:#fbbf24">winner</span> cycle=${cycle} ` +
            `<span style="color:#34d399">${name}</span> ` +
            `<span style="color:#334155" title="Thompson sample: random draw from this reagent's posterior N(μ,σ) used for selection — NOT the molecule score">TS-draw=</span><span style="color:#67e8f9;font-weight:500">${sampled.toFixed(4)}</span> ` +
            `μ=${mu.toFixed(4)} σ=${std.toFixed(4)} n=${sc}` });

        // When both cycle=0 and cycle=1 have arrived, emit iteration events
        if (_iterWinnerBuf.some(w=>w.cycle===0) && _iterWinnerBuf.some(w=>w.cycle===1)) {
            _iterIndex++;
            const w0=_iterWinnerBuf.find(w=>w.cycle===0);
            const w1=_iterWinnerBuf.find(w=>w.cycle===1);
            _iterWinnerBuf=[];

            // The two cycle winners reacted together this iteration → they are
            // each other's partner. Record it so the bars show "reacts with <id>".
            if (!self._cyclePartner) self._cyclePartner = {};
            self._cyclePartner[w0.name] = w1.name;
            self._cyclePartner[w1.name] = w0.name;
            if (self._reagentTable) {
                if (self._reagentTable[w0.name]) self._reagentTable[w0.name].bestPartner = w1.name;
                if (self._reagentTable[w1.name]) self._reagentTable[w1.name].bestPartner = w0.name;
            }

            // Actual molecule score — try sources in priority order:
            // 1. pendingScore: from [evaluate] score=X line (most reliable, INFO level)
            // 2. pendingIterScore: from "score: X | smiles:" line (DEBUG level)
            // 3. w1.sampled: Thompson sample (fallback, can be unrealistically high)
            let actualMolScore = null;
            if (pendingScore != null && pendingScore > 0 && pendingScore < 15) {
                actualMolScore = pendingScore;
            } else if (pendingIterScore != null && pendingIterScore > 0 && pendingIterScore < 15) {
                actualMolScore = pendingIterScore;
            } else if (w1.sampled > 0 && w1.sampled < 15) {
                actualMolScore = w1.sampled;
            } else {
                actualMolScore = Math.max(w0.mu, w1.mu); // last resort: best posterior mean
            }
            // Reset for next iteration
            pendingIterScore = null;
            pendingScore     = null;

            // Always show evaluated score in log (all levels) — include iter for verification.
            // Prefer the backend's authoritative score (self._lastBackendScore) so the log
            // line, the chart, and the persisted JSON all show the same number.
            const logScore = (self._lastBackendScore != null) ? self._lastBackendScore : actualMolScore;
            self.postMessage({ type: 'ts_log', html:
                `<span style="color:#1e3a5f;font-size:9px">iter&thinsp;</span>` +
                `<span style="color:#334155;font-family:monospace;font-size:9px">${_iterIndex}</span>` +
                `&nbsp;<span style="color:#475569">eval score</span> ` +
                `<span style="color:#34d399;font-weight:600;font-size:11px">${logScore.toFixed(4)}</span> ` +
                `<span style="color:#1e3a5f">← ${w0.name} ⊕ ${w1.name}</span>` });

            // Defer best-score assignment: the molecule score for THIS iteration's
            // winners arrives on the [TS:stats] line that comes AFTER these winner
            // lines. Remember the pending winners so [TS:stats] can assign the real
            // (authoritative) score to them — never a Thompson sample.
            self._pendingScoreReagents = [w0.name, w1.name];
            _emitBarsReady();
            // Emit cycle_stats per slot. The mu/std range is cycle-wide (one
            // range per cycle covering all reagents), so apply it to both slots.
            if (_pendingCycleRange) {
                [0, 1].forEach(slot => {
                    self.postMessage({ type: 'cycle_stats', slot,
                        competitors: _pendingCycleRange.competitors ?? '—',
                        muMin: _pendingCycleRange.muMin, muMax: _pendingCycleRange.muMax,
                        stdMin: _pendingCycleRange.stdMin, stdMax: _pendingCycleRange.stdMax,
                    });
                });
            }
            _pendingCycleStats = {};
            _pendingCycleRange = null;
            // NOTE: do NOT call _pushBatchScore here. The chart's batch_stats now
            // comes exclusively from the backend's authoritative [TS:stats] line
            // (handled at the top of parseLine), so the live chart matches the
            // persisted/reloaded chart exactly. This block only drives the log
            // panel (ts_log) and the per-iteration event detail (ts_event).
            self.postMessage({ type: 'ts_event', event: {
                iter: _iterIndex, score: actualMolScore, smiles: `${w0.name}+${w1.name}`,
                winner_idx: null, winner_mu: w0.mu, masked: pendingMasked,
                updates: [
                    {id:w0.name, name:w0.name, mu_before:w0.mu, mu_after:w0.mu, std_after:w0.std, sc:w0.sc,
                     best: (self._reagentBestScore || {})[w0.name]},
                    {id:w1.name, name:w1.name, mu_before:w1.mu, mu_after:w1.mu, std_after:w1.std, sc:w1.sc,
                     best: (self._reagentBestScore || {})[w1.name]},
                ],
            }});
            pendingIterScore = null;
        }
        return;
    }

    // disallow_mask
    const maskM = line.match(/disallow_mask[^:]*:\s*\{([^}]{1,4000})\}/);
    if (maskM) { pendingMasked = (maskM[1].match(/np\.int64/g)||[]).length; return; }

    // --- cycle_id=N | mu range: [X,Y], std range: [A,B] | competitors=N
    const muRangeM = line.match(/---\s*cycle_id=(\d+)\s*\|\s*mu range:\s*\[([\d.eE+\-]+),\s*([\d.eE+\-]+)\],?\s*std range:\s*\[([\d.eE+\-]+),\s*([\d.eE+\-]+)\](?:\s*\|\s*competitors=(\d+))?/i);
    if (muRangeM) {
        const cid = parseInt(muRangeM[1]);
        if (!currentPhase.startsWith('TS')) {
            currentPhase = 'TS inference';
            self.postMessage({ type: 'phase', phase: 'TS inference' });
            self.postMessage({ type: 'switch_tab', tab: 'ts' });
        }
        _pendingCycleRange = {
            muMin: parseFloat(muRangeM[2]), muMax: parseFloat(muRangeM[3]),
            stdMin: parseFloat(muRangeM[4]), stdMax: parseFloat(muRangeM[5]),
            competitors: muRangeM[6] ? parseInt(muRangeM[6]) : null,
        };
        self.postMessage({ type: 'status', text: `μ range [${muRangeM[2]}, ${muRangeM[3]}]` });
        return;
    }
    // Fallback: old format without std range
    const muRangeOldM = line.match(/mu range:\s*\[([\d.]+),\s*([\d.]+)\]/i);
    if (muRangeOldM) {
        if (!currentPhase.startsWith('TS')) {
            currentPhase = 'TS inference';
            self.postMessage({ type: 'phase', phase: 'TS inference' });
            self.postMessage({ type: 'switch_tab', tab: 'ts' });
        }
        self.postMessage({ type: 'status', text: `μ range [${muRangeOldM[1]}, ${muRangeOldM[2]}]` });
        return;
    }

    // Send ALL lines to raw log — gated by debug level
    if (_DEBUG_LEVEL >= 2) {
        self.postMessage({ type: 'debug', msg: '[worker RAW] ' + line.substring(0, 300) });
    } else if (_DEBUG_LEVEL >= 1 && (line.includes('ERROR') || line.includes('Traceback') || line.includes('ValueError'))) {
        self.postMessage({ type: 'debug', msg: '[worker RAW] ' + line.substring(0, 300) });
    }

    // Filter debug/info log lines from status bar
    const isDebug = /^\d{4}-\d{2}-\d{2}/.test(line) || line.includes(' DEBUG ') || line.includes(' INFO ');
    if (!isDebug && line.length < 300 && !line.startsWith('[_update') && !line.startsWith('  [')) {
        self.postMessage({ type: 'status', text: line });
    }
}

// ── Log level control ────────────────────────────────────────────────────────
// 0 = minimal (winner + post-update only)
// 1 = normal  (+ evaluate + add_score after)
// 2 = verbose (+ full Bayesian math)
let _LOG_LEVEL = 0;

// ── Debug level control ──────────────────────────────────────────────────────
// 0 = off (no [worker RAW] spam), 1 = errors only, 2 = all raw lines
// Set via message: { type: 'set_debug', level: 0|1|2 }
let _DEBUG_LEVEL = 0;   // default: silent
console.log('[ts_worker.js] worker started, version:', Date.now());
self.onmessage = (evt) => {
    const msg = evt.data;
    if (msg.type === 'set_iter_offset') {
        _iterIndex = msg.offset ?? 0;
        // Pre-fill the rolling batch window with the last N scores from history
        // so the first live batch_stats message continues the rolling mean smoothly
        const seeds = msg.seedScores || [];
        if (seeds.length > 0) {
            _batchScores.length = 0;
            _batchSum = 0; _batchSumSq = 0;
            const take = seeds.slice(-_BATCH_MAX);
            for (const s of take) {
                _batchScores.push(s);
                _batchSum   += s;
                _batchSumSq += s * s;
            }
            _totalScored = _iterIndex;
            console.log('[worker] batch window pre-filled with', take.length, 'scores, mean=',
                (_batchSum / take.length).toFixed(4));
        }
        return;
    }
    if (msg.type === 'set_log_level') {
        _LOG_LEVEL = msg.level ?? 0;
        return;
    }
    if (msg.type === 'set_debug') {
        _DEBUG_LEVEL = msg.level ?? 0;
        return;
    }
    if (msg.type === 'seed_reagents') {
        // Restore the reagent table from persisted JSON on reconnect so the
        // live bars continue from the saved top-5 (best scores) instead of
        // starting empty and dropping the restored high-scorers until they
        // win again. Seeds _reagentTable, _reagentBestScore, _cyclePartner,
        // and _lastBarIds so the first live emit matches the restored list.
        if (!self._reagentTable)   self._reagentTable = {};
        if (!self._reagentBestScore) self._reagentBestScore = {};
        if (!self._cyclePartner)   self._cyclePartner = {};
        (msg.reagents || []).forEach(r => {
            self._reagentTable[r.name] = {
                id: r.name, name: r.name, mu: r.mu, std: r.std, sc: r.sc,
                delta_mu: 0, best: (r.best != null ? r.best : null),
                bestPartner: r.partner || '—',
            };
            if (r.best != null) self._reagentBestScore[r.name] = r.best;
            if (r.partner) self._cyclePartner[r.name] = r.partner;
        });
        if (Array.isArray(msg.top5Ids)) self._lastBarIds = msg.top5Ids.slice();
        _emitBarsReady();
        return;
    }
    if (msg.type === 'set_batch_size') {
        _BATCH_MAX = Math.max(1, parseInt(msg.batchSize) || 20);
        return;
    }
    if (msg.type === 'reset') {
        Object.keys(reagents).forEach(k => delete reagents[k]);
        top5Ids = new Set(); top5Min = 0; top5Dirty = false; updatesSinceTop5 = 0;
        pendingScore = null; pendingReagents = []; wuSeenForPair = {}; evalPartnerMap = {};
        currentPhase = 'Warmup 1';
        pendingIter = undefined; pendingIterScore = null; pendingIterSmiles = null;
        pendingWinnerIdx = null; pendingWinnerMu = null; pendingMasked = null;
        pendingUpdates = []; pendingUpdateReagent = null;
        _iterWinnerBuf=[]; _iterIndex=0; _totalScored=0;
        Object.keys(_prevNumScores).forEach(k=>delete _prevNumScores[k]);
        _batchScores.length=0; _batchSum=0; _batchSumSq=0;
        _pendingCycleStats={};
        _pendingCycleRange=null;
        self._cycleReagents=[];
        self._reagentTable={};
        self._slotRanges={0: null, 1: null};
        return;
    }
    if (msg.type === 'line') {
        parseLine(msg.data);
    }
};