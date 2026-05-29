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

// ── Parse one line ────────────────────────────────────────────────────────────
function parseLine(rawLine) {
    // Strip Python logging prefix: "YYYY-MM-DD:HH:MM:SS,mmm LEVEL /path:lineno MESSAGE"
    const logPrefixM = rawLine.match(/^\d{4}-\d{2}-\d{2}[:\s][\d:,]+\s+\w+\s+\S+:\d+\s+(.*)/);
    const line = (logPrefixM ? logPrefixM[1] : rawLine).trim();
    if (!line) return;

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

    // [evaluate]
    const evalM = line.match(/\[evaluate\]\s+score=([\d.eE+\-]+)\s+being added to \d+ reagents:\s*\[(.+?)\]/);
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
        self.postMessage({ type: 'debug', msg: '[worker] cycle_id detected, phase='+currentPhase });
        return;
    }

    // === iteration N result ===
    const iterM = line.match(/===\s*iteration\s+(\d+)\s+result\s*===/i);
    if (iterM) {
        pendingIter = parseInt(iterM[1]); pendingUpdates = [];
        self._cycleReagents = [];
        self.postMessage({ type: 'debug', msg: '[worker] === iteration '+pendingIter+' ===' });
        self.postMessage({ type: 'ts_log', html:
            `<span style="color:#1e3a5f">═══ iter </span><span style="color:#67e8f9;font-weight:500">${pendingIter}</span>` });
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

    // [add_score] reagent=ID |
    const rsM = line.match(/\[add_score\]\s+reagent=(\S+)\s+\|/);
    if (rsM) { pendingUpdateReagent = rsM[1]; return; }

    // [add_score] after:
    const afM = line.match(/\[add_score\]\s+after:\s+mu=([\d.eE+\-]+),\s+std=([\d.eE+\-]+)\s+\(delta_mu=([+\-][\d.eE+\-]+),\s+delta_std=([+\-][\d.eE+\-]+),\s+num_scores=(\d+)\)/i);
    if (afM && pendingUpdateReagent) {
        const delta_mu = parseFloat(afM[3]);
        pendingUpdates.push({
            id: pendingUpdateReagent, name: pendingUpdateReagent,
            mu_before: parseFloat(afM[1]) - delta_mu, mu_after: parseFloat(afM[1]),
            std_after: parseFloat(afM[2]), sc: parseInt(afM[5]),
        });
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

    // post-update (INFO level TS) — real format: "post-update | comp=1, reagent=793 (44813781): mu=7.82..."
    const puM = line.match(/post-update\s*\|\s*comp(?:onent)?[=\s]+\d+[,\s]+reagent[=\s]+(\d+)\s+\((\S+)\):\s*mu=([\d.eE+\-]+),\s*std=([\d.eE+\-]+),\s*num_scores=(\d+)/i);
    if (puM) {
        // Always send ts_reagent and log — unconditional bar update
        // Track cycle's reagents to link as partners
        if (!self._cycleReagents) self._cycleReagents = [];
        self._cycleReagents.push({ id: puM[1], name: puM[2], mu: parseFloat(puM[3]), std: parseFloat(puM[4]), sc: parseInt(puM[5]) });
        // When we have both components of the pair, link them as partners and send
        if (self._cycleReagents.length >= 2) {
            const [rA, rB] = self._cycleReagents;
            self._cycleReagents = [];
            self.postMessage({ type: 'ts_reagent', id: rA.id, name: rA.name,
                mu: rA.mu, std: rA.std, sc: rA.sc, bestPartner: rB.name });
            self.postMessage({ type: 'ts_reagent', id: rB.id, name: rB.name,
                mu: rB.mu, std: rB.std, sc: rB.sc, bestPartner: rA.name });
        }
        self.postMessage({ type: 'debug', msg: '[worker] ts_reagent sent: '+puM[2]+' mu='+puM[3] });
        self.postMessage({ type: 'ts_log', html:
            `&nbsp;&nbsp;<span style="color:#0e7490">post-update</span> ` +
            `<span style="color:#34d399">${puM[2]}</span> ` +
            `μ=<span style="color:#67e8f9">${parseFloat(puM[3]).toFixed(4)}</span> ` +
            `σ=${parseFloat(puM[4]).toFixed(4)} n=${puM[5]}` });
        // Only push to pendingUpdates if we have an active iteration
        if (pendingIter !== undefined) {
            pendingUpdates.push({ id: puM[1], name: puM[2],
                mu_before: parseFloat(puM[3]), mu_after: parseFloat(puM[3]),
                std_after: parseFloat(puM[4]), sc: parseInt(puM[5]) });
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

    // winner | cycle_id=N | reagent=NAME | sampled=X | mu=Y | std=Z | num_scores=N
    const winnerLineM = line.match(/winner\s*\|\s*cycle_id=(\d+)\s*\|\s*reagent=(\S+)\s*\|\s*sampled=([\d.eE+\-]+)\s*\|\s*mu=([\d.eE+\-]+)\s*\|\s*std=([\d.eE+\-]+)\s*\|\s*num_scores=(\d+)/i);
    if (winnerLineM) {
        self.postMessage({ type: 'ts_log', html:
            `<span style="color:#fbbf24">winner</span> ` +
            `cycle=${winnerLineM[1]} ` +
            `<span style="color:#34d399">${winnerLineM[2]}</span> ` +
            `sampled=<span style="color:#67e8f9;font-weight:500">${parseFloat(winnerLineM[3]).toFixed(4)}</span> ` +
            `μ=${parseFloat(winnerLineM[4]).toFixed(4)} σ=${parseFloat(winnerLineM[5]).toFixed(4)} n=${winnerLineM[6]}`
        });
        return;
    }

    // disallow_mask
    const maskM = line.match(/disallow_mask[^:]*:\s*\{([^}]{1,4000})\}/);
    if (maskM) { pendingMasked = (maskM[1].match(/np\.int64/g)||[]).length; return; }

    // mu range — appears during TS inference cycles (warmup is finished)
    const muRangeM = line.match(/mu range:\s*\[([\d.]+),\s*([\d.]+)\]/i);
    if (muRangeM) {
        // Signal transition to TS phase if not already there
        if (!currentPhase.startsWith('TS')) {
            currentPhase = 'TS inference';
            self.postMessage({ type: 'phase', phase: 'TS inference' });
            self.postMessage({ type: 'switch_tab', tab: 'ts' });
        }
        self.postMessage({ type: 'status', text: `μ range [${muRangeM[1]}, ${muRangeM[2]}]` });
        return;
    }

    // Filter debug/info log lines from status bar
    const isDebug = /^\d{4}-\d{2}-\d{2}/.test(line) || line.includes(' DEBUG ') || line.includes(' INFO ');
    if (!isDebug && line.length < 300 && !line.startsWith('[_update') && !line.startsWith('  [')) {
        self.postMessage({ type: 'status', text: line });
    }
    // DEBUG: log every line to find what's actually arriving
    if (line.includes('cycle_id') || line.includes('post-update') || line.includes('iteration') || line.includes('flush_score')) {
        self.postMessage({ type: 'debug', msg: '[worker RAW] ' + line.substring(0, 120) });
    }
}

// ── Message handler ───────────────────────────────────────────────────────────
console.log('[ts_worker.js] worker started, version:', Date.now());
self.onmessage = (evt) => {
    const msg = evt.data;
    if (msg.type === 'reset') {
        Object.keys(reagents).forEach(k => delete reagents[k]);
        top5Ids = new Set(); top5Min = 0; top5Dirty = false; updatesSinceTop5 = 0;
        pendingScore = null; pendingReagents = []; wuSeenForPair = {}; evalPartnerMap = {};
        currentPhase = 'Warmup 1';
        pendingIter = undefined; pendingIterScore = null; pendingIterSmiles = null;
        pendingWinnerIdx = null; pendingWinnerMu = null; pendingMasked = null;
        pendingUpdates = []; pendingUpdateReagent = null;
        return;
    }
    if (msg.type === 'line') {
        parseLine(msg.data);
    }
};