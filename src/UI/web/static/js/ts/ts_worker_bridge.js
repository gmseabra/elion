// =============================================================================
// ts/ts_worker_bridge.js — worker message routing and SSE stream management
// Depends on: ts_core.js, ts_chart.js, ts_ui.js, ts_warmup.js
// =============================================================================

// ── Render throttle ───────────────────────────────────────────────────────
let _tsRenderQueued = false;
function _tsQueueRender() {
    if (_tsRenderQueued) return;
    _tsRenderQueued = true;
    requestAnimationFrame(() => { _tsRenderQueued = false; _tsRenderWuBars(); });
}

// ── Worker message handler ────────────────────────────────────────────────
let _tsRenderCount = 0;
function _tsHandleWorkerMsg(msg) {
    switch (msg.type) {
        case 'reagent_update':
            if ((msg._jobIdx ?? 0) === (_ts._activeJobIdx ?? 0)) {
                _ts.reagents[msg.id] = { mu: msg.mu, std: msg.std, count: msg.count, best: msg.best, bestPartner: msg.bestPartner };
                if ((++_tsRenderCount) % 50 === 0) _tsQueueRender();
            }
            break;

        case 'wu_event': {
            const widx = msg._jobIdx ?? 0;
            if (widx === 0) {
                _ts.wuState.events.push(msg.event);
            } else {
                if (!_ts._wuJobEvals) _ts._wuJobEvals = {};
                _ts._wuJobEvals[widx] = (_ts._wuJobEvals[widx] || 0) + 1;
                const ev  = msg.event;
                const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
                set(`tsWuEvals_${widx}`, _ts._wuJobEvals[widx]);
                if (ev.score && ev.score > (_ts._wuJobBest?.[widx] || 0)) {
                    if (!_ts._wuJobBest) _ts._wuJobBest = {};
                    _ts._wuJobBest[widx] = ev.score;
                    set(`tsWuBest_${widx}`, ev.score.toFixed(4));
                }
            }
            break;
        }

        case 'ts_event': {
            const ji = msg._jobIdx ?? 0;
            if (!_ts._jobStates) _ts._jobStates = {};
            if (!_ts._jobStates[ji]) _ts._jobStates[ji] = _tsInitTs();
            _ts._jobStates[ji].events.push(msg.event);
            if (msg.event.score !== undefined) {
                if (!_ts._jobStates[ji].winners) _ts._jobStates[ji].winners = [];
                _ts._jobStates[ji].winners.push({ iter: msg.event.iter, score: msg.event.score, smiles: msg.event.smiles });
            }
            const actualScore = msg.event.score;
            if (actualScore != null && msg.event.updates && ji === (_ts._activeJobIdx ?? 0)) {
                msg.event.updates.forEach(u => {
                    if (_ts.tsState?.reagents[u.id]) {
                        const newBest = u.best ?? actualScore;
                        const prev    = _ts.tsState.reagents[u.id].best;
                        if (prev == null || newBest > prev) _ts.tsState.reagents[u.id].best = newBest;
                    }
                });
            }
            if (ji === (_ts._activeJobIdx ?? 0)) {
                _ts.tsState.events.push(msg.event);
                if (_ts.activeTab === 'warmup') _tsTab('ts');
                if (!_ts._tsDrainQueued) {
                    _ts._tsDrainQueued = true;
                    requestAnimationFrame(function drain() {
                        const deadline = performance.now() + 8;
                        while (_ts.tsState && _ts.tsState.idx < _ts.tsState.events.length && performance.now() < deadline) _tsTsStep();
                        if (_ts.tsState && _ts.tsState.idx < _ts.tsState.events.length) {
                            requestAnimationFrame(drain);
                        } else {
                            _ts._tsDrainQueued = false;
                            // Trim fully-processed events to prevent unbounded memory growth.
                            // At 900 iters × ~3 events each = 2700 objects sitting in RAM.
                            // Keep the last 50 as a safety margin; reset idx to match.
                            if (_ts.tsState && _ts.tsState.idx > 100) {
                                const keep = 50;
                                const trim = Math.max(0, _ts.tsState.idx - keep);
                                _ts.tsState.events.splice(0, trim);
                                _ts.tsState.idx = Math.max(0, _ts.tsState.idx - trim);
                            }
                        }
                    });
                }
            }
            break;
        }

        case 'ts_reagent':
            // Legacy per-reagent message — still update the table for the active job
            // so reconnect/state stays consistent, but bar RENDERING is now driven
            // by 'bars_ready' (pre-sorted in the worker). Kept for compatibility.
            if ((msg._jobIdx ?? 0) === (_ts._activeJobIdx ?? 0) && _ts.tsState) {
                const prev     = _ts.tsState.reagents[msg.id]?.mu ?? msg.mu;
                const prevBest = _ts.tsState.reagents[msg.id]?.best;
                const newBest  = msg.best != null
                    ? (prevBest != null ? Math.max(prevBest, msg.best) : msg.best)
                    : prevBest;
                _ts.tsState.reagents[msg.id] = {
                    name: msg.name, mu: msg.mu, std: msg.std, sc: msg.sc,
                    delta_mu: msg.mu - prev,
                    best: newBest,
                    bestPartner: msg.bestPartner || _ts.tsState.reagents[msg.id]?.bestPartner || '—',
                };
            }
            break;

        case 'bars_ready': {
            // DISABLED as the bar driver. The reagent bars now come from the
            // 5-second /ts_top5 poll (backend's authoritative best-score ranking),
            // not from the worker. We ignore worker-computed bars to avoid the two
            // sources fighting (the worker's live ranking could momentarily differ
            // from the persisted JSON). The worker still maintains its table for
            // other purposes; we simply don't render from it here.
            break;
        }

        case 'cycle_stats': {
            const set    = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
            const slot   = msg.slot;
            const ji     = msg._jobIdx ?? 0;
            const suffix = _ts.jobs?.length > 1 ? `_j${ji}` : '';
            const muId   = slot === 0 ? `tsTsPool0MuRange${suffix}` : `tsTsPool1MuRange${suffix}`;
            const stdId  = slot === 0 ? `tsTsPool0StdRange${suffix}` : `tsTsPool1StdRange${suffix}`;
            // NOTE: the pool BB-count span (tsTsPoolN) is intentionally NOT
            // updated here. It used to show msg.competitors — elion's LIVE
            // eligible count, which shrinks as the DisallowTracker masks
            // reagents and so flickers across iterations. The sidebar now shows
            // the STATIC raw CSV count instead (set once via _tsFetchPoolCounts
            // in ts_ui.js). We still update the μ/σ posterior ranges per cycle.
            set(muId,   isFinite(msg.muMin)  ? `${msg.muMin.toFixed(3)} – ${msg.muMax.toFixed(3)}`  : '—');
            set(stdId,  isFinite(msg.stdMin) ? `${msg.stdMin.toFixed(3)} – ${msg.stdMax.toFixed(3)}` : '—');
            if (ji === 0) {
                set(slot === 0 ? 'tsTsPool0MuRange' : 'tsTsPool1MuRange', isFinite(msg.muMin)  ? `${msg.muMin.toFixed(3)} – ${msg.muMax.toFixed(3)}`  : '—');
                set(slot === 0 ? 'tsTsPool0StdRange' : 'tsTsPool1StdRange', isFinite(msg.stdMin) ? `${msg.stdMin.toFixed(3)} – ${msg.stdMax.toFixed(3)}` : '—');
            }
            break;
        }

        case 'batch_stats': {
            if ((msg._jobIdx ?? 0) !== (_ts._activeJobIdx ?? 0)) {
                const ji = msg._jobIdx ?? 0;
                _tsSparkPush(msg.iter, msg.mean, msg.std, ji, msg.score);
                if (!_ts._jobBatchBest)  _ts._jobBatchBest  = {};
                if (!_ts._jobBatchWorst) _ts._jobBatchWorst = {};
                if (_ts._jobBatchBest[ji]  === undefined || msg.mean > _ts._jobBatchBest[ji])  _ts._jobBatchBest[ji]  = msg.mean;
                if (_ts._jobBatchWorst[ji] === undefined || msg.mean < _ts._jobBatchWorst[ji]) _ts._jobBatchWorst[ji] = msg.mean;
                break;
            }
            const ji  = msg._jobIdx ?? 0;
            const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
            set('tsTsIter',    String(msg.iter));
            set('tsBatchMean', msg.mean.toFixed(4));
            set('tsBatchStd',  msg.std.toFixed(4));
            set('tsBatchMean2', msg.mean.toFixed(4));
            set('tsBatchStd2',  msg.std.toFixed(4));
            if (_ts.evalBatchSize) {
                ['tsBatchN','tsBatchN2'].forEach(id => set(id, String(_ts.evalBatchSize)));
            }
            _tsSparkPush(msg.iter, msg.mean, msg.std, ji, msg.score);
            _tsQueueSparkline();
            if (_ts._batchBest  === undefined || msg.mean > _ts._batchBest)  _ts._batchBest  = msg.mean;
            if (_ts._batchWorst === undefined || msg.mean < _ts._batchWorst) _ts._batchWorst = msg.mean;
            if (!_ts._jobBatchBest)  _ts._jobBatchBest  = {};
            if (!_ts._jobBatchWorst) _ts._jobBatchWorst = {};
            _ts._jobBatchBest[ji]  = _ts._batchBest;
            _ts._jobBatchWorst[ji] = _ts._batchWorst;
            break;
        }

        case 'debug':
            console.log(`[job${msg._jobIdx ?? 0}] ${msg.msg}`);
            _tsRawLog(msg.msg);
            break;

        case 'ts_log': {
            // In 'debug' verbosity the panel shows only backend [DEBUG*] lines —
            // drop the normal TS winner/eval log entirely.
            if (typeof _tsIsDebugOnly === 'function' && _tsIsDebugOnly()) break;
            const ji = msg._jobIdx ?? 0;
            if (!_ts._jobLogLines) _ts._jobLogLines = {};
            if (!_ts._jobLogLines[ji]) _ts._jobLogLines[ji] = [];
            _ts._jobLogLines[ji].push(msg.html);
            if (_ts._jobLogLines[ji].length > 200) _ts._jobLogLines[ji].shift();
            if (ji !== (_ts._activeJobIdx ?? 0)) break;
            if (!_ts._tsLogBuf) _ts._tsLogBuf = [];
            _ts._tsLogBuf.push(msg.html);
            // Enqueue for the tsMiniLogPanel (feeds _tsMiniLogBuf before the rAF flush)
            if (typeof _tsEnqueueLogLine === 'function') _tsEnqueueLogLine(msg.html);
            if (!_ts._tsLogRaf) {
                _ts._tsLogRaf = requestAnimationFrame(() => {
                    _ts._tsLogRaf = null;
                    const logEl = document.getElementById('tsTsLog');
                    if (!logEl || !_ts._tsLogBuf?.length) return;
                    const frag = document.createDocumentFragment();
                    _ts._tsLogBuf.splice(0).forEach(html => {
                        const d = document.createElement('div'); d.innerHTML = html; frag.appendChild(d);
                    });
                    logEl.appendChild(frag);
                    while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
                    logEl.scrollTop = logEl.scrollHeight;
                    // Mirror to the tsMiniLogPanel if it's open
                    if (typeof _tsForwardBufToLogPanel === 'function') _tsForwardBufToLogPanel();
                });
            }
            break;
        }

        case 'phase': {
            const pi  = msg._jobIdx ?? 0;
            if (pi === 0) _ts._currentPhase = msg.phase;
            const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
            set(`tsWuPhase_${pi}`, msg.phase);
            if (pi === 0) {
                set('tsWuPhase', msg.phase);
                const bar = document.getElementById('tsStatusBar');
                if (bar) bar.textContent = msg.phase;
            }
            break;
        }

        case 'switch_tab':
            if ((msg._jobIdx ?? 0) === (_ts._activeJobIdx ?? 0)) {
                _tsTab(msg.tab);
                ['tsWuLiveBadge'].forEach(id => { const e = document.getElementById(id); if (e) { e.textContent = 'done'; e.classList.remove('animate-pulse'); } });
                ['tsTsLiveBadge'].forEach(id => { const e = document.getElementById(id); if (e) { e.textContent = 'live'; e.classList.add('animate-pulse'); } });
            }
            {
                const ji = msg._jobIdx ?? 0;
                const phaseEl = document.getElementById(`tsWuPhase_${ji}`);
                if (phaseEl) phaseEl.textContent = 'TS inference';
            }
            break;

        case 'log':
            _tsWuLog(msg.html);
            break;

        case 'status':
            // Mirror genuine errors into the 💬 mini-log panel for debugging,
            // regardless of which job they came from (errors aren't job-scoped).
            if (msg.text && (msg.text.includes('ERROR') || msg.text.includes('Traceback') ||
                             msg.text.includes('exited with code') || msg.text.includes('ValueError'))) {
                if (typeof _tsMiniLogError === 'function') _tsMiniLogError(msg.text);
            }
            if ((msg._jobIdx ?? 0) === (_ts._activeJobIdx ?? 0)) {
                _ts._pendingBarText = msg.text;
                if (!_ts._barRafQueued) {
                    _ts._barRafQueued = true;
                    const _capturedText = msg.text;
                    setTimeout(() => {
                        requestAnimationFrame(() => {
                            _ts._barRafQueued = false;
                            const bar = document.getElementById('tsStatusBar');
                            if (bar) bar.textContent = _ts._pendingBarText || _capturedText;
                        });
                    }, 500);
                }
            }
            break;

        case 'done':
            // Handled via SSE __DONE__ in _tsStreamAll
            break;
    }
}

// ── Multi-job SSE stream ──────────────────────────────────────────────────
function _tsStreamAll(jobs) {
    if (!jobs || jobs.length === 0) return;

    // Resolve worker URL. Priority:
    // 1. window._HUB_FEATURE_BASE injected by hub.html (most reliable — Flask url_for path)
    //    _HUB_FEATURE_BASE ends in 'js/' so worker is at _HUB_FEATURE_BASE + 'ts/ts_worker.js'
    // 2. src of any already-loaded ts_core.js script tag (strip ts_core.js → get ts/ folder)
    // 3. Hardcoded fallback
    const _base = window._HUB_FEATURE_BASE
        ? window._HUB_FEATURE_BASE + 'ts/'
        : (document.querySelector('script[src*="ts/ts_core.js"]')?.src?.replace(/ts_core\.js.*/, '')
           || '/static/js/ts/');
    const workerUrl = _base + 'ts_worker.js?v=' + Date.now();

    _ts._streams = [];
    _ts._inTraceback = false;   // reset traceback-capture state per run
    let doneCount = 0;

    jobs.forEach((job, idx) => {
        const worker = new Worker(workerUrl);
        // Always send set_batch_size unconditionally — evalBatchSize may not be set yet on reload
        worker.postMessage({ type: 'set_batch_size', batchSize: _ts.evalBatchSize || 256 });
        worker.postMessage({ type: 'set_log_level', level: _tsLogLevel });
        const histPts    = job.history?.points || [];
        const histScores = job.history?.scores || [];
        // Guard against corrupt session files (doubled points from pre-fix ts_routes.py).
        // If pts >> scores by >40%, use scores.length as the true iteration count.
        const isCorrupt  = histPts.length > 2 && histScores.length > 0 &&
                           histPts.length > histScores.length * 1.4;
        const histLen    = isCorrupt ? histScores.length : histPts.length;
        if (isCorrupt) console.warn('[TS] corrupt session detected (pts='+histPts.length+
            ' scores='+histScores.length+') — using scores.length as offset');
        const seedScores = histScores.slice(-(_ts.evalBatchSize || 256));
        if (histLen > 0) worker.postMessage({ type: 'set_iter_offset', offset: histLen, seedScores });

        // Seed the worker's reagent table from persisted JSON so the live bars
        // continue from the saved top-5 (highest best scores) after reconnect
        // instead of starting empty. Prefer top5; fall back to the reagents dict.
        const hist = job.history || {};
        let seedReagents = null, seedTop5Ids = null;
        if (Array.isArray(hist.top5) && hist.top5.length) {
            seedReagents = hist.top5.map(t => ({
                name: t.name, mu: t.mu, std: t.std, sc: t.sc,
                best: t.best, partner: t.partner || '',
            }));
            seedTop5Ids = hist.top5.map(t => t.name);
        } else if (hist.reagents && Object.keys(hist.reagents).length) {
            seedReagents = Object.entries(hist.reagents).map(([name, r]) => ({
                name, mu: r.mu, std: r.std, sc: r.sc,
                best: r.best, partner: r.partner || '',
            }));
        }
        if (seedReagents && seedReagents.length) {
            worker.postMessage({ type: 'seed_reagents', reagents: seedReagents,
                                 top5Ids: seedTop5Ids });
        }

        worker.onmessage = (evt) => {
            const msg = evt.data;
            msg._jobIdx = idx;
            msg._jobKey = job.key || '';
            _tsHandleWorkerMsg(msg);
        };
        worker.onerror = (e) => console.error(`[TS worker ${idx}]`, e);

        const es = new EventSource(`/vina_visualization/ts_status/${job.job_id}`);
        es.onmessage = (evt) => {
            const line = evt.data;
            if (line === '__DONE__') {
                es.close();
                stream.done = true;
                doneCount++;
                if (doneCount >= jobs.length) _tsFinalise();
                return;
            }
            // Forward backend debug lines to the browser console so startup
            // diagnostics ([DEBUG:stdout#N], [STDERR:...], [DEBUG:replay], etc.)
            // are visible in DevTools, not just the TS Iteration Log panel.
            if (line.startsWith('[DEBUG:') || line.startsWith('[STDERR:') ||
                line.startsWith('[TS:history]')) {
                console.log(`[SSE job${idx}] ${line}`);
            }
            // Mirror warmup/reagent diagnostics into the 💬 mini-log panel so they're
            // visible without opening DevTools. ⚠/✗/MISSING lines are surfaced as errors
            // (auto-open, red); routine ✓/info debug lines append quietly in cyan.
            // Always mirror the important, low-volume diagnostics into the 💬 panel.
            // In 'debug' verbosity, mirror EVERY [DEBUG*] line (the panel shows only these).
            const _isImportantDbg = line.startsWith('[DEBUG:warmup]') ||
                                    line.startsWith('[DEBUG:reagents]') ||
                                    line.startsWith('[DEBUG:tsproc]');
            const _debugOnly = (typeof _tsIsDebugOnly === 'function' && _tsIsDebugOnly());
            if (_isImportantDbg || (_debugOnly && line.startsWith('[DEBUG'))) {
                if (/[⚠✗]|MISSING|ZERO|EMPTY/.test(line)) {
                    if (typeof _tsMiniLogError === 'function') _tsMiniLogError(line);
                    else if (typeof _tsMiniLogStatus === 'function') _tsMiniLogStatus(line, '#fca5a5');
                } else if (typeof _tsMiniLogStatus === 'function') {
                    _tsMiniLogStatus(line, _isImportantDbg ? '#7dd3fc' : '#f59e0b');
                }
            }
            // Surface the FULL traceback in the 💬 panel.
            // Strategy: once a "Traceback" line is seen, forward every subsequent
            // [STDERR] line (the exception body). Harmless warnings that appear
            // before the crash are NOT forwarded so they don't flood the panel.
            // Only genuine elion output should flag the run as errored — NOT our own
            // [DEBUG*] diagnostic lines, which may merely MENTION "Traceback"/"ERROR"
            // (e.g. "[DEBUG:tail] no Traceback/Error/Exception lines found").
            if (!line.startsWith('[DEBUG') &&
                (line.includes('Traceback') || line.includes('ERROR') ||
                 line.includes('exited with code'))) {
                _ts._sawError  = true;
                _ts._inTraceback = true;   // start capturing the exception body
            }
            if (_ts._inTraceback && line.startsWith('[STDERR]')) {
                // Forward the body line (File "…", line N, ExceptionType: …)
                const clean = line.replace(/^\[STDERR\]\s*/, '');
                if (clean) {
                    _ts._lastErrorText = _ts._lastErrorText || clean;
                    if (typeof _tsMiniLogError === 'function') _tsMiniLogError(clean);
                }
                // Stop capturing after the blank line that follows a traceback
                if (!clean) _ts._inTraceback = false;
            } else if (line.startsWith('[STDERR]') &&
                       (line.includes('ERROR') || line.includes('exited with code'))) {
                _ts._sawError = true;
                const clean = line.replace(/^\[STDERR\]\s*/, '');
                _ts._lastErrorText = clean;
                if (typeof _tsMiniLogError === 'function') _tsMiniLogError(clean);
            }
            worker.postMessage({ type: 'line', data: line });
        };
        es.onerror = () => {
            es.close();
            if (!stream.done) {
                stream.done = true;
                doneCount++;
                if (doneCount >= jobs.length && _ts.running) _tsFinalise();
            }
        };

        const stream = { es, worker, jobIdx: idx, job_id: job.job_id, done: false };
        _ts._streams.push(stream);
    });

    _ts.worker = _ts._streams[0]?.worker || null;
    _ts.es     = _ts._streams[0]?.es     || null;
}

function _tsStream(job_id) {
    _tsStreamAll([{ job_id, key: '' }]);
}