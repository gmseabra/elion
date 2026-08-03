// =============================================================================
// ts/ts_run.js — show/hide modal, load config, run/cancel/finalise, global exports
// Depends on: ts_core.js, ts_chart.js, ts_ui.js, ts_warmup.js, ts_worker_bridge.js
// =============================================================================

// ── Show / hide ───────────────────────────────────────────────────────────
function _tsShow() {
    const modal = document.getElementById('tsModal');
    if (modal) modal.classList.remove('hidden');

    _tsPatchSidebar();
    _tsPatchWarmupSidebar();
    _tsPatchChartPanel();
    _tsPatchSystemTab();
    _tsInjectKillAllBtn();
    _tsPatchLogLevelBtn();
    _tsCpuStart();
    if (typeof _tsGpuStart === 'function') _tsGpuStart();
    if (typeof _tsSpeedStart === 'function') _tsSpeedStart();
    _tsJobsPollStart();
    if (typeof _tsStartTop5Polling === 'function') _tsStartTop5Polling();   // reagent bars from JSON
    // Only reconnect if no live streams are running.
    // _tsRun() sets _ts.running=true and starts streams before calling _tsShow indirectly.
    // On page reload _ts.running is false (fresh JS state) so reconnect fires correctly.
    // On fresh open after clicking Run TS, _ts.running is already true — skip reconnect.
    if (!_ts.running) {
        setTimeout(() => _tsReconnectActive(), 0);
    } else {
        console.log('[ts] _tsShow: skipping reconnect — run already active');
    }

    const oldSmarts = document.getElementById('tsSmartsInput');
    if (oldSmarts && !document.getElementById('tsOutputDirInput')) {
        const span = oldSmarts.previousElementSibling;
        if (span) span.textContent = 'Output Dir';
        oldSmarts.id          = 'tsOutputDirInput';
        // Placeholder only \u2014 _tsLoadConfig fills the real default from the
        // engine yml a moment later. Assigning a value here would race it.
        oldSmarts.placeholder = 'loading default from input_TS.yml\u2026';
        oldSmarts.value       = '';
        oldSmarts.style.width = '420px';
    }
    requestAnimationFrame(_tsDrawSparkline);

    if (typeof _miniChatSetContext === 'function' && typeof _MINICHAT_TS_THEME !== 'undefined') {
        _miniChatSetContext(_MINICHAT_TS_THEME);
    }
    // NOTE: do NOT auto-show the shared #miniChat here. The TS tool uses its own
    // tsMiniLogPanel (bottom-right, toggled by the 💬 tsMiniLogBtn) which also
    // streams the live iteration log. Auto-showing #miniChat produced a second,
    // duplicate panel on the left. Context is still set above so that if the
    // shared panel is opened elsewhere it carries the TS theme.
    // if (typeof _miniChatShow === 'function') _miniChatShow();

    _tsTab('warmup');
    _tsLoadConfig();
}

function _tsHide() {
    const modal = document.getElementById('tsModal');
    if (modal) modal.classList.add('hidden');
    _tsCpuStop();
    if (typeof _tsGpuStop === 'function') _tsGpuStop();
    if (typeof _tsSpeedStop === 'function') _tsSpeedStop();
    _tsJobsPollStop();
    if (typeof _tsStopTop5Polling === 'function') _tsStopTop5Polling();
    if (typeof _miniChatClose === 'function') _miniChatClose();
    if (typeof _miniChatSetContext === 'function' && typeof _MINICHAT_VINA_THEME !== 'undefined') {
        _miniChatSetContext(_MINICHAT_VINA_THEME);
    }
}

// ── Config load ───────────────────────────────────────────────────────────
function _tsLoadConfig() {
    fetch('/vina_visualization/ts_config')
        .then(r => {
            if (!r.ok) return r.text().then(t => {
                let msg = '';
                try { msg = (JSON.parse(t) || {}).message || ''; } catch (_) { msg = (t || '').slice(0, 200); }
                console.error('[ts_config] HTTP', r.status, msg || t.substring(0, 200));
                return { status: 'error', message: msg };
            });
            return r.json();
        })
        .then(d => {
            if (d.status !== 'ok') {
                // Say so on screen. A silent return here is why a bad ELION_CWD
                // presented as "the Reactions row just isn't there".
                if (typeof _tsEngineBanner === 'function') _tsEngineBanner(d.message || '');
                return;
            }
            const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
            set('tsCfgLogLevel', d.log_level || '—');
            set('tsCfgMode',     d.ts_mode   || '—');
            set('tsCfgIter',     d.iterations !== undefined ? d.iterations.toLocaleString() : '—');
            // Also restore the iterations INPUT field so a custom value the user
            // set (e.g. 1000000) survives a page refresh instead of reverting to
            // the hardcoded 5000 placeholder.
            if (d.iterations !== undefined && d.iterations !== '—' && d.iterations !== null) {
                const iterInput = document.getElementById('tsIterInput');
                if (iterInput) {
                    const iterVal = parseInt(String(d.iterations).replace(/[^0-9]/g, ''));
                    if (isFinite(iterVal) && iterVal > 0) iterInput.value = iterVal;
                }
            }
            set('tsCfgWarmup',   d.warmup !== undefined ? d.warmup : '—');
            set('tsCfgBatch',    d.batch  !== undefined ? d.batch  : '—');
            set('tsCfgSmarts',   d.smarts  || '—');
            set('tsCfgResults',  d.results || '—');
            // Both path boxes take their default from the engine yml. In each
            // case the placeholder is always refreshed but the value only when
            // the box is empty, so a path typed this session is never stomped.
            // Restart Flask to pick up a yml edit: ts_config re-reads the file
            // per request, but _tsLoadConfig only runs when the modal opens.
            //
            // Import-database box ← `visualizer.bb_scan_dir`, else the engine's
            // reagent root. Written to the input when the picker already exists,
            // and stashed on _ts for _tsBbToggle when it does not — the panel can
            // be opened either side of this resolving.
            _ts._bbDefaultDir = d.bb_scan_dir || d.bb_base || _ts._bbDefaultDir || '';
            if (_ts._bbDefaultDir) {
                const bbEl = document.getElementById('_tsBbPathInput');
                if (bbEl) {
                    bbEl.placeholder = _ts._bbDefaultDir;
                    bbEl.title       = _ts._bbDefaultDir;
                    if (!bbEl.value.trim()) bbEl.value = _ts._bbDefaultDir;
                }
            }
            // OUTPUT DIR box ← `visualizer.default_output_dir`, else the
            // directory part of generator.TS.results_filename.
            if (d.default_output_dir) {
                const outEl = document.getElementById('tsOutputDirInput');
                if (outEl) {
                    outEl.placeholder = d.default_output_dir;
                    outEl.title       = d.default_output_dir;
                    if (!outEl.value.trim()) outEl.value = d.default_output_dir;
                }
            }
            if (d.batch !== undefined && d.batch !== '—') {
                _ts.evalBatchSize = parseInt(d.batch) || 20;
                ['tsBatchN','tsBatchN2'].forEach(id => set(id, String(_ts.evalBatchSize)));
                if (_ts.worker) _ts.worker.postMessage({ type: 'set_batch_size', batchSize: _ts.evalBatchSize });
            }
            if (Array.isArray(d.reactions)) {
                _ts._catalogueReactions = d.reactions;
                _tsBuildReactionPicker(d.reactions, d.smarts, d.warmup_cached || {});
            }
            const reagEl = document.getElementById('tsCfgReagents');
            if (reagEl && Array.isArray(d.reagents)) {
                reagEl.innerHTML = d.reagents.map(r => {
                    const fname = r.split('/').pop();
                    return `<div title="${r}" style="color:#475569;cursor:default">· ${fname}</div>`;
                }).join('');
            }
        })
        .catch(() => {});
}

// ── Reset ─────────────────────────────────────────────────────────────────
function _tsResetState() {
    _ts.wuState = _tsInitWu(); _ts.tsState = _tsInitTs();
    _ts.reagents = {}; _ts._currentPhase = 'Warmup 1';
    _tsRenderCount = 0; _tsRenderQueued = false;
    _ts._barRaf = null; _ts._pendingBarText = '';
    _ts._lastBarUpdate = 0;
    _ts._tsDrainQueued = false;
    _ts._tsBarsQueued  = false;
    _tsBarsRafPending = false; _tsWinnersRafPending = false;
    _ts._tsLogBuf = []; _ts._tsLogRaf = null;
    _ts._activeJobIdx = 0; _ts._wuJobMap = {}; _ts.jobs = [];
    _ts._wuJobEvals = {}; _ts._wuJobBest = {}; _ts._streams = [];
    _ts._jobStates = {}; _ts._barRafQueued = false;
    _ts._jobLogLines = {};
    _ts._jobSparkHistory = {}; _ts._jobBatchBest = {}; _ts._jobBatchWorst = {};
    _tsSparkReset();
}

function _tsResetUI() {
    _tsLogBuf.length = 0;
    if (_tsLogRaf) { cancelAnimationFrame(_tsLogRaf); _tsLogRaf = null; }
    // Blank the panels AND drop the cached render signature with them.
    // `_tsRenderTsBars` keeps `el._tsIdSig` (the reagent-set fingerprint) to
    // decide "patch numbers in place" vs "rebuild rows". Clearing innerHTML
    // without clearing the signature means a second run over the same reaction
    // — which usually produces the same top-5 set — takes the in-place branch,
    // patches `el.children[i]` that no longer exist, and paints nothing for the
    // whole run. The signature must never outlive the markup it describes.
    ['tsWuLog','tsTsBars','tsWuBars','tsResultsBody'].forEach(id => {
        const e = document.getElementById(id);
        if (e) { e.innerHTML = ''; e._tsIdSig = null; }
    });
    document.querySelectorAll('[data-wu-panel]').forEach(el => el.remove());
    const defPanel = document.getElementById('tsWuPanel_default');
    if (defPanel) defPanel.style.display = '';
    const sysProc = document.getElementById('tsSysProcTable'); if (sysProc) sysProc.innerHTML = '';
    ['tsWuPhase','tsWuEvals','tsWuReagents','tsWuBest',
     'tsTsIter','tsTsScore','tsTsMasked',
     'tsBatchMean','tsBatchStd',
     'tsBatchMean2','tsBatchStd2',
     'tsTsPool0','tsTsPool0MuRange','tsTsPool0StdRange',
     'tsTsPool1','tsTsPool1MuRange','tsTsPool1StdRange',
     'tsStatusBadge','tsMolCount','tsBestScore','tsIterDone','tsElapsed',
    ].forEach(id => { const e = document.getElementById(id); if (e) e.textContent = '—'; });
    const batchNEl = document.getElementById('tsBatchN');
    if (batchNEl) batchNEl.textContent = _ts.evalBatchSize ? String(_ts.evalBatchSize) : '—';
    const canvas = document.getElementById('tsBatchCanvas');
    if (canvas) { const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height); }
}

// ── Main run ──────────────────────────────────────────────────────────────
function _tsRun() {
    if (_ts.running) return;
    if (_ts.es) { _ts.es.close(); _ts.es = null; }
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);

    _ts.running = true;
    _ts._sawError = false; _ts._lastErrorText = null;   // clear prior-run error state
    _tsResetState(); _tsResetUI();

    const outputDir = document.getElementById('tsOutputDirInput')?.value.trim() || '';
    const iters     = parseInt(document.getElementById('tsIterInput')?.value) || 5000;
    const checkedKeys = [...document.querySelectorAll('[id^="_rxnChk_"]:checked')].map(el => el.value);
    let bodyPayload;
    if (checkedKeys.length > 0) {
        bodyPayload = { reactions: checkedKeys.map(k => ({key:k})), num_ts_iterations: iters, output_dir: outputDir };
    } else {
        bodyPayload = { num_ts_iterations: iters, output_dir: outputDir };
    }

    const btn  = document.getElementById('tsRunBtn'), icon = document.getElementById('tsRunIcon');
    const sBar = document.getElementById('tsSpinnerBar'), bar = document.getElementById('tsStatusBar');
    if (btn)  { btn.disabled = true; btn.title = 'Running…'; btn.onclick = _tsCancel; }
    if (icon) icon.textContent = '⏳';
    if (sBar) sBar.classList.remove('hidden');
    if (bar)  bar.textContent = 'Launching python elion.py -i input_TS.yml …';

    ['tsWuLiveBadge','tsTsLiveBadge'].forEach(id => { const e = document.getElementById(id); if (e) { e.textContent = 'live'; e.classList.add('animate-pulse'); } });
    _tsTab('warmup');

    _ts.wuTimer = setInterval(_tsWuStep, _tsWuInterval());
    _ts.tsTimer = setInterval(_tsTsStep, _tsTsInterval());

    fetch('/vina_visualization/ts_run', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(bodyPayload),
    })
    .then(r => {
        if (!r.ok) return r.text().then(txt => {
            // The server's body names the actual problem — most often
            // {"message": "yml not found: <path>"} from a mis-resolved
            // ELION_CWD. Throwing only the status turns a fixable path error
            // into an indistinguishable "HTTP 404: NOT FOUND", which is what
            // this used to do. Keep the status AND the reason.
            let detail = '';
            try { detail = (JSON.parse(txt) || {}).message || ''; } catch (_) { detail = (txt || '').slice(0, 200); }
            throw new Error(`HTTP ${r.status}: ${r.statusText}${detail ? ' — ' + detail : ''}`);
        });
        return r.json();
    })
    .then(data => {
        if (data.status !== 'started') { _tsError(data.message || 'Failed to start'); return; }
        _ts.jobId = data.job_id;
        _ts.jobs  = data.jobs || [{ job_id: data.job_id, key: '' }];
        _ts._activeJobIdx = 0;
        if (bar) bar.textContent = `${_ts.jobs.length} job(s) started — streaming output`;
        _tsBuildWuPanels(_ts.jobs);
        setTimeout(_tsMakeReactionTabsClickable, 100);
        _tsStreamAll(_ts.jobs);
    })
    .catch(e => { console.error('[ts_run] fetch failed:', e); _tsError(e.message); });
}

// ── Cancel ────────────────────────────────────────────────────────────────
function _tsCancel() {
    (_ts._streams || []).forEach(s => {
        if (s.es)     s.es.close();
        if (s.worker) s.worker.terminate();
    });
    _ts._streams = [];
    if (_ts.es)     { _ts.es.close();          _ts.es     = null; }
    if (_ts.worker) { _ts.worker.terminate();   _ts.worker = null; }
    fetch('/vina_visualization/ts_kill_all', { method: 'POST' }).catch(() => {});
    _tsFinalise(true);
}

// ── Finalise ──────────────────────────────────────────────────────────────
function _tsFinalise(cancelled = false) {
    _ts.running = false;
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);
    _ts.wuTimer = null; _ts.tsTimer = null;

    const btn  = document.getElementById('tsRunBtn'), icon = document.getElementById('tsRunIcon');
    const sBar = document.getElementById('tsSpinnerBar'), bar = document.getElementById('tsStatusBar');
    if (btn)  { btn.disabled = false; btn.title = ''; btn.onclick = _tsRun; }
    if (icon) icon.textContent = '🎲';
    if (sBar) sBar.classList.add('hidden');

    ['tsWuLiveBadge','tsTsLiveBadge'].forEach(id => {
        const e = document.getElementById(id);
        if (e) { e.textContent = cancelled ? 'cancelled' : 'done'; e.classList.remove('animate-pulse'); }
    });

    _tsRenderWuBars();
    if (_ts.wuState) while (_ts.wuState.idx < _ts.wuState.events.length) _tsWuStep();

    function _drainAndFinish() {
        if (_ts.tsState && _ts.tsState.idx < _ts.tsState.events.length) {
            const deadline = performance.now() + 12;
            while (_ts.tsState.idx < _ts.tsState.events.length && performance.now() < deadline) _tsTsStep();
            if (_ts.tsState.idx < _ts.tsState.events.length) { requestAnimationFrame(_drainAndFinish); return; }
        }
        _tsRenderTsBars();
        const winners = [...(_ts.tsState?.winners || [])].sort((a, b) => b.score - a.score);
        const best    = winners.length ? winners[0].score : null;
        const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        set('tsMolCount',    winners.length || '—');
        set('tsBestScore',   best !== null ? best.toFixed(4) : '—');
        set('tsIterDone',    _ts.tsState?.winners.length || '—');

        // Decide success/error/cancelled AFTER pending status updates settle.
        // The error line ("ERROR: elion.py exited…") reaches the status bar via a
        // ~500ms delayed path, but __DONE__ → _tsFinalise fires synchronously, so
        // checking the bar (or writing "✓ Complete") right now races the error and
        // can mask it. Defer past that delay, then read the settled state.
        setTimeout(() => {
            const barTxt = document.getElementById('tsStatusBar')?.textContent || '';
            const barErr = barTxt.includes('ERROR') || barTxt.includes('exited with code') ||
                           barTxt.includes('Traceback') || barTxt.startsWith('❌');
            const errored = _ts._sawError || barErr;

            set('tsStatusBadge', errored ? '❌ Error' : (cancelled ? 'Cancelled' : '✓ Done'));
            if (bar) {
                if (errored)        bar.textContent = _ts._lastErrorText || (barErr ? barTxt : '❌ Run ended with an error');
                else if (cancelled) { /* leave cancelled text as-is */ }
                else                bar.textContent = `✓ Complete — ${winners.length} molecules, best ${best !== null ? best.toFixed(4) : '?'}`;
            }

            // Reflect the run's end state in the 💬 panel so it isn't left showing
            // only the greeting (e.g. when a run errors before producing any log).
            if (errored) {
                const detail = _ts._lastErrorText || (barErr ? barTxt.replace(/^❌\s*/, '') : '');
                if (detail && typeof _tsMiniLogError === 'function') _tsMiniLogError(detail);
                if (typeof _tsMiniLogStatus === 'function')
                    _tsMiniLogStatus('■ Run ended with an error.', '#fca5a5');
            } else if (cancelled) {
                if (typeof _tsMiniLogStatus === 'function')
                    _tsMiniLogStatus('■ Run cancelled.', '#94a3b8');
            } else if (typeof _tsMiniLogStatus === 'function') {
                _tsMiniLogStatus(`✓ Run complete — ${winners.length} molecule${winners.length===1?'':'s'}, ` +
                                 `best score ${best !== null ? best.toFixed(4) : '?'}.`, '#34d399');
            }
        }, 650);
    }
    requestAnimationFrame(_drainAndFinish);
}

function _tsError(msg) {
    _ts.running = false;
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);
    const btn  = document.getElementById('tsRunBtn'), icon = document.getElementById('tsRunIcon');
    const sBar = document.getElementById('tsSpinnerBar');
    if (btn)  { btn.disabled = false; btn.onclick = _tsRun; }
    if (icon) icon.textContent = '🎲';
    if (sBar) sBar.classList.add('hidden');
    const bar   = document.getElementById('tsStatusBar');   if (bar)   bar.textContent   = '❌ ' + (msg || 'Error');
    const badge = document.getElementById('tsStatusBadge'); if (badge) badge.textContent = '❌ Error';
    // Also surface the error in the 💬 mini-log panel for debugging.
    if (typeof _tsMiniLogError === 'function') _tsMiniLogError('❌ ' + (msg || 'Error'));
}

// ── Reconnect to running jobs after page reload ───────────────────────────────
// Fetches /ts_active, restores TS inference state, replays spark history,
// and reconnects live SSE streams. Called from _tsShow() on every modal open.
async function _tsReconnectActive() {
    try {
        const resp = await fetch('/vina_visualization/ts_active');
        if (!resp.ok) return;
        const ct = resp.headers.get('content-type') || '';
        if (!ct.includes('application/json')) return;
        const data = await resp.json();
        const jobs = data.jobs || [];
        // Log _debug array from _load_sessions so session file state is visible in browser console
        if (data._debug && data._debug.length) {
            console.group('[reconnect] _load_sessions debug:');
            data._debug.forEach(line => console.log(line));
            console.groupEnd();
        }
        console.log('[reconnect] jobs:', jobs.length, jobs.map(j => ({name: j.short_name, n: j.history?.scores?.length})));
        if (!jobs.length) {
            console.warn('[reconnect] NO JOBS FOUND — see _load_sessions debug above');
            // Session files may not exist yet if the run just started.
            // Retry up to 5 times with 1s delay to catch files written shortly after page load.
            if (!_tsReconnectActive._retries) _tsReconnectActive._retries = 0;
            if (_tsReconnectActive._retries < 8) {  // up to 16s total (8 × 2s)
                _tsReconnectActive._retries++;
                const _t = new Date().toTimeString().slice(0,8);
                console.log(`[reconnect ${_t}] no jobs yet, retrying in 2s (attempt ${_tsReconnectActive._retries}/8)...`);
                setTimeout(_tsReconnectActive, 2000);
            } else {
                _tsReconnectActive._retries = 0;
                console.log('[reconnect] no running jobs found after retries — this is normal if no run is active');
            }
            return;
        }
        _tsReconnectActive._retries = 0;

        _ts.running          = true;
        _ts.jobs             = jobs;
        _ts._streams         = [];
        _ts._jobStates       = {};
        _ts._jobSparkHistory = {};
        _ts._activeJobIdx    = 0;

        // Rebuild reaction tab pills
        const palette = ['#0891b2','#7c3aed','#0d9488','#b45309','#be185d'];
        const wrapper = document.getElementById('_tsRxnPicker');
        if (wrapper) {
            wrapper.querySelectorAll('[data-rxn-pretab]').forEach(el => el.remove());
            jobs.forEach((job, i) => {
                const color = palette[i % palette.length];
                const pill  = document.createElement('button');
                pill.dataset.rxnPretab = job.key;
                pill.dataset.rxnKey    = job.key;
                pill.id = `_tsRxnTab_${i}`;
                pill.style.cssText = [
                    'display:inline-flex;align-items:center;gap:5px',
                    'padding:3px 10px;border-radius:5px',
                    'font-size:10px;font-weight:600;cursor:pointer;white-space:nowrap;flex-shrink:0',
                    `border:0.5px solid ${color}55;background:${color}18`,
                    `color:${i === 0 ? color : color + '99'}`,
                    `border-bottom:${i === 0 ? '2px solid ' + color : '2px solid transparent'}`,
                    'transition:all 0.15s',
                ].join(';');
                const dot = document.createElement('span');
                dot.style.cssText = `width:6px;height:6px;border-radius:50%;background:${color};display:inline-block;flex-shrink:0`;
                pill.appendChild(dot);
                pill.appendChild(document.createTextNode(job.short_name.toUpperCase()));
                wrapper.appendChild(pill);
            });
        }

        const lbl = document.getElementById('_tsRxnBtnLabel');
        if (lbl) lbl.textContent = jobs.map(j =>
            j.short_name.charAt(0).toUpperCase() + j.short_name.slice(1)).join(', ');
        jobs.forEach(job => {
            const chk = document.getElementById(`_rxnChk_${job.key}`);
            if (chk) chk.checked = true;
        });

        // Restore per-job state and replay spark history (ts_chart.js owns backfill logic)
        if (!_ts._jobBars) _ts._jobBars = {};
        jobs.forEach((job, ji) => {
            const hist = job.history || {scores:[], reagents:{}};
            _ts._jobStates[ji] = _tsInitTs();
            _tsSparkReplayHistory(hist, ji);   // → ts_chart.js
            Object.entries(hist.reagents || {}).forEach(([name, r]) => {
                _ts._jobStates[ji].reagents[name] = {
                    name, mu: r.mu, std: r.std, sc: r.sc,
                    best: r.best, delta_mu: 0,
                    bestPartner: r.partner || '—',
                };
            });
            // Prefer the persisted top5 (highest best-score reagents, maintained
            // by the backend) so the bar panel shows the SAME ranking that was
            // live before reload. Fall back to computing from the reagents dict.
            if (Array.isArray(hist.top5) && hist.top5.length) {
                const maxMu = Math.max(...hist.top5.map(t => t.mu || 0), 0.001);
                _ts._jobBars[ji] = hist.top5.map(t => {
                    const muPct   = +((t.mu || 0) / maxMu * 100).toFixed(1);
                    const stdPct  = +Math.min(((t.std || 0) / maxMu) * 100, 18).toFixed(1);
                    const stdLeft = +Math.max(0, muPct - stdPct / 2).toFixed(1);
                    // Seed the SMILES cache from the JSON so structures/strings
                    // appear immediately without an extra fetch.
                    if (t.smiles) _tsSmilesCache[t.name] = t.smiles;
                    return {
                        id: t.name, name: t.name, mu: t.mu || 0, std: t.std || 0,
                        sc: t.sc || 0, best: t.best,
                        bestPartner: t.partner || '—', delta_mu: 0,
                        muPct, stdPct, stdLeft,
                    };
                });
                // Also prime the worker's notion of what's shown so the first
                // live emit doesn't reshuffle relative to the restored list.
                if (!_ts._restoredTop5Ids) _ts._restoredTop5Ids = {};
                _ts._restoredTop5Ids[ji] = hist.top5.map(t => t.name);
            } else {
                _ts._jobBars[ji] = _tsComputeBarsFromReagents(_ts._jobStates[ji].reagents);
            }
        });

        _ts.tsState       = _ts._jobStates[0];
        _ts._sparkHistory = _ts._jobSparkHistory[0] || [];
        _ts._activeBars   = _ts._jobBars[0] || [];

        // Update sidebar counters
        const set    = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        const hist0  = jobs[0]?.history || {points:[], scores:[]};
        const n0     = hist0.points?.length || 0;   // true count; scores capped at 256
        const scores0 = hist0.scores || [];
        if (n0 > 0) {
            set('tsTsIter',  n0);
            if (scores0.length > 0) set('tsTsScore', scores0[scores0.length - 1].toFixed(4));
        }

        _tsInjectMultiJobPools(jobs);
        const ka = document.getElementById('tsKillAllBtn');
        if (ka) ka.style.display = '';
        _tsTab('ts');
        _tsQueueBarsRender();
        requestAnimationFrame(_tsDrawSparkline);

        _tsStreamAll(jobs);
        setTimeout(_tsMakeReactionTabsClickable, 50);

    } catch(e) {
        console.warn('[ts] reconnect failed:', e);
    }
}

// ── Global exports (called from onclick= in dynamically-injected HTML) ────
window._tsRxnToggle      = _tsRxnToggle;
window._tsRxnSelectAll   = _tsRxnSelectAll;
window._tsRxnDone        = _tsRxnDone;
window._tsRxnUpdateLabel = _tsRxnUpdateLabel;
window._tsMonSubTab      = _tsMonSubTab;
window._tsMetaRefresh    = _tsMetaRefresh;
window._tsShow           = _tsShow;
window._tsHide           = _tsHide;
window._tsRun            = _tsRun;
window._tsCancel         = _tsCancel;
window._tsTab            = _tsTab;
window._tsWarmupClear    = _tsWarmupClear;
window._tsLogLevelCycle  = _tsLogLevelCycle;

window._tsDebug = function(on = true) {
    _tsDebugVerbose = !!on;
    const level = on ? 2 : 0;
    (_ts._streams || []).forEach(s => { if (s.worker) s.worker.postMessage({ type: 'set_debug', level }); });
    console.log(`[ts] debug ${on ? 'ON (verbose)' : 'OFF (errors only)'}`);
};