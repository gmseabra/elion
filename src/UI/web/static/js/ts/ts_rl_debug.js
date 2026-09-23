// =============================================================================
// ts_rl_debug.js — why is TOP TS PRODUCTS empty?
// -----------------------------------------------------------------------------
// Reports every link in the ranking chain into the Pose Generation activity
// panel (#poseMiniChatOutput, the box PoseGen._miniLog writes to), which is
// on-screen inside the RL tab. Console-only diagnostics are useless here: the
// interesting moment is a run FINISHING, which happens while you are looking at
// the tab, not at devtools.
//
// The chain, and what can break at each link:
//
//   GET /vina_visualization/ts_top5   ← server: does the ranking still exist?
//        │                              (jobs:[] here = nothing else can work)
//        ▼
//   _tsPollTop5 in ts_ui.js           ← client: did it store what arrived?
//        │  _tsTop5ByJob[launch_idx]
//        │  _ts._activeTopRaw ← _tsTop5ByJob[rawIdx]      ← INDEX MISMATCH lives here
//        ▼
//   _tsRlRanking() in ts_rl.js        ← which of those two it actually reads
//        ▼
//   _tsRlTopSecRender() → #tsRlTopList
//
// OBSERVES, never wraps. _tsPollTop5 is a top-level function declaration and
// ts_ui.js schedules it as `setInterval(_tsPollTop5, …)` — a bare identifier
// that resolves to the declaration, not to window._tsPollTop5. Reassigning the
// window property would intercept nothing while looking like it worked. So this
// runs its own fetch and reads the resulting globals, which also means a
// disagreement between what the server sent and what the client kept shows up
// as a disagreement between two lines here rather than being invisible.
//
//   window._tsRlDbgOff()     stop
//   window._tsRlDbgOn()      start (auto-starts on load)
//   window._tsRlDbgNow()     one immediate report, unconditionally
//   window._tsRlDbgAll(true) log every tick, not only on change
// =============================================================================

(function () {
    'use strict';
    if (window.__tsRlDebugLoaded) return;
    window.__tsRlDebugLoaded = true;

    var EVERY_MS = 3000;
    var timer = null, lastSig = null, always = false, ticks = 0;

    // Expected in the /ts_top5 payload once the session-retention fix is live.
    // Its ABSENCE is the single most useful thing this file reports: Flask
    // loads route modules at import, so an edited ts_routes.py does nothing at
    // all until the server is restarted — and "I applied the patch and nothing
    // changed" is exactly what that looks like from the browser.
    var WANT_IMPL = 'retention-v3';

    function log(msg, col, mono) {
        var PG = window.PoseGen;
        if (PG && typeof PG._miniLog === 'function') PG._miniLog(msg, col, mono !== false);
        else console.log('[ts-rl-dbg]', msg);
    }

    function n(v) { return (v == null) ? '—' : v; }
    function len(v) { return Array.isArray(v) ? v.length : (v == null ? '—' : '?not-array'); }

    // ── What the client currently holds ──────────────────────────────────────
    function clientState() {
        var s = { ts: (typeof _ts !== 'undefined') };
        if (!s.ts) return s;
        s.activeJobIdx = _ts._activeJobIdx;
        s.polled       = !!_ts._top5Polled;
        s.stateDir     = _ts._top5StateDir || '';
        s.activeTopRaw = len(_ts._activeTopRaw);
        s.activeBars   = len(_ts._activeBars);
        s.byJobKeys    = (typeof _tsTop5ByJob !== 'undefined') ? Object.keys(_tsTop5ByJob) : null;
        s.byJobLens    = {};
        if (s.byJobKeys) s.byJobKeys.forEach(function (k) { s.byJobLens[k] = len(_tsTop5ByJob[k]); });
        // What the RL panel's own resolver returns — the number the badge shows.
        s.ranking = (typeof _tsRlRanking === 'function') ? len(_tsRlRanking()) : 'fn missing';
        s.rxnKey  = (typeof _tsRlRxnKey === 'function') ? _tsRlRxnKey() : 'fn missing';
        return s;
    }

    function domState() {
        var list = document.getElementById('tsRlTopList');
        var bars = document.getElementById('tsTsBars');
        return {
            listPresent: !!list,
            listCards:   list ? list.querySelectorAll('.rlm').length : '—',
            listEmptyMsg: !!(list && /No ranking yet/.test(list.textContent || '')),
            barsRows:    bars ? bars.querySelectorAll('[data-el="ridLabel"]').length : '—',
            barsEmptyMsg: !!(bars && bars.querySelector('#tsTsBarsEmpty')),
            paneVisible: !!(document.getElementById('tsPaneRl') &&
                            !document.getElementById('tsPaneRl').classList.contains('hidden')),
        };
    }

    function report(server, err) {
        var c = clientState(), d = domState();

        // Only speak when something changed, or the panel would fill with
        // identical lines every 3 s and the one that matters would scroll away.
        var sig = JSON.stringify([
            err || null,
            server && server.impl, server && (server.jobs || []).length,
            server && (server.jobs || []).map(function (j) {
                return [j.launch_idx, j.status, (j.top5 || []).length].join('/');
            }),
            c.activeJobIdx, c.activeTopRaw, c.byJobLens, c.ranking,
            d.listCards, d.barsRows,
        ]);
        if (!always && sig === lastSig) return;
        lastSig = sig;

        log('── RL ranking probe #' + (++ticks) + ' ──', '#a78bfa');

        // ── 1. Server ────────────────────────────────────────────────────────
        if (err) {
            log('1 server  GET /ts_top5 FAILED: ' + err, '#fb7185');
        } else {
            var jobs = server.jobs || [];
            log('1 server  jobs=' + jobs.length +
                '  impl=' + n(server.impl) +
                '\n          state_dir=' + n(server.state_dir),
                jobs.length ? '#34d399' : '#fb7185');

            if (server.impl !== WANT_IMPL) {
                log('  ⚠ impl is not "' + WANT_IMPL + '". The patched ts_routes.py is NOT loaded.\n' +
                    '    Flask imports route modules once at startup — restart the server.\n' +
                    '    Until then /ts_top5 still deletes finished sessions and answers jobs=[].',
                    '#fbbf24');
            }
            if (!jobs.length) {
                log('  ⚠ zero jobs. Nothing downstream can work.\n' +
                    '    Before the run:  normal, nothing has been launched yet.\n' +
                    '    After a run:     the session file is missing. Check, in order —\n' +
                    '    • ls ' + n(server.state_dir) + '\n' +
                    '      empty → _run_ts_job deleted it on completion (pre-retention-v3\n' +
                    '      ts_routes.py wrote the final ranking and unlinked it on the next\n' +
                    '      line). Unrecoverable for that run; the next one will persist.\n' +
                    '    • directory itself missing/unreadable → visualizer.output_dir or\n' +
                    '      ELION_CWD is resolving somewhere else\n' +
                    '    • file present but not returned → it is older than retention_h=' +
                    n(server.retention_h) + 'h', '#fbbf24');
            }
            jobs.forEach(function (j) {
                log('  job idx=' + n(j.launch_idx) + ' status=' + n(j.status) +
                    ' top5=' + (j.top5 || []).length + ' rxn=' + n(j.rxn_key) +
                    (j.job_id ? ' id=' + String(j.job_id).slice(0, 8) : ''),
                    (j.top5 || []).length ? '#94a3b8' : '#fbbf24');
            });
        }

        // ── 2. Client store ──────────────────────────────────────────────────
        if (!c.ts) { log('2 client  _ts is undefined — ts_core.js did not load', '#fb7185'); return; }
        log('2 client  _tsTop5ByJob=' + JSON.stringify(c.byJobLens) +
            '\n          _activeJobIdx=' + n(c.activeJobIdx) +
            '  _activeTopRaw=' + c.activeTopRaw +
            '  _activeBars=' + c.activeBars +
            '\n          polled=' + c.polled, '#94a3b8');

        // The failure mode the "single-job index rescue" in _tsPollTop5 exists
        // for, reported explicitly instead of left to be inferred from two
        // numbers that happen to disagree.
        if (c.byJobKeys && c.byJobKeys.length && c.activeTopRaw === 0) {
            var nonEmpty = c.byJobKeys.filter(function (k) { return c.byJobLens[k] > 0; });
            if (nonEmpty.length) {
                log('  ⚠ INDEX MISMATCH: _activeTopRaw is empty but _tsTop5ByJob has data\n' +
                    '    under key(s) [' + nonEmpty.join(', ') + '] while _activeJobIdx=' +
                    n(c.activeJobIdx) + '.\n' +
                    '    Try:  _ts._activeJobIdx = ' + nonEmpty[0] + '   then reopen the RL tab.',
                    '#fbbf24');
            }
        }

        // ── 3. RL resolver ───────────────────────────────────────────────────
        log('3 rl      _tsRlRanking()=' + c.ranking + '  rxnKey=' + (c.rxnKey || '(empty)'),
            (c.ranking > 0) ? '#34d399' : '#fbbf24');
        if (c.ranking === 0 && c.activeTopRaw > 0) {
            log('  ⚠ _tsRlRanking() returns 0 while _activeTopRaw has ' + c.activeTopRaw +
                ' — ts_rl.js is reading a different source than ts_ui.js wrote.', '#fb7185');
        }
        if (!c.rxnKey) {
            log('  note rxnKey is empty: structure SVGs cannot be requested, so cards\n' +
                '       would render without images even if the ranking were populated.', '#94a3b8');
        }

        // ── 4. DOM ───────────────────────────────────────────────────────────
        log('4 dom     #tsRlTopList cards=' + d.listCards + (d.listEmptyMsg ? ' (empty-state shown)' : '') +
            '\n          #tsTsBars rows=' + d.barsRows + (d.barsEmptyMsg ? ' (empty-state shown)' : '') +
            '\n          RL pane visible=' + d.paneVisible, '#94a3b8');
        if (d.barsRows > 0 && c.ranking === 0) {
            log('  ⚠ the belief panel is showing ' + d.barsRows + ' STALE rows over an empty\n' +
                '    ranking — the ts_ui.js clear-on-empty patch is not loaded either.\n' +
                '    Hard-reload the page (Ctrl/⌘+Shift+R) to bypass the cached JS.', '#fbbf24');
        }
        if (!d.paneVisible) {
            log('  note the RL pane is hidden, so _tsRlTick is stopped and the panel is\n' +
                '       not re-rendering. This probe keeps running regardless.', '#94a3b8');
        }
    }

    function tick() {
        fetch('/vina_visualization/ts_top5', { cache: 'no-store' })
            .then(function (r) { return r.ok ? r.json() : Promise.reject('HTTP ' + r.status); })
            .then(function (d) { report(d, null); })
            .catch(function (e) { report(null, String(e)); });
    }

    window._tsRlDbgOn   = function () { if (!timer) { timer = setInterval(tick, EVERY_MS); tick(); }
                                        return 'ts-rl-dbg on (' + EVERY_MS + 'ms)'; };
    window._tsRlDbgOff  = function () { if (timer) { clearInterval(timer); timer = null; }
                                        return 'ts-rl-dbg off'; };
    window._tsRlDbgNow  = function () { lastSig = null; tick(); return 'probing…'; };
    window._tsRlDbgAll  = function (v) { always = (v !== false); return 'log-every-tick=' + always; };

    // Auto-start, but only once the page has settled, so the very first report
    // is not just "everything is undefined, scripts are still loading".
    setTimeout(function () {
        window._tsRlDbgOn();
        log('ts_rl_debug.js armed — probing /ts_top5 every ' + (EVERY_MS / 1000) + 's.\n' +
            'Silence with _tsRlDbgOff() · force a report with _tsRlDbgNow()', '#a78bfa');
    }, 1500);
})();