// =============================================================================
// ts/ts_core.js — shared state object, init helpers, speed, tab switching
// Load order: FIRST (all other ts_*.js files depend on _ts and _tsInitWu/Ts)
// =============================================================================

console.log('%c[ts] LOADED ' + Date.now(), 'background:#0891b2;color:white;padding:2px 6px;border-radius:3px');

/* ─── Shared state ───────────────────────────────────────────────────────── */
const _ts = {
    speed: 3, jobId: null, wuTimer: null, tsTimer: null,
    es: null, worker: null, running: false, activeTab: 'warmup',
    wuState: null, tsState: null,
    reagents: {},
    _barRaf: null, _pendingBarText: '',
    _top5Cache: null,
};

function _tsInitWu() { return { reagents:{}, evals:0, best:0, phase:'—', events:[], idx:0 }; }
function _tsInitTs() { return { reagents:{}, winners:[], events:[], idx:0, masked:0 }; }

/* ─── Speed ──────────────────────────────────────────────────────────────── */
function _tsWuInterval() { return [1200,800,500,300,150][_ts.speed-1]; }
function _tsTsInterval() { return [1400,950,600,350,180][_ts.speed-1]; }
function _tsSetSpeed(v) {
    _ts.speed = +v;
    if (_ts.wuTimer) { clearInterval(_ts.wuTimer); _ts.wuTimer = setInterval(_tsWuStep, _tsWuInterval()); }
    if (_ts.tsTimer) { clearInterval(_ts.tsTimer); _ts.tsTimer = setInterval(_tsTsStep, _tsTsInterval()); }
}

/* ─── Tab switching ──────────────────────────────────────────────────────── */
const _TS_TAB_ACTIVE = {
    warmup:  'bg-violet-900 text-violet-200',
    ts:      'bg-cyan-900   text-cyan-200',
    results: 'bg-emerald-900 text-emerald-200',
};
const _TS_TAB_INACTIVE = 'bg-transparent text-slate-400 hover:bg-slate-800 hover:text-white';
const _TS_PANE = { warmup:'tsPaneWarmup', ts:'tsPaneTs', results:'tsPaneResults' };

function _tsTab(tab) {
    _ts.activeTab = tab;
    document.querySelectorAll('[data-tab]').forEach(btn => {
        const t = btn.dataset.tab;
        if (t === tab) {
            btn.className = `px-4 py-1.5 ${t !== 'results' ? 'border-r border-slate-700 ' : ''}${_TS_TAB_ACTIVE[t]}`;
        } else {
            btn.className = `px-4 py-1.5 ${t !== 'results' ? 'border-r border-slate-700 ' : ''}${_TS_TAB_INACTIVE}`;
        }
    });
    Object.entries(_TS_PANE).forEach(([k, id]) => {
        const pane = document.getElementById(id);
        if (!pane) return;
        if (k === tab) pane.classList.remove('hidden');
        else           pane.classList.add('hidden');
    });
}