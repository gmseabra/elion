// ══ Vina JS ══════════════════════════════════════════════════════════════════
// Build marker, stamped onto every row _renderTable creates (see below) and exposed
// globally, so STEP 0 can compare "what build made this row" against "what build is
// running right now" — the single most direct way to catch a stale/cached deployment,
// which has been the root cause more than once in this debugging session already.
const _VINA_JS_BUILD = 'createElement-v3-2026-07-19';

// ── DOUBLE-LOAD DETECTOR ─────────────────────────────────────────────────────
// Root cause of the "STALE ROW / buildsMatch=false" bug: vina.js is being loaded
// TWICE on the page — a fresh cache-busted copy AND an older cached copy. Because
// both define `function _renderTable`, whichever script is evaluated LAST wins the
// global name, and the stale copy's _renderTable (old onclick="" rows) is the one
// that renders the table. This block counts how many times a vina.js has evaluated on
// the page and reports honestly.
//
// IMPORTANT: the ONLY reliable signal is __vinaLoadCount. An earlier version of this
// block ALSO flagged "a _renderTable already existed before this copy loaded" — that
// was a false-positive bug: `function _renderTable(){}` is a HOISTED declaration, so
// `window._renderTable` (assigned at the bottom of this file) is not the right probe,
// and more importantly the hoisted binding exists before this IIFE runs, making the
// check fire on a totally normal SINGLE load. That false alarm sent debugging down a
// wrong path. Count is the truth: >1 means genuinely loaded more than once.
(function () {
    const prevCount = window.__vinaLoadCount || 0;
    window.__vinaLoadCount = prevCount + 1;
    window.__vinaLoadBuilds = window.__vinaLoadBuilds || [];
    window.__vinaLoadBuilds.push(_VINA_JS_BUILD);

    if (window.__vinaLoadCount > 1) {
        console.error('%c[voxel] ⚠️ vina.js LOADED MORE THAN ONCE — THIS IS THE BUG',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:3px 6px',
            '\nvina.js has now evaluated ' + window.__vinaLoadCount + ' time(s) on this page.'
          + '\nbuilds seen (in load order): ' + JSON.stringify(window.__vinaLoadBuilds)
          + '\n→ The LAST copy to load wins `_renderTable`. If a STALE copy loads after the'
          + '\n  fresh one, its old onclick="" rows are what render, causing buildsMatch=false.'
          + '\n→ FIX: remove the duplicate <script src=".../vina.js"> include so vina.js'
          + '\n  loads EXACTLY ONCE. Network tab (filter "vina.js") must show it only once.');
        if (typeof window._voxelBanner === 'function') {
            window._voxelBanner('⚠️ vina.js LOADED x' + window.__vinaLoadCount
                + ' — duplicate <script> include is the bug', '#ff3333');
        }
    } else {
        // Single load — the healthy case. Say so plainly so it's not mistaken for the bug.
        console.log('%c[voxel] ✓ vina.js loaded exactly once (healthy) — build '
            + _VINA_JS_BUILD, 'color:#22c55e;font-weight:bold');
    }
})();

window._VINA_JS_BUILD = _VINA_JS_BUILD;
console.log('%c[voxel] vina.js BUILD 2026-07-18-debug loaded — marker: ' + _VINA_JS_BUILD,
            'color:#f59e0b;font-weight:bold');

// Copy-paste-safe payload serializer. Plain console.log('label', {some: obj}) prints
// the object as an interactive, COLLAPSED reference in DevTools — copying the visible
// text without clicking each entry open captures just the word "Object" and loses
// every field, which is what happened with the STEP 0 payloads in the last round of
// debugging. _vjson instead returns a plain JSON string, so passing it (not the raw
// object) to console.log/warn/error means the full detail is plain text from the
// moment it's printed — no expanding needed, and a normal copy-paste of the console
// always includes everything. DOM elements (event targets, matched rows) are reduced
// to a safe, serializable summary (tag/id/class/data-atom-idx) since JSON.stringify
// can't handle a live Node.
function _vjson(payload) {
    try {
        return JSON.stringify(payload, function (k, v) {
            if (v instanceof Node) {
                return {
                    __node: v.tagName || v.nodeName || '?',
                    id: v.id || null,
                    className: (v.className && v.className.toString) ? v.className.toString() : null,
                    dataAtomIdx: (v.getAttribute && v.getAttribute('data-atom-idx')) || null,
                };
            }
            return (v === undefined) ? null : v;   // JSON.stringify silently drops undefined
        }, 2);
    } catch (e) {
        return '(could not serialize: ' + e.message + ')';
    }
}
// console.log wrapper using _vjson — for the common case of a styled info line.
function _vlog(label, payload) {
    console.log('%c' + label, 'background:#1e1b4b;color:#c7d2fe;font-weight:bold;padding:1px 4px',
                '\n' + _vjson(payload));
}

// ON-PAGE visual debug trail (now OFF by default). This rendered a fixed banner at the
// top of the page stacking the last several pipeline steps with timestamps — invaluable
// while diagnosing the row-click issue because partial console copy-pastes couldn't be
// trusted. Now that the pipeline works, the banner is removed from the UI: this function
// is a no-op unless window._VOXEL_BANNER === true is set in the console. All the
// _voxelBanner(...) call sites throughout the file remain, harmlessly doing nothing, so
// re-enabling is a one-line flag rather than re-instrumenting.
function _voxelBanner(text, color) {
    if (!window._VOXEL_BANNER) return;   // banner disabled — no fixed overlay is created
    let el = document.getElementById('_voxelDebugBanner');
    if (!el) {
        el = document.createElement('div');
        el.id = '_voxelDebugBanner';
        el.style.cssText = 'position:fixed; top:0; left:0; right:0; z-index:2147483647; '
          + 'background:#000; font:bold 12px/1.5 monospace; padding:4px 8px; '
          + 'max-height:180px; overflow-y:auto; border-bottom:3px solid #22ff22; '
          + 'white-space:pre-wrap; pointer-events:none;';
        if (document.body) document.body.appendChild(el);
        else { document.addEventListener('DOMContentLoaded', () => document.body.appendChild(el)); }
    }
    const line = document.createElement('div');
    line.style.color = color || '#22ff22';
    const t = new Date();
    const ts = String(t.getMinutes()).padStart(2,'0') + ':' + String(t.getSeconds()).padStart(2,'0')
             + '.' + String(t.getMilliseconds()).padStart(3,'0');
    line.textContent = ts + '  ' + text;
    el.insertBefore(line, el.firstChild);
    while (el.children.length > 14) el.removeChild(el.lastChild);
}
window._voxelBanner = _voxelBanner;

// [voxel STEP 0] Ground-truth diagnostic: capture-phase listener directly on
// `document`, registered as early as this script executes — before anything else on
// the page has a chance to intercept the event. This bypasses _renderTable's own
// delegation entirely. If a click on an atom row never prints this line, something
// upstream of our code (a full-page overlay, another script's capture-phase listener,
// a browser extension) is swallowing the click before it reaches the DOM tree we
// control — that's a different bug than anything in vina.js and no amount of logging
// deeper in our own functions will reveal it. If STEP 0 prints but nothing after it
// does, the problem is inside our own code, and the numbered steps below will show
// exactly which one.
document.addEventListener('click', function(ev) {
    const inModal = ev.target.closest && ev.target.closest('#vinaModal');
    if (!inModal) return;   // ignore clicks elsewhere on the page — keep this quiet
    const row = ev.target.closest && ev.target.closest('[data-atom-idx]');
    // STALE-DEPLOYMENT CHECK: compare the build that constructed THIS specific row
    // (stamped by _renderTable at creation time, see below) against the build
    // currently executing. If they differ — or rowBuiltBy is missing entirely — the
    // row on screen was created by an OLDER script than the one now running, meaning
    // the page is serving/caching a stale vina.js and every "why doesn't my listener
    // fire" investigation downstream of this is moot until that's fixed.
    _vlog('[voxel STEP 0] document click (capture, ground truth)', {
        target: ev.target,
        targetTag: ev.target.tagName,
        targetClass: ev.target.className,
        rowFound: !!row,
        rowAtomIdx: row ? row.getAttribute('data-atom-idx') : null,
        rowOnclickAttribute: row ? row.getAttribute('onclick') : null,
        rowOnclickProperty:  row ? (typeof row.onclick) : null,
        rowBuiltByBuild:     row ? (row._voxelBuiltBy || '(no marker — STALE row)') : null,
        currentRunningBuild: _VINA_JS_BUILD,
        buildsMatch:         row ? (row._voxelBuiltBy === _VINA_JS_BUILD) : null,
        defaultPrevented: ev.defaultPrevented,
    });
    if (row) {
        const matches = row._voxelBuiltBy === _VINA_JS_BUILD;
        _voxelBanner('0️⃣ click on row ' + row.getAttribute('data-atom-idx')
            + '  buildsMatch=' + matches + (matches ? '' : '  ⚠️ STALE ROW'),
            matches ? '#22ff22' : '#ff3333');
    }
}, true);   // capture phase — fires on the way DOWN, before bubble-phase listeners

// [voxel STEP 0.5] STEP 0 (document, capture) fires reliably, but nothing downstream
// ever does — not even a row's own INLINE onclick, which needs no bubbling at all and
// fires the instant the event reaches that element. The only thing that explains
// "outermost capture listener fires, but the target's own handler never does" is some
// OTHER listener, somewhere between document and the row, calling stopPropagation() or
// stopImmediatePropagation() on the way down. Rather than guess where, intercept the
// native methods themselves — this fires, with a full stack trace naming the exact
// calling script/line, no matter which element or which script does it.
//
// NOTE: this is now OFF by default. It was invaluable for finding the cause, but the
// thing it kept catching — the modal wrapper's own onclick="event.stopImmediatePropagation()"
// (hub.html), fired both on page load by chembert_chat.js's jQuery .click() and on any
// in-modal click — is confirmed HARMLESS and unrelated to the atom-row handler. Left on,
// its red stack traces drown the one signal that matters (STEP 1 / STEP 1.5). Set
// window._VOXEL_TRACE_STOPPROP = true in the console before repro to re-enable.
if (window._VOXEL_TRACE_STOPPROP) (function () {
    const origSP  = Event.prototype.stopPropagation;
    const origSIP = Event.prototype.stopImmediatePropagation;
    Event.prototype.stopPropagation = function () {
        if (this.type === 'click') {
            console.error('%c[voxel STEP 0.5] stopPropagation() called on a click event',
                'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px',
                '\ncalled on: ' + (this.target && this.target.tagName) + '.' + (this.target && this.target.className)
              + '\nstack:\n' + new Error().stack);
        }
        return origSP.apply(this, arguments);
    };
    Event.prototype.stopImmediatePropagation = function () {
        if (this.type === 'click') {
            console.error('%c[voxel STEP 0.5] stopImmediatePropagation() called on a click event',
                'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px',
                '\ncalled on: ' + (this.target && this.target.tagName) + '.' + (this.target && this.target.className)
              + '\nstack:\n' + new Error().stack);
        }
        return origSIP.apply(this, arguments);
    };
})();

// [voxel STEP 0.6] Corroborating checkpoints at several depths between document and a
// table row, each in capture phase. Whichever of these is the DEEPEST one that still
// fires for a given click pins down exactly which ancestor gap the event dies in —
// independent confirmation alongside STEP 0.5's stack trace. Also opt-in now (same
// reason) — these only fire on a real row click, but keep the default output focused.
function _voxelArmDepthCheckpoints() {
    if (!window._VOXEL_TRACE_DEPTH) return;
    ['body', '#vinaModal', '#vSideA', '#vSideScrollA', '#vTableA'].forEach(sel => {
        const el = sel === 'body' ? document.body : document.querySelector(sel);
        if (!el || el._voxelDepthArmed) return;
        el._voxelDepthArmed = true;
        el.addEventListener('click', function (ev) {
            const row = ev.target.closest && ev.target.closest('[data-atom-idx]');
            if (!row) return;   // keep this quiet — only log when it's plausibly our row
            _vlog('[voxel STEP 0.6] capture reached: ' + sel, {
                target: ev.target, eventPhase: ev.eventPhase, cancelBubble: ev.cancelBubble,
            });
        }, true);
    });
}
const _voxelArmInterval = setInterval(_voxelArmDepthCheckpoints, 500);

let _vinaMode      = false;
let _vinaAtoms     = [];
let _selAtomIdx    = null;
let _currentView   = 'both';
let _recAtoms      = [];
let _pairsByLig    = {};
let _ligAtomsCache = [];
let _globalPairEMin = 0;
let _globalPairEMax = 0;

/* ════════════════════════════════════════════════════════════════════════════
   Modal open / close
   ════════════════════════════════════════════════════════════════════════════ */
function _vinaShow() {
    document.getElementById('vinaModal').classList.remove('hidden');
    document.getElementById('vinaModal').classList.add('flex');
    // Fill the editable docking-box inputs (Box ctr / len) from _GRID on open
    if (typeof _syncGridInputs === 'function') _syncGridInputs();
    // Trigger Plotly resize after modal is visible so vPlotA gets correct dimensions
    setTimeout(() => {
        const vp = document.getElementById('vPlotA');
        if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
    }, 150);
    setTimeout(() => {
        // Restore Vina branding in case another tool was used before
        if (typeof _miniChatSetContext === 'function') _miniChatSetContext(_MINICHAT_VINA_THEME);
        _miniChatShow();
        const out = document.getElementById('miniChatOutput');
        // Always show Vina welcome when opening this modal
        if (out) {
            out.innerHTML = '';
            if (typeof _vinaMiniWelcome === 'function') {
                _vinaMiniWelcome();
            } else {
                _miniChatWelcome();      // fallback
            }
        }
        // Highlight whichever path fields are empty on open
        _highlightEmptyPathFields();
    }, 400);
}

function _highlightEmptyPathFields() {
    const rec = document.getElementById('recPath');
    const lig = document.getElementById('ligPath');
    const missing = [];
    if (rec && !rec.value.trim()) missing.push('recPath');
    if (lig && !lig.value.trim()) missing.push('ligPath');
    if (missing.length) _highlightAmberUntilAction(missing, 'input');
}

function _vinaHide() {
    document.getElementById('vinaModal').classList.add('hidden');
    document.getElementById('vinaModal').classList.remove('flex');
    _closeDrawer();
    _miniChatClose();   // hide vina mini-chat when modal closes
}

function _setView(v) {
    _currentView = v;
    // Segmented control: Ligand | Protein | Both. Dividers (border-r) sit between
    // segments, so every button except the last carries one.
    const activeCls   = 'px-4 py-1.5 bg-cyan-800 text-cyan-100';
    const inactiveCls = 'px-4 py-1.5 text-slate-400 hover:text-white hover:bg-slate-800';
    [ {id:'modeLigand',  view:'ligand',  last:false},
      {id:'modeProtein', view:'protein', last:false},
      {id:'modeBoth',    view:'both',    last:true } ].forEach(b => {
        const el = document.getElementById(b.id);
        if (el) el.className = (b.view===v ? activeCls : inactiveCls) + (b.last ? '' : ' border-r border-slate-700');
    });

    if (v==='ligand') {
        if (_ligAtomsCache.length)
            _render3D('vPlotA', _ligAtomsCache, window._lastBonds||[], false);
    } else if (v==='protein') {
        _renderProteinView(_selAtomIdx);
    } else {   // both — ligand + protein in one scene
        _renderBothView(_selAtomIdx);
    }
}

function _renderProteinView(ligAtomIdx) {
    if (!_recAtoms.length) { _vinaStatus('No receptor atoms — load receptor first', true); return; }
    const pairs = (ligAtomIdx!=null && _pairsByLig[ligAtomIdx]) || [];
    const pairMap = {};
    pairs.forEach(p => { pairMap[p.rec_idx] = p.pair_e; });

    // Split atoms into two groups: background (never changes) and interacting (small, restyled)
    const bgAtoms  = _recAtoms.filter(a => pairMap[a.idx] === undefined);
    const hitAtoms = _recAtoms.filter(a => pairMap[a.idx] !== undefined);

    // Use global scale so colour is consistent across all ligand atoms
    const minE = _globalPairEMin;
    const maxE = _globalPairEMax;
    const rng  = (maxE - minE) || 1;

    function _cwColor(pairE) {
        const norm = (maxE - pairE) / rng;   // more negative -> 1 (red/favourable)
        const t = Math.max(0, Math.min(1, norm));
        let r, g, b;
        if (t < 0.5) { const s=t*2; r=~~(59+s*(221-59)); g=~~(76+s*(220-76)); b=~~(192+s*(220-192)); }
        else          { const s=(t-0.5)*2; r=~~(221+s*(180-221)); g=~~(220+s*(4-220)); b=~~(220+s*(38-220)); }
        return `rgb(${r},${g},${b})`;
    }

    const hitColors = hitAtoms.map(a => _cwColor(pairMap[a.idx]));

    const plotEl = document.getElementById('vPlotA');

    // Check if both traces already exist with the right background size
    const hasBg  = plotEl._fullData && plotEl._fullData.length === 2 && plotEl._fullData[0] && plotEl._fullData[0].name === 'RecBg'
                   && plotEl._fullData[0].x.length === bgAtoms.length;
    const hasHit = plotEl._fullData && plotEl._fullData[1] && plotEl._fullData[1].name === 'RecHit';

    if (hasBg && hasHit) {
        // Background never changes — only restyle the tiny interacting trace (~40 atoms)
        Plotly.restyle('vPlotA', {
            x: [hitAtoms.map(a=>a.x)],
            y: [hitAtoms.map(a=>a.y)],
            z: [hitAtoms.map(a=>a.z)],
            'marker.color': [hitColors],
            customdata: [hitAtoms.map(a=>({
                rec_idx: a.idx,
                pair_e:  (pairMap[a.idx] ?? 0).toFixed(5),
                name: a.name, res: a.resname+a.resseq,
            }))],
        }, [1]);
    } else {
        // First protein draw: build both traces, restore protein camera
        const bgTrace = {
            type:'scatter3d', mode:'markers',
            x:bgAtoms.map(a=>a.x), y:bgAtoms.map(a=>a.y), z:bgAtoms.map(a=>a.z),
            marker:{ size:4, color:'#1e2a3a', opacity:0.18, line:{width:0} },
            customdata:bgAtoms.map(a=>({rec_idx:a.idx,pair_e:'n/a',name:a.name,res:a.resname+a.resseq})),
            hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [%{customdata.rec_idx}]<br>' +
                          'xyz: (%{x:.2f}, %{y:.2f}, %{z:.2f}) Å<extra></extra>',
            name:'RecBg',
        };
        const hitTrace = {
            type:'scatter3d', mode:'markers',
            x:hitAtoms.map(a=>a.x), y:hitAtoms.map(a=>a.y), z:hitAtoms.map(a=>a.z),
            marker:{ size:4, color:hitColors, opacity:1.0, line:{width:0} },
            customdata:hitAtoms.map(a=>({
                rec_idx: a.idx,
                pair_e:  (pairMap[a.idx] ?? 0).toFixed(5),
                name: a.name, res: a.resname+a.resseq,
            })),
            hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [rec %{customdata.rec_idx}]<br>' +
                          'pair_e: %{customdata.pair_e} kcal/mol<br>' +
                          'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
            name:'RecHit',
        };
        // dragmode:'orbit' — Plotly's gl3d default is 'turntable', which pins the
        // scene's up-vector to +z and CLAMPS the polar angle at the poles. Azimuth
        // (left/right) is unbounded, so horizontal drags feel fine while vertical
        // drags hit an invisible wall and stop dead. 'orbit' is free trackball
        // rotation with no clamp — the same mode pose.js already uses. Trade-off:
        // the molecule can roll, i.e. "up" is no longer guaranteed vertical.
        const _sceneP = { bgcolor:'#05070f',
            dragmode:'orbit',
            xaxis:{showgrid:false,zeroline:false,showticklabels:false},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false},
            aspectmode:'data' };
        if (window._savedCameraProtein) _sceneP.camera = window._savedCameraProtein;
        Plotly.react('vPlotA', [bgTrace, hitTrace],
            {paper_bgcolor:'transparent', plot_bgcolor:'transparent', margin:{l:0,r:0,t:0,b:0},
             scene:_sceneP, showlegend:false, font:{color:'#94a3b8'}},
            {responsive:true, displayModeBar:false, displaylogo:false}
        ).then(()=>{
            document.getElementById('vPlotA').style.opacity='1';
            if (!plotEl._protCamListenerAttached) {
                plotEl._protCamListenerAttached = true;
                plotEl.on('plotly_relayout', ev => {
                    if (ev['scene.camera'] && _currentView === 'protein')
                        window._savedCameraProtein = JSON.parse(JSON.stringify(ev['scene.camera']));
                });
            }
            _ensureVoxelListener(); // safe to call repeatedly — attaches only once
        });
    }

    const ligAtom = (_ligAtomsCache.length && ligAtomIdx!=null) ? _ligAtomsCache[ligAtomIdx] : null;
    const label = ligAtom ? ligAtom.symbol : 'all';
    _vinaStatus('Protein view: ' + hitAtoms.length + ' receptor atoms interacting with ' + label);
}

// Combined view: ligand (bonds + labelled atoms) AND receptor (faint cloud + pair_e
// coloured contacts) in one scene. Ligand and receptor share the docking coordinate
// frame, so the ligand sits in its pocket. Trace order is
//   [RecBg, RecHit, Bonds, Atoms]  (receptor present)  or  [Bonds, Atoms]  (no receptor).
// Receptor colouring follows the selected ligand atom, exactly like the protein view;
// with no atom selected the receptor shows as the faint context cloud. Re-selecting an
// atom takes the same restyle fast-path the protein view uses (only trace [1] changes).
function _renderBothView(ligAtomIdx) {
    if (!_ligAtomsCache.length) { _vinaStatus('No ligand loaded yet', true); return; }

    const atoms  = _ligAtomsCache;
    const bonds  = window._lastBonds || [];
    const hasRec = _recAtoms.length > 0;

    // Receptor pair map for the selected ligand atom (empty → whole ligand, faint cloud).
    const pairs = (ligAtomIdx!=null && _pairsByLig[ligAtomIdx]) || [];
    const pairMap = {};
    pairs.forEach(p => { pairMap[p.rec_idx] = p.pair_e; });

    const bgAtoms  = hasRec ? _recAtoms.filter(a => pairMap[a.idx] === undefined) : [];
    const hitAtoms = hasRec ? _recAtoms.filter(a => pairMap[a.idx] !== undefined) : [];

    const minE = _globalPairEMin, maxE = _globalPairEMax, rng = (maxE - minE) || 1;
    function _cwColor(pairE) {
        const norm = (maxE - pairE) / rng;
        const t = Math.max(0, Math.min(1, norm));
        let r, g, b;
        if (t < 0.5) { const s=t*2; r=~~(59+s*(221-59)); g=~~(76+s*(220-76)); b=~~(192+s*(220-192)); }
        else          { const s=(t-0.5)*2; r=~~(221+s*(180-221)); g=~~(220+s*(4-220)); b=~~(220+s*(38-220)); }
        return `rgb(${r},${g},${b})`;
    }
    const hitColors = hitAtoms.map(a => _cwColor(pairMap[a.idx]));

    const plotEl = document.getElementById('vPlotA');
    const fd = plotEl._fullData || [];
    // Structural check only: 4 traces in the right order, same LIGAND atom count.
    // bgAtoms.length/hitAtoms.length are NOT checked here — they legitimately differ
    // for every ligand atom (each has a different contact count), so comparing them
    // was why this fast-path almost never triggered: switching from one atom to
    // another (nearly always a different contact count) made combinedReady false on
    // every click, forcing a full Plotly.react() rebuild each time — which is what
    // was resetting the camera on every row click, independent of any camera-restore
    // logic below.
    const combinedReady = hasRec && fd.length === 4
        && fd[0].name === 'RecBg'
        && fd[1].name === 'RecHit'
        && fd[3].name === 'Atoms'  && fd[3].x.length === atoms.length;

    // [voxel STEP CAM] Confirms which path _renderBothView takes and why — the fast
    // path (Plotly.restyle) never touches the camera; the rebuild path
    // (Plotly.react) can, which is what this whole investigation is about.
    _vlog('[voxel STEP CAM] _renderBothView branch decision', {
        ligAtomIdx, combinedReady, hasRec,
        fdLength: fd.length, fdNames: fd.map(t => t.name),
        bgAtomsCount: bgAtoms.length, hitAtomsCount: hitAtoms.length, atomsCount: atoms.length,
        savedCameraBothExists: !!window._savedCameraBoth,
        pathTaken: combinedReady ? 'FAST (restyle, camera untouched)' : 'REBUILD (react, camera via scene.camera + uirevision)',
    });

    if (combinedReady) {
        // Fast path: recolour BOTH receptor traces. Previously this only restyled
        // trace 1 (RecHit) and left trace 0 (RecBg) showing the PREVIOUS atom's
        // non-contact set — since which receptor atoms are "contact" vs "background"
        // changes per ligand atom, that would have left some atoms duplicated across
        // both traces and others in neither. Plotly.restyle never touches the camera,
        // so this path was always safe — the bug was that it was rarely reached.
        Plotly.restyle('vPlotA', {
            x: [bgAtoms.map(a=>a.x)], y: [bgAtoms.map(a=>a.y)], z: [bgAtoms.map(a=>a.z)],
            customdata: [bgAtoms.map(a=>({rec_idx:a.idx, pair_e:'n/a', name:a.name, res:a.resname+a.resseq}))],
        }, [0]);
        Plotly.restyle('vPlotA', {
            x: [hitAtoms.map(a=>a.x)],
            y: [hitAtoms.map(a=>a.y)],
            z: [hitAtoms.map(a=>a.z)],
            'marker.color': [hitColors],
            customdata: [hitAtoms.map(a=>({
                rec_idx: a.idx,
                pair_e:  (pairMap[a.idx] ?? 0).toFixed(5),
                name: a.name, res: a.resname+a.resseq,
            }))],
        }, [1]);

        // ── LIGAND COORDINATES MUST BE PUSHED TOO ────────────────────────────
        // This fast path used to restyle ONLY traces 0/1 (the receptor). That was
        // fine while the ligand never moved, but it silently broke the docked-pose
        // toggle: _voxelToggleLigFrame swapped every atom's x/y/z in _ligAtomsCache,
        // called this function, hit this branch, and the plotted ligand was never
        // redrawn — so the viewer kept showing the exported .pdbqt coordinates while
        // the code (and the chat card) believed the docked pose was on screen.
        // Verified against real data: C24 stayed at the input's (-27.543, 0.849,
        // 25.079) instead of moving to the docked (-27.338, 0.807, 24.680).
        // Restyling with unchanged values is a visual no-op and never touches the
        // camera, so this is safe to do on every call.
        const _lb = _vinaBondSegments(atoms, bonds);
        Plotly.restyle('vPlotA', {x:[_lb.x], y:[_lb.y], z:[_lb.z]}, [2]);       // Bonds
        Plotly.restyle('vPlotA', {
            x: [atoms.map(a=>a.x)], y: [atoms.map(a=>a.y)], z: [atoms.map(a=>a.z)],
        }, [3]);                                                                 // Atoms
    } else {
        // Full (re)build of the combined scene.
        const _bs = _vinaBondSegments(atoms, bonds);
        const bx = _bs.x, by = _bs.y, bz = _bs.z;
        const bondTrace = {type:'scatter3d',mode:'lines',x:bx,y:by,z:bz,
            line:{color:'#334155',width:3},hoverinfo:'skip',name:'Bonds'};
        const atomTrace = {type:'scatter3d',mode:'markers+text',
            x:atoms.map(a=>a.x), y:atoms.map(a=>a.y), z:atoms.map(a=>a.z),
            text:atoms.map(a=>a.symbol), textfont:{size:10,color:'#ffffff'}, textposition:'top center',
            marker:{ size:atoms.map(a=>7+(a.weight_norm??0)*12), color:atoms.map(a=>a.weight_norm??0),
                colorscale:[[0,'#3b4cc0'],[.25,'#88bbee'],[.5,'#dddddd'],[.75,'#ee8866'],[1,'#b40426']],
                cmin:0, cmax:1, showscale:false, line:{width:1,color:'#0f172a'} },
            customdata:atoms.map(a=>({idx:a.idx,raw:(a.weight_raw??0).toFixed(4),norm:(a.weight_norm??0).toFixed(4)})),
            hovertemplate:'<b>%{text}</b> (idx %{customdata.idx})<br>' +
                          'this_e: %{customdata.raw}  norm: %{customdata.norm}<br>' +
                          'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
            name:'Atoms'};

        const traces = [];
        if (hasRec) {
            const bgTrace = {type:'scatter3d',mode:'markers',
                x:bgAtoms.map(a=>a.x), y:bgAtoms.map(a=>a.y), z:bgAtoms.map(a=>a.z),
                marker:{size:4,color:'#1e2a3a',opacity:0.18,line:{width:0}},
                customdata:bgAtoms.map(a=>({rec_idx:a.idx,pair_e:'n/a',name:a.name,res:a.resname+a.resseq})),
                hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [%{customdata.rec_idx}]<br>' +
                              'xyz: (%{x:.2f}, %{y:.2f}, %{z:.2f}) Å<extra></extra>',
                name:'RecBg'};
            const hitTrace = {type:'scatter3d',mode:'markers',
                x:hitAtoms.map(a=>a.x), y:hitAtoms.map(a=>a.y), z:hitAtoms.map(a=>a.z),
                marker:{size:4,color:hitColors,opacity:1.0,line:{width:0}},
                customdata:hitAtoms.map(a=>({rec_idx:a.idx,pair_e:(pairMap[a.idx]??0).toFixed(5),name:a.name,res:a.resname+a.resseq})),
                hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [rec %{customdata.rec_idx}]<br>' +
                              'pair_e: %{customdata.pair_e} kcal/mol<br>' +
                              'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
                name:'RecHit'};
            traces.push(bgTrace, hitTrace);
        }
        traces.push(bondTrace, atomTrace);

        const _sceneB = { bgcolor:'#05070f',
            dragmode:'orbit',          // free vertical rotation — see _sceneP above
            xaxis:{showgrid:false,zeroline:false,showticklabels:false},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false},
            aspectmode:'data' };
        if (window._savedCameraBoth) _sceneB.camera = window._savedCameraBoth;

        Plotly.react('vPlotA', traces,
            {paper_bgcolor:'transparent', plot_bgcolor:'transparent', margin:{l:0,r:0,t:0,b:0},
             scene:_sceneB, showlegend:false, font:{color:'#94a3b8'},
             uirevision:'vina3d'},   // Plotly-native camera preservation across data
                                     // changes — backs up the manual scene.camera
                                     // restore above, which can be overridden by
                                     // Plotly's own autorange when trace extents shift.
            {responsive:true, displayModeBar:false, displaylogo:false}
        ).then(()=>{
            document.getElementById('vPlotA').style.opacity='1';
            if (!plotEl._bothCamListenerAttached) {
                plotEl._bothCamListenerAttached = true;
                plotEl.on('plotly_relayout', ev => {
                    if (ev['scene.camera'] && _currentView === 'both')
                        window._savedCameraBoth = JSON.parse(JSON.stringify(ev['scene.camera']));
                });
            }
            _ensureVoxelListener();
        });
    }

    const ligAtom = (ligAtomIdx!=null) ? _ligAtomsCache[ligAtomIdx] : null;
    const label   = ligAtom ? (ligAtom.symbol + ' ' + ligAtomIdx) : 'whole ligand';
    _vinaStatus('Both view: ligand + '
              + (hasRec ? (hitAtoms.length + ' receptor contact atoms · ' + label)
                        : 'no receptor loaded'));
}

/* ════════════════════════════════════════════════════════════════════════════
   Load file — validate path exists via backend
   ════════════════════════════════════════════════════════════════════════════ */
function _loadFile(which) {
    const inputId  = which === 'rec' ? 'recPath' : 'ligPath';
    const statusId = which === 'rec' ? 'recStatus' : 'ligStatus';
    const path     = document.getElementById(inputId).value.trim();
    if (!path) return;
    document.getElementById(statusId).textContent = '…';
    fetch('/vina_visualization/vina_check_file', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ path })
    })
    .then(r => r.json())
    .then(data => {
        const el = document.getElementById(statusId);
        if (data.exists) {
            el.textContent = `✓ ${data.n_atoms} atoms`;
            el.style.color = '#4ade80';
        } else {
            el.textContent = '✗ not found';
            el.style.color = '#f87171';
        }
    })
    .catch(() => {
        const el = document.getElementById(statusId);
        el.textContent = '✗ error'; el.style.color = '#f87171';
    });
}

/* ════════════════════════════════════════════════════════════════════════════
   Vina Dock — run docking via backend, then auto-parse log
   ════════════════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════════════════
   Docking live terminal — exact port of vina_ajax.html _showProgressPanel
   ════════════════════════════════════════════════════════════════════════════ */
function _showProgressPanel() {
    let panel = document.getElementById('progressPanel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'progressPanel';
        panel.style.cssText = [
            'position:fixed;bottom:24px;left:420px;',
            'width:680px;height:220px;',
            'background:#020408;',
            'border:1px solid #1e3a5f;border-radius:12px;',
            'font-family:"Courier New",monospace;font-size:10.5px;color:#4ade80;',
            'z-index:10150;display:flex;flex-direction:column;',
            'box-shadow:0 8px 40px rgba(0,0,0,.8);',
            'overflow:hidden;resize:both;min-width:380px;min-height:120px;',
        ].join('');

        const titleBar = document.createElement('div');
        titleBar.id = 'progressTitleBar';
        titleBar.style.cssText = [
            'display:flex;align-items:center;justify-content:space-between;',
            'padding:6px 10px;flex-shrink:0;',
            'background:#060d1a;border-bottom:1px solid #1e3a5f;',
            'cursor:grab;user-select:none;',
        ].join('');
        titleBar.innerHTML = `
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#f87171;"></div>
                <div style="width:10px;height:10px;border-radius:50%;background:#fbbf24;"></div>
                <div style="width:10px;height:10px;border-radius:50%;background:#4ade80;"></div>
                <span id="progressCmd" style="color:#64748b;font-size:9.5px;font-family:'Courier New',monospace;
                      margin-left:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:480px;">
                    ⚗️ Vina Dock
                </span>
            </div>
            <button onclick="document.getElementById('progressPanel').remove()"
                    style="color:#475569;background:none;border:none;cursor:pointer;font-size:13px;
                           line-height:1;padding:0 2px;transition:color .15s;"
                    onmouseover="this.style.color='#e2e8f0'" onmouseout="this.style.color='#475569'">✕</button>
        `;
        panel.appendChild(titleBar);

        const logArea = document.createElement('div');
        logArea.id = 'progressLog';
        logArea.style.cssText = [
            'flex:1;overflow-y:auto;padding:8px 14px;',
            'word-break:break-all;line-height:1.6;',
            'scrollbar-width:none;',
        ].join('');
        const _hss = document.createElement('style');
        _hss.textContent = '#progressLog::-webkit-scrollbar{display:none;}';
        document.head.appendChild(_hss);
        panel.appendChild(logArea);
        document.body.appendChild(panel);

        // Draggable
        let ox=0,oy=0,sx=0,sy=0,drag=false;
        titleBar.addEventListener('mousedown', e => {
            if (e.target.tagName==='BUTTON') return;
            drag=true;
            const r=panel.getBoundingClientRect();
            panel.style.left=r.left+'px'; panel.style.top=r.top+'px';
            panel.style.bottom='auto'; panel.style.right='auto';
            sx=e.clientX; sy=e.clientY; ox=r.left; oy=r.top;
            titleBar.style.cursor='grabbing'; e.preventDefault();
        });
        document.addEventListener('mousemove', e => {
            if (!drag) return;
            panel.style.left=Math.max(0,Math.min(window.innerWidth-panel.offsetWidth,ox+e.clientX-sx))+'px';
            panel.style.top=Math.max(0,Math.min(window.innerHeight-panel.offsetHeight,oy+e.clientY-sy))+'px';
        });
        document.addEventListener('mouseup',()=>{ drag=false; titleBar.style.cursor='grab'; });
    }

    document.getElementById('progressLog').innerHTML = '';

    const rec = document.getElementById('recPath')?.value?.trim() || '';
    const lig = document.getElementById('ligPath')?.value?.trim() || '';
    const recName = rec.split('/').pop();
    const ligName = lig.split('/').pop();
    const cmdEl = document.getElementById('progressCmd');
    if (cmdEl) cmdEl.textContent = `$ vina --receptor ${recName} --ligand ${ligName}`;

    _appendProgress(`$ vina --receptor ${rec}`);
    _appendProgress(`         --ligand   ${lig}`);
    _appendProgress(`         --center_x ${_GRID.cx} --center_y ${_GRID.cy} --center_z ${_GRID.cz}`);
    _appendProgress(`         --size_x ${_GRID.sx} --size_y ${_GRID.sy} --size_z ${_GRID.sz} --exhaustiveness 8`);
    _appendProgress('');
}

function _appendProgress(line) {
    const log = document.getElementById('progressLog');
    if (!log) return;

    // Full log tail block (energy assembly + mode table)
    if (line.startsWith('__LOGBLOCK__')) {
        const text = line.slice(12);
        const wrap = document.createElement('div');
        wrap.style.cssText = [
            'margin:8px 0;padding:10px 12px;',
            'background:#060d1a;border:1px solid #1e3a5f;border-radius:8px;',
            'font-family:"Courier New",monospace;font-size:10.5px;line-height:1.65;',
            'white-space:pre;overflow-x:auto;max-height:340px;overflow-y:auto;',
            'scrollbar-width:thin;scrollbar-color:#1e3a5f #060d1a;',
        ].join('');
        text.split('\n').forEach(tline => {
            const span = document.createElement('div');
            if (/mode\s+\|/.test(tline) || /affinity/.test(tline)) {
                span.style.color = '#fbbf24';
            } else if (/^\s*-+\+/.test(tline)) {
                span.style.color = '#1e3a5f';
            } else if (/^\s*1\s+-/.test(tline)) {
                span.style.color = '#4ade80'; span.style.fontWeight = '700';
            } else if (/^\s*\d+\s+-/.test(tline)) {
                span.style.color = '#94a3b8';
            } else if (/^\[/.test(tline.trim())) {
                span.style.color = '#818cf8';
            } else if (/=\s*-?[\d.]+\s*$/.test(tline)) {
                span.style.color = '#67e8f9';
            } else {
                span.style.color = '#475569';
            }
            span.textContent = tline;
            wrap.appendChild(span);
        });
        log.appendChild(wrap);
        log.scrollTop = log.scrollHeight;
        return;
    }

    // Mode result table
    if (line.startsWith('__MODETABLE__')) {
        const tableText = line.slice(13);
        const block = document.createElement('div');
        block.style.cssText = [
            'margin:6px 0;padding:10px 12px;',
            'background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;',
            'font-family:"Courier New",monospace;font-size:11px;line-height:1.7;white-space:pre;',
        ].join('');
        tableText.split('\n').forEach(tline => {
            const span = document.createElement('div');
            if (tline.includes('mode') && tline.includes('affinity')) span.style.color = '#fbbf24';
            else if (/^\s*1\s/.test(tline)) { span.style.color = '#4ade80'; span.style.fontWeight = '700'; }
            else span.style.color = '#94a3b8';
            span.textContent = tline;
            block.appendChild(span);
        });
        log.appendChild(block);
        log.scrollTop = log.scrollHeight;
        return;
    }

    // Batched bar update — __BARCH__stars:<stars> or __BARCH__sep:<sep>
    if (line.startsWith('__BARCH__')) {
        const colon   = line.indexOf(':');
        const key     = line.slice(9, colon);
        const barContent = line.slice(colon + 1);
        const divId   = 'pbar_' + key;
        let barDiv    = document.getElementById(divId);
        if (!barDiv) {
            barDiv = document.createElement('div');
            barDiv.id = divId;
            barDiv.style.cssText = `font-size:11px;font-family:'Courier New',monospace;white-space:pre;letter-spacing:0;`;
            log.appendChild(barDiv);
        }
        if (key === 'sep') {
            barDiv.style.color = '#1e3a5f';
            barDiv.textContent = '|----|----|----|----|----|----|----|----|----|----|';
        } else {
            barDiv.style.color = '#22d3ee';
            barDiv.textContent = barContent;
        }
        log.scrollTop = log.scrollHeight;
        return;
    }

    // Percentage line
    if (line.startsWith('__BAR__')) {
        const key     = line[7];
        const barContent = line.slice(8);
        const divId   = 'pbar_' + key;
        let barDiv = document.getElementById(divId);
        if (!barDiv) {
            barDiv = document.createElement('div');
            barDiv.id = divId;
            barDiv.style.cssText = `color:#64748b;font-size:11px;font-family:'Courier New',monospace;white-space:pre;`;
            log.appendChild(barDiv);
        }
        barDiv.textContent = barContent;
        log.scrollTop = log.scrollHeight;
        return;
    }

    // Regular lines — colour coding
    let color = '#4ade80';
    if      (line.includes('ERROR') || line.includes('error'))     color = '#f87171';
    else if (line.includes('[non_cache'))                           color = '#22d3ee';
    else if (line.includes('mode |') || line.includes('affinity')) color = '#fbbf24';
    else if (line.includes('✓'))                                    color = '#a3e635';
    else if (line.includes('⚠') || line.includes('WARNING'))       color = '#fb923c';
    else if (line.includes('Performing docking'))                   color = '#818cf8';
    else if (line.includes('Computing Vina'))                       color = '#818cf8';

    const div = document.createElement('div');
    div.style.cssText = `color:${color};font-size:11px;font-family:'Courier New',monospace;white-space:pre-wrap;word-break:break-word;`;
    div.textContent = line;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

/* ════════════════════════════════════════════════════════════════════════════
   Vina Dock — exact port of vina_ajax.html, with vina_tail_log for non_cache lines
   ════════════════════════════════════════════════════════════════════════════ */
function _vinaDock() {
    // Pull the current Box ctr / len inputs into _GRID first, so both the command
    // preview below and the POST body use exactly what's in the box fields right now
    // (covers the case where Dock is clicked before an input's change event fires).
    if (typeof _readGridInputs === 'function') _readGridInputs();

    const btn = document.getElementById('vinaDockBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-ring" style="width:10px;height:10px;border-width:2px;display:inline-block"></span> Docking…';
    _vinaStatus('Starting Vina docking…'); _vinaBusy(true);

    _showProgressPanel();

    // ── SSE 1: vina_dock_progress — progress bar, grid info, mode table ──
    let es = new EventSource('/vina_visualization/vina_dock_progress');
    es.onmessage = ev => {
        if (ev.data === '__DONE__') { es.close(); return; }
        if (ev.data.startsWith(':')) return;
        if (/^[-|]{6,}/.test(ev.data.trim())) return;
        _appendProgress(ev.data);
        if (ev.data.startsWith('__BARCH__stars:')) {
            const stars = ev.data.slice(15).length;
            const pct   = Math.min(100, Math.round((stars / 51) * 100));
            _vinaStatus(`Docking… ${pct}%`);
        } else if (ev.data.includes('mode |')) {
            _vinaStatus('Docking complete — reading results…');
        }
    };
    es.onerror = () => { es.close(); };

    // ── SSE 2: vina_tail_log — [non_cache::clip] + [non_cache::eval lig_atom] lines ──
    let es2 = new EventSource('/vina_visualization/vina_tail_log');
    es2.onmessage = ev => {
        if (ev.data === '__DONE__') { es2.close(); return; }
        if (!ev.data || ev.data.startsWith(':')) return;
        if (ev.data.includes('[non_cache::')) {
            _appendProgress(ev.data);   // colour-coded cyan via _appendProgress
        }
    };
    es2.onerror = () => { es2.close(); };

    // POST the dock request
    fetch('/vina_visualization/vina_dock', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
            receptor_path: document.getElementById('recPath').value.trim(),
            ligand_path:   document.getElementById('ligPath').value.trim(),
            // Editable docking box (from the Box ctr / len inputs). The server must
            // read these for the edited box to affect docking; extra keys are safely
            // ignored by endpoints that only pull receptor_path / ligand_path.
            center_x: _GRID.cx, center_y: _GRID.cy, center_z: _GRID.cz,
            size_x:   _GRID.sx, size_y:   _GRID.sy, size_z:   _GRID.sz,
        })
    })
    .then(r => r.json())
    .then(data => {
        es.close(); es2.close();
        btn.disabled = false;
        btn.innerHTML = '⚗️ Vina Dock';
        _vinaBusy(false);

        if (data.status !== 'success') {
            _vinaStatus('Dock error: '+data.message, true);
            _appendProgress('ERROR: '+data.message);
            return;
        }

        if (data.tail_block) {
            _appendProgress('__LOGBLOCK__' + data.tail_block);
        } else if (data.mode_table) {
            _appendProgress('__MODETABLE__' + data.mode_table);
        } else {
            _appendProgress(`✓ Best affinity = ${data.best_affinity?.toFixed(4)??'?'} kcal/mol`);
        }
        _appendProgress(`✓ Log: ${data.log_file??'—'}`);

        _vinaStatus(`Done · mode 1 = ${data.best_affinity?.toFixed(3)??'?'} kcal/mol · ${data.n_pair_lines??'?'} pairs`);

        setTimeout(() => {
            _clearAllHighlights();
            _highlightBtnUntilClick('vizBtn');
        }, 500);

        const miniEl = document.getElementById('miniChat');
        if (miniEl && miniEl.style.display === 'flex') {
            const aff = data.best_affinity?.toFixed(3) ?? '?';
            _miniChatAppend('ai',
                `Docking complete! Best affinity: **${aff} kcal/mol**\n\nNow click the flashing **Visualize** button to render the 3D per-atom energy decomposition.`
            );
        }

        const vinaLogEl = document.getElementById('vinaLog');
        if (vinaLogEl && data.log_text) vinaLogEl.value = data.log_text;

        if (data.lig_atoms && data.lig_atoms.length > 0) {
            _vinaAtoms     = data.lig_atoms;
            _ligAtomsCache = data.lig_atoms;
            _selAtomIdx    = null;
            _pairsByLig    = data.pairs_by_lig_atom || {};
            const _allPairE = Object.values(_pairsByLig).flatMap(ps => ps.map(p => p.pair_e));
            _globalPairEMin = _allPairE.length ? Math.min(..._allPairE) : -0.1;
            _globalPairEMax = _allPairE.length ? Math.max(..._allPairE) :  0.0;
            window._lastBonds = data.bonds || [];

            _render3D('vPlotA', data.lig_atoms, data.bonds || [], false);
            _renderBar('vBarA', data.weight_vector, 'vPlotA', 'vTableA');
            if (typeof (window._renderTable || _renderTable) === 'function')
                (window._renderTable || _renderTable)('vTableA', data.lig_atoms, 'vBarA', 'vPlotA', data.weight_vector);

            const te = data.total_e;
            document.getElementById('vScoreA').textContent     = te!=null ? te.toFixed(3)+' kcal/mol' : '—';
            document.getElementById('vScoreA').className        = 'score-val '+(te<0?'score-better':'score-worse');
            document.getElementById('vScoreANote').textContent  = 'kcal/mol · Vina lig_grids E';
            document.getElementById('vBarLabel').textContent    = 'this_e · all atoms';
            document.getElementById('vBarNote').innerHTML       = '<code class="text-emerald-400">this_e</code> = Σ pair_e after curl';
            document.getElementById('vStatusA').textContent     = `${data.n_lig_atoms} atoms · Σ = ${te?.toFixed(4)??'?'}`;
            document.getElementById('vSpinnerA').style.display  = 'none';
            document.getElementById('vPlotA').style.opacity     = '1';
            _setView('both');
        } else if (data.message) {
            _appendProgress('⚠ '+data.message);
        }
    })
    .catch(e => {
        es.close(); es2.close();
        btn.disabled = false; btn.innerHTML = '⚗️ Vina Dock';
        _vinaBusy(false); _vinaStatus('Dock error: '+e, true);
        _appendProgress('ERROR: '+e);
    });
}

/* ════════════════════════════════════════════════════════════════════════════
   Vina Visualize — reads log file, builds weight_a-compatible payload, renders identically
   ════════════════════════════════════════════════════════════════════════════ */
function _vinaVisualize() {
    // [voxel STEP -1] Confirms _vinaVisualize was actually invoked. If Visualize was
    // never clicked (or its onclick is somehow broken) this line settles that
    // immediately, rather than us inferring it from the absence of later steps.
    _vlog('[voxel STEP -1] _vinaVisualize() entered', {
        recPath: (document.getElementById('recPath')||{}).value,
        ligPath: (document.getElementById('ligPath')||{}).value,
    });

    const btn = document.getElementById('vizBtn');
    btn.disabled = true;
    _vinaStatus('Reading log file…'); _vinaBusy(true);
    document.getElementById('vSpinnerA').style.display = 'flex';
    document.getElementById('vSpinnerAMsg').textContent = 'Reading vina_non_cache.log…';
    document.getElementById('vPlotA').style.opacity = '0';

    const _reqBody = {
        receptor_path: document.getElementById('recPath').value.trim(),
        ligand_path:   document.getElementById('ligPath').value.trim(),
    };
    _vlog('[voxel STEP -1] fetch → /vina_visualization/vina_parse_log', _reqBody);

    fetch('/vina_visualization/vina_parse_log', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(_reqBody)
    })
    .then(r => {
        // [voxel STEP -1] Raw HTTP response, before attempting to parse JSON — a
        // non-2xx status or a non-JSON body (e.g. an HTML error page from a server
        // crash) would otherwise fail silently inside the next .then().
        _vlog('[voxel STEP -1] fetch response received', {ok: r.ok, status: r.status, statusText: r.statusText});
        return r.json();
    })
    .then(data => {
        btn.disabled = false; _vinaBusy(false);
        // [voxel STEP -1] The parsed payload's status — this is the single most
        // important checkpoint: if this doesn't say 'success', NONE of the render
        // functions below ever run, and #vTableA stays as the static placeholder
        // forever, which is exactly what "zero STEP 1 logs" looks like from outside.
        _vlog('[voxel STEP -1] response parsed', {
            status: data.status, message: data.message,
            atomCount: (data.atoms||[]).length,
            hasWeightVector: !!data.weight_vector,
        });
        if (data.status !== 'success') {
            console.error('%c[voxel STEP -1] VISUALIZE FAILED — _renderTable will NOT run',
                'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px',
                '\nmessage: ' + data.message);
            _vinaStatus('Error: ' + data.message, true);
            document.getElementById('vSpinnerAMsg').textContent = 'Error: ' + data.message;
            return;
        }

        _ligAtomsCache = data.atoms;
        _recAtoms      = data.rec_atoms || [];
        _pairsByLig    = data.pairs_by_lig_atom || {};
        _dockedPosePath = data.docked_pose_path || null;   // which _out.pdbqt was read
        // Compute global pair_e range across ALL lig atoms so colour scale is consistent
        const _allPairE = Object.values(_pairsByLig).flatMap(ps => ps.map(p => p.pair_e));
        _globalPairEMin = _allPairE.length ? Math.min(..._allPairE) : -0.1;
        _globalPairEMax = _allPairE.length ? Math.max(..._allPairE) :  0.0;
        window._lastBonds = data.bonds;
        _render3D('vPlotA', data.atoms, data.bonds, false);
        _renderBar('vBarA', data.weight_vector, 'vPlotA', 'vTableA');
        _vlog('[voxel STEP -1] about to call _renderTable', {atomCount: (data.atoms||[]).length});
        // Call the LOCKED window._renderTable explicitly — not the bare identifier,
        // which a later `function _renderTable(){}` in another script could shadow in
        // this scope even though window._renderTable is locked.
        (window._renderTable || _renderTable)('vTableA', data.atoms, 'vBarA', 'vPlotA', data.weight_vector);
        _setView('both');

        // Score card → Vina affinity (mode 1) — matches the mode table number
        const affinity = data.best_affinity;
        const te       = data.total_e;
        const scoreEl  = document.getElementById('vScoreA');
        scoreEl.textContent = affinity != null ? affinity.toFixed(3) + ' kcal/mol' : '—';
        scoreEl.className   = 'score-val ' + (affinity != null && affinity < 0 ? 'score-better' : 'score-worse');
        // State which pose is on screen. A Vina _out.pdbqt contains one MODEL per
        // docked mode and the viewer renders MODEL 1 (the best); saying so prevents
        // the coordinates being compared against a different MODEL in the same file.
        const nModels = data.lig_model_count || 1;
        const poseTxt = nModels > 1
            ? ` · showing pose ${data.lig_model_shown || 1} of ${nModels} (best)`
            : '';
        document.getElementById('vScoreANote').textContent =
            `Vina affinity · mode 1 (lig_grids = ${te != null ? te.toFixed(3) : '—'})${poseTxt}`;

        // Labels
        document.getElementById('vBarLabel').textContent = 'this_e · top 50';
        document.getElementById('vBarNote').innerHTML =
            '<code class="text-emerald-400">this_e</code> = Σ pair_e  ·  ' +
            '<code class="text-cyan-400">weight_a[i+1]</code> analogue';
        document.getElementById('vStatusA').textContent =
            `${data.atoms.length} heavy atoms · ${data.bonds.length} bonds · Σ = ${te?.toFixed(4)??'?'}`;

        document.getElementById('vSpinnerA').style.display = 'none';
        document.getElementById('vPlotA').style.opacity = '1';
        _vinaStatus(`Loaded ${data.atoms.length} atoms · best mode = ${data.best_affinity?.toFixed(3)??'?'} kcal/mol`);

        // Force Plotly to re-measure dimensions — the modal flex layout may have
        // been zero-height when react() was called
        setTimeout(() => {
            const vp = document.getElementById('vPlotA');
            if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
        }, 80);
    })
    .catch(e => {
        btn.disabled = false; _vinaBusy(false);
        console.error('%c[voxel STEP -1] VISUALIZE THREW / FETCH FAILED — _renderTable will NOT run',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px', e);
        _vinaStatus('Fetch error: ' + e, true);
        document.getElementById('vSpinnerAMsg').textContent = 'Fetch error: ' + e;
    });
}

/* ════════════════════════════════════════════════════════════════════════════
   Pair drawer
   ════════════════════════════════════════════════════════════════════════════ */
function _openDrawer(atom) {
    const la  = atom.lig_atom;
    const lbl = la ? `${la.name} (${la.atype}, xs=${atom.xs})` : `lig_atom[${atom.lig_idx}] ${atom.xs_label}`;
    document.getElementById('drawerTitle').textContent = lbl;
    document.getElementById('drawerSub').textContent   =
        `this_e=${atom.this_e.toFixed(5)} · ${atom.n_pairs} pairs · pair_sum=${atom.pair_sum.toFixed(5)}`;

    const rows = (atom.top_pairs??[]).map((p,pi) => {
        const ra  = p.rec_atom;
        const rLbl= ra ? `${ra.name} ${ra.resname}${ra.resseq}` : `rec[${p.rec_idx}]`;
        const ec  = p.pair_e<0?'text-emerald-400':'text-red-400';
        return `
<div class="pair-row flex items-center gap-2 py-1.5 border-b border-slate-800/50 px-1"
     data-pi="${pi}" onclick="_selectPair(${pi})">
  <span class="text-slate-600 w-4 text-right text-[9px] flex-shrink-0">${pi+1}</span>
  <div class="flex-1 min-w-0">
    <div class="font-mono text-[10px] text-slate-200 truncate">${rLbl}</div>
    <div class="text-[9px] text-slate-500">${p.rec_xs_label} · r=${p.r.toFixed(3)}Å · s=${p.s.toFixed(3)}Å</div>
  </div>
  <span class="font-mono font-semibold text-[10px] flex-shrink-0 ${ec}">${p.pair_e.toFixed(5)}</span>
</div>`;
    }).join('');

    document.getElementById('pairList').innerHTML =
        `<p class="text-[9px] text-slate-600 pt-1 pb-1">top ${atom.top_pairs?.length??0} / ${atom.n_pairs} total</p>`
        + (rows || '<p class="text-[9px] text-slate-700">No pairs.</p>')
        + '<p class="text-[9px] text-slate-700 mt-2">pair_e = w₁G₁+w₂G₂+w₃R+w₄H+w₅B</p>';
    document.getElementById('termPanel').innerHTML = '<p class="text-[9px] text-slate-700">Click a pair.</p>';
    document.getElementById('pairDrawer').classList.remove('hidden');
}

function _closeDrawer() { document.getElementById('pairDrawer').classList.add('hidden'); }

function _selectPair(pi) {
    document.querySelectorAll('.pair-row').forEach((el,i) =>
        el.classList.toggle('selected', i===pi));
    const atom = _vinaAtoms[_selAtomIdx];
    const pair = atom.top_pairs[pi];
    const t    = pair.terms;
    const ra   = pair.rec_atom, la = atom.lig_atom;

    const terms = [
        {name:'Gauss 1',    color:'#60a5fa',w:-0.035579,raw:t.gauss1_raw,     wv:t.gauss1_w},
        {name:'Gauss 2',    color:'#818cf8',w:-0.005156,raw:t.gauss2_raw,     wv:t.gauss2_w},
        {name:'Repulsion',  color:'#f87171',w: 0.840245,raw:t.repulsion_raw,  wv:t.repulsion_w},
        {name:'Hydrophobic',color:'#fb923c',w:-0.035069,raw:t.hydrophobic_raw,wv:t.hydrophobic_w},
        {name:'H-bond',     color:'#34d399',w:-0.587439,raw:t.hbond_raw,      wv:t.hbond_w},
    ];
    const maxAbs = Math.max(...terms.map(tt=>Math.abs(tt.wv)), 1e-9);
    const ligLbl = la?`${la.name}(${la.resname})`:`[${atom.lig_idx}]`;
    const recLbl = ra?`${ra.name}(${ra.resname}${ra.resseq})`:`rec[${pair.rec_idx}]`;

    const rows = terms.map(tt => {
        const pct=(Math.abs(tt.wv)/maxAbs*100).toFixed(1);
        const ec =tt.wv<0?'text-emerald-400':tt.wv>0?'text-red-400':'text-slate-500';
        return `
<div class="py-1 border-b border-slate-800/50">
  <div class="flex justify-between text-[9px] mb-0.5">
    <span class="text-slate-300 font-medium">${tt.name}</span>
    <span class="font-mono ${ec}">${tt.wv.toFixed(5)}</span>
  </div>
  <div class="h-[4px] rounded bg-slate-800">
    <div style="width:${pct}%;background:${tt.color};height:100%;border-radius:3px"></div>
  </div>
  <div class="text-[8px] text-slate-600 mt-0.5">raw=${tt.raw.toFixed(4)} w=${tt.w}</div>
</div>`;
    }).join('');

    const fc=t.pair_e_formula<0?'text-emerald-400':'text-red-400';
    document.getElementById('termPanel').innerHTML = `
<div class="pb-1.5 mb-1 border-b border-slate-800 text-[8px] text-slate-500">
  <div>${ligLbl}↔${recLbl}</div>
  <div>r=${pair.r.toFixed(4)} s=${t.s.toFixed(4)}Å</div>
</div>
${rows}
<div class="mt-2 pt-1.5 border-t border-slate-800 space-y-0.5 text-[9px]">
  <div class="flex justify-between font-semibold">
    <span class="text-slate-400">Σ formula</span>
    <span class="font-mono ${fc}">${t.pair_e_formula.toFixed(5)}</span>
  </div>
  <div class="flex justify-between text-slate-600">
    <span>eval_fast</span><span class="font-mono">${pair.pair_e.toFixed(5)}</span>
  </div>
  <div class="flex justify-between text-slate-600">
    <span>Δ interp</span><span class="font-mono">${(pair.pair_e-t.pair_e_formula).toFixed(5)}</span>
  </div>
</div>`;
}

/* ════════════════════════════════════════════════════════════════════════════
   Utilities
   ════════════════════════════════════════════════════════════════════════════ */
function _cwColor(val, minV, maxV) {
    const t = (val-minV)/(maxV-minV+1e-9);
    if (t < 0.5) {
        const k=t*2;
        return `rgb(${~~(59+k*(221-59))},${~~(76+k*(220-76))},${~~(192+k*(220-192))})`;
    } else {
        const k=(t-0.5)*2;
        return `rgb(${~~(221+k*(180-221))},${~~(220+k*(4-220))},${~~(220+k*(38-220))})`;
    }
}
function _vinaStatus(msg,err=false){
    const el=document.getElementById('vinaStatus');
    el.textContent=msg; el.style.color=err?'#f87171':'#475569';
}
function _vinaBusy(on){
    document.getElementById('vinaSpinner').classList.toggle('hidden',!on);
}

// ── Bond line segments, WITH multiplicity ────────────────────────────────────
// Every bond used to be pushed as one straight line, so a benzo ring drew as six
// identical single lines and there was no way to tell an aromatic/double bond from
// a single one (and a triple bond would have looked the same again). The server now
// sends {begin, end, order, aromatic} — order is the Kekule order 1/2/3 — so draw
// order 2 as two parallel lines and order 3 as three.
//
// The extra lines are offset along a vector perpendicular to the bond axis, chosen
// as the in-plane direction pointing toward the mean of the atoms attached to
// either end. Inside a ring that direction points into the ring, which is where a
// chemist draws the second line; for an acyclic C=O it points toward the
// substituents, which reads correctly too. Offset lines are shortened toward the
// midpoint so they do not poke out past the atom markers.
//
// Returns one flat {x,y,z} triple with null separators — i.e. still a SINGLE
// scatter3d trace, so the two-trace fast paths in _render3D / _renderBothView keep
// working untouched.
function _vinaBondSegments(atoms, bonds) {
    const X = [], Y = [], Z = [];
    const seg = (a, b) => {
        X.push(a[0], b[0], null); Y.push(a[1], b[1], null); Z.push(a[2], b[2], null);
    };
    if (!atoms || !bonds) return {x: X, y: Y, z: Z};

    const OFF = 0.16;                       // Å between parallel lines

    // neighbour map — used only to pick the plane the extra lines lie in
    const nb = new Map();
    bonds.forEach(b => {
        if (!nb.has(b.begin)) nb.set(b.begin, []);
        if (!nb.has(b.end))   nb.set(b.end,   []);
        nb.get(b.begin).push(b.end); nb.get(b.end).push(b.begin);
    });

    bonds.forEach(b => {
        const A = atoms[b.begin], B = atoms[b.end];
        if (!A || !B) return;
        const p = [A.x, A.y, A.z], q = [B.x, B.y, B.z];
        const order = Math.max(1, Math.min(3, Math.round(Number(b.order) || 1)));
        if (order === 1) { seg(p, q); return; }

        let ax = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        const L = Math.hypot(ax[0], ax[1], ax[2]) || 1;
        ax = [ax[0] / L, ax[1] / L, ax[2] / L];
        const mid = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];

        // mean position of everything else bonded to either end
        let rx = 0, ry = 0, rz = 0, k = 0;
        [[b.begin, b.end], [b.end, b.begin]].forEach(pair => {
            (nb.get(pair[0]) || []).forEach(w => {
                if (w === pair[1]) return;
                const W = atoms[w];
                if (!W) return;
                rx += W.x; ry += W.y; rz += W.z; k++;
            });
        });

        let perp;
        if (k) {
            const r = [rx / k - mid[0], ry / k - mid[1], rz / k - mid[2]];
            const d = r[0] * ax[0] + r[1] * ax[1] + r[2] * ax[2];
            perp = [r[0] - d * ax[0], r[1] - d * ax[1], r[2] - d * ax[2]];
        } else {
            perp = [0, 0, 0];
        }
        let pl = Math.hypot(perp[0], perp[1], perp[2]);
        if (pl < 1e-6) {                    // degenerate: any vector off the axis
            const t = Math.abs(ax[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
            perp = [ax[1] * t[2] - ax[2] * t[1],
                    ax[2] * t[0] - ax[0] * t[2],
                    ax[0] * t[1] - ax[1] * t[0]];
            pl = Math.hypot(perp[0], perp[1], perp[2]) || 1;
        }
        perp = [perp[0] / pl, perp[1] / pl, perp[2] / pl];

        const shrink = f => [
            [mid[0] + (p[0] - mid[0]) * f, mid[1] + (p[1] - mid[1]) * f, mid[2] + (p[2] - mid[2]) * f],
            [mid[0] + (q[0] - mid[0]) * f, mid[1] + (q[1] - mid[1]) * f, mid[2] + (q[2] - mid[2]) * f],
        ];
        const shift = (pt, s) => [pt[0] + perp[0] * OFF * s,
                                  pt[1] + perp[1] * OFF * s,
                                  pt[2] + perp[2] * OFF * s];

        seg(p, q);                                          // the bond axis itself
        if (order === 2) {
            const e = shrink(0.80);
            seg(shift(e[0], 1), shift(e[1], 1));
        } else {
            const e = shrink(0.85);
            seg(shift(e[0],  1), shift(e[1],  1));
            seg(shift(e[0], -1), shift(e[1], -1));
        }
    });
    return {x: X, y: Y, z: Z};
}

// ── 3-D scatter + bonds (verbatim from ajax_weight_a.html) ───────────────────
function _render3D(divId, atoms, bonds, compact) {
    const atomTrace = {
        type:'scatter3d', mode:'markers+text',
        x:atoms.map(a=>a.x), y:atoms.map(a=>a.y), z:atoms.map(a=>a.z),
        text:         atoms.map(a=>a.symbol),
        textfont:     { size: compact?8:10, color:'#ffffff' },
        textposition: 'top center',
        marker: {
            size:  atoms.map(a=>(compact?5:7)+(a.weight_norm??0)*12),
            color: atoms.map(a=>a.weight_norm??0),
            colorscale:[[0,'#3b4cc0'],[.25,'#88bbee'],[.5,'#dddddd'],[.75,'#ee8866'],[1,'#b40426']],
            cmin:0, cmax:1, showscale:false,
            line:{width:1,color:'#0f172a'},
        },
        customdata: atoms.map(a=>({idx:a.idx,raw:(a.weight_raw??0).toFixed(4),norm:(a.weight_norm??0).toFixed(4)})),
        hovertemplate:
            '<b>%{text}</b> (idx %{customdata.idx})<br>' +
            'this_e: %{customdata.raw}  norm: %{customdata.norm}<br>' +
            'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
        name:'Atoms',
    };
    const _bs = _vinaBondSegments(atoms, bonds);
    const bx = _bs.x, by = _bs.y, bz = _bs.z;
    const _el3d  = document.getElementById(divId);
    const _existsLig = _el3d._fullData && _el3d._fullData.length === 2
        && _el3d._fullData[1] && _el3d._fullData[1].name === 'Atoms'
        && _el3d._fullData[1].x && _el3d._fullData[1].x.length === atoms.length;

    if (_existsLig) {
        // Same atom count — restyle colours, sizes AND coordinates; camera untouched.
        // The coordinates used to be omitted here (colours/sizes only), which meant a
        // pure coordinate change — exactly what the docked-pose toggle does — rendered
        // as a no-op while the code believed the atoms had moved. Same root cause as
        // the fast path in _renderBothView; see the note there.
        const newColors = atoms.map(a=>a.weight_norm);
        const newSizes  = atoms.map(a=>(compact?5:7)+a.weight_norm*12);
        Plotly.restyle(divId, {x:[bx], y:[by], z:[bz]}, [0]);                    // Bonds
        Plotly.restyle(divId, {
            'marker.color':[newColors], 'marker.size':[newSizes],
            x:[atoms.map(a=>a.x)], y:[atoms.map(a=>a.y)], z:[atoms.map(a=>a.z)],
        }, [1]);                                                                 // Atoms
        document.getElementById(divId).style.opacity='1';
        if (divId === 'vPlotA') _ensureVoxelListener();   // keep atom-click inspector bound
    } else {
        // First draw or atom count changed: full react, then attach camera listener once
        const _sceneLayout = { bgcolor:'#05070f',
            dragmode:'orbit',          // free vertical rotation — see _sceneP above
            xaxis:{showgrid:false,zeroline:false,showticklabels:false},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false},
            aspectmode:'data' };
        if (window._savedCamera) _sceneLayout.camera = window._savedCamera;
        Plotly.react(divId,
            [{type:'scatter3d',mode:'lines',x:bx,y:by,z:bz,
              line:{color:'#334155',width:3},hoverinfo:'skip',name:'Bonds'}, atomTrace],
            { paper_bgcolor:'transparent', plot_bgcolor:'transparent',
              margin:{l:0,r:0,t:0,b:0}, scene:_sceneLayout,
              showlegend:false, font:{color:'#94a3b8'} },
            { responsive:true, displayModeBar:false,
              modeBarButtonsToRemove:['toImage','resetCameraLastSave3d'],
              displaylogo:false }
        ).then(()=>{
            document.getElementById(divId).style.opacity='1';
            // Attach camera-save listener once per plot element
            if (!_el3d._camListenerAttached) {
                _el3d._camListenerAttached = true;
                _el3d.on('plotly_relayout', ev => {
                    if (ev['scene.camera'] && !window._cameraLocked) window._savedCamera = JSON.parse(JSON.stringify(ev['scene.camera']));
                });
            }
            if (divId === 'vPlotA') _ensureVoxelListener();   // keep atom-click inspector bound
        });
    }
}

// ── Bar chart (verbatim from ajax_weight_a.html) ─────────────────────────────
function _renderBar(divId, wv, plotDivId, tableDiv) {
    const top  = [...wv.top_indices].sort((a,b)=>a-b);
    const vals = top.map(i=>wv.values[i]);
    const syms = top.map(i=>(wv.atom_symbols||[])[i]||'?');
    const cols = vals.map(v=>v<=0?'#22d3ee':'#f87171');

    Plotly.react(divId, [{
        type:'bar', x:top, y:vals,
        marker:{color:cols},
        customdata:syms,
        hovertemplate:'<b>%{customdata}%{x}</b><br>this_e: %{y:.4f} kcal/mol<extra></extra>',
        selected:   {marker:{opacity:1}},
        unselected: {marker:{opacity:0.4}},
    }], {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:30,r:2,t:2,b:22},
        xaxis:{title:{text:'Atom idx',font:{color:'#475569',size:8}},
               tickfont:{color:'#475569',size:7},gridcolor:'#1e293b'},
        yaxis:{tickfont:{color:'#475569',size:7},gridcolor:'#1e293b',zerolinecolor:'#334155'},
        bargap:.15, clickmode:'event',
    }, {responsive:true,displayModeBar:false});

    const barEl = document.getElementById(divId);
    barEl.removeAllListeners && barEl.removeAllListeners('plotly_click');
    barEl.on('plotly_click', function(ev) {
        if (!ev.points.length) return;
        _highlightAtom(ev.points[0].x, tableDiv, plotDivId, wv);
    });
}

// ── Atom table (verbatim from ajax_weight_a.html) ────────────────────────────

function _renderTable(divId, atoms, barDivId, plotDivId, wv) {
    // [voxel TRACE] Who called _renderTable, and is THIS the current build's copy? The
    // stack's second frame names the caller script:line. If two vina.js copies ever
    // coexist, the build marker here tells you which one's _renderTable actually ran.
    console.log('%c[voxel TRACE] _renderTable CALLED — build ' + _VINA_JS_BUILD,
        'background:#064e3b;color:#6ee7b7;font-weight:bold;padding:1px 4px',
        '\ndivId=' + divId + '  atoms=' + (atoms && atoms.length)
      + '\ncaller stack:\n' + (new Error().stack || '(no stack)'));
    if (typeof window._voxelBanner === 'function')
        window._voxelBanner('🅣 _renderTable CALLED (build ' + _VINA_JS_BUILD + ')', '#6ee7b7');

    const sorted = [...atoms].sort((a,b)=>b.weight_norm-a.weight_norm);
    const containerEl = document.getElementById(divId);
    if (!containerEl) { console.error('[voxel] _renderTable: #'+divId+' not found'); return; }

    // [voxel STEP 0.5] proved something rewrites/hijacks the inline onclick="..."
    // ATTRIBUTE string before the row is ever clicked — the stack trace showed
    // stopImmediatePropagation() being called from inside "HTMLDivElement.onclick"
    // while our own _vinaRowClickInline never ran (no STEP 1.5). Whatever's doing that
    // targets the HTML ATTRIBUTE specifically (likely a MutationObserver that scans for
    // onclick="..." strings and swaps them). So this version exposes NO onclick
    // attribute at all: every row is a real DOM node built with createElement, and the
    // click handler is attached via addEventListener as a plain JS closure — there is
    // no attribute string for anything to find or rewrite.
    containerEl.textContent = '';   // clear previous render (removes old nodes + their listeners)

    const hdr = document.createElement('p');
    hdr.className = 'text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-1';
    hdr.textContent = 'Atoms ↓  —  most favourable first · click to locate';
    containerEl.appendChild(hdr);

    sorted.forEach(a => {
        const row = document.createElement('div');
        row.dataset.atomIdx = String(a.idx);
        row._voxelBuiltBy = _VINA_JS_BUILD;   // see STEP 0's stale-deployment check
        row.style.cssText = 'cursor:pointer;border-radius:4px;border:1px solid transparent;transition:background .15s';
        row.className = 'flex items-center justify-between py-0.5 px-0.5 border-b border-slate-800/40';

        const left = document.createElement('span');
        left.className = 'flex items-center gap-1';

        const dot = document.createElement('span');
        dot.className = 'w-2 h-2 rounded-full flex-shrink-0';
        dot.style.background = a.color;

        const label = document.createElement('span');
        label.className = 'font-mono';
        label.textContent = a.symbol;
        const sub = document.createElement('sub');
        sub.className = 'text-slate-600';
        sub.textContent = String(a.idx);
        label.appendChild(sub);

        left.appendChild(dot);
        left.appendChild(label);

        const val = document.createElement('span');
        val.className = 'font-mono text-slate-300';
        val.textContent = (a.weight_raw ?? 0).toFixed(4);

        row.appendChild(left);
        row.appendChild(val);

        // Listener attached directly to this row's own element reference — a JS
        // closure, not an HTML attribute. Also listens for mousedown as a fallback in
        // case whatever intercepted onclick targets the 'click' event type
        // specifically; a flag prevents double-firing if both end up delivering.
        let _rowFired = false;
        const fire = (evName) => {
            if (_rowFired) return;
            _rowFired = true;
            setTimeout(() => { _rowFired = false; }, 50);   // allow the NEXT real click through
            _vlog('[voxel STEP 1.5] row listener fired via ' + evName, {atomIdx: a.idx, divId});
            _voxelBanner('✅ 1.5️⃣ ROW LISTENER FIRED (' + evName + ') atomIdx=' + a.idx, '#22ff22');
            _vinaOnAtomRowClick(a.idx, divId, barDivId, plotDivId, wv);
        };
        row.addEventListener('click',     () => fire('click'));
        row.addEventListener('mousedown', () => fire('mousedown'));

        containerEl.appendChild(row);
    });

    // [voxel STEP 1] Confirm the render happened and that rows carry NO onclick
    // attribute this time (should be null — the handler lives only in the closure
    // above, invisible to anything scanning HTML attribute strings).
    _vlog('[voxel STEP 1] _renderTable rendered (createElement + addEventListener, no onclick attr)', {
        divId, atomCount: sorted.length,
        rowsInDom: containerEl.querySelectorAll('[data-atom-idx]').length,
        firstRowOnclickAttr: containerEl.querySelector('[data-atom-idx]') &&
                              containerEl.querySelector('[data-atom-idx]').getAttribute('onclick'),
    });
    _voxelBanner('1️⃣ _renderTable(' + divId + ') → ' + sorted.length + ' rows · build=' + _VINA_JS_BUILD, '#c7d2fe');
}
// ── LOCK _renderTable AGAINST SHADOWING ──────────────────────────────────────
// Another loaded template/script (identified: attn_ajax_.html carries a full inline
// copy of an OLD _renderTable with jQuery onclick="" rows; there may be a similar one
// among hub.js/pose.js/etc. loaded AFTER vina.js) defines its own global
// `function _renderTable`. Because plain function declarations just reassign the global
// name, whichever loads LAST wins — and the stale copy was winning, which is why the
// rendered rows had onclick="" and buildsMatch=false. Locking our version as a
// non-writable / non-configurable property of window makes every later `function
// _renderTable(){…}` reassignment silently no-op, so OURS always renders the table.
(function () {
    try {
        Object.defineProperty(window, '_renderTable', {
            value: _renderTable,
            writable: false,       // later `_renderTable = …` / `function _renderTable` can't replace it
            configurable: false,   // …and it can't be redefined away either
            enumerable: true,
        });
        console.log('%c[voxel] ✓ _renderTable locked (shadow-proof) — build ' + _VINA_JS_BUILD,
            'color:#22c55e;font-weight:bold');
    } catch (e) {
        // If it's already locked (e.g. this file somehow ran twice), that's fine.
        console.warn('[voxel] could not lock _renderTable:', e && e.message);
    }
})();

// The atom-row click behaviour, callable independent of how the click was captured.
function _vinaOnAtomRowClick(atomIdx, divId, barDivId, plotDivId, wv) {
    // [voxel STEP 2] Unconditional entry log — runs regardless of divId, so if STEP
    // 1.5 printed but this doesn't, the function call itself is throwing/blocked
    // before its first statement (extremely unlikely, but this removes all doubt).
    _vlog('[voxel STEP 2] _vinaOnAtomRowClick entered', {atomIdx, divId, barDivId, plotDivId});

    // Populate the voxel contact panel FIRST. The steps below (bar restyle, 3D
    // re-render) can throw, and an exception mid-handler silently aborts every
    // remaining statement. Each step is isolated so one failure can't take out others.
    if (divId === 'vTableA') {
        _vlog('[voxel STEP 3] divId===vTableA, calling _voxelSimulateAtomPick',
                    {atomIdx, fn: typeof _voxelSimulateAtomPick});
        try { _voxelSimulateAtomPick(atomIdx); }
        catch (e) { console.error('%c[voxel STEP 3] _voxelSimulateAtomPick THREW',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px', '\n' + (e && e.stack || e)); }
    } else {
        _vlog('[voxel STEP 3] SKIPPED — divId is not vTableA', {divId});
    }

    if (barDivId) {
        try {
            const barEl = document.getElementById(barDivId);
            if (barEl && barEl.data && barEl.data[0]) {
                const xs = barEl.data[0].x;
                const barPos = xs.indexOf(atomIdx);
                if (barPos >= 0) {
                    Plotly.restyle(barDivId, {selectedpoints: [[barPos]]}, [0]);
                    setTimeout(()=>Plotly.restyle(barDivId,{selectedpoints:[null]},[0]), 2000);
                }
            }
        } catch (e) { console.error('[voxel] bar highlight failed:', e); }
    }

    _selAtomIdx = atomIdx;
    try {
        if (_currentView==='protein') {
            _renderProteinView(atomIdx);
        } else if (_currentView==='both') {
            _renderBothView(atomIdx);
        } else {
            _highlightAtom(atomIdx, divId, plotDivId, wv);
        }
    } catch (e) { console.error('[voxel] 3D update failed:', e); }

    // Keep the row highlight in sync in every view (_highlightAtom only runs in
    // the ligand-only branch above).
    if (divId === 'vTableA') {
        try {
            document.querySelectorAll('#'+divId+' [data-atom-idx]').forEach(r => {
                const on = parseInt(r.dataset.atomIdx) === atomIdx;
                r.style.background  = on ? 'rgba(34,211,238,0.15)' : '';
                r.style.borderColor = on ? '#22d3ee' : 'transparent';
            });
        } catch (e) {}
    }
}

function _highlightAtom(atomIdx, tableDiv, plotDivId, wv) {
    if (tableDiv) {
        const rows = document.querySelectorAll('#'+tableDiv+' [data-atom-idx]');
        rows.forEach(r => {
            const isMatch = parseInt(r.dataset.atomIdx) === atomIdx;
            r.style.background  = isMatch ? 'rgba(34,211,238,0.15)' : '';
            r.style.borderColor = isMatch ? '#22d3ee' : '';
            if (isMatch) r.scrollIntoView({block:'nearest', behavior:'smooth'});
        });
    }
    if (plotDivId) {
        const plotEl = document.getElementById(plotDivId);
        if (!plotEl || !plotEl.data || plotEl.data.length < 2) return;
        const atoms = plotEl.data[1];
        const n = (atoms.x||[]).length;
        const sizes = Array.from({length:n}, (_,i)=>
            i===atomIdx ? 22 : (7 + (atoms.marker.color[i]||0)*12)
        );
        const opacs = Array.from({length:n}, (_,i)=> i===atomIdx ? 1 : 0.55);
        Plotly.restyle(plotDivId, {'marker.size':[sizes], 'marker.opacity':[opacs]}, [1]);
        setTimeout(()=>{
            const defSizes = Array.from({length:n}, (_,i)=>(7+(atoms.marker.color[i]||0)*12));
            Plotly.restyle(plotDivId, {'marker.size':[defSizes], 'marker.opacity':[Array(n).fill(1)]}, [1]);
        }, 2000);
    }
}

// ── Grid box toggle ───────────────────────────────────────────────────────────
const _GRID = { cx:-25.7, cy:0.22, cz:28.39, sx:20, sy:20, sz:20 };
let _gridBoxVisible = false;

function _toggleGridBox() {
    _gridBoxVisible = !_gridBoxVisible;
    const btn  = document.getElementById('gridBoxBtn');
    const icon = document.getElementById('gridBoxIcon');

    if (!_gridBoxVisible) {
        const plotEl = document.getElementById('vPlotA');
        if (plotEl && plotEl.data && plotEl.data.length > 0) {
            const toRemove = [];
            for (let i = plotEl.data.length-1; i >= 0; i--) {
                if ((plotEl.data[i].name||'').includes('Grid')) toRemove.push(i);
            }
            if (toRemove.length) Plotly.deleteTraces('vPlotA', toRemove);
        }
        btn.className = btn.className.replace('border-cyan-500 text-cyan-400','border-slate-600 text-slate-400');
        icon.textContent = '⬜';
        return;
    }

    const { cx, cy, cz, sx, sy, sz } = _GRID;
    const hx=sx/2, hy=sy/2, hz=sz/2;
    const vx=[cx-hx,cx+hx,cx+hx,cx-hx,cx-hx,cx+hx,cx+hx,cx-hx];
    const vy=[cy-hy,cy-hy,cy+hy,cy+hy,cy-hy,cy-hy,cy+hy,cy+hy];
    const vz=[cz-hz,cz-hz,cz-hz,cz-hz,cz+hz,cz+hz,cz+hz,cz+hz];

    const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const ex=[],ey=[],ez=[];
    edges.forEach(([a,b])=>{ ex.push(vx[a],vx[b],null); ey.push(vy[a],vy[b],null); ez.push(vz[a],vz[b],null); });

    const faceTrace={
        type:'mesh3d', x:vx, y:vy, z:vz,
        i:[0,0,1,1,0,0,4,4,0,0,3,3], j:[1,2,2,3,1,5,5,6,4,5,7,6], k:[2,3,3,0,5,4,6,7,5,1,6,5],
        color:'#22d3ee', opacity:0.06, hoverinfo:'skip', name:'Grid box (filled)', showlegend:false,
    };
    const edgeTrace={
        type:'scatter3d', mode:'lines', x:ex, y:ey, z:ez,
        line:{color:'#22d3ee',width:2,dash:'dot'},
        hoverinfo:'skip', name:'Grid box edges', showlegend:false,
    };
    const centreTrace={
        type:'scatter3d', mode:'markers+text',
        x:[cx], y:[cy], z:[cz],
        marker:{size:4,color:'#22d3ee',symbol:'cross'},
        text:[`center (${cx}, ${cy}, ${cz})<br>size ${sx}×${sy}×${sz} Å`],
        textfont:{size:9,color:'#22d3ee'}, textposition:'top center',
        hovertemplate:`Grid box<br>center: (${cx}, ${cy}, ${cz})<br>size: ${sx}×${sy}×${sz} Å<extra></extra>`,
        name:'Grid centre', showlegend:false,
    };

    const plotEl = document.getElementById('vPlotA');
    if (!plotEl || !plotEl.data || plotEl.data.length===0) {
        Plotly.newPlot('vPlotA',[faceTrace,edgeTrace,centreTrace],{
            paper_bgcolor:'transparent', plot_bgcolor:'transparent',
            margin:{l:0,r:0,t:0,b:0},
            scene:{bgcolor:'#05070f',
                xaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''},
                yaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''},
                zaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''}},
            showlegend:false,
        },{responsive:true,displayModeBar:false});
        document.getElementById('vPlotA').style.opacity='1';
        document.getElementById('vSpinnerA').style.display='none';
    } else {
        Plotly.addTraces('vPlotA',[faceTrace,edgeTrace,centreTrace]);
    }

    btn.className = btn.className.replace('border-slate-600 text-slate-400','border-cyan-500 text-cyan-400');
    icon.textContent = '🟦';
}

// ── Editable docking box: read the vinaCx/Cy/Cz + vinaLen inputs into _GRID ────
// Wired to the "Box ctr / len" controls in the Visualize row (onchange). Updates
// the shared _GRID (which the grid overlay, the dock command preview at dock time,
// and the POST body all read from), then redraws the overlay if it's showing.

// Pure reader: pull the four box inputs into _GRID. No redraw, no status — shared
// by _setGridBox (on input change) and _vinaDock (at click time) so the docking
// command always uses exactly what's typed in the Box ctr / len fields, even if
// the field hasn't fired its change event yet (e.g. Enter-less click on Dock).
function _readGridInputs() {
    const num = id => { const el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; };
    const cx = num('vinaCx'), cy = num('vinaCy'), cz = num('vinaCz'), len = num('vinaLen');
    if (Number.isFinite(cx)) _GRID.cx = cx;
    if (Number.isFinite(cy)) _GRID.cy = cy;
    if (Number.isFinite(cz)) _GRID.cz = cz;
    if (Number.isFinite(len) && len >= 4) { _GRID.sx = len; _GRID.sy = len; _GRID.sz = len; }
}

function _setGridBox() {
    _readGridInputs();

    // Redraw the box overlay in place if it's currently visible (off→on = redraw
    // from the new _GRID, reusing the tested trace-building path in _toggleGridBox).
    if (_gridBoxVisible) { _toggleGridBox(); _toggleGridBox(); }

    if (typeof _vinaStatus === 'function')
        _vinaStatus(`Box → center (${_GRID.cx}, ${_GRID.cy}, ${_GRID.cz}) · ${_GRID.sx} Å`);
}

// Populate the box inputs from _GRID (called on modal open). Skips a field the
// user is actively editing so it never clobbers a value mid-type.
function _syncGridInputs() {
    const set = (id, v) => {
        const el = document.getElementById(id);
        if (el && document.activeElement !== el) el.value = v;
    };
    set('vinaCx', _GRID.cx); set('vinaCy', _GRID.cy);
    set('vinaCz', _GRID.cz); set('vinaLen', _GRID.sx);
}

// ── Apply an authoritative docking box (from the YAML protein library) ─────────
// Takes a payload carrying center_x/y/z + size_x/y/z — i.e. exactly what
// /vina_visualization/vina_select_protein and /vina_visualization/vina_defaults
// return out of input_TS.yml — and makes it the live box.
//
// Writes BOTH _GRID and the four Box ctr / len inputs, because they are two
// separate copies of the same number:
//   • _GRID is what _vinaDock() POSTs and what the overlay draws from, and
//     vina_dock prefers the POSTed box over the YAML config — so _GRID left at
//     the old target silently docks the new receptor at the old protein's site.
//   • the inputs are what the user reads, and _syncGridInputs() rewrites them
//     from _GRID on every modal open — so input-only updates get reverted.
//
// The UI exposes a single edge length, so `len` shows size_x while _GRID keeps
// sx/sy/sz per axis: a non-cubic entry still docks with its true dimensions even
// though the input can only display one of them.
//
// opts.label  – target name, prefixed onto the status line
// opts.quiet  – skip the status line (used by the page-load prefill)
// Returns true when a usable box was applied.
function _applyGridBox(box, opts) {
    if (!box) return false;
    opts = opts || {};
    const n  = v => (v === null || v === undefined || v === '') ? NaN : parseFloat(v);
    const cx = n(box.center_x), cy = n(box.center_y), cz = n(box.center_z);
    const sx = n(box.size_x),   sy = n(box.size_y),   sz = n(box.size_z);

    // No usable centre → leave the current box alone rather than writing NaNs.
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz)) return false;

    _GRID.cx = cx; _GRID.cy = cy; _GRID.cz = cz;
    if (Number.isFinite(sx) && sx >= 4) _GRID.sx = sx;
    if (Number.isFinite(sy) && sy >= 4) _GRID.sy = sy;
    if (Number.isFinite(sz) && sz >= 4) _GRID.sz = sz;

    // Unconditional set (unlike _syncGridInputs): an explicit target switch is
    // authoritative and should win even over a field the user was mid-edit in.
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    set('vinaCx', _GRID.cx); set('vinaCy', _GRID.cy);
    set('vinaCz', _GRID.cz); set('vinaLen', _GRID.sx);

    // Redraw the overlay in place if it is showing (off→on rebuilds from _GRID).
    if (_gridBoxVisible) { _toggleGridBox(); _toggleGridBox(); }

    if (!opts.quiet && typeof _vinaStatus === 'function') {
        const dims = (_GRID.sx === _GRID.sy && _GRID.sy === _GRID.sz)
            ? `${_GRID.sx} Å`
            : `${_GRID.sx}×${_GRID.sy}×${_GRID.sz} Å`;
        _vinaStatus(`${opts.label ? opts.label + ' box' : 'Box'} → center (${_GRID.cx}, ${_GRID.cy}, ${_GRID.cz}) · ${dims}`);
    }
    return true;
}

// ── Export best Vina pose as .pdb ──────────────────────────────────────────────
// The server derives <ligand_stem>_out.pdbqt (the docked output), converts the best
// (first) pose to PDB, and returns { ok, filename, pdb_text }. We turn that into a
// Blob and trigger a browser download — no server-side temp download link needed.
function _vinaExportPdb() {
    const lig = document.getElementById('ligPath')?.value?.trim() || '';
    if (!lig) { _vinaStatus('Load a ligand and dock first — no output pose to export.', true); return; }

    const btn  = document.getElementById('vinaExportBtn');
    const orig = btn ? btn.innerHTML : '';
    if (btn) { btn.disabled = true; btn.innerHTML = '⏳ Exporting…'; }

    // Prefer the _out.pdbqt the viewer actually read — /vina_parse_log reports it
    // as docked_pose_path and it is cached in _dockedPosePath. Re-deriving the path
    // from #ligPath can disagree with what is on screen, and when #ligPath already
    // points at an _out file the derivation misses the file entirely.
    const _body = _dockedPosePath ? { out_path: _dockedPosePath, ligand_path: lig }
                                  : { ligand_path: lig };
    fetch('/vina_visualization/vina_export_pdb', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(_body)
    })
    .then(r => r.json())
    .then(d => {
        if (btn) { btn.disabled = false; btn.innerHTML = orig; }
        if (!d || !d.ok) { _vinaStatus('Export failed: ' + ((d && d.err) || 'unknown error'), true); return; }
        const blob = new Blob([d.pdb_text], { type:'chemical/x-pdb' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = d.filename || 'vina_best_pose.pdb';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        _vinaStatus('Exported ' + (d.filename || 'best pose') + ' — best pose → PDB'
                    + (d.n_conect ? ' · ' + d.n_conect + ' CONECT records' : ''));
    })
    .catch(e => {
        if (btn) { btn.disabled = false; btn.innerHTML = orig; }
        _vinaStatus('Export error: ' + e, true);
    });
}


// ══ Qwen Chat (Claude-style centered) ════════════════════════════════════════

_chatStream  = null;
_msgCount    = 0;

// _chatToggle now just opens the visualizer (the chat IS the landing page)
function _chatToggle() { _vinaShow(); }

// _chatAppend defined in attn JS


// _chatShowTyping defined in attn JS

// _chatHideTyping defined in attn JS


// _chatGetContext defined in attn JS


// _chatSend defined in attn JS


// _qprompt defined in attn JS


// _chatClear defined in attn JS


// Open chat panel with keyboard shortcut: Cmd/Ctrl + K
document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        _chatToggle();
    }
});


// ══ CoT UI Action Handler ═════════════════════════════════════════════════════

// Maps action_id → button to highlight (NO auto-execute — user must click)
const _VINA_ACTIONS = {
    open_visualizer:      { btnId: 'adjWeightBtn' },
    open_vina:            { btnId: 'vinaBtn' },
    open_voxel_inspector: { btnId: 'voxelInspectBtn' },
    toggle_grid_box:      { btnId: 'gridBoxBtn' },
    run_visualization:    { btnId: 'vizBtn' },
    run_docking:          { btnId: 'vinaDockBtn' },
    switch_ligand_view:   { btnId: 'modeLigand' },
    switch_protein_view:  { btnId: 'modeProtein' },
    load_ligand:          { btnId: 'ligLoadBtn' },
    load_receptor:        { btnId: 'recLoadBtn' },
    vina_dock_guided:     { btnId: 'vinaDockBtn' },  // highlights paths + dock btn
};

// ── Unified gold highlight system ────────────────────────────────────────────
// All highlights use the same goldPulse CSS — persistent until user interacts.

const _ALL_HIGHLIGHTABLE = ['adjWeightBtn','recPath','recLoadBtn','ligPath','ligLoadBtn',
                             'vinaDockBtn','vizBtn','gridBoxBtn','voxelInspectBtn',
                             'modeLigand','modeProtein','askElionBtn','sidebarPdb2PdbqtBtn'];

// _clearAllHighlights defined in attn JS


// Persistent gold pulse — stops when user clicks/interacts
// _highlightBtnUntilClick defined in attn JS


// Gold pulse for multiple elements — stops on click or input change
function _highlightAmberUntilAction(ids, stopEvent) {
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.remove('btn-pulse-wait', 'amber-pulse', 'btn-highlight', 'btn-glow');
        void el.offsetWidth;
        el.classList.add('amber-pulse');

        const stop = () => {
            el.classList.remove('amber-pulse', 'btn-glow');
            el.removeEventListener(stopEvent || 'click', stop);
        };
        el.addEventListener(stopEvent || 'click', stop);
    });
}

function _highlightBtn(btnId, confidence) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.classList.remove('btn-highlight', 'btn-glow', 'btn-pulse-wait');
    void btn.offsetWidth;
    btn.classList.add('btn-highlight');
    const holdMs = confidence === 'high' ? 5000 : 3000;
    setTimeout(() => {
        btn.classList.remove('btn-highlight');
        btn.classList.add('btn-glow');
        setTimeout(() => btn.classList.remove('btn-glow'), holdMs);
    }, 2100);
}

// ── Guided flow state ─────────────────────────────────────────────────────────
let _guidedStep = null;   // tracks multi-step guided flow

function handleVinaUiAction(ui_action) {
    if (!ui_action || ui_action.action === 'none') return;

    // Clear any existing highlights before applying new ones
    setTimeout(() => _clearAllHighlights(), 0);

    if (ui_action.action === 'open_visualizer') {
        setTimeout(() => {
            _chatAppend('ai', 'Now click the flashing **Open Visualizer** button above ↑ to launch the 3D docking view.');
            _highlightBtnUntilClick('adjWeightBtn');
            _guidedStep = 'await_visualizer_open';
        }, 400);
        return;
    }

    if (ui_action.action === 'load_ligand') {
        setTimeout(() => {
            _highlightAmberUntilAction(['ligPath', 'ligLoadBtn'], 'click');
        }, 400);
        return;
    }

    if (ui_action.action === 'load_receptor') {
        setTimeout(() => {
            _highlightAmberUntilAction(['recPath', 'recLoadBtn'], 'click');
        }, 400);
        return;
    }

    if (ui_action.action === 'vina_dock_guided') {
        setTimeout(() => {
            _highlightEmptyPathFields();
            _highlightBtnUntilClick('vinaDockBtn');
        }, 400);
        return;
    }

    if (ui_action.action === 'run_visualization') {
        // Persistent gold pulse on Visualize until user clicks
        setTimeout(() => _highlightBtnUntilClick('vizBtn'), 400);
        return;
    }

    if (ui_action.action === 'open_vina') {
        setTimeout(() => {
            _chatAppend('ai', 'Click **🧬 Ask Elion** → **🔬 Vina Docking** to open the docking visualizer.');
            _highlightBtnUntilClick('askElionBtn');
        }, 400);
        return;
    }

    if (ui_action.action === 'guide_pdb_conversion') {
        setTimeout(() => _guidePdbConversionFlow(), 400);
        return;
    }

    // All other actions: persistent gold pulse
    const entry = _VINA_ACTIONS[ui_action.action];
    if (!entry) return;
    setTimeout(() => _highlightBtnUntilClick(entry.btnId), 400);
}

// ── Multi-step guided PDB → PDBQT conversion flow ────────────────────────────
// Step 1: pulse Ask Elion + explain in mini-chat
// Step 2: when user clicks Ask Elion, sidebar opens → pulse PDB→PDBQT button + explain
// Step 3: when user clicks PDB→PDBQT, converter opens → instruct to upload
function _guidePdbConversionFlow() {
    _clearAllHighlights();

    // ── STEP 1: Highlight Ask Elion, explain in mini-chat ────────────────────
    _miniChatAppend('ai',
        '📋 **Step 1 of 3 — Open the Tool Menu**\n\n' +
        'Click the glowing **Ask Elion** button in the top-right corner. ' +
        'It will open the tool menu where the converter lives.'
    );

    const askBtn = document.getElementById('askElionBtn');
    if (!askBtn) return;

    // Pulse Ask Elion
    askBtn.classList.add('btn-pulse-wait');

    // Once user clicks Ask Elion → move to step 2
    function onAskElionClick() {
        askBtn.classList.remove('btn-pulse-wait');
        askBtn.removeEventListener('click', onAskElionClick);
        // Small delay so sidebar finishes sliding in
        setTimeout(_guidePdbStep2, 350);
    }
    askBtn.addEventListener('click', onAskElionClick);
}

function _guidePdbStep2() {
    // ── STEP 2: Sidebar is open → highlight PDB→PDBQT button + explain ───────
    _miniChatAppend('ai',
        '📋 **Step 2 of 3 — Open the Converter**\n\n' +
        'Now click the glowing **⚙️ PDB → PDBQT** button in the Tools section of the menu.'
    );

    const pdbBtn = document.getElementById('sidebarPdb2PdbqtBtn');
    if (!pdbBtn) {
        // Sidebar may not be rendered yet, retry once
        setTimeout(_guidePdbStep2, 300);
        return;
    }

    pdbBtn.classList.add('btn-pulse-wait');

    function onPdbBtnClick() {
        pdbBtn.classList.remove('btn-pulse-wait');
        pdbBtn.removeEventListener('click', onPdbBtnClick);
        // _sidebarOpenConverter keeps mini-chat alive and posts step-3 guidance
        // (step 3 message is injected by _sidebarOpenConverter already, so skip _guidePdbStep3 here)
    }
    pdbBtn.addEventListener('click', onPdbBtnClick);
}

function _guidePdbStep3() {
    // Dropzone glow — _sidebarOpenConverter already posted the text guidance
    const dz = document.getElementById('p2p_dropzone');
    if (dz) {
        dz.style.borderColor = '#34d399';
        dz.style.background  = '#071410';
        dz.style.transition  = 'border-color .3s,background .3s';
        const fileInput = document.getElementById('p2p_file_input');
        if (fileInput) {
            fileInput.addEventListener('change', () => {
                dz.style.borderColor = '#1e3a2a';
                dz.style.background  = '#060e08';
            }, { once: true });
        }
    }
}

// ── Mini chat panel ────────────────────────────────────────────────────────────
function _miniChatShow() {
    document.getElementById('miniChat').style.display = 'flex';
}
function _miniChatClose() {
    document.getElementById('miniChat').style.display = 'none';
}

// Toggle the Vina docking mini-chat (💬 button in the modal header). When opening,
// restore the Vina branding/context first, mirroring how _vinaShow launches it.
function _vinaMiniToggle() {
    const el = document.getElementById('miniChat');
    if (!el) return;
    if (el.style.display === 'flex') {
        _miniChatClose();
    } else {
        if (typeof _miniChatSetContext === 'function') _miniChatSetContext(_MINICHAT_VINA_THEME);
        _miniChatShow();
    }
}

function _miniChatAppend(role, text) {
    const out = document.getElementById('miniChatOutput');
    const div = document.createElement('div');
    div.style.cssText = role === 'user'
        ? 'align-self:flex-end;max-width:85%;background:#0e7490;color:#ecfeff;border-radius:12px 12px 3px 12px;padding:7px 11px;font-size:12px;line-height:1.5;'
        : 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    div.innerHTML = text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
        .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:11px">$1</code>')
        .replace(/\n/g,'<br>');
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _miniChatSend() {
    const inp  = document.getElementById('miniChatInput');
    const text = inp.value.trim();
    if (!text) return;
    if (_chatStream === 'mini') return; // already streaming in mini-chat
    inp.value = '';
    _miniChatAppend('user', text);

    // Show typing indicator in mini-chat
    const out = document.getElementById('miniChatOutput');
    const typingEl = document.createElement('div');
    typingEl.id = 'miniTyping';
    typingEl.style.cssText = 'color:#64748b;font-size:12px;padding:4px 0;';
    typingEl.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    out.appendChild(typingEl);
    out.scrollTop = out.scrollHeight;

    const ctx = _chatGetContext();
    ctx.guided_mode = true;  // enforce short step-by-step responses

    // Inject path fields from Vina modal into context
    const recVal = document.getElementById('recPath')?.value?.trim();
    const ligVal = document.getElementById('ligPath')?.value?.trim();
    if (recVal) ctx.receptor_path = recVal;
    if (ligVal) ctx.ligand_path   = ligVal;

    // Tell LLM which fields are still empty so it can offer to convert
    if (!recVal) ctx.receptor_missing = true;
    if (!ligVal) ctx.ligand_missing   = true;

    // Inject last conversion so LLM knows a file was just converted and auto-filled
    if (_lastConversion) ctx.last_conversion = _lastConversion;

    let aiDiv      = null;
    let fullText   = '';
    let firstToken = true;
    _chatStream = 'mini'; // block concurrent mini-chat sends

    fetch('/vina_visualization/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        function pump() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    document.getElementById('miniTyping')?.remove();
                    _chatStream = null;
                    return;
                }
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split('\n\n');
                buf = parts.pop();
                for (const part of parts) {
                    const evM   = part.match(/^event:\s*(\w+)/m);
                    const dataM = part.match(/^data:\s*(.+)$/m);
                    if (!evM || !dataM) continue;
                    const ev = evM[1];
                    let payload;
                    try { payload = JSON.parse(dataM[1]); } catch { payload = dataM[1]; }

                    if (ev === 'token') {
                        if (firstToken) {
                            document.getElementById('miniTyping')?.remove();
                            firstToken = false;
                            aiDiv = document.createElement('div');
                            aiDiv.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
                            out.appendChild(aiDiv);
                            fullText = '';
                        }
                        fullText += payload;
                        aiDiv.innerHTML = fullText
                            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                            .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
                            .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:11px">$1</code>')
                            .replace(/\n/g,'<br>');
                        out.scrollTop = out.scrollHeight;

                    } else if (ev === 'ui_action') {
                        handleVinaUiAction(payload);
                    } else if (ev === 'done') {
                        document.getElementById('miniTyping')?.remove();
                        _chatStream = null;
                    } else if (ev === 'error') {
                        document.getElementById('miniTyping')?.remove();
                        _chatStream = null;
                        _miniChatAppend('ai', '⚠️ ' + (payload || 'Error'));
                    }
                }
                pump();
            }).catch(err => {
                document.getElementById('miniTyping')?.remove();
                _chatStream = null;
                _miniChatAppend('ai', '⚠️ Stream error: ' + err.message);
            });
        }
        pump();
    }).catch(() => {
        document.getElementById('miniTyping')?.remove();
        _chatStream = null;
        _miniChatAppend('ai', '⚠️ Could not reach Qwen server.');
    });
}

// ── Hook into _adjShow to detect visualizer opening ──────────────────────────
const _origAdjShow = _adjShow;
_adjShow = function() {
    _origAdjShow();
    if (_guidedStep === 'await_visualizer_open') {
        _guidedStep = null;
        // Show mini-chat in bottom-left
        setTimeout(() => {
            _miniChatShow();
            // Give next guided instruction
            const ligPath = document.getElementById('ligPath') ? document.getElementById('ligPath').value : '/path/to/ligand.pdbqt';
            _miniChatAppend('ai',
                'Great! The visualizer is open.\n\n' +
                'Now, **enter the ligand .pdbqt path** in the LIGAND field and click **Load**. For example:\n\n' +
                '`' + ligPath + '`'
            );
            // Highlight the ligand load button persistently
            setTimeout(() => _highlightBtnUntilClick('ligLoadBtn'), 600);
        }, 500);
    }
};



// ══ Wire header buttons ═══════════════════════════════════════════════════════
$(function() {
    $('#adjWeightBtn').on('click', function() {
        _activeTool = 'attn';
        _clearAllHighlights();
        _adjShow();
    });
    $('#vinaBtn').on('click', function() {
        _activeTool = 'vina';
        _clearAllHighlights();
        _vinaShow();
    });
});

/* ════════════════════════════════════════════════════════════════════════════
   VOXEL INSPECTOR
   ─────────────────────────────────────────────────────────────────────────────
   Toggle #voxelInspectBtn (🔬) to enter pick mode, then click atoms in #vPlotA.
   Clicking a ligand atom fills the #voxelPanel (bottom-right) with the full list of
   protein atoms it contacts; clicking a contact row draws the atom–atom distance in
   3D and posts that pair_e's term-by-term breakdown into the docking mini-chat
   (#miniChatOutput). Clicking a receptor atom posts its single pair_e to the chat.
   _ensureVoxelListener binds the plotly_click handler after every #vPlotA render;
   #voxelPickOverlay is the crosshair hint shown while picking.

   Click routing keys off the clicked TRACE, not the current view:
     'Atoms'            → a ligand atom  → list every protein atom it pairs with
     'RecHit' / 'RecBg' → a receptor atom → that atom's single pair_e detail
   Each contact row is itself clickable (_voxelPairClick): it draws the atom–atom
   distance in 3D and posts a term-by-term breakdown of that pair_e to the chat.

   pair_e originates in vina_non_cache.log and is parsed server-side. The 3D-viz
   response we hold in memory carries, per pair, everything the breakdown needs:
     _pairsByLig = { ligandAtomIdx : [ {rec_idx, pair_e, rec_xs, lig_xs, r, s}, ...] }
     _recAtoms   = [ {idx, name, resname, resseq, x, y, z}, ... ]   (labels + coords)
   So clicking an atom or a row needs no backend call.
   ════════════════════════════════════════════════════════════════════════════ */

let _voxelPickActive = false;   // is "click an atom to inspect" mode currently on?

// Attach the plotly_click handler to #vPlotA. Re-attaches cleanly on every call
// (mirrors the codebase's own removeAllListeners('plotly_click') pattern), so it
// survives Plotly.newPlot and is safe to call after each render.
function _ensureVoxelListener() {
    const plotEl = document.getElementById('vPlotA');
    if (!plotEl || typeof plotEl.on !== 'function') return;   // plot not drawn yet
    if (plotEl.removeAllListeners) plotEl.removeAllListeners('plotly_click');
    plotEl.on('plotly_click', function (ev) {
        if (!_voxelPickActive) return;                        // only pick when mode is on
        if (!ev || !ev.points || !ev.points.length) return;
        _voxelHandleClick(ev.points[0]);
    });
}

// Route a clicked point by which trace it belongs to.
// A single physical click can hit two stacked traces (a VoxelPick marker sitting on top
// of the RecBg atom it marks) and be dispatched twice, which toggles a measurement on
// and immediately back off — the "blink". Ignore a repeat of the same point in quick
// succession; a deliberate second click to clear is always slower than this window.
let _voxelLastClick = { key:'', t:0 };

function _voxelHandleClick(pt) {
    const traceName = (pt.data && pt.data.name) || '';
    const rid = (pt.customdata && pt.customdata.rec_idx != null) ? pt.customdata.rec_idx
              : (typeof pt.customdata === 'number' ? pt.customdata : pt.pointNumber);
    const key = traceName + ':' + rid;
    const now = Date.now();

    // [voxel STEP 5] Unconditional entry log, BEFORE the de-dupe guard below — that
    // guard used to return silently with no trace at all, which is exactly the kind of
    // gap that makes a real failure indistinguishable from "working as designed."
    _vlog('[voxel STEP 5] _voxelHandleClick entered',
        {traceName, rid, key, lastKey: _voxelLastClick.key, msSinceLast: now - _voxelLastClick.t});

    if (key === _voxelLastClick.key && (now - _voxelLastClick.t) < 400) {
        console.warn('%c[voxel STEP 5] STOPPED by de-dupe guard (same target within 400ms)',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px', '\n' + _vjson({key}));
        return;
    }
    _voxelLastClick = { key, t: now };

    if (traceName === 'Atoms') {
        _vlog('[voxel STEP 5] routing → _voxelShowLigandContacts', {});
        _voxelShowLigandContacts(pt);
    } else if (traceName === 'VoxelPick') {
        _vlog('[voxel STEP 5] routing → _voxelPickAtomClick', {});
        _voxelPickAtomClick(pt);
    } else if (traceName === 'RecHit' || traceName === 'RecBg') {
        // Any receptor atom is measurable once a ligand atom is selected; otherwise
        // fall back to showing that atom's own pair_e detail.
        if (_voxelHL.ligIdx != null && rid != null) _voxelPairClick(_voxelHL.ligIdx, rid);
        else                                        _voxelShowReceptorAtom(pt);
    } else {
        console.warn('%c[voxel STEP 5] no route matched this traceName — ignored',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px', '\n' + _vjson({traceName}));
    }
    // ignore bond lines, grid-box traces, etc.
}

// A highlighted receptor marker (box / Σ / overlap / pair dot) was clicked: measure it
// against the currently selected ligand atom, exactly like clicking its contact row.
function _voxelPickAtomClick(pt) {
    const recIdx = pt.customdata;
    const ligIdx = _voxelHL.ligIdx;
    if (recIdx == null) return;
    if (ligIdx == null) { _vinaStatus('Pick a ligand atom first to measure against.', true); return; }
    _voxelPairClick(ligIdx, recIdx);
}

/* ── Ligand atom clicked → every protein atom that has a pair_e with it ──────── */
function _voxelShowLigandContacts(pt) {
    const cd     = pt.customdata || {};
    const ligIdx = (cd.idx != null) ? cd.idx : pt.pointNumber;   // = key into _pairsByLig
    // [voxel STEP 6] Unconditional entry log, before the null-check bail.
    _vlog('[voxel STEP 6] _voxelShowLigandContacts entered',
        {customdata: cd, pointNumber: pt.pointNumber, resolvedLigIdx: ligIdx});
    if (ligIdx == null) {
        console.error('%c[voxel STEP 6] STOPPING — ligIdx resolved to null/undefined',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px', '\n' + _vjson({cd}));
        return;
    }
    _voxelShowContactsFor(ligIdx);
}

// Build the contact panel for a ligand atom index. Called by the 🔬 pick handler above
// and by a click on a row of the sidebar atom list, so both routes show the same panel.
function _voxelShowContactsFor(ligIdx) {
    // [voxel STEP 7] Unconditional entry log, before the null-check bail — the previous
    // detailed log further down only fired AFTER this guard, so a null ligIdx reaching
    // here would have produced zero output despite getting this far in the pipeline.
    _vlog('[voxel STEP 7] _voxelShowContactsFor entered', {ligIdx});
    _voxelBanner('7️⃣ building contact panel for ligIdx=' + ligIdx, '#fb923c');
    if (ligIdx == null) {
        console.error('%c[voxel STEP 7] STOPPING — ligIdx is null/undefined',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px');
        _voxelBanner('❌ 7️⃣ STOPPED — ligIdx is null', '#ff3333');
        return;
    }

    const ligAtom = _ligAtomsCache[ligIdx];
    const ligSym  = ligAtom ? (ligAtom.symbol || 'atom') : 'atom';
    const ligName = ligSym + ' #' + ligIdx;

    // Full pose table for this ligand atom, most-favourable (most negative) first.
    const pairs = (_pairsByLig[ligIdx] || []).slice()
                    .sort((a, b) => a.pair_e - b.pair_e);
    const sumE  = pairs.reduce((s, p) => s + (p.pair_e || 0), 0);

    // ── The full contact list goes in the #voxelPanel (bottom-right). ──
    const title = document.getElementById('voxelPanelTitle');
    _vlog('[voxel STEP 7] _voxelShowContactsFor detail', {
        ligIdx, ligName,
        pairs: pairs.length,
        pairsByLigKeys: Object.keys(_pairsByLig || {}).length,
        ligAtomFound: !!ligAtom,
        panels_with_that_id: document.querySelectorAll('#voxelPanel').length,
        titleEl: !!title,
        bodyEl: !!document.getElementById('voxelPanelBody'),
        panelEl: !!document.getElementById('voxelPanel'),
    });
    if (title) title.textContent = ligName + '  ·  ' + pairs.length
        + ' contact' + (pairs.length === 1 ? '' : 's') + '  ·  Σ pair_e ' + sumE.toFixed(3);

    const body = document.getElementById('voxelPanelBody');
    if (body) {
        if (!pairs.length) {
            body.innerHTML = '<p class="text-slate-600 text-[10px]">No receptor contacts recorded for '
                           + ligName + '.</p>';
        } else {
            const recBy = _voxelRecByIdx();
            const rows  = pairs.map(p => {
                const rec   = recBy[p.rec_idx];
                const label = rec ? (rec.name + ' ' + rec.resname + rec.resseq) : ('rec ' + p.rec_idx);
                return _voxelContactRow(label, p.rec_idx, p.pair_e, ligIdx);
            }).join('');
            // Σ row (highlight every contact atom) + docking-box row, pinned after the list.
            const actionRow = (onclick, kind, title, leftHtml, rightHtml) =>
                '<div onclick="' + onclick + '" data-vox-row="' + kind + '" '
              +      'onmouseover="_voxelRowHover(this,true)" onmouseout="_voxelRowHover(this,false)" '
              +      'title="' + title + '" '
              +      'style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;'
              +             'cursor:pointer;border-radius:4px;padding:2px 4px;transition:background .12s;">'
              +   leftHtml + rightHtml + '</div>';
            const actions =
                '<div style="border-top:1px solid #1e293b;margin-top:6px;padding-top:5px;'
              +   'display:flex;flex-direction:column;gap:2px;">'
              +   actionRow('_voxelHighlightAllContacts(' + ligIdx + ')', 'all',
                    'Toggle: highlight all ' + pairs.length + ' contact atoms in 3D',
                    '<span style="color:#fbbf24;font-weight:600;">Σ&nbsp; highlight all ' + pairs.length + ' atoms</span>',
                    '<span style="color:#67e8f9;font-weight:600;font-variant-numeric:tabular-nums;">' + sumE.toFixed(4) + ' kcal/mol</span>')
              +   actionRow('_voxelHighlightBox()', 'box',
                    'Toggle: highlight every receptor atom inside the docking box',
                    '<span style="color:#34d399;font-weight:600;">⬜ all atoms in docking box</span>',
                    '<span style="color:#475569;">▶</span>')
              +   actionRow('_voxelToggleVoxelGrid()', 'grid',
                    'Toggle: draw the szv_grid voxels (3 Å cells) the box is divided into',
                    '<span style="color:#38bdf8;font-weight:600;">▦ show docking-box voxels</span>',
                    '<span style="color:#475569;">▶</span>')
              +   actionRow('_voxelToggleLigFrame()', 'ligframe',
                    'Toggle: show the docked ligand pose Vina actually scored — from '
                  + '&lt;ligand&gt;_out.pdbqt if present, else vina_non_cache.log — instead of the exported .pdbqt',
                    '<span style="color:#f472b6;font-weight:600;">⇄ show docked pose (in box)</span>',
                    '<span style="color:#475569;">▶</span>')
              +   '<div style="color:#334155;font-size:8px;padding:2px 4px 0;">'
              +     'highlights stay until clicked again · '
              +     '<span style="color:#e879f9;">magenta</span> = in both sets</div>'
              + '</div>';
            body.innerHTML =
                '<p class="text-slate-600 text-[9px] mb-1">protein atoms with pair_e · from vina_non_cache.log · click a row to measure &amp; break down</p>'
              + '<div style="display:flex;flex-direction:column;gap:2px;">' + rows + '</div>'
              + actions;
        }
    }

    // Switching to a different ligand atom: the per-pair single-line highlights are
    // specific to the old atom's rows, so clear those. But the Σ "highlight all" toggle
    // and the box/grid toggles are VIEW MODES the user turned on — carry them over so
    // clicking from one row to another keeps showing the fan + overlap for the newly
    // selected atom (image2), instead of resetting to the plain view (image1).
    console.log('[voxel STEP 7] body written; entering redraw/reveal',
        {prevLig: _voxelHL.ligIdx, ligIdx, keepAll: _voxelHL.all, keepBox: _voxelHL.box, keepGrid: _voxelHL.grid});
    if (_voxelHL.ligIdx !== ligIdx) {
        _voxelHL.ligIdx = ligIdx;
        _voxelHL.pairs.clear();          // per-atom single-pair lines don't carry over
        // _voxelHL.all / .box / .grid intentionally preserved (persistent view modes)
        try { _voxelRedrawHighlights(); }   // re-applies the active modes to the new atom
        catch (e) { console.error('[voxel] _voxelRedrawHighlights threw:', e); }
    }
    try { _voxelSyncRowStyles(); }       // repaint the Σ/box/grid rows as active if on
    catch (e) { console.error('[voxel] _voxelSyncRowStyles threw:', e); }

    _voxelSetHidden('voxelPanel', false);          // show the contact panel
    _voxelSetHidden('voxelPickOverlay', true);

    // Final report: is the panel actually painted, and where?
    const _p = document.getElementById('voxelPanel');
    if (_p) {
        const cs = getComputedStyle(_p), r = _p.getBoundingClientRect();
        const onScreen = r.width > 0 && r.height > 0 &&
                          r.bottom > 0 && r.right > 0 &&
                          r.top < innerHeight && r.left < innerWidth;
        _vlog('[voxel STEP 8] panel state (final visibility check)', {
            classes: _p.className,
            display: cs.display, visibility: cs.visibility, opacity: cs.opacity,
            zIndex: cs.zIndex, position: cs.position,
            rect: {top: Math.round(r.top), left: Math.round(r.left),
                   w: Math.round(r.width), h: Math.round(r.height)},
            onScreen,
            titleText: (document.getElementById('voxelPanelTitle')||{}).textContent,
            bodyLen: (document.getElementById('voxelPanelBody')||{}).innerHTML?.length,
        });
        _voxelBanner((onScreen ? '✅ ' : '⚠️ ') + '8️⃣ panel display=' + cs.display
            + ' onScreen=' + onScreen + ' ' + (r.width|0)+'x'+(r.height|0),
            onScreen ? '#22ff22' : '#ff9933');
    } else {
        console.error('[voxel STEP 8] #voxelPanel NOT FOUND in the DOM');
        _voxelBanner('❌ 8️⃣ #voxelPanel NOT FOUND IN DOM', '#ff3333');
    }

    _vinaStatus(ligName + ': ' + pairs.length
              + ' receptor contact' + (pairs.length === 1 ? '' : 's') + '.');
    console.log('[voxel STEP 9] _voxelShowContactsFor finished OK');
}

// Render a voxel result as a message inside the docking mini-chat (#miniChatOutput),
// styled like an assistant bubble but as a bordered "data" card. Opens the chat if it
// was closed. Bypasses _miniChatAppend because that helper HTML-escapes its input,
// which would strip the coloured contact rows.
function _voxelChatMessage(headingText, bodyHtml) {
    if (typeof _miniChatShow === 'function') _miniChatShow();
    const out = document.getElementById('miniChatOutput');
    if (!out) return;
    const div = document.createElement('div');
    div.style.cssText = 'align-self:flex-start;width:97%;box-sizing:border-box;'
      + 'background:#0b1120;border:1px solid #164e63;border-radius:12px 12px 12px 3px;'
      + 'padding:8px 10px;font-size:11px;line-height:1.6;color:#cbd5e1;';
    div.innerHTML =
        '<div style="color:#67e8f9;font-weight:600;font-size:12px;margin-bottom:5px;">' + headingText + '</div>'
      + bodyHtml;
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _voxelContactRow(label, recIdx, pairE, ligIdx) {
    const val = (pairE <= 0) ? pairE.toFixed(4) : ('+' + pairE.toFixed(4));
    const col = (pairE < 0) ? '#5eead4' : '#f87171';   // favourable teal / unfavourable red
    return '<div onclick="_voxelPairClick(' + ligIdx + ',' + recIdx + ')" '
         +      'data-vox-row="pair" data-rec="' + recIdx + '" '
         +      'onmouseover="_voxelRowHover(this,true)" onmouseout="_voxelRowHover(this,false)" '
         +      'title="Click to measure &amp; break down · click again to clear the highlight" '
         +      'style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;'
         +             'cursor:pointer;border-radius:4px;padding:1px 4px;transition:background .12s;">'
         +   '<span style="color:#94a3b8;">' + label
         +     ' <span style="color:#475569;">[' + recIdx + ']</span></span>'
         +   '<span style="color:' + col + ';font-variant-numeric:tabular-nums;">' + val + ' kcal/mol</span>'
         + '</div>';
}

/* ── Persistent highlight state ──────────────────────────────────────────────
   Every highlight stays on screen until its own row is clicked again. Because the
   Σ and docking-box sets can overlap (and the overlap is drawn in its own colour),
   the whole overlay is rebuilt from this state on each toggle rather than being
   patched incrementally. */
const _voxelHL = {
    ligIdx: null,        // ligand atom the pair/Σ highlights belong to
    pairs:  new Set(),   // rec_idx of individually toggled contact rows
    all:    false,       // Σ row active → every contact of ligIdx
    box:    false,       // docking-box row active
    grid:   false,       // szv_grid voxel wireframe active
    ligFrame: false,     // false = exported-pose xyz; true = docked-pose xyz
};
const _VOX_C = { pair:'#fbbf24', all:'#fbbf24', box:'#34d399', overlap:'#e879f9', grid:'#38bdf8', ligframe:'#f472b6' };
// Lighter tint of each highlight colour, used for the distance label text.
const _VOX_LC = { '#fbbf24':'#fde68a', '#34d399':'#6ee7b7', '#e879f9':'#f5d0fe' };

// Hover shading that doesn't clobber the active-highlight background.
function _voxelRowHover(el, on) {
    const active = el.getAttribute('data-hl-active') === '1';
    el.style.background = on ? (active ? '#164e63' : '#0e2536') : (active ? '#0c3a4a' : '');
}

// Paint every row in the panel to match _voxelHL.
function _voxelSyncRowStyles() {
    const body = document.getElementById('voxelPanelBody');
    if (!body) return;
    body.querySelectorAll('[data-vox-row]').forEach(el => {
        const kind = el.getAttribute('data-vox-row');
        let active = false, color = _VOX_C.pair;
        if (kind === 'pair')     active = _voxelHL.pairs.has(parseInt(el.getAttribute('data-rec'), 10));
        else if (kind === 'all') active = _voxelHL.all;
        else if (kind === 'box') { active = _voxelHL.box;  color = _VOX_C.box; }
        else if (kind === 'grid') { active = _voxelHL.grid; color = _VOX_C.grid; }
        else if (kind === 'ligframe') { active = _voxelHL.ligFrame; color = _VOX_C.ligframe; }
        el.setAttribute('data-hl-active', active ? '1' : '0');
        el.style.background = active ? '#0c3a4a' : '';
        el.style.boxShadow  = active ? ('inset 3px 0 0 ' + color) : '';
    });
}

// Rebuild the whole 'VoxelMeasure' overlay from _voxelHL, in one camera-preserving pass.
function _voxelRedrawHighlights() {
    const recBy   = _voxelRecByIdx();
    const ligIdx  = _voxelHL.ligIdx;
    const ligAtom = (ligIdx != null) ? _ligAtomsCache[ligIdx] : null;
    const traces  = [];

    // Σ set: every receptor atom this ligand atom contacts.
    const allSet = new Set();
    if (_voxelHL.all && ligIdx != null) {
        (_pairsByLig[ligIdx] || []).forEach(p => { if (recBy[p.rec_idx]) allSet.add(p.rec_idx); });
    }
    // Box set: every receptor atom inside the current Box ctr / len.
    const boxSet = new Set();
    let box = null;
    if (_voxelHL.box) {
        if (typeof _readGridInputs === 'function') _readGridInputs();   // honour what's typed
        const { cx, cy, cz, sx, sy, sz } = _GRID;
        box = { cx, cy, cz, hx: sx/2, hy: sy/2, hz: sz/2 };
        _recAtoms.forEach(a => {
            if (a.x >= cx-box.hx && a.x <= cx+box.hx &&
                a.y >= cy-box.hy && a.y <= cy+box.hy &&
                a.z >= cz-box.hz && a.z <= cz+box.hz) boxSet.add(a.idx);
        });
    }
    // Atoms in BOTH get their own colour.
    const overlap = new Set();
    allSet.forEach(i => { if (boxSet.has(i)) overlap.add(i); });

    if (box) traces.push(_voxelBoxEdges(box.cx, box.cy, box.cz, box.hx, box.hy, box.hz));

    // szv_grid voxel wireframe: the box divided into the same ~3 Å cells Vina bins with.
    if (_voxelHL.grid) {
        if (typeof _readGridInputs === 'function') _readGridInputs();
        traces.push(_voxelGridEdges(_GRID));
    }

    // Σ fan: faint lines from the ligand atom out to each contact.
    if (allSet.size && ligAtom) {
        const x=[], y=[], z=[];
        allSet.forEach(i => {
            const a = recBy[i];
            x.push(ligAtom.x, a.x, null); y.push(ligAtom.y, a.y, null); z.push(ligAtom.z, a.z, null);
        });
        traces.push({ type:'scatter3d', mode:'lines', x, y, z,
            line:{ color:_VOX_C.all, width:2 }, opacity:0.4, hoverinfo:'skip',
            name:'VoxelMeasure', showlegend:false });
    }

    // Receptor-atom markers are clickable: each carries its rec_idx so a click can be
    // measured against the selected ligand atom. Hover shows the atom's label + xyz.
    const markers = (ids, color, size) => {
        const keep = ids.filter(i => recBy[i]);
        const a    = keep.map(i => recBy[i]);
        return { type:'scatter3d', mode:'markers',
            x:a.map(p=>p.x), y:a.map(p=>p.y), z:a.map(p=>p.z),
            marker:{ size, color, line:{ width:1, color:'#0f172a' } },
            customdata: keep,
            text: a.map(p => p.name + ' ' + p.resname + p.resseq + ' [' + p.idx + ']'),
            hovertemplate: '%{text}<br>xyz: (%{x:.2f}, %{y:.2f}, %{z:.2f}) Å<extra>click for distance</extra>',
            name:'VoxelPick', showlegend:false };
    };
    const boxOnly = [...boxSet].filter(i => !overlap.has(i));
    const allOnly = [...allSet].filter(i => !overlap.has(i));
    if (boxOnly.length)  traces.push(markers(boxOnly, _VOX_C.box, 5));
    if (allOnly.length)  traces.push(markers(allOnly, _VOX_C.all, 6));
    if (overlap.size)    traces.push(markers([...overlap], _VOX_C.overlap, 8));

    // Individually toggled pairs: dashed line + distance label, drawn on top. Each
    // measurement takes the colour of the atom it points at — green for a docking-box
    // atom, magenta when it's in both sets, amber for a plain contact — so the line
    // always matches the ball that was clicked. Lines are grouped per colour because a
    // single Plotly trace can't vary colour segment by segment.
    if (_voxelHL.pairs.size && ligAtom) {
        const groups = {};
        _voxelHL.pairs.forEach(recIdx => {
            const a = recBy[recIdx];
            if (!a) return;
            const color = overlap.has(recIdx) ? _VOX_C.overlap
                        : boxSet.has(recIdx)  ? _VOX_C.box
                        : _VOX_C.pair;
            const g = groups[color] || (groups[color] =
                { px:[],py:[],pz:[], tx:[],ty:[],tz:[],tt:[], ids:[] });
            g.px.push(ligAtom.x, a.x, null);
            g.py.push(ligAtom.y, a.y, null);
            g.pz.push(ligAtom.z, a.z, null);
            // Distance MUST be the geometric distance between the two plotted atoms —
            // that's literally what the drawn line spans. The log's pr.r is in a
            // different frame (it comes out ~2x off / mismatched vs the exported-pose
            // coordinates this viewer plots), so trusting it made the label disagree
            // with the line. Always measure from coordinates.
            const r  = _voxelDist(ligAtom, a);
            g.tx.push((ligAtom.x+a.x)/2); g.ty.push((ligAtom.y+a.y)/2); g.tz.push((ligAtom.z+a.z)/2);
            g.tt.push(r.toFixed(2) + ' Å');
            g.ids.push(recIdx);
        });
        Object.keys(groups).forEach(color => {
            const g = groups[color];
            traces.push({ type:'scatter3d', mode:'lines', x:g.px, y:g.py, z:g.pz,
                line:{ color, width:6, dash:'dot' }, hoverinfo:'skip',
                name:'VoxelMeasure', showlegend:false });
            traces.push(markers(g.ids, color, 8));
            traces.push({ type:'scatter3d', mode:'text', x:g.tx, y:g.ty, z:g.tz, text:g.tt,
                textfont:{ color:(_VOX_LC[color] || '#fde68a'), size:13 }, textposition:'middle center',
                hoverinfo:'skip', name:'VoxelMeasure', showlegend:false });
        });
    }

    // Mark the ligand atom itself whenever anything hangs off it.
    if (ligAtom && (allSet.size || _voxelHL.pairs.size)) {
        traces.push({ type:'scatter3d', mode:'markers',
            x:[ligAtom.x], y:[ligAtom.y], z:[ligAtom.z],
            marker:{ size:10, color:'#f59e0b', symbol:'diamond', line:{width:1, color:'#78350f'} },
            hoverinfo:'skip', name:'VoxelMeasure', showlegend:false });
    }

    _voxelSetMeasureTraces(traces);
    return { allSet, boxSet, overlap };
}

// rec_idx → receptor atom, for labelling contact rows.
function _voxelRecByIdx() {
    const m = {};
    for (let i = 0; i < _recAtoms.length; i++) m[_recAtoms[i].idx] = _recAtoms[i];
    return m;
}

/* ════════════════════════════════════════════════════════════════════════════
   pair_e breakdown — a faithful JS port of the backend's _term_breakdown
   (vina_dock_routes.py). AutoDock Vina's empirical score is a weighted sum of
   distance-dependent terms evaluated at the surface distance s = r − (Ri+Rj):
     gauss1      = exp(−(s/0.5)²)
     gauss2      = exp(−((s−3)/2)²)
     repulsion   = s²      (only when s < 0)
     hydrophobic = piecewise ramp on s   (only between two hydrophobic atoms)
     hbond       = piecewise ramp on s   (only across a donor/acceptor pair)
     pair_e      = Σ wᵢ·termᵢ
   ════════════════════════════════════════════════════════════════════════════ */
const _VINA_W = { gauss1:-0.035579, gauss2:-0.005156, repulsion:0.840245,
                  hydrophobic:-0.035069, hbond:-0.587439 };
// xs type → [label, hydrophobic, hbond_donor, hbond_acceptor]
const _XS_META = {
    0:['C_H',true,false,false],   1:['C_P',true,false,false],
    2:['N_P',false,false,false],  3:['N_D',false,true,false],
    4:['N_A',false,false,true],   5:['N_DA',false,true,true],
    6:['O_P',false,false,false],  7:['O_D',false,true,false],
    8:['O_A',false,false,true],   9:['O_DA',false,true,true],
    10:['S_P',false,false,false], 11:['P_P',false,false,false],
    12:['F_H',false,false,false], 13:['Cl_H',false,false,false],
    14:['Br_H',false,false,false],15:['I_H',false,false,false],
    16:['Met',false,false,false],
};

function _voxelTermBreakdown(xs1, xs2, s) {
    const g1 = Math.exp(-Math.pow(s / 0.5, 2));
    const g2 = Math.exp(-Math.pow((s - 3.0) / 2.0, 2));
    const rp = s < 0 ? s * s : 0.0;
    const m1 = _XS_META[xs1] || ['?', false, false, false];
    const m2 = _XS_META[xs2] || ['?', false, false, false];
    const hp = (m1[1] && m2[1]) ? (s <= 0.5 ? 1.0 : s >= 1.5 ? 0.0 : 1.0 - (s - 0.5)) : 0.0;
    const hbOk = (m1[2] && m2[3]) || (m1[3] && m2[2]);
    const hb = hbOk ? (s <= -0.7 ? 1.0 : s >= 0.0 ? 0.0 : 1.0 - s / (-0.7)) : 0.0;
    const W = _VINA_W;
    const rows = [
        ['gauss1',      g1, W.gauss1],
        ['gauss2',      g2, W.gauss2],
        ['repulsion',   rp, W.repulsion],
        ['hydrophobic', hp, W.hydrophobic],
        ['hbond',       hb, W.hbond],
    ];
    let sum = 0;
    rows.forEach(row => { row[3] = row[1] * row[2]; sum += row[3]; });   // row[3] = weighted term
    return { rows, sum, xs1Label: m1[0], xs2Label: m2[0] };
}

// Toggle a ligand↔receptor measurement. Reached from a contact row (always a scored
// pair) or from clicking a highlighted receptor marker (which may be a box atom beyond
// Vina's cutoff — no pair_e). Turning it on draws the distance in 3D and posts either
// the term-by-term pair_e breakdown (scored contacts) or a distance-only note (atoms
// past the cutoff); clicking again clears it.
function _voxelPairClick(ligIdx, recIdx) {
    const ligAtom = _ligAtomsCache[ligIdx];
    const recAtom = _voxelRecByIdx()[recIdx];
    if (!ligAtom || !recAtom) return;

    const pair    = (_pairsByLig[ligIdx] || []).find(p => p.rec_idx === recIdx);  // undefined if beyond cutoff
    const ligName = (ligAtom.symbol || 'atom') + ' #' + ligIdx;
    const recName = recAtom.name + ' ' + recAtom.resname + recAtom.resseq;
    // Geometric distance between the plotted atoms — the true separation in the
    // exported pose. (The log's pair.r is in a mismatched frame; see the label code.)
    const r = _voxelDist(ligAtom, recAtom);

    if (_voxelHL.ligIdx !== ligIdx) {          // switched ligand atom → drop its predecessor's rows
        _voxelHL.ligIdx = ligIdx; _voxelHL.pairs.clear(); _voxelHL.all = false;
    }
    if (_voxelHL.pairs.has(recIdx)) {          // second click on the same atom → clear it
        _voxelHL.pairs.delete(recIdx);
        _voxelRedrawHighlights();
        _voxelSyncRowStyles();
        _vinaStatus('Cleared ' + ligName + ' → ' + recName + '.');
        return;
    }
    _voxelHL.pairs.add(recIdx);
    _voxelRedrawHighlights();
    _voxelSyncRowStyles();

    let bodyHtml;
    if (pair && pair.s != null && pair.rec_xs != null && pair.lig_xs != null) {
        // The breakdown's terms derive from pair.s (= log r − opt), so show the log's
        // own r here for internal consistency of this card. The drawn line + the
        // measurement headline use the geometric distance.
        const rForFormula = (pair.r != null) ? pair.r : r;
        const t = _voxelTermBreakdown(pair.lig_xs, pair.rec_xs, pair.s);
        bodyHtml = _voxelFormulaHtml(rForFormula, pair.s, t, pair.pair_e);
    } else if (pair) {
        // Scored contact, but the payload predates rec_xs/lig_xs/s — show the distance.
        bodyHtml = '<div style="color:#94a3b8;font-size:11px;">distance r = <b style="color:#fde68a">'
                 + r.toFixed(3) + ' Å</b> · pair_e = ' + Number(pair.pair_e).toFixed(4) + ' kcal/mol'
                 + '<div style="color:#64748b;font-size:9px;margin-top:3px;">Re-run Visualize to load the'
                 + ' term breakdown for this pair.</div></div>';
    } else {
        // Not in this ligand atom's pair list at all.
        bodyHtml = '<div style="color:#94a3b8;font-size:11px;">distance r = <b style="color:#fde68a">'
                 + r.toFixed(3) + ' Å</b> <span style="color:#64748b;">(measured from coordinates)</span>'
                 + '<div style="color:#64748b;font-size:9px;margin-top:3px;">Not in ' + ligName + '\'s scored'
                 + ' pair list, so it contributes pair_e = 0.</div></div>';
    }

    // If the log recorded its own r for this pair and it disagrees with the geometric
    // distance, surface both — the geometric one (shown above, and drawn) is the true
    // separation of the plotted atoms; the log value comes from a different frame.
    if (pair && pair.r != null && Math.abs(pair.r - r) > 0.1) {
        bodyHtml = '<div style="color:#94a3b8;font-size:9px;line-height:1.5;margin-bottom:5px;'
                 + 'border-left:2px solid #475569;padding-left:6px;">geometric distance = '
                 + r.toFixed(3) + ' Å (plotted-pose separation, shown on the line). '
                 + 'vina_non_cache.log recorded r = ' + Number(pair.r).toFixed(3) + ' Å for this pair '
                 + '(different coordinate frame).</div>' + bodyHtml;
    }

    _voxelChatMessage('🔬 ' + ligName + ' → ' + recName + ' [' + recIdx + ']', bodyHtml);
    _vinaStatus('Measuring ' + ligName + ' → ' + recName + ' : ' + r.toFixed(2) + ' Å');
}

function _voxelDist(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function _voxelFormulaHtml(r, s, t, loggedPairE) {
    const rowHtml = t.rows.map(row => {
        const name = row[0], raw = row[1], w = row[2], wv = row[3];
        const c = wv < 0 ? '#5eead4' : (wv > 0 ? '#f87171' : '#475569');
        return '<div style="display:flex;gap:6px;">'
             +   '<span style="color:#94a3b8;width:72px;flex-shrink:0;">' + name + '</span>'
             +   '<span style="color:#64748b;width:50px;text-align:right;">' + raw.toFixed(3) + '</span>'
             +   '<span style="color:#475569;width:62px;text-align:right;">×' + w.toFixed(4) + '</span>'
             +   '<span style="color:' + c + ';flex:1;text-align:right;font-variant-numeric:tabular-nums;">'
             +     (wv >= 0 ? '+' : '') + wv.toFixed(4) + '</span>'
             + '</div>';
    }).join('');
    return ''
      + '<div style="color:#64748b;font-size:9px;margin-bottom:5px;">'
      +   'r = <span style="color:#fde68a;">' + r.toFixed(3) + ' Å</span>'
      +   ' · s = r−opt = ' + s.toFixed(3) + ' Å · types ' + t.xs1Label + '↔' + t.xs2Label
      + '</div>'
      + '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;'
      +   'display:flex;flex-direction:column;gap:2px;">'
      +   '<div style="display:flex;gap:6px;color:#334155;font-size:9px;">'
      +     '<span style="width:72px;flex-shrink:0;">term</span>'
      +     '<span style="width:50px;text-align:right;">raw</span>'
      +     '<span style="width:62px;text-align:right;">× weight</span>'
      +     '<span style="flex:1;text-align:right;">kcal/mol</span></div>'
      +   rowHtml
      +   '<div style="border-top:1px solid #1e293b;margin-top:3px;padding-top:3px;display:flex;justify-content:space-between;">'
      +     '<span style="color:#67e8f9;font-weight:600;">Σ formula</span>'
      +     '<span style="color:#67e8f9;font-weight:600;font-variant-numeric:tabular-nums;">'
      +       (t.sum >= 0 ? '+' : '') + t.sum.toFixed(4) + ' kcal/mol</span></div>'
      +   '<div style="display:flex;justify-content:space-between;color:#64748b;font-size:9px;">'
      +     '<span>logged pair_e</span><span>' + Number(loggedPairE).toFixed(4) + ' kcal/mol</span></div>'
      + '</div>';
}

// Replace the 'VoxelMeasure' overlay with the given traces, preserving the user's
// camera. Adding/removing traces makes Plotly re-fit the gl3d scene (snapping the
// camera back to "fit all data"), so we snapshot the camera and restore it after.
let _voxelRenderTimer = null, _voxelPendingTraces = null;

function _voxelSetMeasureTraces(traces) {
    const plotEl = document.getElementById('vPlotA');
    if (!plotEl || typeof Plotly === 'undefined' || !plotEl.data) return;

    // Swap the overlay in ONE render. The old path deleted traces one-by-one and then
    // called addTraces + relayout, so a single click could trigger ~10 sequential gl3d
    // rebuilds; that re-creates the camera controller mid-event and its drag listeners
    // end up detached (rotation stops working). Plotly.react diffs in a single pass, and
    // uirevision tells Plotly to preserve camera/zoom across data changes.
    const cam = _voxelCurrentCamera(plotEl);
    _voxelPendingTraces = traces || [];
    if (_voxelRenderTimer) return;                  // coalesce: last state wins, one render

    // Defer past the current click event so Plotly finishes its own mouse bookkeeping
    // before the scene is touched.
    _voxelRenderTimer = setTimeout(() => {
        _voxelRenderTimer = null;
        const gd = document.getElementById('vPlotA');
        if (!gd || !gd.data) return;
        const base   = (gd.data || []).filter(t => t.name !== 'VoxelMeasure' && t.name !== 'VoxelPick');
        const layout = gd.layout || {};
        layout.uirevision = 'vina3d';               // keep user camera across updates
        if (cam) { layout.scene = layout.scene || {}; layout.scene.camera = cam; }
        try { Plotly.react('vPlotA', base.concat(_voxelPendingTraces || []), layout); } catch (e) {}
    }, 0);
}

// Σ row: toggle the "every contact of this ligand atom" highlight (amber).
function _voxelHighlightAllContacts(ligIdx) {
    const pairs   = _pairsByLig[ligIdx] || [];
    const ligAtom = _ligAtomsCache[ligIdx];
    if (!ligAtom || !pairs.length) return;
    const ligName = (ligAtom.symbol || 'atom') + ' #' + ligIdx;

    if (_voxelHL.ligIdx !== ligIdx) {          // switched ligand atom → drop stale rows
        _voxelHL.ligIdx = ligIdx; _voxelHL.pairs.clear(); _voxelHL.all = false;
    }
    _voxelHL.all = !_voxelHL.all;
    const sets = _voxelRedrawHighlights();
    _voxelSyncRowStyles();

    if (!_voxelHL.all) { _vinaStatus(ligName + ': cleared the all-contacts highlight.'); return; }

    const sumE = pairs.reduce((s, p) => s + (p.pair_e || 0), 0);
    const n    = sets.allSet.size, ov = sets.overlap.size;
    _voxelChatMessage('🔬 ' + ligName + ' · all ' + n + ' contacts',
        '<div style="color:#94a3b8;font-size:11px;">Highlighted every receptor atom this ligand atom '
      + 'contacts (<span style="color:' + _VOX_C.all + ';font-weight:600;">amber</span>). '
      + '<span style="color:#67e8f9;font-weight:600;">Σ pair_e = ' + sumE.toFixed(4) + ' kcal/mol</span>.'
      + (ov ? ' <span style="color:' + _VOX_C.overlap + ';font-weight:600;">' + ov + '</span> of them also sit '
            + 'inside the docking box and are drawn in magenta.' : '')
      + '</div>');
    _vinaStatus(ligName + ': highlighted all ' + n + ' contacts (Σ ' + sumE.toFixed(3) + ').');
}

// Docking-box row: toggle the "every receptor atom inside Box ctr / len" highlight (green).
function _voxelHighlightBox() {
    _voxelHL.box = !_voxelHL.box;
    const sets = _voxelRedrawHighlights();     // re-reads the box inputs when turning on
    _voxelSyncRowStyles();

    if (!_voxelHL.box) { _vinaStatus('Cleared the docking-box highlight.'); return; }

    const { cx, cy, cz, sx } = _GRID;
    const n = sets.boxSet.size, ov = sets.overlap.size;
    if (!n) {
        _vinaStatus('No receptor atoms inside the docking box.', true);
        _voxelChatMessage('⬜ Docking box',
            '<div style="color:#94a3b8;font-size:11px;">No receptor atoms fall inside the box '
          + '(centre ' + cx + ', ' + cy + ', ' + cz + ' · ' + sx + ' Å).</div>');
        return;
    }
    _voxelChatMessage('⬜ Docking box · ' + n + ' receptor atoms',
        '<div style="color:#94a3b8;font-size:11px;">Highlighted <span style="color:' + _VOX_C.box
      + ';font-weight:600;">' + n + ' receptor atoms</span> inside the box (centre ' + cx + ', ' + cy
      + ', ' + cz + ' · ' + sx + ' Å edge).'
      + (ov ? ' <span style="color:' + _VOX_C.overlap + ';font-weight:600;">' + ov + '</span> of them also '
            + 'contact the selected ligand atom and are drawn in magenta.' : '')
      + '</div>');
    _vinaStatus('Docking box: ' + n + ' receptor atoms highlighted'
              + (ov ? ' · ' + ov + ' overlap with contacts.' : '.'));
}

// Voxel-grid row: toggle the szv_grid cell wireframe over the docking box.
function _voxelToggleVoxelGrid() {
    _voxelHL.grid = !_voxelHL.grid;
    _voxelRedrawHighlights();
    _voxelSyncRowStyles();
    if (!_voxelHL.grid) { _vinaStatus('Hid the docking-box voxels.'); return; }

    if (typeof _readGridInputs === 'function') _readGridInputs();
    const nx = _voxelVoxelCount(_GRID.sx), ny = _voxelVoxelCount(_GRID.sy), nz = _voxelVoxelCount(_GRID.sz);
    _voxelChatMessage('▦ Docking-box voxels · ' + nx + '×' + ny + '×' + nz,
        '<div style="color:#94a3b8;font-size:11px;">Drew the <span style="color:' + _VOX_C.grid
      + ';font-weight:600;">' + (nx*ny*nz) + ' voxels</span> (' + nx + '×' + ny + '×' + nz
      + ') the box is divided into — Vina bins receptor atoms into these ~3 Å cells to build each '
      + 'ligand atom\'s neighbour shortlist (szv_grid).'
      + '<div style="color:#64748b;font-size:9px;margin-top:3px;">Cell count uses int((edge)/3) per axis, '
      + 'matching szv_grid_dims.</div></div>');
    _vinaStatus('Docking box: ' + (nx*ny*nz) + ' voxels (' + nx + '×' + ny + '×' + nz + ').');
}

// Voxels per axis — mirrors szv_grid_dims: floor(edge/3), at least 1.
function _voxelVoxelCount(edge) {
    const n = Math.floor(edge / 3);
    return n < 1 ? 1 : n;
}

// Resolved path of the docked pose the backend actually read (from vina_parse_log).
let _dockedPosePath = null;

// Push _ligAtomsCache coordinates into the plotted ligand traces. Belt-and-braces
// guarantee on top of the two render fast-path fixes: locates traces by NAME (not a
// hardcoded index), so it is correct in every view regardless of trace order. restyle
// never moves the camera. Returns true if the Atoms trace was found and updated.
function _voxelApplyLigandCoords() {
    const gd = document.getElementById('vPlotA');
    if (!gd || !gd.data || typeof Plotly === 'undefined') return false;
    if (!Array.isArray(_ligAtomsCache) || !_ligAtomsCache.length) return false;
    const atoms = _ligAtomsCache;

    let atomsIdx = -1, bondsIdx = -1;
    gd.data.forEach((t, i) => {
        if (t.name === 'Atoms' && (t.x || []).length === atoms.length) atomsIdx = i;
        else if (t.name === 'Bonds' && bondsIdx < 0) bondsIdx = i;
    });
    if (atomsIdx < 0) return false;

    try {
        Plotly.restyle('vPlotA', {
            x: [atoms.map(a => a.x)], y: [atoms.map(a => a.y)], z: [atoms.map(a => a.z)],
        }, [atomsIdx]);
    } catch (e) { console.error('[voxel] ligand coord restyle failed:', e); return false; }

    if (bondsIdx >= 0) {
        const _bs = _vinaBondSegments(atoms, window._lastBonds || []);
        try { Plotly.restyle('vPlotA', { x: [_bs.x], y: [_bs.y], z: [_bs.z] }, [bondsIdx]); } catch (e) {}
    }
    return true;
}

// Read a ligand atom's coordinate back OUT of the chart, so a toggle can VERIFY that
// its swap reached the screen instead of asserting that it did. This is what turns the
// old "trust the message" behaviour into "prove it against the plotted data."
function _voxelPlottedLigCoord(i) {
    const gd = document.getElementById('vPlotA');
    if (!gd || !gd.data || !Array.isArray(_ligAtomsCache)) return null;
    for (const t of gd.data) {
        if (t.name === 'Atoms' && (t.x || []).length === _ligAtomsCache.length)
            return { x: t.x[i], y: t.y[i], z: t.z[i] };
    }
    return null;
}

// ── Toggle ligand atom positions: exported pose ⇄ Vina's real docked pose ──────
// The exported PDBQT and the frame Vina actually scored place the ligand differently
// (receptor matches; ligand can be off by several Å). This toggle prefers the REAL
// docked pose from <ligand>_out.pdbqt (written by Vina itself — authoritative, no
// reconstruction), falling back to the coordinates parsed out of vina_non_cache.log
// only if the backend couldn't find/match that file. It swaps x/y/z in-place, forces
// the plotted traces to match via _voxelApplyLigandCoords (the two render fast paths
// alone previously left the ligand un-moved — the root cause of the false claim), then
// READS THE COORDINATE BACK OFF THE CHART and reports the real numbers instead of
// asserting success.
function _voxelToggleLigFrame() {
    if (!Array.isArray(_ligAtomsCache) || !_ligAtomsCache.length) return;

    const nWithDocked = _ligAtomsCache.filter(a => a && a.docked_x != null).length;
    const nWithLog    = _ligAtomsCache.filter(a => a && a.log_x    != null).length;
    const source = nWithDocked > 0 ? 'docked' : (nWithLog > 0 ? 'log' : null);
    const nWithFrame = source === 'docked' ? nWithDocked : nWithLog;

    if (!source) {
        _vinaStatus('No docked-pose coordinates — run Vina Dock, then re-run Visualize.', true);
        _voxelChatMessage('⇄ Docked pose',
            '<div style="color:#94a3b8;font-size:11px;">The backend sent no docked coordinate: neither '
          + '<code>&lt;ligand&gt;_out.pdbqt</code> nor a matching <code>log_xyz</code> was found. Run '
          + '<b>⚗️ Vina Dock</b>, then <b>⚡ Visualize</b>.</div>');
        return;
    }

    _voxelHL.ligFrame = !_voxelHL.ligFrame;

    // Swap in the cache, measuring how far each atom really moves so the report below
    // can quote numbers instead of asserting success. probeI = first atom with a swap.
    let moved = 0, sumD = 0, maxD = 0, probeI = -1;
    _ligAtomsCache.forEach((a, i) => {
        if (!a) return;
        const x0 = a.x, y0 = a.y, z0 = a.z;
        if (_voxelHL.ligFrame) {
            const dx = (source === 'docked') ? a.docked_x : a.log_x;
            const dy = (source === 'docked') ? a.docked_y : a.log_y;
            const dz = (source === 'docked') ? a.docked_z : a.log_z;
            if (dx == null) return;
            if (a._poseX == null) { a._poseX = a.x; a._poseY = a.y; a._poseZ = a.z; }
            a.x = dx; a.y = dy; a.z = dz;
        } else {
            if (a._poseX == null) return;
            a.x = a._poseX; a.y = a._poseY; a.z = a._poseZ;
        }
        const d = Math.sqrt((a.x-x0)*(a.x-x0) + (a.y-y0)*(a.y-y0) + (a.z-z0)*(a.z-z0));
        if (d > 1e-6) { moved++; sumD += d; if (d > maxD) maxD = d; }
        if (probeI < 0) probeI = i;
    });

    // Re-render, then FORCE the plotted coordinates to match the cache (belt+braces).
    const plotEl = document.getElementById('vPlotA');
    const cam = plotEl && plotEl._fullLayout && plotEl._fullLayout.scene
              ? _voxelCurrentCamera(plotEl) : null;
    if (_currentView === 'both')         _renderBothView(_voxelHL.ligIdx != null ? _voxelHL.ligIdx : _selAtomIdx);
    else if (_currentView === 'protein') _renderProteinView(_selAtomIdx);
    else                                 _render3D('vPlotA', _ligAtomsCache, window._lastBonds || [], false);
    _voxelApplyLigandCoords();
    if (cam) { try { Plotly.relayout('vPlotA', {'scene.camera': cam}); } catch (e) {} }
    _voxelRedrawHighlights();      // lines/fan follow the new atom positions
    _voxelSyncRowStyles();

    // ── Verify against the chart rather than asserting ────────────────────────
    const probe = probeI >= 0 ? _ligAtomsCache[probeI] : null;
    const shown = probeI >= 0 ? _voxelPlottedLigCoord(probeI) : null;
    const fmt = c => c ? '(' + Number(c.x).toFixed(3) + ', ' + Number(c.y).toFixed(3)
                             + ', ' + Number(c.z).toFixed(3) + ')' : '—';
    const landed = !!(probe && shown
        && Math.abs(shown.x - probe.x) < 1e-3
        && Math.abs(shown.y - probe.y) < 1e-3
        && Math.abs(shown.z - probe.z) < 1e-3);

    if (!landed) {
        _voxelChatMessage('⚠️ Docked pose — NOT applied',
            '<div style="color:#fca5a5;font-size:11px;">The swap did not reach the chart. Intended <b>'
          + (probe ? probe.symbol : '?') + '</b> at ' + fmt(probe) + ' but the plot shows ' + fmt(shown)
          + '.<div style="color:#64748b;font-size:9px;margin-top:3px;">Reported rather than claimed '
          + 'success — the coordinate was read back out of the chart.</div></div>');
        _vinaStatus('Docked-pose toggle did not reach the chart — see the chat card.', true);
        return;
    }

    if (_voxelHL.ligFrame) {
        const srcLabel = source === 'docked'
            ? 'the <b>real docked pose</b> from <code>' + (_dockedPosePath || '&lt;ligand&gt;_out.pdbqt') + '</code>'
            : 'coordinates <b>parsed from vina_non_cache.log</b> (no matching _out.pdbqt found)';
        _voxelChatMessage('⇄ Docked pose (in box)',
            '<div style="color:#94a3b8;font-size:11px;">Ligand is now at ' + srcLabel + '.'
          + '<div style="margin-top:4px;font-family:ui-monospace,monospace;font-size:10px;color:#cbd5e1;">'
          +   'moved <b>' + moved + '</b>/' + nWithFrame + ' atoms · mean '
          +   (sumD / Math.max(moved, 1)).toFixed(3) + ' Å · max ' + maxD.toFixed(3) + ' Å<br>'
          +   (probe ? (probe.symbol + ': ' + fmt({x:probe._poseX, y:probe._poseY, z:probe._poseZ})
                        + ' → <span style="color:#f472b6;">' + fmt(shown) + '</span> (verified on chart)') : '')
          + '</div>'
          + '<div style="color:#64748b;font-size:9px;margin-top:4px;">Click again for the exported .pdbqt. '
          + 'Receptor unchanged. Re-run Visualize after any new dock, or this shows the previous run\'s pose.'
          + '</div></div>');
        _vinaStatus('Docked pose on · ' + moved + ' atoms moved (max ' + maxD.toFixed(2) + ' Å).');
    } else {
        _vinaStatus('Showing exported .pdbqt pose (' + moved + ' atoms restored).');
    }
}

// Wireframe of the szv_grid: the box split into nx×ny×nz cells. One line trace holds
// every grid line as null-separated segments (thin, so the box edges still read).
function _voxelGridEdges(g) {
    const nx = _voxelVoxelCount(g.sx), ny = _voxelVoxelCount(g.sy), nz = _voxelVoxelCount(g.sz);
    const x0 = g.cx - g.sx/2, y0 = g.cy - g.sy/2, z0 = g.cz - g.sz/2;
    const xs = [], ys = [], zs = [];
    for (let i = 0; i <= nx; i++) xs.push(x0 + g.sx * i / nx);
    for (let j = 0; j <= ny; j++) ys.push(y0 + g.sy * j / ny);
    for (let k = 0; k <= nz; k++) zs.push(z0 + g.sz * k / nz);
    const X=[], Y=[], Z=[];
    const seg = (ax,ay,az, bx,by,bz) => { X.push(ax,bx,null); Y.push(ay,by,null); Z.push(az,bz,null); };
    // Lines along X (vary x, fixed y,z), along Y, along Z.
    ys.forEach(y => zs.forEach(z => seg(xs[0],y,z, xs[nx],y,z)));
    xs.forEach(x => zs.forEach(z => seg(x,ys[0],z, x,ys[ny],z)));
    xs.forEach(x => ys.forEach(y => seg(x,y,zs[0], x,y,zs[nz])));
    return { type:'scatter3d', mode:'lines', x:X, y:Y, z:Z,
        line:{ color:_VOX_C.grid, width:1 }, opacity:0.28,
        hoverinfo:'skip', name:'VoxelMeasure', showlegend:false };
}

// 12 edges of the docking-box cube as one line trace (null-separated segments).
function _voxelBoxEdges(cx, cy, cz, hx, hy, hz) {
    const v = [
        [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
        [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
    ];
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const x=[], y=[], z=[];
    edges.forEach(([a,b]) => {
        x.push(v[a][0], v[b][0], null);
        y.push(v[a][1], v[b][1], null);
        z.push(v[a][2], v[b][2], null);
    });
    return { type:'scatter3d', mode:'lines', x, y, z,
        line:{ color:'#34d399', width:2, dash:'dot' }, opacity:0.6,
        hoverinfo:'skip', name:'VoxelMeasure', showlegend:false };
}

// Best-effort read of the live gl3d camera (eye/center/up). Falls back to the camera
// our relayout listener last saved for the combined view.
function _voxelCurrentCamera(gd) {
    try {
        const s = gd && gd._fullLayout && gd._fullLayout.scene;
        if (s) {
            if (s._scene && typeof s._scene.getCamera === 'function') return s._scene.getCamera();
            if (s.camera) return s.camera;
        }
    } catch (e) {}
    return window._savedCameraBoth || null;
}

// Remove every overlay trace — same single-render path as setting them.
function _voxelClearMeasure() {
    _voxelSetMeasureTraces([]);
}

/* ── Receptor atom clicked → single pair_e detail (into the mini-chat) ───────── */
function _voxelShowReceptorAtom(pt) {
    const cd    = pt.customdata || {};
    const recId = (cd.rec_idx != null) ? cd.rec_idx : '—';
    const name  = cd.name || 'atom';
    const res   = cd.res  || '—';
    const pairE = (cd.pair_e == null || cd.pair_e === 'n/a')
                    ? 'n/a (non-interacting)'
                    : cd.pair_e + ' kcal/mol';
    const xyz   = [pt.x, pt.y, pt.z]
                    .map(v => (typeof v === 'number' ? v.toFixed(2) : '?'))
                    .join(', ');
    const ligAtom  = (_selAtomIdx != null) ? _ligAtomsCache[_selAtomIdx] : null;
    const ligLabel = ligAtom ? ((ligAtom.symbol || 'atom') + _selAtomIdx) : '—';

    const heading  = '🔬 ' + name + ' ' + res + '  ·  rec ' + recId;
    const bodyHtml =
        '<div style="color:#64748b;font-size:9px;margin-bottom:4px;">pair_e vs ligand atom ' + ligLabel
      +   ' · from vina_non_cache.log</div>'
      + '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;flex-direction:column;gap:2px;">'
      +   _voxelRow('pair_e',  pairE, '#5eead4')
      +   _voxelRow('xyz (Å)', xyz,   '#e2e8f0')
      + '</div>';

    _voxelChatMessage(heading, bodyHtml);
    _voxelSetHidden('voxelPickOverlay', true);
    _vinaStatus('Inspecting ' + name + ' ' + res + ' (rec ' + recId + ') → chat.');
}

/* ── Toggle / activate / deactivate ─────────────────────────────────────────── */
function _toggleVoxelPick() {
    _voxelPickActive = !_voxelPickActive;
    if (_voxelPickActive) _voxelActivate();
    else                  _voxelDeactivate();
}

function _voxelActivate() {
    _voxelStyleButton(true);
    _ensureVoxelListener();                                   // bind now if plot already exists
    const plotEl = document.getElementById('vPlotA');
    if (plotEl) plotEl.style.cursor = 'crosshair';
    _voxelSetHidden('voxelPickOverlay', false);              // show the crosshair hint

    if (_currentView === 'protein')
        _vinaStatus('Voxel inspect on — click a receptor atom for its pair_e.');
    else
        _vinaStatus('Voxel inspect on — click a ligand atom to list its protein contacts.');
}

// Automates the two manual steps — "click 🔬 Voxel" then "click this atom in 3D" — for
// a given ligand atom index. Used by the sidebar atom-row click so the row produces the
// exact same state a real pick would: pick mode ON (button glow, crosshair cursor), and
// the contact panel populated via the SAME router (_voxelHandleClick) a genuine
// plotly_click event uses, rather than a parallel code path that could drift from it.
function _voxelSimulateAtomPick(atomIdx) {
    // [voxel STEP 4] Unconditional entry log.
    _vlog('[voxel STEP 4] _voxelSimulateAtomPick entered',
        {atomIdx, voxelPickActiveBefore: _voxelPickActive});
    _voxelBanner('4️⃣ _voxelSimulateAtomPick(' + atomIdx + ')', '#facc15');

    // Step 1 — "click 🔬 Voxel": only if it isn't already on. _toggleVoxelPick() is a
    // blind on/off switch, so calling it unconditionally here would flip pick mode OFF
    // on every second row click — turn it on directly instead, via the same activation
    // the button itself uses.
    if (!_voxelPickActive) {
        _voxelPickActive = true;
        _voxelActivate();
        _vlog('[voxel STEP 4] pick mode activated (was off)', {});
    } else {
        _vlog('[voxel STEP 4] pick mode already on, skipped activation', {});
    }

    // Step 2 — "click this atom in the 3D view": build the same point shape Plotly
    // hands to a real plotly_click on the 'Atoms' trace, and route it through the
    // identical handler a genuine click would hit.
    const a = _ligAtomsCache[atomIdx];
    // This lookup previously failed SILENTLY (`if (!a) return;` with no log) — that is
    // a real, plausible failure mode: _ligAtomsCache is only populated inside the
    // Visualize response handlers, so a click before/without a successful Visualize
    // would hit this exact line and stop here with no trace at all.
    _vlog('[voxel STEP 4] _ligAtomsCache lookup', {
        atomIdx,
        found: !!a,
        cacheLength: Array.isArray(_ligAtomsCache) ? _ligAtomsCache.length : Object.keys(_ligAtomsCache||{}).length,
        cacheIsArray: Array.isArray(_ligAtomsCache),
        cacheSample: (_ligAtomsCache && _ligAtomsCache[0]) || null,
    });
    if (!a) {
        console.error('%c[voxel STEP 4] STOPPING — no atom at _ligAtomsCache['+atomIdx+']',
            'background:#7f1d1d;color:#fff;font-weight:bold;padding:2px 4px',
            '\nThis means Visualize has not successfully populated _ligAtomsCache yet — '
          + 'check for [voxel STEP -1] logs above to see whether Visualize ran/succeeded.');
        return;
    }
    const pt = {
        data: { name: 'Atoms' },
        customdata: { idx: atomIdx, raw: (a.weight_raw ?? 0).toFixed(4), norm: (a.weight_norm ?? 0).toFixed(4) },
        pointNumber: atomIdx,
        x: a.x, y: a.y, z: a.z,
    };
    _vlog('[voxel STEP 4] synthetic point built, calling _voxelHandleClick', pt);
    _voxelHandleClick(pt);
}

// Full exit: toggling the button off. Hides the contact panel, clears the crosshair
// and the 3D measurement, and un-highlights the button.
function _voxelDeactivate() {
    _voxelPickActive = false;
    _voxelStyleButton(false);
    const plotEl = document.getElementById('vPlotA');
    if (plotEl) plotEl.style.cursor = '';
    _voxelSetHidden('voxelPickOverlay', true);
    _voxelSetHidden('voxelPanel', true);
    _voxelResetHighlights();
    _vinaStatus('Voxel inspect off.');
}

// The ✕ on #voxelPanel — hides the contact panel and clears the 3D measurement, but
// leaves pick mode on so another atom can be inspected. (The 🔬 button is the master
// off switch.)
function _closeVoxelPanel() {
    _voxelSetHidden('voxelPanel', true);
    _voxelResetHighlights();
}

// Drop every persistent highlight and wipe the overlay.
function _voxelResetHighlights() {
    _voxelHL.ligIdx = null;
    _voxelHL.pairs.clear();
    _voxelHL.all = false;
    _voxelHL.box = false;
    _voxelHL.grid = false;
    // If the ligand was swapped into the docked frame, put it back to the exported
    // pose and re-render, so closing the inspector returns to the original view.
    if (_voxelHL.ligFrame) {
        _voxelHL.ligFrame = false;
        if (Array.isArray(_ligAtomsCache)) _ligAtomsCache.forEach(a => {
            if (a && a._poseX != null) { a.x = a._poseX; a.y = a._poseY; a.z = a._poseZ; }
        });
        const gd = document.getElementById('vPlotA');
        const cam = gd && gd._fullLayout && gd._fullLayout.scene ? _voxelCurrentCamera(gd) : null;
        try {
            if (_currentView === 'both')         _renderBothView(_selAtomIdx);
            else if (_currentView === 'protein') _renderProteinView(_selAtomIdx);
            else                                 _render3D('vPlotA', _ligAtomsCache, window._lastBonds || [], false);
            _voxelApplyLigandCoords();   // fast paths alone won't move the atoms back
            if (cam) Plotly.relayout('vPlotA', {'scene.camera': cam});
        } catch (e) {}
    }
    _voxelClearMeasure();
    _voxelSyncRowStyles();
}

// ── Make #voxelPanel draggable by its header ─────────────────────────────────
// The panel is position:fixed with top/left; dragging just updates those. Bound once
// (guarded) on the header so it survives the panel being shown/hidden repeatedly.
function _voxelInitPanelDrag() {
    const panel  = document.getElementById('voxelPanel');
    const handle = document.getElementById('voxelPanelHeader');
    if (!panel || !handle || handle._dragBound) return;
    handle._dragBound = true;

    let startX = 0, startY = 0, startLeft = 0, startTop = 0, dragging = false;

    const onMove = (e) => {
        if (!dragging) return;
        const dx = e.clientX - startX, dy = e.clientY - startY;
        let left = startLeft + dx, top = startTop + dy;
        // Keep it on-screen (leave a little margin so the header is always grabbable).
        const w = panel.offsetWidth, h = panel.offsetHeight;
        left = Math.max(4, Math.min(left, window.innerWidth  - w - 4));
        top  = Math.max(4, Math.min(top,  window.innerHeight - h - 4));
        panel.style.left = left + 'px';
        panel.style.top  = top + 'px';
        panel.style.right = 'auto';   // in case it was ever set from the right edge
        e.preventDefault();
    };
    const onUp = () => {
        dragging = false;
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
    };
    handle.addEventListener('mousedown', (e) => {
        // Ignore drags that start on the ✕ close button.
        if (e.target.closest('button')) return;
        dragging = true;
        const r = panel.getBoundingClientRect();
        startX = e.clientX; startY = e.clientY;
        startLeft = r.left;  startTop = r.top;
        // Switch to explicit px left/top so dragging is absolute, not tied to any right/bottom anchor.
        panel.style.left = r.left + 'px';
        panel.style.top  = r.top + 'px';
        panel.style.right = 'auto'; panel.style.bottom = 'auto';
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        e.preventDefault();
    });
}
// Arm as soon as the DOM is ready (and again lazily, since the modal markup is present
// at load — the header exists even while the panel is hidden).
if (document.readyState !== 'loading') _voxelInitPanelDrag();
else document.addEventListener('DOMContentLoaded', _voxelInitPanelDrag);

// ── Fullscreen the Vina 3D plot ──────────────────────────────────────────────
// Toggles the browser Fullscreen API on the plot wrapper, then resizes Plotly so the
// gl3d scene fills the new dimensions (Plotly doesn't auto-resize on fullscreen).
function _vinaTogglePlotFullscreen() {
    const wrap = document.getElementById('vPlotWrap');
    if (!wrap) return;
    const resizePlot = () => {
        setTimeout(() => {
            const gd = document.getElementById('vPlotA');
            if (gd && gd._fullLayout && typeof Plotly !== 'undefined') Plotly.Plots.resize(gd);
        }, 120);
    };
    if (!document.fullscreenElement) {
        (wrap.requestFullscreen ? wrap.requestFullscreen()
            : wrap.webkitRequestFullscreen ? wrap.webkitRequestFullscreen()
            : Promise.reject()).then(resizePlot).catch(() => {});
    } else {
        (document.exitFullscreen ? document.exitFullscreen()
            : document.webkitExitFullscreen ? document.webkitExitFullscreen()
            : Promise.reject()).then(resizePlot).catch(() => {});
    }
}
// Resize the plot whenever fullscreen state changes (covers Esc-to-exit too).
document.addEventListener('fullscreenchange', () => {
    const gd = document.getElementById('vPlotA');
    if (gd && gd._fullLayout && typeof Plotly !== 'undefined')
        setTimeout(() => Plotly.Plots.resize(gd), 120);
});

/* ── Small helpers ──────────────────────────────────────────────────────────── */
function _voxelRow(label, value, valueColor) {
    return '<div style="display:flex;justify-content:space-between;gap:14px;">'
         +   '<span style="color:#64748b;">' + label + '</span>'
         +   '<span style="color:' + valueColor + ';">' + value + '</span>'
         + '</div>';
}

// Toggle Tailwind's .hidden (display:none) on/off. The inline display is set too, so a
// stale/global rule can't leave the panel invisible after the class is removed.
function _voxelSetHidden(id, hidden) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('hidden', hidden);
    el.style.display = hidden ? '' : 'block';
}

// Button "active" look via inline styles (independent of Tailwind's compiled classes).
function _voxelStyleButton(on) {
    const btn = document.getElementById('voxelInspectBtn');
    if (!btn) return;
    if (on) {
        btn.style.background  = '#083344';
        btn.style.borderColor = '#22d3ee';
        btn.style.color       = '#67e8f9';
        btn.style.boxShadow   = '0 0 0 1px #22d3ee, 0 0 12px rgba(34,211,238,.35)';
        btn.setAttribute('aria-pressed', 'true');
    } else {
        ['background', 'border-color', 'color', 'box-shadow'].forEach(p => btn.style.removeProperty(p));
        btn.setAttribute('aria-pressed', 'false');
    }
}