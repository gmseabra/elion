// =============================================================================
// ts/ts_warmup.js — warmup animation step, TS animation step, reaction picker
// Depends on: ts_core.js, ts_ui.js, ts_chart.js
// =============================================================================

// ── Animation step (warmup) ───────────────────────────────────────────────
function _tsWuStep() {
    const wu = _ts.wuState;
    if (!wu || wu.idx >= wu.events.length) return;
    const ev  = wu.events[wu.idx++];
    wu.evals++;
    if (ev.score > wu.best) wu.best = ev.score;
    const livePhase = _ts._currentPhase || ev.phase || wu.phase || '—';
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('tsWuPhase', livePhase);
    set('tsWuEvals', wu.evals);
    const idx = _ts._activeJobIdx ?? 0;
    set(`tsWuPhase_${idx}`, livePhase);
    set(`tsWuEvals_${idx}`, wu.evals);
    if (wu.best > 0) set(`tsWuBest_${idx}`, wu.best.toFixed(4));
}

// ── TS animation step ─────────────────────────────────────────────────────
function _tsTsStep() {
    const ts = _ts.tsState;
    if (!ts || ts.idx >= ts.events.length) return;
    const ev = ts.events[ts.idx++];
    (ev.updates || []).forEach(u => {
        const prev = ts.reagents[u.id]?.mu ?? u.mu_before;
        ts.reagents[u.id] = { name: u.name, mu: u.mu_after, std: u.std_after, sc: u.sc, delta_mu: u.mu_after - prev };
    });
    if (ev.score !== undefined) ts.winners.push({ iter: ev.iter, score: ev.score, smiles: ev.smiles || null });
    if (ev.masked !== undefined) ts.masked = ev.masked;
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    // NOTE: 'Current iter' (tsTsIter) is owned exclusively by the batch_stats
    // handler in ts_worker_bridge.js, which is active-job-guarded and uses the
    // backend's authoritative [TS:stats] iteration count. Updating it here too
    // (from the worker's own ev.iter, which counts winner-pairs and can differ)
    // caused the sidebar iteration to flicker between the two jobs' counts.
    set('tsTsScore',  ev.score !== undefined ? ev.score.toFixed(4) : '—');
    set('tsTsMasked', ts.masked || '—');
    _tsQueueBarsRender();
    const bar = document.getElementById('tsStatusBar');
    if (bar && ev.iter !== undefined) bar.textContent = `Iteration ${ev.iter} · score ${(ev.score || 0).toFixed(4)}`;
}

// ── Warmup badge refresh ──────────────────────────────────────────────────
// `warmup_cached` is computed server-side inside /ts_config, and ts_run.js
// fetches that exactly once — when the modal opens. So a checkpoint written
// DURING a run never lit its badge: the run that created
// `suzuki_<ts>_warmup.json` still showed "no cache", and only a page reload
// fixed it. Re-fetch after every completed run.
//
// _tsBuildReactionPicker already has an "picker exists -> just repaint the
// badges and return" branch, so calling it again is cheap and non-destructive:
// it does not rebuild the dropdown or disturb the user's checkbox selection.
function _tsRefreshWarmupBadges() {
    fetch('/vina_visualization/ts_config')
        .then(r => (r.ok ? r.json() : null))
        .then(d => {
            if (!d || !d.reactions) return;
            _tsBuildReactionPicker(d.reactions, d.smarts, d.warmup_cached || {});
        })
        .catch(() => {});     // a stale badge is not worth surfacing an error for
}

// ── Warmup clear ──────────────────────────────────────────────────────────
function _tsWarmupClear(rxnKey) {
    fetch('/vina_visualization/ts_warmup_clear', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({rxn_key: rxnKey}),
    }).then(r => r.json()).then(() => {
        const badge    = document.getElementById(`_wuCacheBadge_${rxnKey}`);
        const clearBtn = document.getElementById(`_wuClearBtn_${rxnKey}`);
        if (badge)    badge.style.display    = 'none';
        if (clearBtn) clearBtn.style.display = 'none';
        if (_ts._warmupCached) _ts._warmupCached[rxnKey] = false;
    }).catch(() => {});
}

// ── Engine-not-found banner ───────────────────────────────────────────────
// Rendered in the slot the reaction picker would occupy. Without it, a
// mis-resolved ELION_CWD looks exactly like a missing feature: /ts_config
// answers 404, the picker is never built, and the Reactions row is simply
// absent with nothing on screen saying why.
function _tsEngineBanner(detail) {
    document.getElementById('_tsRxnPicker')?.remove();
    const existing = document.getElementById('_tsEngineWarn');
    if (existing) existing.remove();

    const toolbarRow = document.getElementById('tsOutputDirInput')?.closest('div')
                    || document.getElementById('tsSmartsInput')?.closest('div');
    if (!toolbarRow) return;

    const bar = document.createElement('div');
    bar.id = '_tsEngineWarn';
    bar.style.cssText = [
        'display:flex', 'align-items:center', 'gap:8px', 'padding:5px 24px',
        'background:rgba(69,26,3,0.5)', 'border-bottom:0.5px solid rgba(245,158,11,0.35)',
        'font-size:10px', 'color:#fbbf24', 'min-height:30px',
    ].join(';');
    bar.innerHTML =
        '<span style="flex-shrink:0">⚠</span>' +
        '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
        'Elion engine config not found &mdash; reactions unavailable. ' +
        '<span style="color:#94a3b8">' + (detail || '') + '</span>' +
        ' Set <code style="color:#fbbf24">ELION_CWD</code> to the directory holding ' +
        '<code style="color:#fbbf24">input_TS.yml</code>.</span>';
    bar.title = detail || '';
    toolbarRow.parentNode.insertBefore(bar, toolbarRow.nextSibling);
}

// ── Reaction picker ───────────────────────────────────────────────────────
function _tsBuildReactionPicker(rxnKeys, currentSmarts, warmupCached = {}) {
    document.getElementById('_tsEngineWarn')?.remove();
    _ts._warmupCached = warmupCached;
    const existing       = document.getElementById('_tsRxnPicker');
    const dropdownExists = document.getElementById('_tsRxnDropdown');
    if (existing && dropdownExists) {
        rxnKeys.forEach(k => {
            const badge    = document.getElementById(`_wuCacheBadge_${k}`);
            const clearBtn = document.getElementById(`_wuClearBtn_${k}`);
            const show     = !!warmupCached[k];
            if (badge)    badge.style.display    = show ? 'inline' : 'none';
            if (clearBtn) clearBtn.style.display = show ? 'inline' : 'none';
        });
        existing.style.flexWrap  = 'nowrap';
        existing.style.minHeight = '32px';
        existing.style.display   = 'flex';
        return;
    }
    existing?.remove();

    const toolbarRow = document.getElementById('tsOutputDirInput')?.closest('div')
                    || document.getElementById('tsSmartsInput')?.closest('div');
    if (!toolbarRow) return;

    const labels = {
        rxn101_amide:'Amide (101)', rxn102_buchwald:'Buchwald (102)',
        rxn108_sonogashira:'Sonogashira (108)', rxn110_suzuki:'Suzuki (110)',
        rxn113_sulfonamide:'Sulfonamide (113)', rxn208_snar:'SnAr (208)',
    };

    const wrapper = document.createElement('div');
    wrapper.id = '_tsRxnPicker';
    wrapper.style.cssText = [
        'display:flex','align-items:center','flex-wrap:nowrap',
        'overflow:hidden','gap:6px','padding:4px 24px',
        'background:rgba(2,4,10,0.6)',
        'border-bottom:0.5px solid rgba(30,41,59,0.7)',
        'position:relative','min-height:32px',
    ].join(';');

    wrapper.innerHTML = `
      <button id="_tsBbImportBtn" onclick="_tsBbToggle(event)"
        title="Scan a directory of building-block CSVs and report how many blocks are eligible as each reaction's first and second reagent"
        style="display:flex;align-items:center;gap:5px;padding:3px 10px;flex-shrink:0;
               background:rgba(8,145,178,0.10);border:0.5px solid rgba(8,145,178,0.35);
               border-radius:5px;cursor:pointer;font-size:10px;color:#22d3ee;white-space:nowrap">
        <span style="font-size:10px">⛁</span><span>Import database</span>
      </button>
      <button id="_tsBbRestoreBtn" onclick="_tsBbReportRestore(event)"
        title="Reopen the building-block scan report"
        style="display:none;align-items:center;gap:5px;padding:3px 10px;flex-shrink:0;
               background:rgba(52,211,153,0.10);border:0.5px solid rgba(52,211,153,0.35);
               border-radius:5px;cursor:pointer;font-size:10px;color:#34d399;white-space:nowrap">
        <span style="font-size:10px">⛁</span><span id="_tsBbRestoreLabel">report</span>
      </button>
      <span style="font-size:9px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.05em;white-space:nowrap;flex-shrink:0">Reactions</span>
      <button id="_tsRxnBtn" onclick="_tsRxnToggle(event)"
        style="display:flex;align-items:center;gap:6px;padding:3px 10px 3px 8px;flex-shrink:0;
               background:rgba(30,41,59,0.6);border:0.5px solid rgba(71,85,105,0.5);
               border-radius:5px;cursor:pointer;font-size:10px;color:#94a3b8;white-space:nowrap">
        <span id="_tsRxnBtnLabel">None selected</span>
        <span style="font-size:8px;opacity:0.5">▼</span>
      </button>
      <div id="_tsBbPanel"
        style="display:none;position:fixed;z-index:99999;
               background:#0a0f1e;border:0.5px solid rgba(71,85,105,0.5);border-radius:6px;
               padding:10px 12px;width:min(560px,90vw);box-shadow:0 8px 24px rgba(0,0,0,0.6)">
        <div style="font-size:9px;font-weight:600;color:#475569;text-transform:uppercase;
                    letter-spacing:0.05em;margin-bottom:6px">Building-block directory</div>
        <div style="display:flex;gap:6px;align-items:center">
          <input id="_tsBbPathInput" class="path-input" spellcheck="false"
                 placeholder="/path/to/Building_Blocks"
                 style="flex:1;min-width:0"
                 onkeydown="if(event.key==='Enter'){event.preventDefault();_tsBbScan();}">
          <button id="_tsBbScanBtn" onclick="_tsBbScan()"
            style="flex-shrink:0;padding:4px 12px;font-size:10px;font-weight:600;color:#22d3ee;
                   background:rgba(8,145,178,0.12);border:0.5px solid rgba(8,145,178,0.4);
                   border-radius:4px;cursor:pointer">Scan</button>
        </div>
        <div id="_tsBbPanelMsg" style="margin-top:6px;font-size:9px;color:#475569;line-height:1.5">
          Every <code style="color:#64748b">.csv</code> under this directory is read and each
          building block is matched against every reaction's SMARTS — reporting how many are
          eligible as reagent&nbsp;1 and how many as reagent&nbsp;2.
        </div>
      </div>
      <div id="_tsRxnDropdown"
        style="display:none;position:fixed;z-index:99999;
               background:#0a0f1e;border:0.5px solid rgba(71,85,105,0.5);border-radius:6px;
               padding:6px 0;min-width:210px;box-shadow:0 8px 24px rgba(0,0,0,0.6)">
        ${rxnKeys.map(k => {
            const lbl    = labels[k] || k;
            const cached = warmupCached[k];
            return `<label style="display:flex;align-items:center;gap:8px;padding:6px 14px;cursor:pointer;font-size:11px;color:#64748b;transition:background 0.15s"
                          onmouseover="this.style.background='rgba(30,41,59,0.8)'" onmouseout="this.style.background=''"
                          onclick="event.stopPropagation()">
              <input type="checkbox" id="_rxnChk_${k}" value="${k}"
                     onchange="_tsRxnUpdateLabel()"
                     style="accent-color:#0891b2;cursor:pointer;width:13px;height:13px">
              <span style="flex:1">${lbl}</span>
              <span id="_wuCacheBadge_${k}" title="Warmup cached — will skip warmup"
                style="display:${cached?'inline':'none'};font-size:9px;color:#f59e0b;
                       background:rgba(245,158,11,0.12);padding:1px 5px;border-radius:3px;
                       border:0.5px solid rgba(245,158,11,0.3);white-space:nowrap">⚡ cached</span>
              <button id="_wuClearBtn_${k}" title="Clear warmup cache"
                onclick="event.preventDefault();event.stopPropagation();_tsWarmupClear('${k}')"
                style="display:${cached?'inline':'none'};font-size:9px;color:#475569;
                       background:none;border:none;cursor:pointer;padding:0 2px;opacity:0.6">✕</button>
            </label>`;
        }).join('')}
        <div style="border-top:0.5px solid rgba(30,41,59,0.8);margin:4px 0 2px"></div>
        <div style="display:flex;gap:6px;padding:4px 10px">
          <button onclick="_tsRxnSelectAll()" style="flex:1;padding:3px;font-size:9px;color:#22d3ee;background:rgba(8,145,178,0.1);border:0.5px solid rgba(8,145,178,0.3);border-radius:3px;cursor:pointer">All</button>
          <button onclick="_tsRxnDone()" style="flex:1;padding:3px;font-size:9px;color:#34d399;background:rgba(52,211,153,0.08);border:0.5px solid rgba(52,211,153,0.3);border-radius:3px;cursor:pointer;font-weight:600">Done</button>
        </div>
      </div>`;

    toolbarRow.parentNode.insertBefore(wrapper, toolbarRow.nextSibling);

    if (!document._tsRxnCloseListenerAdded) {
        document._tsRxnCloseListenerAdded = true;
        document.addEventListener('click', (e) => {
            const picker = document.getElementById('_tsRxnPicker');
            const dd     = document.getElementById('_tsRxnDropdown');
            if (!dd || dd.style.display === 'none') return;
            if (!(picker && picker.contains(e.target)) && !dd.contains(e.target)) dd.style.display = 'none';
        });
    }
}

function _tsRxnToggle(e) {
    e.stopPropagation();
    const dd = document.getElementById('_tsRxnDropdown');
    if (!dd) return;
    const isOpen = dd.style.display !== 'none' && dd.style.display !== '';
    if (isOpen) {
        dd.style.display = 'none';
    } else {
        if (dd.parentElement !== document.body) document.body.appendChild(dd);
        const btn = document.getElementById('_tsRxnBtn');
        if (btn) {
            const r = btn.getBoundingClientRect();
            dd.style.position = 'fixed';
            dd.style.top      = (r.bottom + 4) + 'px';
            dd.style.left     = r.left + 'px';
            dd.style.zIndex   = '99999';
        }
        dd.style.display = 'block';
    }
}

function _tsRxnUpdateLabel() {
    const palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    const shortLabels = { rxn101_amide:'Amide', rxn102_buchwald:'Buchwald', rxn108_sonogashira:'Sonogashira', rxn110_suzuki:'Suzuki', rxn113_sulfonamide:'Sulfonamide', rxn208_snar:'SnAr' };
    const checked = [...document.querySelectorAll('[id^="_rxnChk_"]:checked')];
    const names   = checked.map(el => shortLabels[el.value] || el.value);
    const lbl = document.getElementById('_tsRxnBtnLabel');
    if (lbl) lbl.textContent = names.length ? names.join(', ') : 'None selected';
    const btn = document.getElementById('_tsRxnBtn');
    if (btn) btn.style.color = names.length ? '#22d3ee' : '#94a3b8';
    _tsRxnRebuildTabs(checked, palette, shortLabels);
}

function _tsRxnRebuildTabs(checkedEls, palette, shortLabels) {
    const wrapper = document.getElementById('_tsRxnPicker');
    if (!wrapper) return;
    wrapper.style.flexWrap  = 'nowrap';
    wrapper.style.overflowX = 'auto';
    wrapper.style.overflowY = 'hidden';
    wrapper.querySelectorAll('[data-rxn-pretab]').forEach(el => el.remove());
    if (!checkedEls.length) return;
    checkedEls.forEach((el, i) => {
        const key      = el.value;
        const label    = shortLabels[key] || key;
        const color    = palette[i % palette.length];
        const isActive = i === (_ts._activeJobIdx ?? 0);
        const pill     = document.createElement('button');
        pill.dataset.rxnPretab = key;
        pill.dataset.rxnKey    = key;
        pill.id = `_tsRxnTab_${i}`;
        pill.style.cssText = [
            'display:inline-flex;align-items:center;gap:5px',
            'padding:3px 10px;border-radius:5px',
            'font-size:10px;font-weight:600;cursor:pointer;white-space:nowrap;flex-shrink:0',
            `border:0.5px solid ${color}55;background:${color}18`,
            `color:${isActive ? color : color + '99'}`,
            `border-bottom:${isActive ? '2px solid ' + color : '2px solid transparent'}`,
            'transition:all 0.15s',
        ].join(';');
        const dot = document.createElement('span');
        dot.dataset.rxnDot = '1';
        dot.style.cssText = `width:6px;height:6px;border-radius:50%;background:${color};display:inline-block;flex-shrink:0`;
        pill.appendChild(dot);
        pill.appendChild(document.createTextNode(label));
        pill.addEventListener('click', () => {
            if (_ts.jobs?.length > 1) {
                const ji = _ts.jobs.findIndex(j => j.key === key);
                if (ji >= 0) _tsSetActiveJob(ji);
            }
            wrapper.querySelectorAll('[data-rxn-pretab]').forEach((p, idx) => {
                const c = palette[idx % palette.length];
                const isMe = p.dataset.rxnKey === key;
                p.style.color        = isMe ? c : c + '99';
                p.style.borderBottom = isMe ? `2px solid ${c}` : '2px solid transparent';
            });
        });
        wrapper.appendChild(pill);
    });
}

function _tsRxnDone() {
    const dd = document.getElementById('_tsRxnDropdown');
    if (dd) dd.style.display = 'none';
}
function _tsRxnSelectAll() {
    document.querySelectorAll('[id^="_rxnChk_"]').forEach(el => el.checked = true);
    _tsRxnUpdateLabel();
}
function _tsRxnSelectNone() {
    document.querySelectorAll('[id^="_rxnChk_"]').forEach(el => el.checked = false);
    _tsRxnUpdateLabel();
}

// _tsReconnectActive → ts_run.js
// ══ Building-block database scan ═══════════════════════════════════════════
// Import-database button next to the reaction picker. Takes a directory, streams
// /vina_visualization/ts_scan_bb, and renders a per-file × per-reaction table of
// how many building blocks can serve as reagent 1 and reagent 2.
//
// The report lands in #tsBbReportWrap — the slot the Warmup pane's "Raw log
// stream" terminal used to occupy.

let _tsBbEs = null;               // live EventSource, so a second scan cancels the first
let _tsBbState = null;            // {reactions, nReactants, files: [], totals}

function _tsBbToggle(e) {
    if (e) e.stopPropagation();
    const panel = document.getElementById('_tsBbPanel');
    const btn   = document.getElementById('_tsBbImportBtn');
    if (!panel || !btn) return;
    if (panel.style.display === 'block') { panel.style.display = 'none'; return; }
    // Position under the button. position:fixed (not absolute) because the
    // picker row is inside an overflow:hidden flex column — an absolutely
    // positioned panel would be clipped by it, exactly as the reaction
    // dropdown above is.
    const r = btn.getBoundingClientRect();
    panel.style.top  = (r.bottom + 6) + 'px';
    panel.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 580)) + 'px';
    panel.style.display = 'block';
    const input = document.getElementById('_tsBbPathInput');
    if (input) {
        // Seed from the engine's reagent directory when we know it, so the common
        // case is one click. _tsLoadConfig stashes it; fall back to the OUTPUT DIR.
        if (_ts._bbDefaultDir) {
            input.placeholder = _ts._bbDefaultDir;
            input.title       = _ts._bbDefaultDir;
        }
        if (!input.value.trim()) {
            input.value = _ts._bbDefaultDir
                       || document.getElementById('tsOutputDirInput')?.value.trim()
                       || '';
        }
        input.focus();
        input.select();
    }
    if (!document._tsBbCloser) {
        document._tsBbCloser = true;
        document.addEventListener('click', ev => {
            const p = document.getElementById('_tsBbPanel');
            const b = document.getElementById('_tsBbImportBtn');
            if (!p || p.style.display !== 'block') return;
            if (!p.contains(ev.target) && !(b && b.contains(ev.target))) p.style.display = 'none';
        });
    }
}

function _tsBbReportOpen() {
    const wrap = document.getElementById('tsBbReportWrap');
    if (wrap) wrap.style.display = 'flex';
    _tsBbSyncRestoreBtn();
}

// `close` HIDES the report; it does not cancel the scan.
//
// A scan over a real library is minutes of work — closing the panel to see the
// warmup bars underneath must not throw that away, and there is no way to tell
// from the button which of the two it would do. Cancelling is a separate,
// explicitly-labelled action (`stop`, shown only while a scan is running), and
// the restore chip keeps a hidden scan visible by counting up in the picker row.
function _tsBbReportClose() {
    const wrap = document.getElementById('tsBbReportWrap');
    if (wrap) wrap.style.display = 'none';
    _tsBbSyncRestoreBtn();
}

function _tsBbReportRestore(e) {
    if (e) e.stopPropagation();
    if (!_tsBbState) return;
    _tsBbReportOpen();
    _tsBbRenderReport();
}

function _tsBbStop(e) {
    if (e) e.stopPropagation();
    if (_tsBbEs) { _tsBbEs.close(); _tsBbEs = null; }
    if (_tsBbState) _tsBbState.stopped = true;
    _tsBbStatus(`stopped · ${(_tsBbState?.files || []).length} file(s) scanned`, '#f59e0b');
    _tsBbSyncRestoreBtn();
}

function _tsBbScanning() {
    return !!_tsBbEs;
}

/** Keep the picker-row chip and the header's stop button in step with state. */
function _tsBbSyncRestoreBtn() {
    const wrap    = document.getElementById('tsBbReportWrap');
    const chip    = document.getElementById('_tsBbRestoreBtn');
    const label   = document.getElementById('_tsBbRestoreLabel');
    const stopBtn = document.getElementById('tsBbStopBtn');
    const hidden  = !wrap || wrap.style.display === 'none';
    const st      = _tsBbState;

    if (chip) {
        // Offer the chip only when there is something to restore.
        chip.style.display = (st && hidden) ? 'flex' : 'none';
        if (label && st) {
            label.textContent = _tsBbScanning()
                ? `${(st.files || []).length}/${st.nFiles || '?'}`
                : 'report';
            chip.title = _tsBbScanning()
                ? `Scan running in the background — ${(st.files || []).length} of ${st.nFiles || '?'} files. Click to watch.`
                : 'Reopen the building-block scan report';
        }
    }
    if (stopBtn) stopBtn.style.display = _tsBbScanning() ? 'inline' : 'none';
}

function _tsBbStatus(text, colour) {
    const el = document.getElementById('tsBbReportStatus');
    if (el) { el.textContent = text; el.style.color = colour || '#475569'; }
    _tsBbSyncRestoreBtn();
}

function _tsBbScan() {
    const input = document.getElementById('_tsBbPathInput');
    const path  = (input?.value || '').trim();
    if (!path) { _tsBbPanelMsg('Enter a directory path first.', '#f87171'); return; }

    if (_tsBbEs) { _tsBbEs.close(); _tsBbEs = null; }
    const panel = document.getElementById('_tsBbPanel');
    if (panel) panel.style.display = 'none';

    _tsBbState = { reactions: {}, nReactants: {}, files: [], totals: null,
                   root: path, scanId: null };
    _tsBbReportOpen();
    _tsBbStatus('scanning…', '#22d3ee');
    const rep = document.getElementById('tsBbReport');
    if (rep) rep.innerHTML = '<div style="color:#475569">Reading ' + _tsBbEsc(path) + ' …</div>';

    _tsBbEs = new EventSource('/vina_visualization/ts_scan_bb?path=' + encodeURIComponent(path));
    _tsBbSyncRestoreBtn();          // the stream now exists: reveal `stop`
    _tsBbEs.onmessage = ev => {
        if (ev.data === '__DONE__') {
            if (_tsBbEs) { _tsBbEs.close(); _tsBbEs = null; }
            if (_tsBbState && !_tsBbState.totals && !_tsBbState.errored) {
                _tsBbStatus('ended without a result', '#f59e0b');
            }
            // Clear `stop` and switch the chip from a progress count back to
            // "report" — _tsBbEs is null only now, so the earlier `done`
            // handler could not have done it.
            _tsBbSyncRestoreBtn();
            return;
        }
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_) { return; }
        _tsBbHandle(msg);
    };
    // A stream that dies without __DONE__ (server restart, proxy timeout) must
    // not leave the panel saying "scanning…" forever.
    _tsBbEs.onerror = () => {
        if (_tsBbEs) { _tsBbEs.close(); _tsBbEs = null; }
        if (_tsBbState && !_tsBbState.totals) _tsBbStatus('connection lost', '#f87171');
        _tsBbSyncRestoreBtn();
    };
}

function _tsBbPanelMsg(text, colour) {
    const el = document.getElementById('_tsBbPanelMsg');
    if (el) { el.textContent = text; el.style.color = colour || '#475569'; }
}

function _tsBbHandle(msg) {
    const st = _tsBbState;
    if (!st) return;
    if (msg.type === 'error') {
        st.errored = true;
        _tsBbStatus('failed', '#f87171');
        const rep = document.getElementById('tsBbReport');
        if (rep) rep.innerHTML = '<div style="color:#f87171">⚠ ' + _tsBbEsc(msg.message || 'Scan failed.') + '</div>';
        return;
    }
    if (msg.type === 'warning') {
        (st.warnings = st.warnings || []).push(msg.message);
    } else if (msg.type === 'scan_id') {
        st.scanId = msg.scan_id;
    } else if (msg.type === 'start') {
        st.reactions  = msg.reactions || {};
        st.nReactants = msg.n_reactants || {};
        st.nFiles     = msg.n_files;
        st.capped     = msg.files_capped;
        st.dedup      = msg.dedup !== false;
    } else if (msg.type === 'file') {
        msg._i = st.files.length;
        st.files.push(msg);
        _tsBbStatus(`${st.files.length}/${st.nFiles || '?'} files`, '#22d3ee');
    } else if (msg.type === 'file_error') {
        st.files.push({ file: msg.file, error: msg.message, counts: {} });
    } else if (msg.type === 'done') {
        st.totals       = msg.totals;
        st.seconds      = msg.seconds;
        st.parsedTotal  = msg.parsed_total;
        st.dupTotal     = msg.duplicate_total;
        st.uniqueTotal  = msg.unique_total;
        st.dedupCapped  = msg.dedup_capped;
        st.maxUnique    = msg.max_unique;
        _tsBbStatus(`${msg.n_files} file(s) · ${msg.seconds}s`, '#34d399');
    }
    _tsBbRenderReport();
}

function _tsBbEsc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _tsBbRenderReport() {
    const el = document.getElementById('tsBbReport');
    const st = _tsBbState;
    if (!el || !st) return;

    const keys = Object.keys(st.reactions || {});
    if (!keys.length) return;

    const cell = 'padding:3px 8px;white-space:nowrap;';
    const head = 'padding:4px 8px;position:sticky;top:0;background:#05070f;color:#475569;' +
                 'font-weight:600;text-transform:uppercase;letter-spacing:0.04em;font-size:9px;';

    // One column pair per reaction: eligible as reagent 1 / reagent 2.
    let h = '<table style="border-collapse:collapse;font-family:ui-monospace,monospace;font-size:10px;width:100%">';
    h += '<thead><tr>';
    h += `<th style="${head}text-align:left">file</th>`;
    h += `<th style="${head}text-align:right">rows</th>`;
    if (st.dedup !== false) h += `<th style="${head}text-align:right" title="blocks repeating a molecule already counted — in an earlier file, or earlier in this same file">dup</th>`;
    keys.forEach(k => {
        const n = st.nReactants[k] || 2;
        h += `<th colspan="${n}" style="${head}text-align:center;border-left:0.5px solid rgba(30,41,59,0.9)">`
           + _tsBbEsc(st.reactions[k]) + '</th>';
    });
    h += '</tr><tr>';
    h += `<th style="${head}"></th><th style="${head}"></th>`;
    if (st.dedup !== false) h += `<th style="${head}"></th>`;
    keys.forEach(k => {
        const n = st.nReactants[k] || 2;
        for (let i = 0; i < n; i++) {
            h += `<th style="${head}text-align:right;${i === 0 ? 'border-left:0.5px solid rgba(30,41,59,0.9)' : ''}">R${i + 1}</th>`;
        }
    });
    h += '</tr></thead><tbody>';

    st.files.forEach(f => {
        h += '<tr style="border-bottom:0.5px solid rgba(148,163,184,0.06)">';
        if (f.error) {
            const span = 2 + (st.dedup !== false ? 1 : 0)
                       + keys.reduce((a, k) => a + (st.nReactants[k] || 2), 0);
            h += `<td style="${cell}color:#f87171" colspan="${span}">`
               + _tsBbEsc(f.file) + ' — ' + _tsBbEsc(f.error) + '</td></tr>';
            return;
        }
        const warn = f.unparseable ? ` title="${f.unparseable} SMILES did not parse"` : '';
        h += `<td style="${cell}color:#cbd5e1"${warn}>` + _tsBbEsc(f.file)
           + (f.unparseable ? ' <span style="color:#f59e0b">!</span>' : '') + '</td>';
        h += `<td style="${cell}text-align:right;color:#64748b">` + (f.parsed ?? 0)
           + (f.truncated ? '<span style="color:#f59e0b" title="row cap reached">+</span>' : '') + '</td>';
        if (st.dedup !== false) {
            const d = f.duplicates ?? 0;
            const n = (f.dup_details || []).length;
            const tip = `${d} of this file's ${f.parsed ?? 0} blocks repeat a molecule already counted`
                      + ` — in an earlier file, or earlier in this same file. `
                      + `${(f.parsed ?? 0) - d} were new.`
                      + (n ? `  Click to see ${n} of them.` : '');
            if (d && n) {
                // Only clickable when examples actually came back, so a live
                // cursor never leads to an empty panel.
                h += `<td style="${cell}text-align:right;padding:0">`
                   + `<button onclick="_tsBbShowDupes(${f._i})" title="${tip}"`
                   + ` style="background:none;border:none;padding:3px 8px;cursor:pointer;`
                   + `font:inherit;color:#f59e0b;text-decoration:underline;`
                   + `text-underline-offset:2px;text-decoration-style:dotted">${d}</button></td>`;
            } else {
                h += `<td style="${cell}text-align:right;color:${d ? '#f59e0b' : '#334155'}" title="${tip}">${d}</td>`;
            }
        }
        keys.forEach(k => {
            const cs = (f.counts || {})[k] || [];
            const n  = st.nReactants[k] || 2;
            for (let i = 0; i < n; i++) {
                const v = cs[i] ?? 0;
                // Zero is the answer that matters most here — a file with no
                // eligible blocks for a reaction cannot serve as its reagent
                // source — so it is de-emphasised, not hidden.
                const colour = v > 0 ? '#22d3ee' : '#334155';
                h += `<td style="${cell}text-align:right;color:${colour};${i === 0 ? 'border-left:0.5px solid rgba(30,41,59,0.9)' : ''}">${v}</td>`;
            }
        });
        h += '</tr>';
    });

    if (st.totals) {
        h += '<tr style="border-top:0.5px solid rgba(71,85,105,0.6)">';
        h += `<td style="${cell}color:#94a3b8;font-weight:600">total</td>`;
        h += `<td style="${cell}text-align:right;color:#64748b">`
           + st.files.reduce((a, f) => a + (f.parsed || 0), 0) + '</td>';
        if (st.dedup !== false) {
            h += `<td style="${cell}text-align:right;color:#f59e0b;font-weight:600">`
               + (st.dupTotal ?? st.files.reduce((a, f) => a + (f.duplicates || 0), 0)) + '</td>';
        }
        keys.forEach(k => {
            const cs = st.totals[k] || [];
            const n  = st.nReactants[k] || 2;
            for (let i = 0; i < n; i++) {
                h += `<td style="${cell}text-align:right;color:#67e8f9;font-weight:600;${i === 0 ? 'border-left:0.5px solid rgba(30,41,59,0.9)' : ''}">${cs[i] ?? 0}</td>`;
            }
        });
        h += '</tr>';

        // ── Duplicate / unique summary ────────────────────────────────────
        // Two rows below the totals, because the totals row answers "how many
        // eligible blocks did I count" and these answer "how many DISTINCT
        // blocks do I actually have" — the reagent-pool size that matters when
        // several files are combined. A block in three files is one unique and
        // two duplicates.
        if (st.dedup !== false && st.uniqueTotal != null) {
            const nCols = 2 + 1 + keys.reduce((a, k) => a + (st.nReactants[k] || 2), 0);
            const parsed = st.parsedTotal ?? st.files.reduce((a, f) => a + (f.parsed || 0), 0);
            const dup    = st.dupTotal ?? 0;
            const pct    = parsed ? ((dup / parsed) * 100).toFixed(1) : '0.0';

            h += '<tr>';
            h += `<td style="${cell}color:#f59e0b;font-weight:600">duplicates</td>`;
            h += `<td style="${cell}text-align:right;color:#f59e0b;font-weight:600">${dup}</td>`;
            h += `<td style="${cell}color:#475569" colspan="${nCols - 2}">`
               + `repeated building blocks across all files — ${pct}% of parsed rows`
               + '</td></tr>';

            h += '<tr style="border-top:0.5px solid rgba(103,232,249,0.35)">';
            h += `<td style="${cell}color:#67e8f9;font-weight:700">unique blocks</td>`;
            h += `<td style="${cell}text-align:right;color:#67e8f9;font-weight:700">${st.uniqueTotal}</td>`;
            h += `<td style="${cell}color:#475569" colspan="${nCols - 2}">`
               + 'distinct molecules in the whole library (canonical SMILES)'
               + (st.dedupCapped
                    ? ` <span style="color:#f59e0b">— cap of ${st.maxUnique} reached, this is a LOWER BOUND</span>`
                    : '')
               + '</td></tr>';
        }
    }
    h += '</tbody></table>';

    h += `<div style="margin-top:8px;color:#334155;font-size:9px;line-height:1.6">`
       + `R1 / R2 = building blocks eligible as that reaction's first / second reagent `
       + `(substructure match against the reactant templates of its SMARTS). `
       + `Duplicates are matched on CANONICAL SMILES, so the same molecule written `
       + `two ways counts once. `
       + `Scanned <code style="color:#475569">${_tsBbEsc(st.root)}</code>.`;
    if (st.capped) h += ` <span style="color:#f59e0b">File cap reached — not every CSV was read.</span>`;
    (st.warnings || []).forEach(w => {
        h += `<div style="color:#f59e0b;margin-top:3px">⚠ ${_tsBbEsc(w)}</div>`;
    });
    h += '</div>';

    el.innerHTML = h;
}

window._tsBbToggle        = _tsBbToggle;
window._tsBbScan          = _tsBbScan;
window._tsBbReportClose   = _tsBbReportClose;
window._tsBbReportRestore = _tsBbReportRestore;
window._tsBbStop          = _tsBbStop;

// ══ Duplicate detail cards, paged ══════════════════════════════════════════
// Clicking a file's `dup` count renders its duplicate rows into #tsWuBars —
// the Warmup pane's bars area, which is empty until a run starts.
//
// Paged against /ts_scan_dupes rather than the inline `dup_details` the SSE
// stream carries: a single file can hold 10k+ duplicates, so the records live
// on disk and the browser reads a slice. The inline copy is the fallback for
// when the scan directory has been pruned.
//
// Structures come from /ts_smiles_svg (keyed on the SMILES itself), not
// /ts_mol_svg/<rxn_key>/<id>: a scanned directory need not be the engine's
// reagent index, so there is no rxn_key to resolve an id against.

const _TS_BB_PAGE_SIZES = [5, 10, 25, 50, 100];
let _tsBbPage = null;   // {fileIdx, offset, limit, total, rows, err}

function _tsBbShowDupes(fileIdx, offset, limit) {
    const st = _tsBbState;
    const f  = st && st.files && st.files[fileIdx];
    if (!f) return;
    const lim = limit || (_tsBbPage && _tsBbPage.limit) || _TS_BB_PAGE_SIZES[0];
    const off = Math.max(0, offset || 0);
    _tsBbPage = { fileIdx, offset: off, limit: lim,
                  total: f.duplicates || 0, rows: null, err: null };
    _tsBbRenderDupes();     // paint the frame immediately, then fill it

    if (!st.scanId) { _tsBbUseInline(f, 'no scan id — showing the inline sample'); return; }
    fetch(`/vina_visualization/ts_scan_dupes?scan=${encodeURIComponent(st.scanId)}`
          + `&file=${fileIdx}&offset=${off}&limit=${lim}`)
        .then(r => r.json().then(d => ({ ok: r.ok, d })))
        .then(({ ok, d }) => {
            if (!_tsBbPage || _tsBbPage.fileIdx !== fileIdx || _tsBbPage.offset !== off) return;
            if (!ok || d.error) { _tsBbUseInline(f, d.error || 'could not read the records'); return; }
            _tsBbPage.rows  = d.rows || [];
            _tsBbPage.total = d.total ?? _tsBbPage.total;
            _tsBbRenderDupes();
        })
        .catch(e => _tsBbUseInline(f, String(e)));
}

/** Fall back to the sample that rode the SSE stream. */
function _tsBbUseInline(f, why) {
    if (!_tsBbPage) return;
    const inline = f.dup_details || [];
    _tsBbPage.rows   = inline.slice(_tsBbPage.offset, _tsBbPage.offset + _tsBbPage.limit);
    _tsBbPage.total  = inline.length;
    _tsBbPage.err    = why;
    _tsBbPage.inline = true;
    _tsBbRenderDupes();
}

function _tsBbSetPageSize(n) {
    if (!_tsBbPage) return;
    _tsBbShowDupes(_tsBbPage.fileIdx, 0, parseInt(n) || _TS_BB_PAGE_SIZES[0]);
}

function _tsBbGoPage(p) {
    if (!_tsBbPage) return;
    _tsBbShowDupes(_tsBbPage.fileIdx, (p - 1) * _tsBbPage.limit, _tsBbPage.limit);
}

/** Page numbers to render: a window around the current page, always with 1. */
function _tsBbPageWindow(cur, last, span = 10) {
    if (last <= span) return Array.from({ length: last }, (_, i) => i + 1);
    let from = Math.max(1, cur - Math.floor(span / 2));
    let to   = from + span - 1;
    if (to > last) { to = last; from = last - span + 1; }
    return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

function _tsBbRenderDupes() {
    const el = document.getElementById('tsWuBars');
    const st = _tsBbState;
    const pg = _tsBbPage;
    if (!el || !st || !pg) return;
    const f = st.files[pg.fileIdx] || {};
    const esc = _tsBbEsc;
    const svg = (smi, w, h) =>
        `/vina_visualization/ts_smiles_svg?smi=${encodeURIComponent(smi)}&w=${w}&h=${h}`;

    const total = pg.total || 0;
    const last  = Math.max(1, Math.ceil(total / pg.limit));
    const cur   = Math.floor(pg.offset / pg.limit) + 1;
    const from  = total ? pg.offset + 1 : 0;
    const to    = Math.min(pg.offset + pg.limit, total);

    let h = `<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px;
                         padding-bottom:6px;border-bottom:0.5px solid rgba(148,163,184,0.12)">
        <span style="font-size:11px;font-weight:600;color:#f59e0b">Duplicates in ${esc(f.file || '')}</span>
        <span style="font-size:10px;color:#475569">${from}–${to} of ${total}</span>
        <label style="margin-left:auto;font-size:9px;color:#475569;display:flex;align-items:center;gap:4px">
          rows
          <select onchange="_tsBbSetPageSize(this.value)"
                  style="background:#0b1120;border:0.5px solid #1e3a5f;border-radius:4px;
                         color:#94a3b8;font-size:9px;font-family:inherit;padding:1px 3px;cursor:pointer">
            ${_TS_BB_PAGE_SIZES.map(n =>
                `<option value="${n}"${n === pg.limit ? ' selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
        <button onclick="document.getElementById('tsWuBars').innerHTML=''"
                style="background:none;border:none;cursor:pointer;font-size:9px;color:#475569">clear</button>
      </div>`;

    if (pg.err) {
        h += `<div style="font-size:9px;color:#f59e0b;margin-bottom:8px">⚠ ${esc(pg.err)}</div>`;
    }
    if (f.dup_truncated) {
        h += `<div style="font-size:9px;color:#475569;margin-bottom:8px">`
           + `Records on disk are capped per file; the count in the table is exact.</div>`;
    }

    if (pg.rows === null) {
        h += `<div style="font-size:10px;color:#475569;padding:12px 0">Loading…</div>`;
    } else if (!pg.rows.length) {
        h += `<div style="font-size:10px;color:#475569;padding:12px 0">No records on this page.</div>`;
    } else {
        h += pg.rows.map(r => `
      <div style="padding:6px 0;border-bottom:0.5px solid rgba(148,163,184,0.07)">
        <div style="display:grid;grid-template-columns:104px 1fr auto;align-items:start;gap:10px">
          <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
            <object type="image/svg+xml" data="${svg(r.smiles, 96, 48)}" width="96" height="48"
                    style="pointer-events:none;display:block;margin:0 auto"
                    aria-label="structure of ${esc(r.id || r.row)}"></object>
            <span style="font-family:monospace;font-size:10px;color:#94a3b8;overflow:hidden;
                         text-overflow:ellipsis;white-space:nowrap;max-width:100px"
                  title="${esc(r.id || '(no id column)')}">${esc(r.id || '—')}</span>
          </div>
          <div style="display:flex;flex-direction:column;gap:3px;min-width:0">
            <code style="font-family:monospace;font-size:11px;color:#67e8f9;
                         background:rgba(8,145,178,0.08);border:0.5px solid rgba(8,145,178,0.18);
                         border-radius:4px;padding:3px 7px;word-break:break-all;line-height:1.4;
                         user-select:all;cursor:text">${esc(r.smiles)}</code>
            ${r.raw ? `<code title="how this file writes it — different text, same molecule"
                 style="font-family:monospace;font-size:10px;color:#a78bfa;
                        background:rgba(124,58,237,0.08);border:0.5px solid rgba(124,58,237,0.2);
                        border-radius:4px;padding:2px 6px;word-break:break-all;user-select:all;
                        cursor:text">as written: ${esc(r.raw)}</code>` : ''}
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
              <span style="font-size:9px;padding:1px 5px;border-radius:3px;
                           background:rgba(8,145,178,0.15);color:#67e8f9">${esc(f.file || '')}</span>
              <span style="font-size:9px;color:#94a3b8">row ${r.row}</span>
              <span style="font-size:9px;color:#64748b" title="duplicate of">⊕ duplicate of</span>
              <span style="font-size:9px;padding:1px 5px;border-radius:3px;
                           background:rgba(245,158,11,0.12);color:#f59e0b">${esc(r.of_file)}</span>
              <span style="font-size:9px;color:#94a3b8">row ${r.of_row}</span>
              ${r.of_id ? `<span style="font-family:monospace;font-size:9px;color:#0f766e">${esc(r.of_id)}</span>` : ''}
            </div>
          </div>
          <span title="${r.raw ? 'Different SMILES text, same molecule — a text-level dedup would have missed this row'
                              : 'Identical SMILES text'}"
                style="font-size:10px;font-weight:600;white-space:nowrap;
                       color:${r.raw ? '#a78bfa' : '#475569'}">${r.raw ? 'rewritten' : 'exact'}</span>
        </div>
      </div>`).join('');
    }

    // ── Pager ──────────────────────────────────────────────────────────────
    if (last > 1) {
        const link = (label, page, on) => on
            ? `<button onclick="_tsBbGoPage(${page})"
                       style="background:none;border:none;cursor:pointer;font-family:inherit;
                              font-size:11px;padding:2px 5px;color:#22d3ee;text-decoration:underline;
                              text-underline-offset:3px">${label}</button>`
            : `<span style="font-size:11px;padding:2px 5px;color:#475569">${label}</span>`;
        h += `<div style="display:flex;align-items:center;justify-content:center;gap:2px;
                          flex-wrap:wrap;padding:10px 0 4px">`;
        h += link('‹ Prev', cur - 1, cur > 1);
        _tsBbPageWindow(cur, last).forEach(pnum => {
            h += pnum === cur
                ? `<span style="font-size:11px;padding:2px 6px;color:#f59e0b;font-weight:700">${pnum}</span>`
                : link(String(pnum), pnum, true);
        });
        h += link('Next ›', cur + 1, cur < last);
        h += `<span style="font-size:9px;color:#334155;margin-left:8px">page ${cur} of ${last}</span>`;
        h += '</div>';
    }

    el.innerHTML = h;
    el.scrollTop = 0;
}

window._tsBbShowDupes   = _tsBbShowDupes;
window._tsBbSetPageSize = _tsBbSetPageSize;
window._tsBbGoPage      = _tsBbGoPage;