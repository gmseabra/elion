// =============================================================================
// ts/ts_ui.js — DOM patches, bar renderers, log panel, CPU monitor, system tab
// Depends on: ts_core.js, ts_chart.js
// =============================================================================

// ── Sidebar patch ─────────────────────────────────────────────────────────
function _tsPatchSidebar() {
    // Guard on the stable container, NOT tsTsPool0: in multi-job runs the
    // static pool div (which holds tsTsPool0) gets replaced by
    // _tsMultiPoolContainer, so tsTsPool0 disappears. Keying the early-out on
    // tsTsPool0 would then let this re-inject the static "Competition pool"
    // block on reopen, duplicating the per-job sections. _tsSidebarExtra
    // persists for the lifetime of the patched sidebar, so it's the right guard.
    if (document.getElementById('_tsSidebarExtra')) return;

    const iterEl = document.getElementById('tsTsIter');
    if (!iterEl) return;
    const sectionDiv = iterEl.closest('div.px-4');
    if (!sectionDiv) return;

    const rowsEl = sectionDiv.querySelector('div.space-y-2');
    if (rowsEl) {
        rowsEl.innerHTML = `
          <div class="flex justify-between"><span class="text-slate-500">Current iter</span><span id="tsTsIter" class="text-cyan-400 font-semibold">—</span></div>
          <div class="flex justify-between"><span class="text-slate-500">Last score</span><span id="tsTsScore" class="text-emerald-400 font-semibold">—</span></div>
          <div class="flex justify-between">
            <span class="text-slate-500" title="Reagents permanently retired by disallow_mask.">Masked reagents</span>
            <span id="tsTsMasked" class="text-slate-500">—</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-500" title="Search speed: seconds per iteration (rolling average over recent iterations). Lower is faster; it/s shown alongside.">Speed</span>
            <span id="tsTsSpeed" class="text-amber-400 font-semibold">—</span>
          </div>`;
    }

    const competitionHTML = `
      <div id="_tsSidebarExtra" class="flex flex-col flex-1 min-h-0 overflow-y-auto ts-minilog-scroll">
        <div class="px-4 pt-3 pb-3 border-b border-slate-800 flex-shrink-0">
          <p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2"
             title="Each TS iteration draws one winner per reaction slot from the eligible reagent pool.">
            Competition pool
          </p>
          <div class="space-y-1.5 text-[11px]">
            <div class="flex justify-between items-center">
              <span id="tsTsPool0Label" class="text-slate-600 text-[10px]">BB · slot 0</span>
              <span id="tsTsPool0" class="text-cyan-300 font-semibold">—</span>
            </div>
            <div class="flex justify-between text-[10px]">
              <span class="text-slate-600">μ</span>
              <span id="tsTsPool0MuRange" class="text-emerald-400 font-mono">—</span>
            </div>
            <div class="flex justify-between text-[10px] mb-1">
              <span class="text-slate-600">σ</span>
              <span id="tsTsPool0StdRange" class="text-amber-400 font-mono">—</span>
            </div>
            <div class="flex justify-between items-center border-t border-slate-800/60 pt-1.5">
              <span id="tsTsPool1Label" class="text-slate-600 text-[10px]">BB · slot 1</span>
              <span id="tsTsPool1" class="text-cyan-300 font-semibold">—</span>
            </div>
            <div class="flex justify-between text-[10px]">
              <span class="text-slate-600">μ</span>
              <span id="tsTsPool1MuRange" class="text-emerald-400 font-mono">—</span>
            </div>
            <div class="flex justify-between text-[10px]">
              <span class="text-slate-600">σ</span>
              <span id="tsTsPool1StdRange" class="text-amber-400 font-mono">—</span>
            </div>
          </div>
        </div>
      </div>`;

    sectionDiv.insertAdjacentHTML('afterend', competitionHTML);

    const allPx4 = document.querySelectorAll('.px-4.pt-3.pb-1');
    allPx4.forEach(el => {
        const p = el.querySelector('p');
        if (p && p.textContent.trim().toLowerCase().startsWith('best mol')) el.remove();
    });
    const winnersEl = document.getElementById('tsTsWinners');
    if (winnersEl) winnersEl.remove();
    document.querySelectorAll('p, h3').forEach(el => {
        if (el.textContent.trim().toLowerCase() === 'best molecules') el.closest('div')?.remove();
    });
}

// ── Warmup sidebar patch ──────────────────────────────────────────────────
function _tsPatchWarmupSidebar() {
    document.querySelectorAll('p, span').forEach(el => {
        if (el.textContent.trim().toLowerCase().startsWith('config ·')) {
            el.closest('div.px-4, div[class*="px-4"]')?.remove();
        }
    });
    if (!document.getElementById('tsWuProgressList')) {
        const wuPhaseEl = document.getElementById('tsWuPhase');
        if (!wuPhaseEl) return;
        const sectionDiv = wuPhaseEl.closest('div.px-4, div[class*="px-4"]');
        if (!sectionDiv) return;
        const list = document.createElement('div');
        list.id = 'tsWuProgressList';
        list.style.cssText = 'flex:1;overflow-y:auto;';
        list.className = 'divide-y divide-slate-800';
        const existingRows = sectionDiv.querySelector('div.space-y-2');
        const defaultPanel = document.createElement('div');
        defaultPanel.id = 'tsWuPanel_default';
        defaultPanel.style.cssText = 'padding:10px 16px 12px;';
        if (existingRows) { defaultPanel.appendChild(existingRows.cloneNode(true)); existingRows.remove(); }
        list.appendChild(defaultPanel);
        sectionDiv.innerHTML = '';
        sectionDiv.appendChild(list);
    }
}

// ── System tab patch ──────────────────────────────────────────────────────
function _tsPatchSystemTab() {
    const tabBtn = document.getElementById('tsTabResults');
    if (tabBtn && !tabBtn.dataset.sysPatched) {
        tabBtn.textContent = '📡 Monitor';
        tabBtn.dataset.sysPatched = '1';
    }
    if (!window._tsTabOriginal) {
        window._tsTabOriginal = window._tsTab;
        window._tsTab = function(tab) {
            const rxnPicker = document.getElementById('_tsRxnPicker');
            if (rxnPicker) rxnPicker.style.display = (tab === 'results') ? 'none' : 'flex';
            if (window._tsTabOriginal) window._tsTabOriginal(tab);
        };
    }
    if (document.getElementById('tsSysSubTabs')) return;
    const pane = document.getElementById('tsPaneResults');
    if (!pane) return;

    const TH  = 'text-align:left;padding:8px 14px 6px 0;font-size:9px;color:#334155;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;border-bottom:0.5px solid rgba(30,41,59,0.9)';
    const THR = TH.replace('text-align:left','text-align:right');

    pane.innerHTML = `
      <div class="flex-1 relative flex flex-col overflow-hidden" style="background:#05070f">
        <div id="tsSysSubTabs" style="display:flex;align-items:center;border-bottom:0.5px solid rgba(30,41,59,0.9);flex-shrink:0;padding:0 20px">
          <button id="tsMonSubSys" onclick="_tsMonSubTab('sys')"
            style="padding:8px 16px 7px;font-size:11px;font-weight:600;color:#22d3ee;border-bottom:2px solid #22d3ee;background:none;border-top:none;border-left:none;border-right:none;cursor:pointer;letter-spacing:0.04em">cpu</button>
          <button id="tsMonSubGpu" onclick="_tsMonSubTab('gpu')"
            style="padding:8px 16px 7px;font-size:11px;font-weight:600;color:#475569;border-bottom:2px solid transparent;background:none;border-top:none;border-left:none;border-right:none;cursor:pointer;letter-spacing:0.04em">gpu</button>
          <button id="tsMonSubMeta" onclick="_tsMonSubTab('meta')"
            style="padding:8px 16px 7px;font-size:11px;font-weight:600;color:#475569;border-bottom:2px solid transparent;background:none;border-top:none;border-left:none;border-right:none;cursor:pointer;letter-spacing:0.04em">meta</button>
          <div id="tsMonSysStats" style="margin-left:auto;display:flex;gap:16px;align-items:center;font-size:10px;font-family:monospace">
            <span style="color:#334155">load</span><span id="tsSysLoadAvg" style="color:#94a3b8">—</span>
            <span style="color:#334155">CPU</span><span id="tsSysCpuTotal" style="color:#22d3ee;font-weight:600">—%</span>
            <span style="color:#334155">MEM</span><span id="tsSysMemUsed" style="color:#c084fc;font-weight:600">—</span>
            <span style="color:#334155">/</span><span id="tsSysMemTotal" style="color:#475569">—</span>
          </div>
          <div id="tsMonMetaCtrl" style="margin-left:auto;display:none;align-items:center;gap:10px">
            <span id="tsMetaDirLabel" style="font-size:9px;color:#1e3a5f;font-family:monospace;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
            <button onclick="_tsMetaRefresh()" style="padding:3px 10px;font-size:10px;color:#22d3ee;background:rgba(8,145,178,0.1);border:0.5px solid rgba(8,145,178,0.3);border-radius:4px;cursor:pointer">↻ Refresh</button>
          </div>
          <div id="tsMonGpuStats" style="margin-left:auto;display:none;gap:16px;align-items:center;font-size:10px;font-family:monospace">
            <span style="color:#334155">GPU</span><span id="tsGpuUtil" style="color:#22d3ee;font-weight:600">—%</span>
            <span style="color:#334155">MEM</span><span id="tsGpuMemUsed" style="color:#c084fc;font-weight:600">—</span>
            <span style="color:#334155">/</span><span id="tsGpuMemTotal" style="color:#475569">—</span>
            <span style="color:#334155">TEMP</span><span id="tsGpuTemp" style="color:#34d399;font-weight:600">—</span>
            <span style="color:#334155">PWR</span><span id="tsGpuPower" style="color:#f59e0b;font-weight:600">—</span>
          </div>
        </div>
        <div id="tsMonPaneSys" style="flex:1;overflow-y:auto;padding:0 20px 12px;display:flex;flex-direction:column">
          <table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:11px">
            <thead><tr style="position:sticky;top:0;background:#05070f;z-index:1">
              <th style="${TH}">PID</th><th style="${TH}">Job</th><th style="${TH}">Command</th>
              <th style="${THR}">%CPU</th><th style="${THR}">%MEM</th><th style="${THR}">VIRT</th>
              <th style="${THR}">RES</th><th style="${THR}">TIME+</th>
              <th style="${THR}" title="Core last ran on">CPU#</th>
              <th style="${THR}" title="Pinned affinity">Affinity</th>
              <th style="text-align:center;padding:8px 0 6px;font-size:9px;color:#334155;font-weight:600;text-transform:uppercase;border-bottom:0.5px solid rgba(30,41,59,0.9)">S</th>
            </tr></thead>
            <tbody id="tsSysProcTable"></tbody>
          </table>
        </div>
        <div id="tsMonPaneMeta" style="flex:1;overflow-y:auto;padding:20px 24px 16px;display:none">
          <div id="tsMetaContent" style="display:flex;flex-direction:column;gap:16px">
            <div style="color:#1e3a5f;font-size:11px;font-family:monospace">Switch to meta tab and click ↻ Refresh to load results.</div>
          </div>
        </div>
        <div id="tsMonPaneGpu" style="flex:1;overflow-y:auto;padding:16px 20px 12px;display:none;flex-direction:column">
          <div id="tsGpuDevices" style="display:flex;flex-direction:column;gap:12px"></div>
          <div id="tsGpuProcWrap" style="margin-top:16px">
            <p style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px">GPU Processes</p>
            <table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:11px">
              <thead><tr style="position:sticky;top:0;background:#05070f;z-index:1">
                <th style="${TH}">PID</th><th style="${TH}">Job</th><th style="${TH}">Process</th>
                <th style="${THR}" title="Compute / Graphics">Type</th>
                <th style="${THR}">GPU Mem</th>
              </tr></thead>
              <tbody id="tsGpuProcTable"></tbody>
            </table>
          </div>
          <div id="tsGpuMsg" style="display:none;color:#64748b;font-size:11px;font-family:monospace;padding:24px 0;text-align:center"></div>
        </div>
      </div>
      <div style="flex-shrink:0;width:220px;border-left:0.5px solid rgba(51,65,85,0.6);background:#030509;display:flex;flex-direction:column;overflow:hidden">
        <div style="padding:14px 14px 10px;border-bottom:0.5px solid rgba(30,41,59,0.8);flex-shrink:0">
          <p style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">Run Summary</p>
          <div style="display:flex;flex-direction:column;gap:6px;font-size:11px">
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Status</span><span id="tsStatusBadge" style="color:#94a3b8">—</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Molecules</span><span id="tsMolCount" style="color:#c084fc;font-weight:600">—</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Best score</span><span id="tsBestScore" style="color:#34d399;font-weight:600">—</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Iterations</span><span id="tsIterDone" style="color:#94a3b8">—</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Elapsed</span><span id="tsElapsed" style="color:#94a3b8">—</span></div>
          </div>
        </div>
      </div>`;
}

function _tsMonSubTab(which) {
    const sysPan  = document.getElementById('tsMonPaneSys');
    const gpuPan  = document.getElementById('tsMonPaneGpu');
    const metaPan = document.getElementById('tsMonPaneMeta');
    const sysBtn  = document.getElementById('tsMonSubSys');
    const gpuBtn  = document.getElementById('tsMonSubGpu');
    const metaBtn = document.getElementById('tsMonSubMeta');
    const sysStats  = document.getElementById('tsMonSysStats');
    const gpuStats  = document.getElementById('tsMonGpuStats');
    const metaCtrl  = document.getElementById('tsMonMetaCtrl');

    const ACTIVE = '#22d3ee', IDLE = '#475569';
    const on  = (el, disp) => { if (el) el.style.display = disp; };
    const lit = (btn, active) => {
        if (!btn) return;
        btn.style.color = active ? ACTIVE : IDLE;
        btn.style.borderBottom = active ? `2px solid ${ACTIVE}` : '2px solid transparent';
    };

    // Hide all panes + header strips, de-activate all buttons.
    on(sysPan, 'none'); on(gpuPan, 'none'); on(metaPan, 'none');
    on(sysStats, 'none'); on(gpuStats, 'none'); on(metaCtrl, 'none');
    lit(sysBtn, false); lit(gpuBtn, false); lit(metaBtn, false);

    if (which === 'gpu') {
        on(gpuPan, 'flex'); on(gpuStats, 'flex'); lit(gpuBtn, true);
    } else if (which === 'meta') {
        on(metaPan, 'flex'); on(metaCtrl, 'flex'); lit(metaBtn, true);
        _tsMetaRefresh();
    } else { // 'sys' (the cpu tab) — default
        on(sysPan, 'flex'); on(sysStats, 'flex'); lit(sysBtn, true);
    }
}

function _tsMonSub(tab) {
    const isSys = tab === 'sys';
    const sys  = document.getElementById('tsMonPaneSys');
    const meta = document.getElementById('tsMonPaneMeta');
    const btnS = document.getElementById('tsMonSubSys');
    const btnM = document.getElementById('tsMonSubMeta');
    const stats   = document.getElementById('tsMonSysStats');
    const refresh = document.getElementById('tsMonMetaRefresh');
    if (sys)     sys.style.display     = isSys ? 'flex'  : 'none';
    if (meta)    meta.style.display    = isSys ? 'none'  : 'block';
    if (btnS)    { btnS.style.color    = isSys ? '#22d3ee' : '#475569'; btnS.style.borderBottom = isSys ? '2px solid #22d3ee' : '2px solid transparent'; }
    if (btnM)    { btnM.style.color    = isSys ? '#475569' : '#22d3ee'; btnM.style.borderBottom = isSys ? '2px solid transparent' : '2px solid #22d3ee'; }
    if (stats)   stats.style.display   = isSys ? 'flex'  : 'none';
    if (refresh) refresh.style.display = isSys ? 'none'  : 'inline-block';
    if (!isSys) _tsMetaLoad();
}

// ── Meta results tab ──────────────────────────────────────────────────────
function _tsMetaRefresh() {
    const outputDir = document.getElementById('tsOutputDirInput')?.value.trim()
                   || document.getElementById('tsSmartsInput')?.value.trim() || '';
    const content = document.getElementById('tsMetaContent');
    const dirLbl  = document.getElementById('tsMetaDirLabel');
    if (dirLbl) dirLbl.textContent = outputDir || '(no output dir set)';
    if (!content) return;
    if (!outputDir) {
        content.innerHTML = `<div style="color:#334155;font-size:11px;font-family:monospace">No output dir set.</div>`;
        return;
    }
    content.innerHTML = `<div style="color:#334155;font-size:11px;font-family:monospace">Loading…</div>`;
    fetch(`/vina_visualization/ts_meta?dir=${encodeURIComponent(outputDir)}`)
        .then(r => r.json())
        .then(d => _tsMetaRender(d, outputDir))
        .catch(e => { content.innerHTML = `<div style="color:#ef4444;font-size:11px;font-family:monospace">Error: ${e.message}</div>`; });
}

function _tsMetaRender(data, outputDir) {
    const content = document.getElementById('tsMetaContent');
    if (!content) return;
    const files = data.files || [];
    if (files.length === 0) {
        content.innerHTML = `<div style="color:#334155;font-size:11px;font-family:monospace">No result CSV files found in:<br><span style="color:#0f766e">${outputDir}</span></div>`;
        return;
    }
    const palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    const groups = {};
    files.forEach(f => { const r = f.reaction||'unknown'; if (!groups[r]) groups[r]=[]; groups[r].push(f); });
    let html = `<div style="font-size:9px;color:#1e3a5f;font-family:monospace;margin-bottom:12px">${files.length} file(s) · <span style="color:#0f4c75">${outputDir}</span></div>`;
    Object.entries(groups).forEach(([rxn, runs], gi) => {
        const color = palette[gi % palette.length];
        runs.sort((a,b) => (b.timestamp||'').localeCompare(a.timestamp||''));
        html += `<div style="margin-bottom:20px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block;flex-shrink:0"></span>
            <span style="font-size:12px;font-weight:700;color:${color};text-transform:uppercase;letter-spacing:0.06em">${rxn}</span>
            <span style="font-size:10px;color:#1e3a5f">${runs.length} run(s)</span>
          </div><div style="display:flex;flex-direction:column;gap:6px">`;
        runs.forEach(f => {
            const mean = f.mean!=null ? f.mean.toFixed(4) : '—';
            const std  = f.std !=null ? f.std.toFixed(4)  : '—';
            const n    = f.n   !=null ? f.n               : '—';
            const barW = Math.min(100, ((f.mean||0)/10)*100).toFixed(1);
            const stdW = Math.min(30,  ((f.std ||0)/10)*100).toFixed(1);
            const stdL = Math.max(0, parseFloat(barW)-parseFloat(stdW)/2).toFixed(1);
            const ts   = (f.timestamp||'').replace('_',' ');
            html += `<div style="background:rgba(15,23,42,0.8);border:0.5px solid rgba(30,41,59,0.9);border-radius:6px;padding:10px 14px">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
                <span style="font-size:9px;color:#334155;font-family:monospace">${ts}</span>
                <span style="font-size:9px;color:#1e3a5f;font-family:monospace;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${f.filename}</span>
              </div>
              <div style="position:relative;height:10px;background:rgba(30,41,59,0.9);border-radius:3px;overflow:hidden;margin-bottom:6px">
                <div style="height:100%;width:${barW}%;background:${color};border-radius:3px;opacity:0.7"></div>
                <div style="position:absolute;top:1px;height:8px;left:${stdL}%;width:${stdW}%;background:rgba(251,191,36,0.4);border-radius:2px"></div>
              </div>
              <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
                <span style="font-size:13px;font-weight:700;color:#e2e8f0;font-family:monospace">${mean}</span>
                <span style="font-size:10px;color:#94a3b8">μ</span>
                <span style="font-size:11px;color:#fbbf24;font-family:monospace">±${std}</span>
                <span style="font-size:10px;color:#94a3b8">σ</span>
                <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:rgba(8,145,178,0.12);color:#67e8f9">n=${n}</span>
                <a href="/vina_visualization/ts_download?path=${encodeURIComponent(f.path)}"
                   style="margin-left:auto;font-size:9px;color:#0891b2;text-decoration:none;opacity:0.7" download>⬇ csv</a>
              </div>
            </div>`;
        });
        html += `</div></div>`;
    });
    content.innerHTML = html;
}

function _tsMetaLoad() {
    const outDir = document.getElementById('tsOutputDirInput')?.value.trim() || '';
    const content = document.getElementById('tsMetaContent');
    if (!content) return;
    if (!outDir) { content.innerHTML = `<span style="color:#334155">No output directory set.</span>`; return; }
    content.innerHTML = `<span style="color:#334155;font-style:italic">Scanning ${outDir} …</span>`;
    fetch(`/vina_visualization/ts_meta?dir=${encodeURIComponent(outDir)}`)
        .then(r => r.json())
        .then(d => _tsMetaRender(d, outDir))
        .catch(e => { content.innerHTML = `<span style="color:#ef4444">Error: ${e.message}</span>`; });
}

// ── CPU monitor ───────────────────────────────────────────────────────────
let _tsCpuEs   = null;
let _tsCpuRaf  = null;
let _tsCpuData = null;

function _tsCpuStart() {
    if (_tsCpuEs) return;
    _tsCpuEs = new EventSource('/vina_visualization/ts_cpu');
    _tsCpuEs.onmessage = (evt) => {
        try { _tsCpuData = JSON.parse(evt.data); } catch {}
        if (!_tsCpuRaf) _tsCpuRaf = requestAnimationFrame(_tsCpuRender);
    };
    _tsCpuEs.onerror = () => { _tsCpuStop(); };
}

function _tsCpuStop() {
    if (_tsCpuEs) { _tsCpuEs.close(); _tsCpuEs = null; }
    if (_tsCpuRaf) { cancelAnimationFrame(_tsCpuRaf); _tsCpuRaf = null; }
}

function _tsQueueCpuRender() {
    if (!_tsCpuRaf) _tsCpuRaf = requestAnimationFrame(_tsCpuRender);
}

function _tsCpuRender() {
    _tsCpuRaf = null;
    const d = _tsCpuData;
    if (!d) return;
    const rows = d.procs || [];
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('tsSysCpuTotal', d.total.toFixed(1) + '%');
    set('tsCpuTotal',    d.total.toFixed(1) + '%');
    if (d.load_avg) set('tsSysLoadAvg', d.load_avg.join('  '));
    if (d.mem) { set('tsSysMemUsed', d.mem.used); set('tsSysMemTotal', d.mem.total); }

    const sysTable = document.getElementById('tsSysProcTable');
    if (!sysTable) { setTimeout(() => { if (_tsCpuData) _tsQueueCpuRender(); }, 300); return; }

    if (rows.length === 0) {
        sysTable.innerHTML = `<tr><td colspan="10" style="color:#1e3a5f;padding:20px 0;font-size:10px;text-align:center;font-family:monospace">no app.py processes found</td></tr>`;
        return;
    }

    const existingRows = {};
    for (const tr of sysTable.querySelectorAll('tr[data-pid]')) existingRows[tr.dataset.pid] = tr;
    const seenPids = new Set();
    const fragment = document.createDocumentFragment();
    const _jobColors = {}, _palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    let _palIdx = 0;
    const _jobColor = tag => {
        if (!tag) return null;
        if (!_jobColors[tag]) _jobColors[tag] = _palette[_palIdx++ % _palette.length];
        return _jobColors[tag];
    };

    rows.forEach(r => {
        const pid = String(r.pid);
        seenPids.add(pid);
        const cpuColor  = r.cpu > 80 ? '#ef4444' : r.cpu > 40 ? '#f59e0b' : '#64748b';
        const statColor = r.status === 'R' ? '#34d399' : r.status === 'Z' ? '#ef4444' : '#475569';
        const cpuNumColor = r.cpu > 80 ? '#ef4444' : r.cpu > 40 ? '#f59e0b' : r.cpu > 5 ? '#22d3ee' : '#1e3a5f';
        const label = r.cmd || r.name;
        const nameColor = r.job_tag ? '#94a3b8' : '#475569';
        const cpuBarW = Math.min(100, r.cpu).toFixed(1);
        const cpuNumStr = r.cpu_num != null ? String(r.cpu_num) : '—';
        let affinityStr = '—';
        if (r.cpu_affinity && r.cpu_affinity.length > 0) {
            const a = r.cpu_affinity;
            affinityStr = a.length === 1 ? String(a[0]) : `${a[0]}–${a[a.length-1]}`;
        }
        let jobHTML = '';
        if (r.job_tag) {
            const jc = _jobColor(r.job_tag);
            jobHTML = r.is_elion_root
                ? `<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:9px;font-weight:700;background:${jc}22;color:${jc};border:0.5px solid ${jc}55;white-space:nowrap">elion·${r.job_tag}</span>`
                : `<span style="display:inline-block;padding:1px 5px;border-radius:3px;font-size:9px;color:${jc}88;border:0.5px solid ${jc}33;white-space:nowrap">↳ ${r.job_tag}</span>`;
        }
        let tr = existingRows[pid];
        if (!tr) {
            tr = document.createElement('tr');
            tr.dataset.pid = pid;
            tr.style.cssText = 'border-bottom:0.5px solid rgba(15,23,42,0.9)';
            tr.innerHTML = `
              <td data-col="pid"      style="padding:5px 14px 5px 0;color:#334155;font-family:monospace;white-space:nowrap"></td>
              <td data-col="job"      style="padding:5px 14px 5px 0;white-space:nowrap"></td>
              <td data-col="cmd"      style="padding:5px 14px 5px 0;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace"></td>
              <td data-col="cpu"      style="padding:5px 14px 5px 0;text-align:right;font-family:monospace">
                <span data-sub="val"></span>
                <div style="height:2px;background:rgba(30,41,59,0.8);border-radius:1px;margin-top:2px;width:40px;margin-left:auto">
                  <div data-sub="bar" style="height:100%;width:0%;border-radius:1px;transition:width 0.8s ease,background 0.4s ease;opacity:0.7"></div>
                </div>
              </td>
              <td data-col="mem"      style="padding:5px 14px 5px 0;text-align:right;color:#64748b;font-family:monospace"></td>
              <td data-col="virt"     style="padding:5px 14px 5px 0;text-align:right;color:#334155;font-family:monospace"></td>
              <td data-col="res"      style="padding:5px 14px 5px 0;text-align:right;color:#475569;font-family:monospace"></td>
              <td data-col="time"     style="padding:5px 14px 5px 0;text-align:right;color:#1e293b;font-family:monospace"></td>
              <td data-col="cpu_num"  style="padding:5px 14px 5px 0;text-align:right;font-family:monospace;font-weight:600"></td>
              <td data-col="affinity" style="padding:5px 14px 5px 0;text-align:right;color:#1e3a5f;font-family:monospace;font-size:10px"></td>
              <td data-col="status"   style="padding:5px 0;text-align:center;font-weight:700;font-family:monospace"></td>`;
        }
        const col = name => tr.querySelector(`[data-col="${name}"]`);
        col('pid').textContent      = pid;
        col('job').innerHTML        = jobHTML;
        col('cmd').textContent      = label;
        col('cmd').style.color      = nameColor;
        col('cmd').title            = label;
        col('mem').textContent      = r.mem.toFixed(1);
        col('virt').textContent     = r.virt;
        col('res').textContent      = r.res;
        col('time').textContent     = r.time || '—';
        col('cpu_num').textContent  = cpuNumStr;
        col('cpu_num').style.color  = cpuNumColor;
        col('affinity').textContent = affinityStr;
        col('status').textContent   = r.status;
        col('status').style.color   = statColor;
        const cpuVal = tr.querySelector('[data-sub="val"]');
        const cpuBar = tr.querySelector('[data-sub="bar"]');
        if (cpuVal) { cpuVal.textContent = r.cpu.toFixed(1); cpuVal.style.color = cpuColor; cpuVal.style.fontWeight = r.cpu > 40 ? '600' : '400'; }
        if (cpuBar) { cpuBar.style.width = cpuBarW + '%'; cpuBar.style.background = cpuColor; }
        fragment.appendChild(tr);
    });
    for (const [pid, tr] of Object.entries(existingRows)) { if (!seenPids.has(pid)) tr.remove(); }
    sysTable.appendChild(fragment);
}

// ── GPU monitor ───────────────────────────────────────────────────────────
let _tsGpuEs   = null;
let _tsGpuRaf  = null;
let _tsGpuData = null;

function _tsGpuStart() {
    if (_tsGpuEs) return;
    _tsGpuEs = new EventSource('/vina_visualization/ts_gpu');
    _tsGpuEs.onmessage = (evt) => {
        try { _tsGpuData = JSON.parse(evt.data); } catch {}
        if (!_tsGpuRaf) _tsGpuRaf = requestAnimationFrame(_tsGpuRender);
    };
    _tsGpuEs.onerror = () => { _tsGpuStop(); };
}

function _tsGpuStop() {
    if (_tsGpuEs) { _tsGpuEs.close(); _tsGpuEs = null; }
    if (_tsGpuRaf) { cancelAnimationFrame(_tsGpuRaf); _tsGpuRaf = null; }
}

// ── Search speed (sec/it) ──────────────────────────────────────────────────
// Self-contained: samples the live iteration count on a timer and derives a
// rolling seconds-per-iteration. Reads the active job's spark history length
// (one entry per [TS:stats], i.e. per iteration) — the same source the sidebar
// iteration counter uses — so it needs no hook into the per-iteration worker
// message path. Writes "X.X s/it (Y.Y it/s)" to #tsTsSpeed.
let _tsSpeedTimer = null;
let _tsSpeedSamples = [];   // rolling [{t: ms, iter: n}]
const _TS_SPEED_WINDOW_MS = 20000;   // average over the last ~20 s
const _TS_SPEED_MIN_SAMPLES = 2;

function _tsCurrentIterCount() {
    // Prefer the active job's spark history length; fall back to the displayed
    // counter text. Returns a number or null.
    try {
        const idx = (_ts && _ts._activeJobIdx != null) ? _ts._activeJobIdx : 0;
        const sh  = _ts && _ts._jobSparkHistory ? _ts._jobSparkHistory[idx] : null;
        if (sh && sh.length) return sh.length;
        // fall back across all jobs (take the max length seen)
        if (_ts && _ts._jobSparkHistory) {
            let mx = 0;
            for (const k in _ts._jobSparkHistory) {
                const a = _ts._jobSparkHistory[k];
                if (a && a.length > mx) mx = a.length;
            }
            if (mx > 0) return mx;
        }
    } catch {}
    // last resort: parse the on-screen counter
    const el = document.getElementById('tsTsIter');
    if (el) { const n = parseInt(el.textContent, 10); if (!isNaN(n)) return n; }
    return null;
}

function _tsSpeedTick() {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    const iter = _tsCurrentIterCount();
    const now  = Date.now();
    if (iter == null) { return; }

    _tsSpeedSamples.push({ t: now, iter });
    // drop samples older than the window
    const cutoff = now - _TS_SPEED_WINDOW_MS;
    while (_tsSpeedSamples.length > 2 && _tsSpeedSamples[0].t < cutoff) {
        _tsSpeedSamples.shift();
    }
    if (_tsSpeedSamples.length < _TS_SPEED_MIN_SAMPLES) { set('tsTsSpeed', '—'); return; }

    const first = _tsSpeedSamples[0];
    const last  = _tsSpeedSamples[_tsSpeedSamples.length - 1];
    const dIter = last.iter - first.iter;
    const dSec  = (last.t - first.t) / 1000;

    if (dIter <= 0) {
        // No progress in the window — either finished or stalled. Show idle.
        set('tsTsSpeed', dSec > 5 ? 'idle' : '—');
        return;
    }
    const secPerIt = dSec / dIter;
    const itPerSec = dIter / dSec;
    // Choose a readable format: if very fast, lead with it/s.
    let txt;
    if (secPerIt < 1) {
        txt = `${itPerSec.toFixed(1)} it/s`;
    } else {
        txt = `${secPerIt.toFixed(2)} s/it (${itPerSec.toFixed(2)} it/s)`;
    }
    set('tsTsSpeed', txt);
}

function _tsSpeedStart() {
    if (_tsSpeedTimer) return;
    _tsSpeedSamples = [];
    _tsSpeedTick();   // prime immediately
    _tsSpeedTimer = setInterval(_tsSpeedTick, 1000);
}

function _tsSpeedStop() {
    if (_tsSpeedTimer) { clearInterval(_tsSpeedTimer); _tsSpeedTimer = null; }
    _tsSpeedSamples = [];
}

function _tsGpuRender() {
    _tsGpuRaf = null;
    const d = _tsGpuData;
    if (!d) return;
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };

    const devWrap  = document.getElementById('tsGpuDevices');
    const procTbl  = document.getElementById('tsGpuProcTable');
    const procWrap = document.getElementById('tsGpuProcWrap');
    const msg      = document.getElementById('tsGpuMsg');
    if (!devWrap) { setTimeout(() => { if (_tsGpuData) { if (!_tsGpuRaf) _tsGpuRaf = requestAnimationFrame(_tsGpuRender); } }, 300); return; }

    // Fallback: no GPU / pynvml unavailable
    if (!d.available) {
        devWrap.innerHTML = '';
        if (procWrap) procWrap.style.display = 'none';
        if (msg) { msg.style.display = 'block'; msg.textContent = d.error || 'No NVIDIA GPU detected.'; }
        set('tsGpuUtil', '—%'); set('tsGpuMemUsed', '—'); set('tsGpuMemTotal', '—');
        set('tsGpuTemp', '—'); set('tsGpuPower', '—');
        return;
    }
    if (msg) msg.style.display = 'none';
    if (procWrap) procWrap.style.display = 'block';

    const gpus = d.gpus || [];
    // Header strip reflects GPU 0 (primary)
    if (gpus[0]) {
        const g0 = gpus[0];
        set('tsGpuUtil',     g0.util != null ? g0.util + '%' : '—%');
        set('tsGpuMemUsed',  g0.mem_used_str || '—');
        set('tsGpuMemTotal', g0.mem_total_str || '—');
        set('tsGpuTemp',     g0.temp != null ? g0.temp + '°C' : '—');
        set('tsGpuPower',    g0.power != null ? g0.power + 'W' : '—');
    }

    // Device cards (one per GPU)
    const utilColor = u => u == null ? '#475569' : u > 80 ? '#ef4444' : u > 40 ? '#f59e0b' : u > 5 ? '#22d3ee' : '#64748b';
    devWrap.innerHTML = gpus.map(g => {
        const memPct = g.mem_pct != null ? g.mem_pct : 0;
        const memBarColor = memPct > 85 ? '#ef4444' : memPct > 60 ? '#f59e0b' : '#c084fc';
        const utilBarW = g.util != null ? Math.min(100, g.util) : 0;
        const pwrStr = (g.power != null && g.power_limit != null)
            ? `${g.power} / ${g.power_limit} W`
            : (g.power != null ? `${g.power} W` : '—');
        return `
        <div style="background:#0a0e1a;border:0.5px solid rgba(30,41,59,0.9);border-radius:8px;padding:12px 14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
            <span style="font-size:12px;font-weight:600;color:#e2e8f0">${g.name || ('GPU ' + g.index)}</span>
            <span style="font-size:9px;color:#475569;font-family:monospace">GPU ${g.index}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-family:monospace;font-size:11px">
            <div>
              <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:#475569">Util</span>
                <span style="color:${utilColor(g.util)};font-weight:600">${g.util != null ? g.util + '%' : '—'}</span>
              </div>
              <div style="height:4px;background:rgba(30,41,59,0.8);border-radius:2px;overflow:hidden">
                <div style="height:100%;width:${utilBarW}%;background:${utilColor(g.util)};transition:width 0.6s ease"></div>
              </div>
            </div>
            <div>
              <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:#475569">Mem</span>
                <span style="color:#c084fc;font-weight:600">${g.mem_used_str || '—'} / ${g.mem_total_str || '—'}</span>
              </div>
              <div style="height:4px;background:rgba(30,41,59,0.8);border-radius:2px;overflow:hidden">
                <div style="height:100%;width:${memPct}%;background:${memBarColor};transition:width 0.6s ease"></div>
              </div>
            </div>
            <div style="display:flex;justify-content:space-between">
              <span style="color:#475569">Temp</span>
              <span style="color:#34d399;font-weight:600">${g.temp != null ? g.temp + '°C' : '—'}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
              <span style="color:#475569">Power</span>
              <span style="color:#f59e0b;font-weight:600">${pwrStr}</span>
            </div>
          </div>
        </div>`;
    }).join('');

    // Process table (flatten all GPUs' procs)
    if (procTbl) {
        const allProcs = [];
        gpus.forEach(g => (g.procs || []).forEach(p => allProcs.push(Object.assign({gpu: g.index}, p))));
        if (allProcs.length === 0) {
            procTbl.innerHTML = `<tr><td colspan="5" style="color:#1e3a5f;padding:16px 0;font-size:10px;text-align:center;font-family:monospace">no GPU processes</td></tr>`;
        } else {
            const _palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
            const _jobColors = {}; let _palIdx = 0;
            const _jobColor = tag => { if (!tag) return null; if (!_jobColors[tag]) _jobColors[tag] = _palette[_palIdx++ % _palette.length]; return _jobColors[tag]; };
            procTbl.innerHTML = allProcs.map(p => {
                let jobHTML = '';
                if (p.job_tag) {
                    const jc = _jobColor(p.job_tag);
                    jobHTML = `<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:9px;font-weight:700;background:${jc}22;color:${jc};border:0.5px solid ${jc}55;white-space:nowrap">elion·${p.job_tag}</span>`;
                }
                const typeColor = p.type === 'C' ? '#22d3ee' : '#475569';
                const nameColor = p.job_tag ? '#94a3b8' : '#475569';
                return `<tr style="border-bottom:0.5px solid rgba(15,23,42,0.9)">
                  <td style="padding:5px 14px 5px 0;color:#334155;font-family:monospace;white-space:nowrap">${p.pid}</td>
                  <td style="padding:5px 14px 5px 0;white-space:nowrap">${jobHTML}</td>
                  <td style="padding:5px 14px 5px 0;color:${nameColor};font-family:monospace;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.name||''}">${p.name||'?'}</td>
                  <td style="padding:5px 14px 5px 0;text-align:right;color:${typeColor};font-family:monospace;font-weight:600" title="${p.type==='C'?'Compute':'Graphics'}">${p.type||'—'}</td>
                  <td style="padding:5px 14px 5px 0;text-align:right;color:#c084fc;font-family:monospace">${p.mem||'—'}</td>
                </tr>`;
            }).join('');
        }
    }
}

// ── Kill All button ───────────────────────────────────────────────────────
function _tsKillAll() {
    const btn = document.getElementById('tsKillAllBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Stopping…'; }
    fetch('/vina_visualization/ts_kill_all', {method:'POST'})
        .then(r => r.json())
        .then(d => {
            const n = d.killed?.length || 0;
            if (btn) { btn.disabled = false; btn.textContent = `☠ Kill All (${n} stopped)`; }
            setTimeout(() => { if (btn) btn.textContent = '☠ Kill All'; }, 3000);
        })
        .catch(() => { if (btn) { btn.disabled = false; btn.textContent = '☠ Kill All'; } });
}

let _tsJobsPollTimer = null;
function _tsJobsPollStart() {
    if (_tsJobsPollTimer) return;
    _tsJobsPollTimer = setInterval(() => {
        fetch('/vina_visualization/ts_jobs_status')
            .then(r => r.json())
            .then(d => {
                const running = (d.jobs || []).filter(j => j.status === 'running');
                const btn = document.getElementById('tsKillAllBtn');
                if (btn) {
                    btn.style.display = running.length > 0 ? 'inline-flex' : 'none';
                    if (running.length > 0) btn.textContent = `☠ Kill All (${running.length})`;
                }
                const monBtn = document.getElementById('tsTabResults');
                if (monBtn) monBtn.textContent = running.length > 0 ? `📡 Monitor ● ${running.length}` : '📡 Monitor';
            })
            .catch(() => {});
    }, 5000);
}
function _tsJobsPollStop() {
    if (_tsJobsPollTimer) { clearInterval(_tsJobsPollTimer); _tsJobsPollTimer = null; }
}

function _tsInjectKillAllBtn() {
    if (document.getElementById('tsKillAllBtn')) return;
    const runBtn = document.getElementById('tsRunBtn');
    if (!runBtn) return;
    const btn = document.createElement('button');
    btn.id = 'tsKillAllBtn';
    btn.textContent = '☠ Kill All';
    btn.title = 'Send SIGTERM to all running elion.py jobs';
    btn.onclick = _tsKillAll;
    btn.style.cssText = [
        'display:none','align-items:center','gap:4px','padding:6px 14px','border-radius:6px',
        'font-size:12px','font-weight:600','cursor:pointer',
        'background:rgba(239,68,68,0.15)','color:#ef4444','border:1px solid rgba(239,68,68,0.4)',
        'transition:background 0.2s','margin-right:8px',
    ].join(';');
    btn.onmouseover = () => btn.style.background = 'rgba(239,68,68,0.25)';
    btn.onmouseout  = () => btn.style.background = 'rgba(239,68,68,0.15)';
    runBtn.parentNode.insertBefore(btn, runBtn);
}

// ── Log level ─────────────────────────────────────────────────────────────
const _TS_LOG_LEVELS = [
    { name: 'minimal', level: 0, color: '#475569', desc: 'winner + post-update only' },
    { name: 'normal',  level: 1, color: '#0891b2', desc: 'adds evaluate + Bayesian update result' },
    { name: 'verbose', level: 2, color: '#7c3aed', desc: 'adds full Bayesian math (before/after/weights)' },
    { name: 'debug',   level: 3, color: '#f59e0b', desc: 'debug statements only (hides TS log)' },
];
let _tsLogLevel = 0;

// True when the verbosity toggle is on 'debug': the 💬 panel then shows ONLY
// backend [DEBUG*] statements and suppresses the normal TS winner/eval log.
function _tsIsDebugOnly() {
    return (_TS_LOG_LEVELS[_tsLogLevel] || {}).name === 'debug';
}
window._tsIsDebugOnly = _tsIsDebugOnly;

function _tsLogLevelCycle() {
    _tsLogLevel = (_tsLogLevel + 1) % _TS_LOG_LEVELS.length;
    const cfg = _TS_LOG_LEVELS[_tsLogLevel];
    const btn = document.getElementById('tsLogLevelBtn');
    if (btn) { btn.textContent = cfg.name; btn.style.color = cfg.color; btn.style.border = `0.5px solid ${cfg.color}44`; btn.title = `Log verbosity: ${cfg.desc} — click to change`; }
    (_ts._streams || []).forEach(s => { if (s.worker) s.worker.postMessage({ type: 'set_log_level', level: cfg.level }); });
}

function _tsPatchLogLevelBtn() {
    if (document.getElementById('tsLogLevelBtn')) return;
    // The TS log section is hidden (the right-side 💬 panel shows the log now),
    // so place the verbosity toggle in the sidebar footer next to tsTsStatus.
    const statusEl = document.getElementById('tsTsStatus');
    const footer   = statusEl?.parentElement;
    if (!footer) return;
    const btn = document.createElement('button');
    btn.id = 'tsLogLevelBtn';
    btn.textContent = 'minimal';
    btn.onclick = _tsLogLevelCycle;
    btn.title = 'Log verbosity: winner + post-update only — click to change';
    btn.style.cssText = 'font-size:9px;padding:1px 6px;border-radius:3px;border:0.5px solid rgba(71,85,105,0.4);background:rgba(30,41,59,0.6);color:#64748b;cursor:pointer;transition:all 0.2s;margin-right:6px;flex-shrink:0';
    // Lay out footer as a row: [minimal btn] [status text …]
    footer.style.display    = 'flex';
    footer.style.alignItems = 'center';
    statusEl.style.flex     = '1';
    statusEl.style.minWidth = '0';   // allow truncate to work inside flex
    footer.insertBefore(btn, statusEl);
}

// ── TS Iteration Log + activity mini-chat (tsMiniLogPanel in hub.html) ────
// The 💬 button (tsMiniLogBtn) and panel (tsMiniLogPanel) are static HTML in
// hub.html, matching the Vina #miniChat layout. On open it shows greeting +
// suggestion bubbles; once a run streams log lines, the bubbles are cleared
// and replaced by the live log. The bottom input answers TS questions via a
// local KB, mirroring pose.js _miniAnswer. This is the TS tool's OWN panel —
// distinct from the shared #miniChat (whose auto-show is suppressed in ts_run.js).

// ── Model architecture viewer (🧬 button) ──────────────────────────────────
// Fetches /ts_model_arch and shows the current CHEMBERT .pt architecture in a
// floating panel, including whether it's a usable fine-tuned model or a bare
// pretrained backbone (which explains the pretrained_model.pt load behavior).
function _tsShowModelArch() {
    let panel = document.getElementById('tsModelArchPanel');
    if (panel && panel.style.display !== 'none') {  // toggle closed if open
        panel.style.display = 'none';
        return;
    }
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'tsModelArchPanel';
        panel.style.cssText =
            'position:fixed;top:120px;right:24px;z-index:10300;width:440px;max-height:70vh;' +
            'background:#0f172a;border:1px solid #4c1d95;border-radius:16px;' +
            'box-shadow:0 8px 32px rgba(0,0,0,0.6);display:flex;flex-direction:column;overflow:hidden;';
        panel.innerHTML =
            '<div style="padding:10px 14px;border-bottom:1px solid #1e293b;display:flex;align-items:center;gap:8px;background:#13091f;">' +
              '<div style="width:22px;height:22px;border-radius:6px;background:linear-gradient(135deg,#7c3aed,#4c1d95);display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;">🧬</div>' +
              '<span style="font-size:12px;font-weight:600;color:#f1f5f9;flex:1;">CHEMBERT Model Architecture</span>' +
              '<button onclick="document.getElementById(\'tsModelArchPanel\').style.display=\'none\'" style="background:none;border:none;cursor:pointer;color:#475569;font-size:14px;padding:2px 4px;">✕</button>' +
            '</div>' +
            '<div id="tsModelArchBody" class="ts-minilog-scroll" style="overflow-y:auto;padding:14px;font-size:12px;color:#cbd5e1;scrollbar-width:thin;scrollbar-color:#1e3a5f #05070f;"></div>';
        document.body.appendChild(panel);
    }
    panel.style.display = 'flex';
    const body = document.getElementById('tsModelArchBody');
    body.innerHTML = '<div style="color:#64748b;font-size:11px;">Loading model architecture…</div>';

    fetch('/vina_visualization/ts_model_arch')
        .then(r => r.json())
        .then(d => _tsRenderModelArch(d))
        .catch(e => { body.innerHTML = `<div style="color:#fca5a5;">Failed to load: ${e}</div>`; });
}

function _tsRenderModelArch(d) {
    const body = document.getElementById('tsModelArchBody');
    if (!body) return;
    if (d.status !== 'ok') {
        let dbgHtml = '';
        if (d.debug && (d.debug.top_level_keys || d.debug.reward_function)) {
            dbgHtml =
                `<div style="margin-top:10px;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">input_TS.yml structure</div>` +
                `<div style="background:#0a0f1a;border:1px solid #1e293b;border-radius:8px;padding:8px 10px;margin-top:6px;font-family:monospace;font-size:10px;color:#94a3b8;white-space:pre-wrap;word-break:break-word;">` +
                `top-level keys: ${JSON.stringify(d.debug.top_level_keys || [])}\n\n` +
                `Reward_function:\n${JSON.stringify(d.debug.reward_function, null, 2)}` +
                `</div>` +
                `<div style="margin-top:8px;font-size:10px;color:#64748b;">Tip: tell me the key path above that holds the .pt file, or click 🧬 again after setting it.</div>`;
        }
        body.innerHTML =
            `<div style="color:#fca5a5;background:rgba(239,68,68,0.08);border:0.5px solid rgba(239,68,68,0.3);border-radius:8px;padding:10px;">` +
            `<strong>Could not inspect model</strong><br>${(d.message||'unknown error').replace(/</g,'&lt;')}</div>` +
            dbgHtml;
        return;
    }
    const s = d.summary || {};
    const v = d.verdict || {};
    const fmt = n => (n == null ? '—' : n.toLocaleString());

    // Verdict banner
    const vColor = v.usable ? '#34d399' : '#f59e0b';
    const vBg    = v.usable ? 'rgba(52,211,153,0.08)' : 'rgba(245,158,11,0.08)';
    const vBord  = v.usable ? 'rgba(52,211,153,0.35)' : 'rgba(245,158,11,0.4)';
    const vIcon  = v.usable ? '✓' : '⚠';

    let html =
        `<div style="background:${vBg};border:0.5px solid ${vBord};border-radius:8px;padding:10px 12px;margin-bottom:12px;">` +
          `<div style="color:${vColor};font-weight:700;font-size:12px;margin-bottom:4px;">${vIcon} ${v.kind || 'unknown'}${v.usable ? ' — usable' : ' — NOT usable for scoring'}</div>` +
          `<div style="color:#94a3b8;font-size:11px;line-height:1.55;">${(v.message||'').replace(/</g,'&lt;')}</div>` +
        `</div>`;

    // File + summary table
    html +=
        `<div style="font-size:10px;color:#64748b;word-break:break-all;margin-bottom:10px;">${(d.model_file||'').replace(/</g,'&lt;')}</div>` +
        `<table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:14px;">` +
        _tsArchRow('Architecture', s.architecture) +
        _tsArchRow('Total parameters', fmt(s.total_params)) +
        _tsArchRow('Weight tensors', fmt(s.tensors)) +
        _tsArchRow('Transformer layers', s.transformer_layers) +
        _tsArchRow('File size', s.file_size_mb != null ? s.file_size_mb + ' MB' : '—') +
        _tsArchRow("Has 'bert.' prefix", _tsArchBool(s.has_bert_prefix)) +
        _tsArchRow('Has regression head (linear.*)', _tsArchBool(s.has_linear_head)) +
        `</table>`;

    // Key groups
    if (Array.isArray(d.groups) && d.groups.length) {
        html += `<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">State dict groups</div>`;
        for (const g of d.groups) {
            html +=
                `<div style="background:#0a0f1a;border:1px solid #1e293b;border-radius:8px;padding:8px 10px;margin-bottom:6px;">` +
                  `<div style="display:flex;justify-content:space-between;align-items:center;">` +
                    `<span style="color:#a78bfa;font-family:monospace;font-weight:600;">${g.name}</span>` +
                    `<span style="color:#64748b;font-size:10px;">${fmt(g.params)} params · ${g.keys} keys</span>` +
                  `</div>`;
            if (Array.isArray(g.sample) && g.sample.length) {
                html += `<div style="margin-top:4px;">`;
                for (const sm of g.sample) {
                    html += `<div style="color:#475569;font-family:monospace;font-size:9px;">${sm.key} <span style="color:#334155;">[${(sm.shape||[]).join('×')}]</span></div>`;
                }
                html += `</div>`;
            }
            html += `</div>`;
        }
    }

    body.innerHTML = html;
}

function _tsArchRow(label, value) {
    return `<tr style="border-bottom:0.5px solid rgba(30,41,59,0.6);">` +
           `<td style="color:#64748b;padding:5px 0;">${label}</td>` +
           `<td style="color:#e2e8f0;text-align:right;font-family:monospace;padding:5px 0;">${value == null ? '—' : value}</td></tr>`;
}
function _tsArchBool(b) {
    return b ? '<span style="color:#34d399;">✓ yes</span>' : '<span style="color:#f59e0b;">✗ no</span>';
}

let _tsMiniLogOpen     = false;
let _tsMiniLogStreaming = false;  // true once log lines have replaced bubbles

function _tsMiniLogToggle() {
    const panel = document.getElementById('tsMiniLogPanel');
    const btn   = document.getElementById('tsMiniLogBtn');
    const live  = document.getElementById('tsMiniLogLive');
    if (!panel) return;

    _tsMiniLogOpen = !_tsMiniLogOpen;
    panel.style.display = _tsMiniLogOpen ? 'flex' : 'none';

    if (_tsMiniLogOpen) {
        // Clear unread glow
        if (btn) btn.style.boxShadow = 'none';
        if (live) live.style.display = _ts.running ? 'inline' : 'none';
        // If a run is already active, drop greeting and show the log history
        if (_ts.running || _tsMiniLogStreaming) {
            _tsMiniLogClearGreeting();
            const src  = document.getElementById('tsTsLog');
            const dest = document.getElementById('tsMiniLogContent');
            if (src && dest && !dest.querySelector('[data-ts-logline]')) {
                // Seed from the main log's current content
                [...src.children].forEach(node => {
                    const d = document.createElement('div');
                    d.dataset.tsLogline = '1';
                    d.style.cssText = 'font-family:ui-monospace,monospace;font-size:10px;line-height:1.6;';
                    d.innerHTML = node.innerHTML;
                    dest.appendChild(d);
                });
            }
            if (dest) dest.scrollTop = dest.scrollHeight;
        }
    }
}

// Remove the greeting/suggestion bubbles the first time real log lines arrive
function _tsMiniLogClearGreeting() {
    const dest = document.getElementById('tsMiniLogContent');
    if (!dest) return;
    dest.querySelectorAll('[data-ts-greeting]').forEach(el => el.remove());
}

// Called by ts_worker_bridge.js after each rAF log flush. Streams new lines
// into the panel body (clearing greeting on first line) or, when the panel is
// closed, glows the 💬 button to signal unread activity.
let _tsMiniLogBuf = [];
function _tsForwardBufToLogPanel() {
    if (!_tsMiniLogBuf.length) return;
    const lines = _tsMiniLogBuf.splice(0);
    _tsMiniLogStreaming = true;

    if (_tsMiniLogOpen) {
        const dest = document.getElementById('tsMiniLogContent');
        if (dest) {
            _tsMiniLogClearGreeting();
            const frag = document.createDocumentFragment();
            lines.forEach(html => {
                const d = document.createElement('div');
                d.dataset.tsLogline = '1';
                d.style.cssText = 'font-family:ui-monospace,monospace;font-size:10px;line-height:1.6;';
                d.innerHTML = html;
                frag.appendChild(d);
            });
            dest.appendChild(frag);
            while (dest.querySelectorAll('[data-ts-logline]').length > 200) {
                dest.querySelector('[data-ts-logline]').remove();
            }
            dest.scrollTop = dest.scrollHeight;
        }
    } else {
        // Glow the 💬 button (matches poseMiniBtn unread pattern)
        const btn = document.getElementById('tsMiniLogBtn');
        if (btn) btn.style.boxShadow = '0 0 0 2px rgba(34,211,238,0.45)';
    }
}

// Enqueue each incoming log line BEFORE the rAF DOM flush so the forwarder
// sees it. Called from ts_worker_bridge.js ts_log handler.
function _tsEnqueueLogLine(html) {
    _tsMiniLogBuf.push(html);
}

// Surface an error/important message in the mini-log panel for debugging.
// Called from the error paths (process exit, failed start, fetch failure) so
// errors that show in the bottom status bar are also visible in the 💬 panel.
// Auto-opens the panel and clears the greeting so the error can't be missed.
function _tsMiniLogError(text) {
    if (!text) return;
    // Remember that an error occurred this run so end-of-run finalisation can
    // reflect it in the panel even if the live error line was missed/deduped.
    if (typeof _ts !== 'undefined' && _ts) { _ts._sawError = true; _ts._lastErrorText = String(text); }
    // Dedup: the same error can arrive via several paths (SSE line, status
    // message, raw log). Skip if identical to the last error shown recently.
    const now = Date.now();
    if (_tsMiniLogError._last === text && (now - (_tsMiniLogError._lastT || 0)) < 3000) return;
    _tsMiniLogError._last = text;
    _tsMiniLogError._lastT = now;
    const dest = document.getElementById('tsMiniLogContent');
    if (!dest) return;
    _tsMiniLogClearGreeting();
    const escaped = String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const row = document.createElement('div');
    row.dataset.tsLogline = '1';
    row.style.cssText = 'font-family:ui-monospace,monospace;font-size:10px;line-height:1.5;' +
                        'color:#fca5a5;background:rgba(239,68,68,0.08);' +
                        'border:0.5px solid rgba(239,68,68,0.3);border-radius:6px;' +
                        'padding:6px 9px;word-break:break-word;white-space:pre-wrap;';
    row.innerHTML = `<span style="color:#ef4444;font-weight:700">⚠ </span>${escaped}`;
    dest.appendChild(row);
    while (dest.querySelectorAll('[data-ts-logline]').length > 200) {
        dest.querySelector('[data-ts-logline]').remove();
    }
    dest.scrollTop = dest.scrollHeight;
    _tsMiniLogStreaming = true;

    // Make sure the error is seen: open the panel if closed, else glow the button.
    if (!_tsMiniLogOpen && typeof _tsMiniLogToggle === 'function') {
        _tsMiniLogToggle();
    } else {
        const btn = document.getElementById('tsMiniLogBtn');
        if (btn) btn.style.boxShadow = '0 0 0 2px rgba(239,68,68,0.55)';
    }
}

// Post a non-error status line to the panel (e.g. run complete / no log
// produced). Clears the greeting so the panel reflects the run's end state
// instead of being stuck on the welcome bubbles. `accent` sets the left color.
function _tsMiniLogStatus(text, accent) {
    if (!text) return;
    const dest = document.getElementById('tsMiniLogContent');
    if (!dest) return;
    _tsMiniLogClearGreeting();
    const col = accent || '#34d399';
    const escaped = String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const row = document.createElement('div');
    row.dataset.tsLogline = '1';
    row.style.cssText = 'font-family:ui-monospace,monospace;font-size:10px;line-height:1.5;' +
                        `color:${col};background:rgba(148,163,184,0.06);` +
                        'border:0.5px solid rgba(148,163,184,0.18);border-radius:6px;' +
                        'padding:6px 9px;word-break:break-word;white-space:pre-wrap;';
    row.innerHTML = escaped;
    dest.appendChild(row);
    while (dest.querySelectorAll('[data-ts-logline]').length > 200) {
        dest.querySelector('[data-ts-logline]').remove();
    }
    dest.scrollTop = dest.scrollHeight;
    _tsMiniLogStreaming = true;
}

// ── Mini-chat Q&A — local KB, mirrors pose.js _miniAnswer ─────────────────
function _tsMiniLogAsk(preset) {
    const inp = document.getElementById('tsMiniLogInput');
    const q = (preset != null ? preset : (inp ? inp.value : '') || '').trim();
    if (!q) return;
    if (inp && preset == null) inp.value = '';
    _tsMiniLogBubble('you: ' + q, '#67e8f9');
    _tsMiniLogBubble(_tsMiniAnswer(q), '#cbd5e1');
}

// Append a chat bubble (not a log line) to the panel body
function _tsMiniLogBubble(text, color) {
    const dest = document.getElementById('tsMiniLogContent');
    if (!dest) return;
    const d = document.createElement('div');
    d.style.cssText = 'align-self:flex-start;max-width:92%;color:' + (color || '#cbd5e1') +
                      ';font-size:12px;line-height:1.6;padding:6px 9px;background:#0a0f1a;' +
                      'border:1px solid #1e293b;border-radius:8px;word-break:break-word;white-space:pre-wrap;';
    d.textContent = text;
    dest.appendChild(d);
    dest.scrollTop = dest.scrollHeight;
}

function _tsMiniAnswer(q) {
    q = q.toLowerCase();
    const KB = [
        { k: ['smarts', 'reaction', 'write', 'pattern', 'rxn'],
          a: 'A reaction SMARTS maps reactant atoms to product atoms with numbered labels. Format: [reactant1].[reactant2]>>[product]. Example (amide coupling): [#6:1](=[#8:2])[#8].[#7:3]>>[#6:1](=[#8:2])[#7:3] — the acid carbon (:1), carbonyl oxygen (:2), and amine nitrogen (:3) are tracked through the reaction. The atoms keeping the same map number on both sides are conserved; unmapped atoms (like the leaving -OH) are removed. The picker presets (rxn101 amide, rxn110 Suzuki, etc.) give you tested patterns to start from.' },
        { k: ['reagent', 'building block', 'bb', 'file', 'input', 'sdf', 'smi'],
          a: 'Reagents are building-block files keyed by reaction slot — e.g. rxn101_1 / rxn101_2 are the two slots for the amide coupling. Each slot is a .smi or .sdf list of compatible fragments. The TS run enumerates the combinatorial product of the slots, so a 67,735 × 40,416 amide pair gives ~2.7B candidate molecules. Point the config at your reagent directory; the eligible counts shown in the sidebar (after warmup masking) tell you how many survived RDKit parsing + the DisallowTracker.' },
        { k: ['warmup', 'belief', 'prior', 'μ', 'mu', 'sigma', 'σ', 'form'],
          a: 'Warmup forms the initial belief. It scores random reagent pairs and assigns each reagent slot a posterior N(μ, σ): μ is the mean reward seen so far, σ the uncertainty (high when n is low). These priors seed the Thompson Sampling phase, where each iteration draws a sample from every reagent\u2019s posterior and picks the combination with the best draw.' },
        { k: ['thompson', ' ts', 'sampling', 'draw', 'posterior', 'select'],
          a: 'Thompson Sampling balances exploration and exploitation. Each iteration: draw one random sample from every reagent\u2019s posterior N(μ, σ), assemble the highest-scoring combination from those draws, score the real molecule, then update the chosen reagents\u2019 posteriors with the observed reward. High-σ (uncertain) reagents occasionally win on a lucky draw — that\u2019s exploration; as σ shrinks with more observations, the search exploits the proven winners. The TS-draw value in the log is the sampled number, NOT the molecule score.' },
        { k: ['elion', 'score', 'estimator', 'sascore', 'qed', 'chembert', 'reward'],
          a: 'Scoring uses Elion estimators combined into the reward signal: SAScore (synthetic accessibility), QED (drug-likeness), and a ChemBERT property model. The eval score in the log is this composite for the assembled molecule; the TS posteriors are updated from it. Reagents that consistently produce high-reward molecules accumulate high μ and get sampled more often.' },
        { k: ['mask', 'disallow', 'eligible', 'count', 'fluctuat', 'retire', '34875', '34843'],
          a: 'The eligible count = initial slot size − masked. The DisallowTracker permanently retires a reagent once it exhausts its valid partner combinations or fails RDKit parsing. Because warmup samples reagents in a random order each run, slightly different sets get retired — so a fresh run lands at a slightly different eligible count (e.g. 34,875 vs 34,843). It\u2019s expected run-to-run variance, not a bug.' },
        { k: ['lag', 'slow', 'freeze', 'performance', '900', 'iteration'],
          a: 'Lag around 900+ iterations came from three accumulating sources: the iteration-log DOM growing unbounded (now capped at ~200 visible lines), the sparkline re-allocating temp arrays every frame (now a single O(n) pass), and the server history array growing without limit (now capped at 512 points). With those fixes the render cost stays flat regardless of iteration count.' },
    ];
    for (const e of KB)
        for (const kw of e.k)
            if (q.indexOf(kw) >= 0) return e.a;
    return 'I can explain: writing reaction SMARTS, reagent/building-block files, the Warmup belief phase, Thompson Sampling draws, Elion scoring (SAScore · QED · ChemBERT), the eligible-count masking, or the iteration lag fixes — ask about any of those.';
}

// ── Raw / WU log helpers ──────────────────────────────────────────────────
let _tsDebugVerbose = false;
function _tsRawLog(text) {
    if (!text) return;
    const isImportant = text.includes('ERROR') || text.includes('Traceback')
        || text.includes('[LOADER]') || text.includes('[WRAPPER]')
        || text.includes('[DEBUG]') || text.includes('ValueError');
    if (!isImportant && !_tsDebugVerbose) return;
    const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const color   = text.includes('ERROR') || text.includes('Traceback') ? '#ef4444' : '#475569';
    const html    = `<div style="font-size:9px;color:${color};font-family:monospace">${escaped}</div>`;
    const tsLog   = document.getElementById('tsIterLog') || document.getElementById('tsWuLog');
    if (tsLog) { tsLog.insertAdjacentHTML('beforeend', html); tsLog.scrollTop = tsLog.scrollHeight; }
    // Mirror genuine errors into the 💬 mini-log panel for debugging.
    if (text.includes('ERROR') || text.includes('Traceback') || text.includes('ValueError')) {
        if (typeof _tsMiniLogError === 'function') _tsMiniLogError(text);
    }
}

const _tsLogBuf = [], _TS_LOG_MAX = 150;
let _tsLogRaf = null;
function _tsWuLog(html) {
    _tsLogBuf.push(html);
    if (!_tsLogRaf) _tsLogRaf = requestAnimationFrame(_tsFlushLog);
}
function _tsFlushLog() {
    _tsLogRaf = null;
    if (!_tsLogBuf.length) return;
    const el = document.getElementById('tsWuLog');
    if (!el) { _tsLogBuf.length = 0; return; }
    const frag = document.createDocumentFragment();
    _tsLogBuf.splice(0).forEach(html => { const d = document.createElement('div'); d.innerHTML = html; frag.appendChild(d); });
    el.appendChild(frag);
    while (el.children.length > _TS_LOG_MAX) el.removeChild(el.firstChild);
    el.scrollTop = el.scrollHeight;
}

// ── Bar renderers ─────────────────────────────────────────────────────────
function _tsGetTop5() {
    const entries = Object.entries(_ts.reagents).sort((a,b) => b[1].best - a[1].best).slice(0,5);
    return new Set(entries.map(([id]) => id));
}

function _tsRenderWuBars() {
    const sorted = Object.entries(_ts.reagents).sort((a,b) => b[1].best - a[1].best).slice(0,5);
    const maxMu  = Math.max(...sorted.map(([,r]) => r.mu), 0.001);
    const el     = document.getElementById('tsWuBars');
    if (!el) return;
    el.innerHTML = sorted.map(([id, r]) => {
        const muPct   = (r.mu / maxMu * 100).toFixed(1);
        const stdPct  = Math.min((r.std / maxMu) * 100, 18).toFixed(1);
        const stdLeft = Math.max(0, +muPct - +stdPct / 2).toFixed(1);
        const stdNorm = Math.min(r.std / (r.mu || 1), 1);
        const stdColor = stdNorm < 0.15 ? '#34d399' : stdNorm < 0.4 ? '#fbbf24' : '#94a3b8';
        const best    = (r.best ?? r.mu).toFixed(4);
        const partner = r.bestPartner || '—';
        return `<div style="padding:5px 0;border-bottom:0.5px solid rgba(148,163,184,0.07)">
          <div style="display:grid;grid-template-columns:130px 1fr auto;align-items:center;gap:8px;margin-bottom:3px">
            <span style="font-family:monospace;font-size:10px;color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${id}">${id}</span>
            <div style="position:relative;height:12px;background:rgba(30,41,59,0.8);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${muPct}%;background:#7c3aed;border-radius:3px;transition:width 0.55s ease"></div>
              <div style="position:absolute;top:2px;height:8px;left:${stdLeft}%;width:${stdPct}%;background:rgba(180,180,80,0.38);border-radius:2px"></div>
            </div>
            <span style="font-size:11px;font-weight:500;color:#34d399;white-space:nowrap">${best}</span>
          </div>
          <div style="display:grid;grid-template-columns:130px 1fr;gap:8px">
            <span></span>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
              <span style="font-size:9px;padding:1px 5px;border-radius:3px;background:rgba(124,58,237,0.15);color:#a78bfa">n=${r.count}</span>
              <span style="font-size:9px;color:#94a3b8">μ=${r.mu.toFixed(4)}</span>
              ${r.count >= 2 ? `<span style="font-size:9px;color:${stdColor}">σ=${r.std.toFixed(4)}</span>` : `<span style="font-size:9px;color:#1e3a5f">σ=—</span>`}
              <span style="font-size:9px;color:#1e3a5f">⊕</span>
              <span style="font-family:monospace;font-size:9px;color:#0f766e;white-space:nowrap">${partner}</span>
            </div>
          </div>
        </div>`;
    }).join('');
    const n    = Object.keys(_ts.reagents).length;
    const best = sorted.length ? sorted[0][1].best : 0;
    const set  = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('tsWuReagents', n);
    set('tsWuBest', best > 0 ? best.toFixed(3) : '—');
    const idx = _ts._activeJobIdx ?? 0;
    set(`tsWuReagents_${idx}`, n);
    if (best > 0) set(`tsWuBest_${idx}`, best.toFixed(4));
    const barEl = document.getElementById(`tsWuBar_${idx}`);
    if (barEl && best > 0) barEl.style.width = Math.min(100, best * 10).toFixed(1) + '%';
}

// Compute the pre-sorted top-5 bars from a reagents dict — mirrors the worker's
// _emitBarsReady geometry. Used on reconnect to seed bars from the restored JSON
// reagents before any live post-update arrives. (Live updates come pre-computed
// from the worker; this is only the cold-start path.)
function _tsComputeBarsFromReagents(reagents) {
    if (!reagents) return [];
    const arr = Object.values(reagents);
    if (!arr.length) return [];
    // Rank by best molecule score (monotonic; matches the live worker), with
    // unscored reagents ordered by μ below all scored ones.
    const scored   = arr.filter(r => r.best != null).sort((a, b) => b.best - a.best);
    const unscored = arr.filter(r => r.best == null).sort((a, b) => b.mu - a.mu);
    const top5 = [...scored, ...unscored].slice(0, 5);
    const maxMu = Math.max(...top5.map(r => r.mu), 0.001);
    return top5.map(r => {
        const muPct   = +(r.mu / maxMu * 100).toFixed(1);
        const stdPct  = +Math.min(((r.std || 0) / maxMu) * 100, 18).toFixed(1);
        const stdLeft = +Math.max(0, muPct - stdPct / 2).toFixed(1);
        return {
            id: r.name, name: r.name, mu: r.mu, std: r.std || 0, sc: r.sc || 0,
            best: r.best, bestPartner: r.bestPartner || '—', delta_mu: r.delta_mu || 0,
            muPct, stdPct, stdLeft,
        };
    });
}

// SMILES client-side cache: reagent id -> smiles string ('' = none, undefined = not fetched)
const _tsSmilesCache = {};
function _tsFetchSmiles(rxnKey, id) {
    if (!rxnKey || id == null) return;
    if (_tsSmilesCache[id] !== undefined) return;        // already fetched (or in-flight)
    _tsSmilesCache[id] = '';                              // mark in-flight to avoid dupes
    fetch(`/vina_visualization/ts_mol_smiles/${encodeURIComponent(rxnKey)}/${encodeURIComponent(id)}`)
        .then(r => r.ok ? r.json() : { smiles: '' })
        .then(d => {
            _tsSmilesCache[id] = d.smiles || '';
            // Patch any visible row for this id without a full re-render
            const el = document.getElementById('tsTsBars');
            if (!el) return;
            const cell = el.querySelector(`[data-smiles-for="${CSS.escape(String(id))}"]`);
            if (cell && _tsSmilesCache[id]) cell.textContent = _tsSmilesCache[id];
        })
        .catch(() => { _tsSmilesCache[id] = ''; });
}

// PRODUCT cache: keyed "rid|partner" -> product SMILES.
//   undefined = not fetched, '' = fetched-but-no-clean-product (show fallback),
//   non-empty = the reaction product SMILES of rid ⊕ partner.
// The <code> cell shows the PRODUCT (rid reacted with its partner under the
// reaction SMARTS), not rid's own building-block SMILES. Cells are tagged
// data-prod-for="rid|partner" so the async result patches the right row.
const _tsProductCache = {};
function _tsProdKey(rid, partner) { return String(rid) + '|' + String(partner); }
function _tsFetchProduct(rxnKey, rid, partner) {
    if (!rxnKey || rid == null || !partner || partner === '—') return;
    const key = _tsProdKey(rid, partner);
    if (_tsProductCache[key] !== undefined) return;      // already fetched / in-flight
    _tsProductCache[key] = '';                            // mark in-flight
    fetch(`/vina_visualization/ts_product_smiles/${encodeURIComponent(rxnKey)}/${encodeURIComponent(rid)}/${encodeURIComponent(partner)}`)
        .then(r => r.ok ? r.json() : { smiles: '' })
        .then(d => {
            _tsProductCache[key] = d.smiles || '';
            const el = document.getElementById('tsTsBars');
            if (!el) return;
            const cell = el.querySelector(`[data-prod-for="${CSS.escape(key)}"]`);
            if (cell) {
                if (_tsProductCache[key]) {
                    cell.textContent = _tsProductCache[key];
                    cell.title = `Product: ${rid} ⊕ ${partner}`;
                    cell.style.color = '#67e8f9';
                } else {
                    cell.textContent = 'no product';
                    cell.title = `${rid} ⊕ ${partner} — no clean product under this reaction`;
                    cell.style.color = '#475569';
                }
            }
        })
        .catch(() => { _tsProductCache[key] = ''; });
}

// ── Reaction hover-card ────────────────────────────────────────────────────
// Hovering a reagent tile pops a reaction scheme: this building block + its
// best-scoring partner → the rendered product, with each reactant's BB id and
// source CSV path, and the product's Elion score. Mirrors the uploaded scheme
// (substrate + reagent library → product). CLICK a tile to PIN the card so its
// paths/score can be selected and copied; click the ✕ or outside to unpin.
let _tsHoverCardEl   = null;
let _tsHoverPinned   = false;
let _tsHoverPinKey   = null;   // `${rid}|${partner}` of the pinned card
const _tsCsvCache    = {};     // `${rxn}/${id}` -> {csv_file, csv_path, slot}

function _tsHoverCardEnsure() {
    if (_tsHoverCardEl && document.body.contains(_tsHoverCardEl)) return _tsHoverCardEl;
    const c = document.createElement('div');
    c.id = '_tsHoverCard';
    c.style.cssText = [
        'position:fixed', 'z-index:100000', 'display:none', 'pointer-events:none',
        'background:#0a0f1e', 'border:1px solid rgba(71,85,105,0.6)', 'border-radius:10px',
        'box-shadow:0 12px 40px rgba(0,0,0,0.7)', 'padding:12px 14px', 'max-width:600px',
    ].join(';');
    document.body.appendChild(c);
    _tsHoverCardEl = c;
    // Click anywhere outside a PINNED card closes it (tile clicks stopPropagation,
    // so they re-pin instead of closing).
    if (!document._tsHoverPinCloser) {
        document._tsHoverPinCloser = true;
        document.addEventListener('click', (e) => {
            if (_tsHoverPinned && _tsHoverCardEl && !_tsHoverCardEl.contains(e.target)) _tsHoverUnpin();
        });
    }
    return c;
}

// Fetch + fill a reagent's source CSV path into the given span.
function _tsCsvLabel(rxnKey, id, spanId) {
    const key   = rxnKey + '/' + id;
    const patch = (info) => {
        const el = document.getElementById(spanId);
        if (!el || !info) return;
        el.textContent = info.csv_path || info.csv_file || '—';
        if (info.slot >= 0) el.textContent += `  (slot ${info.slot})`;
    };
    if (_tsCsvCache[key]) { patch(_tsCsvCache[key]); return; }
    fetch(`/vina_visualization/ts_mol_smiles/${encodeURIComponent(rxnKey)}/${encodeURIComponent(id)}`)
        .then(r => r.ok ? r.json() : null)
        .then(info => { if (info) { _tsCsvCache[key] = info; patch(info); } })
        .catch(() => {});
}

// Build the card's content. `pinned` => interactive: selectable text + ✕ button.
function _tsHoverBuild(c, rxnKey, rid, partner, score, pinned) {
    const hasPartner = partner && partner !== '—';
    const scoreTxt   = (score != null && !isNaN(score)) ? Number(score).toFixed(4) : '—';
    // Transparent images matching the inline tiles — RDKit renders light-on-dark,
    // so a white background washes the structures out; let the dark card show through.
    const imgStyle = 'display:block;margin:0 auto;pointer-events:none';
    const sel      = pinned ? 'user-select:all;-webkit-user-select:all;cursor:text' : '';

    const molImg = (id, w, h) =>
        `<object type="image/svg+xml" data="/vina_visualization/ts_mol_svg/${encodeURIComponent(rxnKey)}/${encodeURIComponent(id)}?w=${w}&h=${h}"
                 width="${w}" height="${h}" style="${imgStyle}"></object>`;
    const prodImg = hasPartner
        ? `<object type="image/svg+xml" data="/vina_visualization/ts_product_svg/${encodeURIComponent(rxnKey)}/${encodeURIComponent(rid)}/${encodeURIComponent(partner)}?w=180&h=96"
                   width="180" height="96" style="${imgStyle}"></object>`
        : `<div style="width:180px;height:96px;display:flex;align-items:center;justify-content:center;color:#475569;font-size:11px">no partner yet</div>`;

    const idLine   = (id)     => `<div style="font-family:monospace;font-size:11px;color:#e2e8f0;font-weight:600;${sel}">BB ${id}</div>`;
    const pathLine = (spanId) => `<div id="${spanId}" style="font-family:monospace;font-size:9px;color:#64748b;max-width:150px;word-break:break-all;text-align:center;line-height:1.3;${sel}">…</div>`;
    const op       = (ch)     => `<div style="font-size:22px;color:#475569;align-self:center;padding:0 2px">${ch}</div>`;
    const reactant = (id, spanId) =>
        `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">${molImg(id,120,72)}${idLine(id)}${pathLine(spanId)}</div>`;

    const partnerCol = hasPartner
        ? reactant(partner, '_tsHovCsvB')
        : `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
             <div style="width:120px;height:72px;display:flex;align-items:center;justify-content:center;color:#475569;font-size:11px;border:0.5px dashed #334155;border-radius:4px">—</div>
             <div style="font-family:monospace;font-size:11px;color:#475569">no partner</div></div>`;

    const closeBtn = pinned
        ? `<button onclick="_tsHoverUnpin()" title="Close"
                   style="position:absolute;top:6px;right:8px;background:none;border:none;color:#64748b;font-size:14px;cursor:pointer;pointer-events:auto;padding:2px 4px">✕</button>`
        : '';
    const header = pinned
        ? `Reaction · ${rxnKey} <span style="color:#22d3ee">· 📌 pinned — select to copy</span>`
        : `Reaction · ${rxnKey}`;

    c.innerHTML = `
      ${closeBtn}
      <div style="font-size:10px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em">${header}</div>
      <div style="display:flex;align-items:flex-start;gap:6px">
        ${reactant(rid, '_tsHovCsvA')}
        ${op('+')}
        ${partnerCol}
        ${op('→')}
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
          ${prodImg}
          <div style="font-size:10px;color:#94a3b8;font-weight:600">Products</div>
          <div style="font-size:13px;color:#f59e0b;font-weight:700;${sel}">Elion score: ${scoreTxt}</div>
        </div>
      </div>`;

    _tsCsvLabel(rxnKey, rid, '_tsHovCsvA');
    if (hasPartner) _tsCsvLabel(rxnKey, partner, '_tsHovCsvB');
}

function _tsHoverPosition(c, targetEl) {
    const t = targetEl.getBoundingClientRect();
    let left = t.right + 12, top = t.top;
    const cw = c.offsetWidth, ch = c.offsetHeight;
    if (left + cw > window.innerWidth  - 8) left = Math.max(8, t.left - cw - 12);
    if (top  + ch > window.innerHeight - 8) top  = Math.max(8, window.innerHeight - ch - 8);
    c.style.left = left + 'px';
    c.style.top  = top + 'px';
}

// Hover preview (non-interactive; hides on mouseleave). Does not disturb a pin.
function _tsHoverCard(ev, rxnKey, rid, partner, score) {
    if (_tsHoverPinned) return;
    if (!rxnKey || rid == null || rid === '') return;
    const c = _tsHoverCardEnsure();
    c.style.pointerEvents = 'none';
    _tsHoverBuild(c, rxnKey, rid, partner, score, false);
    c.style.display = 'block';
    _tsHoverPosition(c, ev.currentTarget);
}
function _tsHoverCardHide() { if (!_tsHoverPinned && _tsHoverCardEl) _tsHoverCardEl.style.display = 'none'; }

// Click to PIN (interactive/copyable). Clicking the same pinned tile toggles off.
function _tsHoverPin(ev, rxnKey, rid, partner, score) {
    if (ev) ev.stopPropagation();
    if (!rxnKey || rid == null || rid === '') return;
    const key = String(rid) + '|' + String(partner);
    if (_tsHoverPinned && _tsHoverPinKey === key) { _tsHoverUnpin(); return; }
    const c = _tsHoverCardEnsure();
    _tsHoverPinned = true;
    _tsHoverPinKey = key;
    c.style.pointerEvents = 'auto';
    _tsHoverBuild(c, rxnKey, rid, partner, score, true);
    c.style.display = 'block';
    _tsHoverPosition(c, ev.currentTarget);
}
function _tsHoverUnpin() {
    _tsHoverPinned = false;
    _tsHoverPinKey = null;
    if (_tsHoverCardEl) { _tsHoverCardEl.style.display = 'none'; _tsHoverCardEl.style.pointerEvents = 'none'; }
}

window._tsHoverCard     = _tsHoverCard;
window._tsHoverCardHide = _tsHoverCardHide;
window._tsHoverPin      = _tsHoverPin;
window._tsHoverUnpin    = _tsHoverUnpin;

// STATIC pool counts: rxn_key -> [n_slot0, n_slot1] (raw CSV building-block
// counts). Unlike the live "competitors" eligible count (which shrinks as the
// DisallowTracker masks reagents and so fluctuates), this is fixed for a run.
// We show this fixed number in the sidebar pool fields.
const _tsPoolCountCache = {};
function _tsFetchPoolCounts(rxnKey, ji) {
    if (!rxnKey) return;
    const apply = (counts) => {
        if (!counts || !counts.length) return;
        // Write to the per-job spans (multi-job) AND the single-job spans.
        const ids0 = [`tsTsPool0_j${ji}`, ...(ji === 0 ? ['tsTsPool0'] : [])];
        const ids1 = [`tsTsPool1_j${ji}`, ...(ji === 0 ? ['tsTsPool1'] : [])];
        ids0.forEach(id => { const e = document.getElementById(id); if (e && counts[0] != null) e.textContent = String(counts[0]); });
        ids1.forEach(id => { const e = document.getElementById(id); if (e && counts[1] != null) e.textContent = String(counts[1]); });
    };
    if (_tsPoolCountCache[rxnKey]) { apply(_tsPoolCountCache[rxnKey]); return; }
    fetch(`/vina_visualization/ts_pool_counts/${encodeURIComponent(rxnKey)}`)
        .then(r => r.ok ? r.json() : { counts: [] })
        .then(d => { _tsPoolCountCache[rxnKey] = d.counts || []; apply(_tsPoolCountCache[rxnKey]); })
        .catch(() => {});
}

// ── Top-5 polling from the session JSON (via /ts_top5) ──────────────────────
// The reagent bars are driven by polling the backend's persisted top-5 every
// 5 seconds rather than computing them live in the worker. The backend maintains
// top5 ranked by best molecule score (authoritative, from [evaluate]), so this
// guarantees the UI matches the JSON exactly and never shows Thompson-sample
// inflated rankings. One poll covers all jobs; we pick the active job's top5.
let _tsTop5Timer = null;
let _tsTop5ByJob = {};   // launch_idx -> top5 array from last poll
// Previous rank position of each reagent per job, to show movement arrows:
// _tsPrevRank[jobIdx][reagentId] = position index (0 = top) at last render.
let _tsPrevRank = {};

function _tsTop5ToBars(top5, jobIdx) {
    if (!Array.isArray(top5) || !top5.length) return [];
    const maxMu = Math.max(...top5.map(t => t.mu || 0), 0.001);
    // Previous positions for this job (id -> rank index). undefined on first poll.
    const prev = (jobIdx != null && _tsPrevRank[jobIdx]) ? _tsPrevRank[jobIdx] : null;
    const bars = top5.map((t, pos) => {
        const muPct   = +((t.mu || 0) / maxMu * 100).toFixed(1);
        const stdPct  = +Math.min(((t.std || 0) / maxMu) * 100, 18).toFixed(1);
        const stdLeft = +Math.max(0, muPct - stdPct / 2).toFixed(1);
        if (t.smiles) _tsSmilesCache[t.name] = t.smiles;   // prime SMILES cache
        // Movement vs the previous render:
        //   'up'    — moved to a higher position (smaller index) OR newly entered
        //   'down'  — moved to a lower position (larger index)
        //   'same'  — unchanged position (no arrow shown)
        let move = 'same';
        if (prev) {
            const prevPos = prev[t.name];
            if (prevPos === undefined)      move = 'up';     // newly entered the list
            else if (pos < prevPos)         move = 'up';
            else if (pos > prevPos)         move = 'down';
            else                            move = 'same';
        }
        // On the very first poll (no prev), default to 'same' (no arrow) — we
        // don't know movement yet.
        return {
            id: t.name, name: t.name, mu: t.mu || 0, std: t.std || 0,
            sc: t.sc || 0, best: t.best,
            bestPartner: t.partner || '—', delta_mu: 0,
            muPct, stdPct, stdLeft, move,
        };
    });
    // Record current positions as the new "previous" for the next poll.
    if (jobIdx != null) {
        const cur = {};
        top5.forEach((t, pos) => { cur[t.name] = pos; });
        _tsPrevRank[jobIdx] = cur;
    }
    return bars;
}

function _tsPollTop5() {
    fetch('/vina_visualization/ts_top5')
        .then(r => r.ok ? r.json() : { jobs: [] })
        .then(d => {
            const jobs = d.jobs || [];
            _ts._top5StateDir = d.state_dir || '';
            _ts._top5Polled   = true;
            _tsTop5ByJob = {};
            if (!_ts._jobBars) _ts._jobBars = {};
            // Convert each job's top5 ONCE (so _tsPrevRank updates exactly once
            // per job per poll), storing the resulting bars per job.
            jobs.forEach(j => {
                const ji = j.launch_idx ?? 0;
                _tsTop5ByJob[ji] = j.top5 || [];
                _ts._jobBars[ji] = _tsTop5ToBars(j.top5 || [], ji);
            });
            // Render the currently-active job's bars
            const idx = _ts._activeJobIdx ?? 0;
            _ts._activeBars = _ts._jobBars[idx] || [];
            _tsDebugSend('poll_top5', {
                jobs: jobs.length,
                perJob: jobs.map(j => ({ idx: j.launch_idx, st: j.status, n: (j.top5 || []).length })),
                activeJobIdx: idx,
                jobBarsKeys: Object.keys(_ts._jobBars),
                activeBars: _ts._activeBars.length,
                elFound: !!document.getElementById('tsTsBars'),
            });
            _tsQueueBarsRender();
        })
        .catch(e => _tsDebugSend('poll_top5_failed', { err: String(e) }));
}

// ── Debug bridge ──────────────────────────────────────────────────────────
// Mirrors the browser end of the reagent-panel chain into the server's debug
// directory, so <debug>/client.log and <debug>/top5.log interleave and one file
// listing shows where the data stopped. Fire-and-forget: never awaited, never
// throws, and silently inert when the endpoint is absent or debug is off.
// Turn off from the console with `_ts._debugOff = true`.
function _tsDebugSend(tag, data) {
    try {
        if (_ts._debugOff) return;
        fetch('/vina_visualization/ts_debug_client', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tag: tag, data: data }),
            keepalive: true,
        }).catch(() => {});
    } catch (_) { /* debug must never break a run */ }
}
window._tsDebugSend = _tsDebugSend;

function _tsStartTop5Polling() {
    if (_tsTop5Timer) return;          // already polling
    _tsPollTop5();                     // immediate first fetch
    _tsTop5Timer = setInterval(_tsPollTop5, 5000);   // then every 5s
}

function _tsStopTop5Polling() {
    if (_tsTop5Timer) { clearInterval(_tsTop5Timer); _tsTop5Timer = null; }
}

function _tsRenderTsBars() {
    // Bars come from the 5-second /ts_top5 poll (backend's persisted best-score
    // ranking). The main thread just paints _ts._activeBars.
    const bars = _ts._activeBars;
    const el = document.getElementById('tsTsBars');
    if (!el) return;
    _tsDebugSend('render_bars', {
        bars: (bars || []).length,
        polled: !!_ts._top5Polled,
        rowsNow: el.querySelectorAll('[data-el="ridLabel"]').length,
        activeJobIdx: _ts._activeJobIdx ?? 0,
        jobs: (_ts.jobs || []).map(j => j.key || j.short_name || '?'),
        sig: el._tsIdSig || null,
        offsetHeight: el.offsetHeight,
        display: getComputedStyle(el).display,
    });
    if (!bars || !bars.length) {
        // Bailing silently here is what made an empty reagent panel unreadable:
        // no rows, no message, nothing to distinguish "the run hasn't produced
        // a session yet" from "/ts_top5 is scanning the wrong directory because
        // ELION_CWD is mis-resolved". Say which directory was scanned.
        if (_ts._top5Polled && !el.querySelector('[data-el="ridLabel"]')) {
            const dir = _ts._top5StateDir || '(unknown)';
            el.innerHTML =
                '<div id="tsTsBarsEmpty" style="padding:18px 6px;font-size:11px;color:#475569;line-height:1.7">' +
                '<div style="color:#64748b;font-weight:600;margin-bottom:4px">No reagent rankings yet</div>' +
                'Rows appear once a run writes a session file. Scanned: ' +
                '<code style="color:#64748b;word-break:break-all">' + dir + '</code>' +
                '<div style="margin-top:6px">If that path looks wrong, it comes from ' +
                '<code style="color:#64748b">visualizer.output_dir</code> in the engine\'s ' +
                '<code style="color:#64748b">input_TS.yml</code> — check <code style="color:#64748b">ELION_CWD</code>.</div>' +
                '</div>';
        }
        return;
    }
    if (el.querySelector('#tsTsBarsEmpty')) {
        // Drop the placeholder AND the cached signature together. Clearing the
        // markup alone would let the `sameSet` branch below patch children that
        // no longer exist, leaving the panel permanently blank.
        el.innerHTML = '';
        el._tsIdSig  = null;
    }

    const activeJob = (_ts.jobs || [])[_ts._activeJobIdx ?? 0] || {};
    const rxnKey    = activeJob.key || activeJob.short_name || '';

    // Signature of the visible reagent SET + partners — if unchanged, update
    // numbers in place; if a reagent or its partner changed, rebuild (so the
    // partner's structure image refreshes).
    const idSig = bars.map(b => b.name + '>' + (b.bestPartner || '—')).join('|') + '::' + rxnKey;
    const sameSet = (el._tsIdSig === idSig);

    if (sameSet) {
        bars.forEach((b, i) => {
            const row = el.children[i];
            if (!row) return;
            const set = (sel, txt) => { const e = row.querySelector(sel); if (e) e.textContent = txt; };
            set('[data-el="nVal"]',  'n=' + (b.sc || 0));
            const sEl = row.querySelector('[data-el="sVal"]');
            if (sEl) sEl.textContent = (b.sc >= 2) ? 'σ=' + (b.std || 0).toFixed(4) : 'σ=—';
            // Right-side best score with movement arrow
            const bestEl = row.querySelector('[data-el="bestVal"]');
            if (bestEl) {
                const arrow = b.move === 'up'   ? '<span style="color:#34d399" title="moved up">⬆</span>'
                            : b.move === 'down' ? '<span style="color:#f87171" title="moved down">⬇</span>'
                            : '';
                const bestText = (b.best != null) ? b.best.toFixed(4) : '—';
                bestEl.innerHTML = arrow + bestText;
            }
            // Ensure the reaction product is fetched (no-op if cached). The
            // data-prod-for cell already exists in this row; _tsFetchProduct
            // patches it when the result arrives.
            const partner = b.bestPartner || '—';
            if (partner && partner !== '—') _tsFetchProduct(rxnKey, b.name, partner);
        });
        return;
    }
    el._tsIdSig = idSig;

    el.innerHTML = bars.map(b => {
        const sc      = b.sc || 0;
        const std     = b.std || 0;
        const stdNorm = Math.min(std / (b.mu || 1), 1);
        const stdColor = stdNorm < 0.15 ? '#34d399' : stdNorm < 0.4 ? '#fbbf24' : '#94a3b8';
        const partner = b.bestPartner || '—';
        const rid     = b.name;
        // Show the reaction PRODUCT (rid ⊕ partner under the reaction SMARTS) in
        // place of rid's own building-block SMILES. Fetched async; shows '…'
        // until it arrives, then patched to the product or 'no product'.
        const prodKey  = _tsProdKey(rid, partner);
        const prodSmi  = _tsProductCache[prodKey];
        if (partner && partner !== '—') _tsFetchProduct(rxnKey, rid, partner);
        const smilesText = (partner === '—')
            ? '—'
            : (prodSmi === undefined ? '…'
               : (prodSmi ? prodSmi : 'no product'));
        const smilesColor = (prodSmi && prodSmi.length) ? '#67e8f9' : '#475569';

        // Right-side value: BEST molecule score, prefixed with a movement arrow
        // showing how this building block's rank changed since the last 5s refresh.
        const arrow = b.move === 'up'   ? '<span style="color:#34d399" title="moved up">⬆</span>'
                    : b.move === 'down' ? '<span style="color:#f87171" title="moved down">⬇</span>'
                    : '';   // unchanged position → no arrow
        const bestText = (b.best != null) ? b.best.toFixed(4) : '—';

        // RDKit 2D structure for this building block (id column), with the id
        // shown underneath. The <object> loads the rendered SVG; if the id has
        // no SMILES (204) the inner fallback text shows instead.
        const molUrl = rxnKey ? `/vina_visualization/ts_mol_svg/${encodeURIComponent(rxnKey)}/${encodeURIComponent(rid)}?w=96&h=48` : '';
        const molBlock = molUrl
            ? `<object type="image/svg+xml" data="${molUrl}" width="96" height="48"
                       style="pointer-events:none;display:block;margin:0 auto" aria-label="structure of ${rid}"></object>`
            : '';

        // Partner building block — the reagent that reacts with this one. Show its
        // own small structure above its id so the pairing is visible.
        const partnerUrl = (rxnKey && partner && partner !== '—')
            ? `/vina_visualization/ts_mol_svg/${encodeURIComponent(rxnKey)}/${encodeURIComponent(partner)}?w=80&h=40` : '';
        const partnerBlock = (partner && partner !== '—')
            ? `<span style="display:inline-flex;flex-direction:column;align-items:center;gap:1px;vertical-align:middle">
                 ${partnerUrl ? `<object type="image/svg+xml" data="${partnerUrl}" width="80" height="40" style="pointer-events:none;display:block" aria-label="structure of ${partner}"></object>` : ''}
                 <span style="font-family:monospace;font-size:9px;color:#0f766e">${partner}</span>
               </span>`
            : `<span style="font-family:monospace;font-size:9px;color:#1e3a5f">—</span>`;

        return `<div style="padding:6px 0;border-bottom:0.5px solid rgba(148,163,184,0.07)">
          <div style="display:grid;grid-template-columns:104px 1fr auto;align-items:start;gap:10px">
            <div style="display:flex;flex-direction:column;align-items:center;gap:2px;cursor:help"
                 onmouseenter="_tsHoverCard(event,'${rxnKey}','${rid}','${partner}',${b.best != null ? b.best : 'null'})"
                 onmouseleave="_tsHoverCardHide()"
                 onclick="_tsHoverPin(event,'${rxnKey}','${rid}','${partner}',${b.best != null ? b.best : 'null'})">
              ${molBlock}
              <span data-el="ridLabel" style="font-family:monospace;font-size:10px;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100px">${rid}</span>
            </div>
            <div style="display:flex;flex-direction:column;gap:3px;min-width:0">
              <code data-prod-for="${prodKey}" title="Product: ${rid} ⊕ ${partner}"
                    style="font-family:monospace;font-size:11px;color:${smilesColor};background:rgba(8,145,178,0.08);
                           border:0.5px solid rgba(8,145,178,0.18);border-radius:4px;padding:3px 7px;
                           word-break:break-all;line-height:1.4;user-select:all;cursor:text">${smilesText}</code>
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                <span data-el="nVal" style="font-size:9px;padding:1px 5px;border-radius:3px;background:rgba(8,145,178,0.15);color:#67e8f9">n=${sc}</span>
                <span style="font-size:9px;color:#94a3b8">μ=${b.mu.toFixed(4)}</span>
                <span data-el="sVal" style="font-size:9px;color:${(sc>=2)?stdColor:'#1e3a5f'}">${sc >= 2 ? 'σ='+std.toFixed(4) : 'σ=—'}</span>
                <span style="font-size:9px;color:#64748b" title="reacts with">⊕ reacts with</span>
                ${partnerBlock}
              </div>
            </div>
            <span data-el="bestVal" title="Best single molecule score"
                  style="font-size:11px;font-weight:600;color:#f59e0b;white-space:nowrap;display:inline-flex;align-items:center;gap:2px">${arrow}${bestText}</span>
          </div>
        </div>`;
    }).join('');
}

function _tsRenderTsWinners() {
    const el = document.getElementById('tsTsWinners');
    if (!el) return;
    el.innerHTML = _ts.tsState.winners.slice(-5).reverse().map(w =>
        `<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:6px 8px;margin-bottom:4px">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span style="font-size:9px;color:#64748b">iter ${w.iter}</span>
            <span style="font-size:11px;font-weight:500;color:#34d399">${w.score.toFixed(4)}</span>
          </div>
          <div style="font-family:monospace;font-size:9px;color:#67e8f9;word-break:break-all">${w.smiles || '—'}</div>
        </div>`
    ).join('');
}

// ── Throttled RAF gates ───────────────────────────────────────────────────
let _tsBarsRafPending    = false;
let _tsWinnersRafPending = false;
function _tsQueueBarsRender() {
    if (_tsBarsRafPending) return;
    _tsBarsRafPending = true;
    requestAnimationFrame(() => { _tsBarsRafPending = false; _tsRenderTsBars(); });
}
function _tsQueueWinnersRender() {
    if (_tsWinnersRafPending) return;
    _tsWinnersRafPending = true;
    requestAnimationFrame(() => { _tsWinnersRafPending = false; _tsRenderTsWinners(); });
}

// ── Multi-job pool injection ──────────────────────────────────────────────
function _tsInjectMultiJobPools(jobs) {
    const pool0 = document.getElementById('tsTsPool0');
    if (!pool0) return;
    const existingSection = pool0.closest('div[id="_tsSidebarExtra"], div') || pool0.parentElement?.parentElement;
    if (!existingSection) return;
    const palette   = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    const container = document.createElement('div');
    container.id = '_tsMultiPoolContainer';
    jobs.forEach((job, ji) => {
        const color     = palette[ji % palette.length];
        const shortName = job.short_name || job.key || `job${ji}`;
        const bbNames   = job.bb_names || [];
        const bb0 = bbNames[0] || 'slot 0', bb1 = bbNames[1] || 'slot 1';
        const sec = document.createElement('div');
        sec.style.cssText = 'padding:10px 16px 12px;border-bottom:0.5px solid rgba(30,41,59,0.7)';
        sec.innerHTML = `
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
            <span style="width:7px;height:7px;border-radius:50%;background:${color};display:inline-block"></span>
            <p style="font-size:10px;font-weight:600;color:${color};text-transform:uppercase;letter-spacing:0.05em;margin:0">${shortName}</p>
          </div>
          <div style="display:flex;flex-direction:column;gap:4px;font-size:11px">
            <div style="display:flex;justify-content:space-between"><span style="color:#475569" title="${bb0}.csv">BB · ${bb0}</span><span id="tsTsPool0_j${ji}" style="color:#22d3ee;font-weight:600">—</span></div>
            <div style="display:flex;justify-content:space-between;font-size:10px"><span style="color:#334155">μ</span><span id="tsTsPool0MuRange_j${ji}" style="color:#34d399;font-family:monospace">—</span></div>
            <div style="display:flex;justify-content:space-between;font-size:10px"><span style="color:#334155">σ</span><span id="tsTsPool0StdRange_j${ji}" style="color:#f59e0b;font-family:monospace">—</span></div>
            <div style="display:flex;justify-content:space-between;margin-top:3px"><span style="color:#475569" title="${bb1}.csv">BB · ${bb1}</span><span id="tsTsPool1_j${ji}" style="color:#22d3ee;font-weight:600">—</span></div>
            <div style="display:flex;justify-content:space-between;font-size:10px"><span style="color:#334155">μ</span><span id="tsTsPool1MuRange_j${ji}" style="color:#34d399;font-family:monospace">—</span></div>
            <div style="display:flex;justify-content:space-between;font-size:10px"><span style="color:#334155">σ</span><span id="tsTsPool1StdRange_j${ji}" style="color:#f59e0b;font-family:monospace">—</span></div>
          </div>`;
        container.appendChild(sec);
        // Fill the pool numbers with the STATIC raw CSV counts (not the live
        // fluctuating eligible count). job.key is the rxn_key (e.g. rxn110_suzuki).
        _tsFetchPoolCounts(job.key || job.short_name || '', ji);
    });
    const sidebarExtra = document.getElementById('_tsSidebarExtra');
    if (sidebarExtra) {
        // Idempotent insert: clear any prior multi-job container AND the static
        // single-job "Competition pool" div first. Without this, a second call
        // (tab reopen / reconnect) leaves the static block in place above the
        // per-job sections — the duplicate "COMPETITION POOL / slot 0 / slot 1"
        // block seen on reopen.
        sidebarExtra.querySelector('#_tsMultiPoolContainer')?.remove();
        // The static pool div is the .px-4 block that contains tsTsPool0Label.
        const staticPool = sidebarExtra.querySelector('#tsTsPool0Label')?.closest('div[class*="px-4"]');
        if (staticPool) staticPool.remove();
        sidebarExtra.prepend(container);
    }
}

function _tsBuildWuPanels(jobs) {
    const list = document.getElementById('tsWuProgressList');
    if (!list) return;
    const palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    if (!jobs || jobs.length <= 1) {
        const def = document.getElementById('tsWuPanel_default');
        if (def) def.style.display = '';
        return;
    }
    const def = document.getElementById('tsWuPanel_default');
    if (def) def.style.display = 'none';
    list.querySelectorAll('[data-wu-panel]').forEach(el => el.remove());
    jobs.forEach((job, i) => {
        const color = palette[i % palette.length];
        const key   = job.key || `job${i}`;
        const sid   = job.job_id.slice(0, 6);
        const panel = document.createElement('div');
        panel.dataset.wuPanel = job.job_id;
        panel.style.cssText = 'padding:10px 16px 12px;';
        panel.innerHTML = `
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:7px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></span>
            <span style="font-size:10px;font-weight:700;color:${color};text-transform:uppercase;letter-spacing:0.05em">${key}</span>
            <span style="font-size:9px;color:#1e3a5f;margin-left:auto">${sid}</span>
          </div>
          <div style="display:flex;flex-direction:column;gap:5px;font-size:11px">
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Phase</span><span id="tsWuPhase_${i}" style="color:#a78bfa;font-weight:600">—</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Combos evaluated</span><span id="tsWuEvals_${i}" style="color:#94a3b8">0</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Active reagents</span><span id="tsWuReagents_${i}" style="color:#94a3b8">0</span></div>
            <div style="display:flex;justify-content:space-between"><span style="color:#475569">Best score</span><span id="tsWuBest_${i}" style="color:#34d399;font-weight:600">—</span></div>
            <div style="height:3px;background:rgba(30,41,59,0.8);border-radius:2px;margin-top:2px;overflow:hidden">
              <div id="tsWuBar_${i}" style="height:100%;width:0%;background:${color};border-radius:2px;transition:width 0.4s ease;opacity:0.7"></div>
            </div>
          </div>`;
        list.appendChild(panel);
    });
    _ts._wuJobMap = {}; _ts._wuJobEvals = {}; _ts._wuJobBest = {};
    jobs.forEach((job, i) => { _ts._wuJobMap[job.job_id] = i; });
    if (jobs.length === 1 && jobs[0].bb_names?.length) {
        const bb = jobs[0].bb_names;
        const lbl0 = document.getElementById('tsTsPool0Label');
        const lbl1 = document.getElementById('tsTsPool1Label');
        if (lbl0) lbl0.textContent = `BB · ${bb[0] || 'slot 0'}`;
        if (lbl1) lbl1.textContent = `BB · ${bb[1] || 'slot 1'}`;
        // Fill the pool numbers with the STATIC raw CSV counts.
        _tsFetchPoolCounts(jobs[0].key || jobs[0].short_name || '', 0);
    }
    if (jobs.length > 1) _tsInjectMultiJobPools(jobs);
}

// ── Job switching ─────────────────────────────────────────────────────────
function _tsSetActiveJob(idx) {
    if (!_ts.jobs || idx >= _ts.jobs.length) return;
    if (idx === _ts._activeJobIdx) return;
    _ts._activeJobIdx = idx;
    const jobState = _ts._jobStates?.[idx];
    if (jobState) {
        _ts.tsState.events   = [...jobState.events];
        _ts.tsState.idx      = jobState.events.length;
        _ts.tsState.reagents = {...(jobState.reagents || {})};
        _ts.tsState.winners  = [...(jobState.winners  || [])];
    } else {
        _ts.tsState = _tsInitTs();
    }
    if (!_ts._jobSparkHistory) _ts._jobSparkHistory = {};
    _ts._sparkHistory    = _ts._jobSparkHistory[idx] || [];
    _ts._sparkRafPending = false;
    _ts._batchBest  = (_ts._jobBatchBest  || {})[idx];
    _ts._batchWorst = (_ts._jobBatchWorst || {})[idx];
    _ts.reagents = {...(_ts.tsState.reagents || {})};
    // Immediately reflect the newly-active job's iteration count and last score
    // in the sidebar, so switching tabs doesn't briefly show the previous job's
    // numbers (which then get corrected only on the next batch_stats). Derived
    // from this job's own winners/sparkHistory.
    {
        const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        const winners = _ts.tsState.winners || [];
        const sparkH  = _ts._jobSparkHistory[idx] || [];
        // Iteration count: prefer the spark history length (one entry per
        // [TS:stats]), fall back to winners length.
        const iterN = sparkH.length || winners.length;
        if (iterN > 0) set('tsTsIter', String(iterN));
        else set('tsTsIter', '—');
        const lastScore = winners.length ? winners[winners.length - 1].score : null;
        set('tsTsScore', (lastScore != null) ? lastScore.toFixed(4) : '—');
    }
    // Switch the visible bars to the newly-active job's pre-computed set (from
    // the worker). Reset the render signature so the next render rebuilds fully.
    _ts._activeBars = (_ts._jobBars || {})[idx] || [];
    const barsEl = document.getElementById('tsTsBars');
    if (barsEl) barsEl._tsIdSig = null;
    _tsQueueBarsRender();
    _tsQueueWinnersRender();
    _tsQueueSparkline();
    const logEl = document.getElementById('tsTsLog');
    if (logEl) {
        const lines = (_ts._jobLogLines || {})[idx] || [];
        logEl.innerHTML = '';
        if (lines.length) {
            const frag = document.createDocumentFragment();
            lines.forEach(html => { const d = document.createElement('div'); d.innerHTML = html; frag.appendChild(d); });
            logEl.appendChild(frag);
            logEl.scrollTop = logEl.scrollHeight;
        }
    }
    const wrapper = document.getElementById('_tsRxnPicker');
    const palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    (_ts.jobs || []).forEach((job, i) => {
        if (wrapper) {
            const pill = wrapper.querySelector(`[data-rxn-key="${job.key}"]`);
            if (pill) { const c = palette[i % palette.length]; pill.style.color = i === idx ? c : c + '88'; pill.style.borderBottom = i === idx ? `2px solid ${c}` : '2px solid transparent'; }
        }
        const btn = document.getElementById(`_tsRxnTab_${i}`);
        if (btn && !btn.dataset.rxnPretab) { btn.style.borderBottom = i === idx ? '2px solid #22d3ee' : '2px solid transparent'; btn.style.color = i === idx ? '#22d3ee' : '#64748b'; }
    });
    const bar = document.getElementById('tsStatusBar');
    if (bar) bar.textContent = `Viewing: ${(_ts.jobs || [])[idx]?.key || `job ${idx}`}`;
}

function _tsMakeReactionTabsClickable() {
    const wrapper = document.getElementById('_tsRxnPicker');
    if (!wrapper || !_ts.jobs?.length) return;
    const palette = ['#0891b2','#7c3aed','#059669','#d97706','#db2777','#2563eb'];
    const pills = [...wrapper.querySelectorAll('[data-rxn-pretab]')];
    if (pills.length !== _ts.jobs.length) {
        const shortLabels = {rxn101_amide:'Amide',rxn102_buchwald:'Buchwald',rxn108_sonogashira:'Sonogashira',rxn110_suzuki:'Suzuki',rxn113_sulfonamide:'Sulfonamide',rxn208_snar:'SnAr'};
        _tsRxnRebuildTabs([...document.querySelectorAll('[id^="_rxnChk_"]:checked')], palette, shortLabels);
    }
    _ts.jobs.forEach((job, ji) => {
        const pill = wrapper.querySelector(`[data-rxn-key="${job.key}"]`);
        if (!pill) return;
        pill.id = `_tsRxnTab_${ji}`;
        const fresh = pill.cloneNode(true);
        pill.parentNode.replaceChild(fresh, pill);
        fresh.addEventListener('click', () => {
            _tsSetActiveJob(ji);
            wrapper.querySelectorAll('[data-rxn-pretab]').forEach((p, idx) => {
                const c = palette[idx % palette.length];
                const isMe = p.dataset.rxnKey === job.key;
                p.style.color        = isMe ? c : c + '88';
                p.style.borderBottom = isMe ? `2px solid ${c}` : '2px solid transparent';
            });
        });
    });
}