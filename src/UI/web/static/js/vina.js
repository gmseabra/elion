// ══ Vina JS ══════════════════════════════════════════════════════════════════
let _vinaMode      = false;
let _vinaAtoms     = [];
let _selAtomIdx    = null;
let _currentView   = 'ligand';
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
    // Trigger Plotly resize after modal is visible so vPlotA gets correct dimensions
    setTimeout(() => {
        const vp = document.getElementById('vPlotA');
        if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
    }, 150);
    setTimeout(() => {
        // Restore Vina branding in case DeepAtom was used before
        if (typeof _miniChatSetContext === 'function') _miniChatSetContext(_MINICHAT_VINA_THEME);
        _miniChatShow();
        const out = document.getElementById('miniChatOutput');
        // Show Vina welcome if chat is empty OR was showing DeepAtom content
        const isDeepAtomContent = out.innerHTML.includes('DeepAtom');
        if (out.children.length === 0 || isDeepAtomContent) {
            out.innerHTML = '';
            _miniChatWelcome();
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
    document.getElementById('modeLigand').className =
        v==='ligand'
        ? 'px-4 py-1.5 border-r border-slate-700 bg-cyan-800 text-cyan-100'
        : 'px-4 py-1.5 border-r border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800';
    document.getElementById('modeProtein').className =
        v==='protein'
        ? 'px-4 py-1.5 bg-cyan-800 text-cyan-100'
        : 'px-4 py-1.5 text-slate-400 hover:text-white hover:bg-slate-800';
    if (v==='ligand') {
        if (_ligAtomsCache.length)
            _render3D('vPlotA', _ligAtomsCache, window._lastBonds||[], false);
    } else {
        _renderProteinView(_selAtomIdx);
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
    const hasBg  = plotEl._fullData && plotEl._fullData[0] && plotEl._fullData[0].name === 'RecBg'
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
                pair_e:  pairMap[a.idx].toFixed(5),
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
            hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [%{customdata.rec_idx}]<extra></extra>',
            name:'RecBg',
        };
        const hitTrace = {
            type:'scatter3d', mode:'markers',
            x:hitAtoms.map(a=>a.x), y:hitAtoms.map(a=>a.y), z:hitAtoms.map(a=>a.z),
            marker:{ size:4, color:hitColors, opacity:1.0, line:{width:0} },
            customdata:hitAtoms.map(a=>({
                rec_idx: a.idx,
                pair_e:  pairMap[a.idx].toFixed(5),
                name: a.name, res: a.resname+a.resseq,
            })),
            hovertemplate:'<b>%{customdata.name} %{customdata.res}</b> [rec %{customdata.rec_idx}]<br>' +
                          'pair_e: %{customdata.pair_e} kcal/mol<br>' +
                          'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
            name:'RecHit',
        };
        const _sceneP = { bgcolor:'#05070f',
            xaxis:{showgrid:false,zeroline:false,showticklabels:false},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false},
            aspectmode:'data' };
        if (window._savedCameraProtein) _sceneP.camera = window._savedCameraProtein;
        Plotly.react('vPlotA', [bgTrace, hitTrace],
            {paper_bgcolor:'transparent', plot_bgcolor:'transparent', margin:{l:0,r:0,t:0,b:0},
             scene:_sceneP, showlegend:false, font:{color:'#94a3b8'}},
            {responsive:true, displayModeBar:true, displaylogo:false}
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
    // Fetch live grid config from backend so it always reflects the active protein
    fetch('/vina_visualization/vina_defaults')
        .then(r => r.json())
        .then(cfg => {
            if (cfg.status === 'success') {
                _GRID.cx = cfg.center_x; _GRID.cy = cfg.center_y; _GRID.cz = cfg.center_z;
                _GRID.sx = cfg.size_x;   _GRID.sy = cfg.size_y;   _GRID.sz = cfg.size_z;
            }
            _appendProgress(`         --center_x ${_GRID.cx} --center_y ${_GRID.cy} --center_z ${_GRID.cz}`);
            _appendProgress(`         --size_x ${_GRID.sx} --size_y ${_GRID.sy} --size_z ${_GRID.sz} --exhaustiveness ${cfg.exhaustiveness || 8}`);
            _appendProgress('');
        })
        .catch(() => {
            _appendProgress(`         --center_x ${_GRID.cx} --center_y ${_GRID.cy} --center_z ${_GRID.cz}`);
            _appendProgress(`         --size_x ${_GRID.sx} --size_y ${_GRID.sy} --size_z ${_GRID.sz} --exhaustiveness 8`);
            _appendProgress('');
        });
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
            if (typeof _renderTable === 'function')
                _renderTable('vTableA', data.lig_atoms, 'vBarA', 'vPlotA', data.weight_vector);

            const te = data.total_e;
            document.getElementById('vScoreA').textContent     = te!=null ? te.toFixed(3)+' kcal/mol' : '—';
            document.getElementById('vScoreA').className        = 'score-val '+(te<0?'score-better':'score-worse');
            document.getElementById('vScoreANote').textContent  = 'kcal/mol · Vina lig_grids E';
            document.getElementById('vBarLabel').textContent    = 'this_e · all atoms';
            document.getElementById('vBarNote').innerHTML       = '<code class="text-emerald-400">this_e</code> = Σ pair_e after curl';
            document.getElementById('vStatusA').textContent     = `${data.n_lig_atoms} atoms · Σ = ${te?.toFixed(4)??'?'}`;
            document.getElementById('vSpinnerA').style.display  = 'none';
            document.getElementById('vPlotA').style.opacity     = '1';
            _setView('ligand');
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
    const btn = document.getElementById('vizBtn');
    btn.disabled = true;
    _vinaStatus('Reading log file…'); _vinaBusy(true);
    document.getElementById('vSpinnerA').style.display = 'flex';
    document.getElementById('vSpinnerAMsg').textContent = 'Reading vina_non_cache.log…';
    document.getElementById('vPlotA').style.opacity = '0';

    fetch('/vina_visualization/vina_parse_log', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            receptor_path: document.getElementById('recPath').value.trim(),
            ligand_path:   document.getElementById('ligPath').value.trim(),
        })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false; _vinaBusy(false);
        if (data.status !== 'success') {
            _vinaStatus('Error: ' + data.message, true);
            document.getElementById('vSpinnerAMsg').textContent = 'Error: ' + data.message;
            return;
        }

        _ligAtomsCache = data.atoms;
        _recAtoms      = data.rec_atoms || [];
        _pairsByLig    = data.pairs_by_lig_atom || {};
        // Compute global pair_e range across ALL lig atoms so colour scale is consistent
        const _allPairE = Object.values(_pairsByLig).flatMap(ps => ps.map(p => p.pair_e));
        _globalPairEMin = _allPairE.length ? Math.min(..._allPairE) : -0.1;
        _globalPairEMax = _allPairE.length ? Math.max(..._allPairE) :  0.0;
        window._lastBonds = data.bonds;
        _render3D('vPlotA', data.atoms, data.bonds, false);
        _renderBar('vBarA', data.weight_vector, 'vPlotA', 'vTableA');
        _renderTable('vTableA', data.atoms, 'vBarA', 'vPlotA', data.weight_vector);
        _setView('ligand');

        // Score card → Vina affinity (mode 1) — matches the mode table number
        const affinity = data.best_affinity;
        const te       = data.total_e;
        const scoreEl  = document.getElementById('vScoreA');
        scoreEl.textContent = affinity != null ? affinity.toFixed(3) + ' kcal/mol' : '—';
        scoreEl.className   = 'score-val ' + (affinity < 0 ? 'score-better' : 'score-worse');
        document.getElementById('vScoreANote').textContent =
            `Vina affinity · mode 1 (lig_grids = ${te != null ? te.toFixed(3) : '—'})`;

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

// ── 3-D scatter + bonds (verbatim from ajax_weight_a.html) ───────────────────
function _render3D(divId, atoms, bonds, compact) {
    const atomTrace = {
        type:'scatter3d', mode:'markers+text',
        x:atoms.map(a=>a.x), y:atoms.map(a=>a.y), z:atoms.map(a=>a.z),
        text:         atoms.map(a=>a.symbol),
        textfont:     { size: compact?8:10, color:'#ffffff' },
        textposition: 'top center',
        marker: {
            size:  atoms.map(a=>(compact?5:7)+a.weight_norm*12),
            color: atoms.map(a=>a.weight_norm),
            colorscale:[[0,'#3b4cc0'],[.25,'#88bbee'],[.5,'#dddddd'],[.75,'#ee8866'],[1,'#b40426']],
            cmin:0, cmax:1, showscale:false,
            line:{width:1,color:'#0f172a'},
        },
        customdata: atoms.map(a=>({idx:a.idx,raw:a.weight_raw.toFixed(4),norm:a.weight_norm.toFixed(4)})),
        hovertemplate:
            '<b>%{text}</b> (idx %{customdata.idx})<br>' +
            'this_e: %{customdata.raw}  norm: %{customdata.norm}<br>' +
            'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
        name:'Atoms',
    };
    const bx=[],by=[],bz=[];
    bonds.forEach(b=>{
        bx.push(atoms[b.begin].x,atoms[b.end].x,null);
        by.push(atoms[b.begin].y,atoms[b.end].y,null);
        bz.push(atoms[b.begin].z,atoms[b.end].z,null);
    });
    const _el3d  = document.getElementById(divId);
    const _existsLig = _el3d._fullData && _el3d._fullData.length === 2
        && _el3d._fullData[1] && _el3d._fullData[1].name === 'Atoms'
        && _el3d._fullData[1].x && _el3d._fullData[1].x.length === atoms.length;

    if (_existsLig) {
        // Same atom count — only restyle colors+sizes; camera untouched
        const newColors = atoms.map(a=>a.weight_norm);
        const newSizes  = atoms.map(a=>(compact?5:7)+a.weight_norm*12);
        Plotly.restyle(divId, {'marker.color':[newColors],'marker.size':[newSizes]}, [1]);
        document.getElementById(divId).style.opacity='1';
    } else {
        // First draw or atom count changed: full react, then attach camera listener once
        const _sceneLayout = { bgcolor:'#05070f',
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
            { responsive:true, displayModeBar:true,
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
    const sorted = [...atoms].sort((a,b)=>b.weight_norm-a.weight_norm);
    const hdr  = '<p class="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-1">Atoms ↓  —  most favourable first · click to locate</p>';
    const rows = sorted.map(a=>
        '<div data-atom-idx="'+a.idx+'" '+
             'style="cursor:pointer;border-radius:4px;border:1px solid transparent;transition:background .15s" '+
             'class="flex items-center justify-between py-0.5 px-0.5 border-b border-slate-800/40">' +
            '<span class="flex items-center gap-1">' +
                '<span class="w-2 h-2 rounded-full flex-shrink-0" style="background:'+a.color+'"></span>' +
                '<span class="font-mono">'+a.symbol+'<sub class="text-slate-600">'+a.idx+'</sub></span>' +
            '</span>' +
            '<span class="font-mono text-slate-300">'+a.weight_raw.toFixed(4)+'</span>' +
        '</div>'
    ).join('');
    const container = $('#'+divId);
    container.html(hdr+rows);
    container.find('[data-atom-idx]').on('click', function() {
        const atomIdx = parseInt($(this).data('atom-idx'));
        if (barDivId) {
            const barEl = document.getElementById(barDivId);
            if (barEl && barEl.data) {
                const xs = barEl.data[0].x;
                const barPos = xs.indexOf(atomIdx);
                if (barPos >= 0) {
                    Plotly.restyle(barDivId, {selectedpoints: [[barPos]]}, [0]);
                    setTimeout(()=>Plotly.restyle(barDivId,{selectedpoints:[null]},[0]), 2000);
                }
            }
        }
        _selAtomIdx = atomIdx;
        if (_currentView==='protein') {
            _renderProteinView(atomIdx);
        } else {
            _highlightAtom(atomIdx, divId, plotDivId, wv);
        }
    });
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
// _GRID may be pre-populated by the Ligand Center tool before the Vina modal loads.
const _GRID = (window._pendingGrid)
    ? { ...window._pendingGrid }
    : { cx:-25.7, cy:0.22, cz:28.39, sx:20, sy:20, sz:20 };
window._pendingGrid = null;
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

    if (ui_action.action === 'select_protein') {
        const protein_id = ui_action.protein_id;
        if (!protein_id) return;
        fetch('/vina_visualization/vina_select_protein', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ protein_id }),
        })
        .then(r => r.json())
        .then(data => {
            if (data.status !== 'success') {
                _miniChatAppend('ai', `⚠️ Could not select protein: ${data.message}`);
                return;
            }
            // Update _GRID so Grid Box + Vina dock use the correct coordinates
            _GRID.cx = data.center_x;
            _GRID.cy = data.center_y;
            _GRID.cz = data.center_z;
            _GRID.sx = data.size_x;
            _GRID.sy = data.size_y;
            _GRID.sz = data.size_z;

            // Redraw grid box if it is currently visible
            if (_gridBoxVisible) {
                _gridBoxVisible = false;
                _toggleGridBox();
            }

            // Fill receptor / ligand path fields
            const rec = document.getElementById('recPath');
            const lig = document.getElementById('ligPath');
            if (rec && data.default_receptor) rec.value = data.default_receptor;
            if (lig && data.default_ligand)   lig.value = data.default_ligand;

            _miniChatAppend('ai',
                `✅ **${data.protein_id}** is now active ✓\n\n` +
                `Receptor and ligand paths have been pre-filled.\n` +
                `Verify the paths in the fields above, then click 🌐 **Vina Dock** to run docking.\n\n` +
                `**Grid box config:**\n` +
                `\`\`\`\ncenter_x: ${data.center_x}\ncenter_y: ${data.center_y}\ncenter_z: ${data.center_z}\nsize_x:   ${data.size_x}\nsize_y:   ${data.size_y}\nsize_z:   ${data.size_z}\n\`\`\``
            );
            // Pulse the Vina Dock button to guide the user
            setTimeout(() => _highlightBtnUntilClick('vinaDockBtn'), 300);
        })
        .catch(err => _miniChatAppend('ai', `⚠️ Network error selecting protein: ${err.message}`));
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