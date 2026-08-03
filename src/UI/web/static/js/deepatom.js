// ══ deepatom.js — DeepAtom Single-Ligand CNN Saliency Inspector
// Mirrors ChemBERT/Vina layout: input row → Visualize → 3D + bar + atom list
// ═══════════════════════════════════════════════════════════════════════════
//
// Open/close + mini-chat panel wiring → deepatom_chat.js
// Theme + welcome function              → elion_mini_chat.js

// ── State ─────────────────────────────────────────────────────────────────────
let _daAtoms    = [];
let _daWorker   = null;
let _daLogEvtSource = null;

// ══════════════════════════════════════════════════════════════════════════════
// Core: Visualize — compute CNN saliency for one ligand
// ══════════════════════════════════════════════════════════════════════════════
async function _deepatomVisualize() {
    const ligInput  = document.getElementById('daLigandPath');
    const dirInput  = document.getElementById('daDataDir');
    const ligVal    = (ligInput?.value || '').trim();
    const dataDir   = (dirInput?.value || '').trim();

    if (!ligVal) {
        document.getElementById('deepatomStatus').textContent =
            '⚠ Enter a compound ID or path to .atomtypes file above.';
        ligInput?.focus();
        return;
    }

    // Show spinner
    const spinner = document.getElementById('daSpinner');
    spinner.style.display = 'flex';
    document.getElementById('daSpinnerMsg').textContent = 'Computing CNN saliency for ' + ligVal + '…';
    document.getElementById('daPlot').style.opacity = '0';
    document.getElementById('daStatusSpinner').classList.remove('hidden');
    document.getElementById('daSpinnerBar').classList.remove('hidden');
    document.getElementById('deepatomStatus').textContent = 'Running ShuffleNetV3 forward+backward pass…';
    document.getElementById('daScore').textContent = '—';
    document.getElementById('daBar').innerHTML = '';
    document.getElementById('daTableA').innerHTML =
        '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-2">' +
        'Atoms ↓ — most favourable first · click to locate</p>';
    document.getElementById('daStatusA').textContent = '—';
    document.getElementById('daCompoundPill').textContent = ligVal;
    document.getElementById('daCompoundPill').title = ligVal;

    // Show terminal log
    const terminal = document.getElementById('daTerminal');
    terminal.classList.remove('hidden');
    const termCmd = document.getElementById('daTerminalCmd');
    if (termCmd) termCmd.textContent = '$ deepatom saliency ' + ligVal;
    const logBody = document.getElementById('daLogBody');
    if (logBody) {
        logBody.innerHTML = '';
        _daLogLine('[deepatom] compound: ' + ligVal, '#67e8f9');
        _daLogLine('[deepatom] data_dir: ' + (dataDir || '(using default)'), '#475569');
        _daLogLine('[deepatom] model: ShuffleNetV3 × 2.0  (input=24ch, grid=32³)', '#475569');
        _daLogLine('', '');
    }

    // POST to saliency backend
    const body = { compound_id: ligVal, data_dir: dataDir };
    // If ligVal looks like a full path, send as pdb_path instead
    if (ligVal.startsWith('/')) {
        body.pdb_path = ligVal;
        delete body.compound_id;
    }

    let data;
    try {
        console.group('[DeepAtom] Saliency');
        console.log('POST /vina_visualization/deepatom_saliency', body);
        const resp = await fetch('/vina_visualization/deepatom_saliency', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        console.log('HTTP', resp.status);
        const rawText = await resp.text();
        console.log('Raw (500ch):', rawText.slice(0, 500));
        try { data = JSON.parse(rawText); }
        catch (_) {
            data = { status: 'error', message: 'Server returned non-JSON: ' + rawText.slice(0, 120) };
        }
        console.log('status:', data.status, 'atoms:', data.atoms?.length, 'bonds:', data.bonds?.length);
        if (data.status !== 'success') {
            console.error('[DeepAtom] Error type:', data.type || 'unknown');
            console.error('[DeepAtom] Error message:', data.message);
            if (data.traceback) {
                console.group('[DeepAtom] Python traceback');
                data.traceback.split('\n').forEach(l => console.log(l));
                console.groupEnd();
            }
        }
        console.groupEnd();
    } catch(err) {
        console.error('[DeepAtom]', err);
        console.groupEnd();
        _daFinishWithError('Network error: ' + err.message);
        return;
    }

    document.getElementById('daStatusSpinner').classList.add('hidden');
    document.getElementById('daSpinnerBar').classList.add('hidden');

    if (data.status !== 'success') {
        _daFinishWithError(data.message || 'Unknown error');
        return;
    }

    _daLogLine('[deepatom] grid source: ' + (data.grid_source || 'unknown'), '#6ee7b7');
    _daLogLine('[deepatom] atoms: ' + data.atoms.length + '  bonds: ' + (data.bonds||[]).length, '#6ee7b7');
    _daLogLine('[deepatom] predicted pK: ' + data.pred_pk, '#fbbf24');

    // Update UI
    _daAtoms = data.atoms;
    _daAtoms = data.atoms;
    // Convert pK → ΔG (kcal/mol): ΔG = -1.36 × pK
    const _pk = data.pred_pk;
    const _dg = _pk != null ? (-1.36 * _pk) : null;
    const _scoreEl = document.getElementById('daScore');
    _scoreEl.textContent = _dg != null ? _dg.toFixed(3) + ' kcal/mol' : '—';
    _scoreEl.style.fontSize = '1.1rem';
    _scoreEl.style.color = _dg != null && _dg < 0 ? '#34d399' : '#fbbf24';
    document.getElementById('daScoreNote').textContent =
        'ΔG (kcal/mol)  ·  pK = ' + (_pk != null ? _pk.toFixed(2) : '—') +
        '  ·  ligand: ' + (data.atoms||[]).length + ' atoms';

    _deepatomRenderSaliency3D(data.atoms, data.bonds || []);
    _deepatomRenderBar(data.atoms);
    _deepatomRenderAtomTable(data.atoms);

    document.getElementById('daStatusA').textContent =
        data.atoms.length + ' atoms · ' + (data.bonds||[]).length + ' bonds · pK ' +
        (data.pred_pk?.toFixed(2) ?? '—');
    document.getElementById('deepatomStatus').textContent =
        'CNN saliency for ' + ligVal + ' · ' + data.atoms.length +
        ' atoms · pK ' + (data.pred_pk?.toFixed(2) ?? '—') +
        '  ·  grid: ' + (data.grid_source || '?');

    _daLogLine('', '');
    _daLogLine('─── Done ───', '#fbbf24');
}

function _daFinishWithError(msg) {
    document.getElementById('daStatusSpinner').classList.add('hidden');
    document.getElementById('daSpinnerBar').classList.add('hidden');
    document.getElementById('daSpinner').style.display = 'flex';
    document.getElementById('daSpinnerMsg').innerHTML =
        '<span style="color:#f87171;font-size:11px;">⚠ ' + msg + '</span>';
    document.getElementById('deepatomStatus').textContent = 'Error: ' + msg;
    _daLogLine('⚠ ' + msg, '#f87171');
}

// ── Terminal log helper ───────────────────────────────────────────────────────
function _daLogLine(text, color) {
    const body = document.getElementById('daLogBody');
    if (!body) return;
    if (!text) { body.appendChild(document.createElement('br')); return; }
    const d = document.createElement('div');
    d.style.cssText = 'white-space:pre-wrap;line-height:1.5;color:' + (color||'#64748b') + ';';
    d.textContent = text;
    body.appendChild(d);
    body.scrollTop = body.scrollHeight;
}

// ── 3D molecular render — mirrors Vina ligand view exactly ─────────────────
// Colour ramp: blue (low importance) → white → red (high importance)
// Size:        proportional to importance_norm
// Labels:      atom name (e.g. "O1", "N4")
// Bonds:       grey sticks
function _deepatomRenderSaliency3D(atoms, bonds) {
    const xs   = atoms.map(a => a.x);
    const ys   = atoms.map(a => a.y);
    const zs   = atoms.map(a => a.z);
    const imps = atoms.map(a => a.importance_norm);
    // Use full atom name (e.g. "C13", "O1", "N3") — matches Vina label style
    const labels = atoms.map(a => a.name || 'X');
    const tips = atoms.map(a =>
        '<b>' + a.name + '</b> (' + a.atom_type + ')<br>importance: ' +
        a.importance_norm.toFixed(3) + '<br>rank: #' +
        (atoms.indexOf(a) + 1) + ' / ' + atoms.length);

    // Blue→white→red colour ramp — relative to THIS compound's range
    const impMin = Math.min(...imps);
    const impMax = Math.max(...imps);
    const impRange = impMax - impMin || 1e-9;
    const colors = imps.map(v => {
        const t = (v - impMin) / impRange;  // 0=least important, 1=most important
        if (t < 0.5) {
            const s = t * 2;
            // blue → white
            return 'rgb(' + Math.round(30 + s*225) + ',' + Math.round(100 + s*155) + ',' + Math.round(220) + ')';
        } else {
            const s = (t - 0.5) * 2;
            // white → red
            return 'rgb(' + Math.round(255) + ',' + Math.round(255 - s*225) + ',' + Math.round(220 - s*220) + ')';
        }
    });
    const sizes = imps.map(v => 6 + v * 16);

    const traces = [];

    // Bond lines
    if (bonds.length) {
        const bx=[], by=[], bz=[];
        bonds.forEach(([i,j]) => { bx.push(xs[i],xs[j],null); by.push(ys[i],ys[j],null); bz.push(zs[i],zs[j],null); });
        traces.push({ type:'scatter3d', mode:'lines', x:bx, y:by, z:bz,
            line:{color:'#1e3a5f',width:3}, hoverinfo:'none', showlegend:false });
    }

    // Atom spheres + labels
    traces.push({
        type:'scatter3d', mode:'markers+text',
        x:xs, y:ys, z:zs,
        marker:{ size:sizes, color:colors, opacity:0.95, line:{color:'rgba(0,0,0,0.25)',width:1} },
        text:labels,
        textfont:{ size:9, color:'#e2e8f0' },
        textposition:'top center',
        customdata:tips,
        hovertemplate:'%{customdata}<extra></extra>',
        showlegend:false,
    });

    const layout = {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:0,r:0,t:0,b:0},
        scene:{
            bgcolor:'#05070f',
            xaxis:{showgrid:false,zeroline:false,showticklabels:false,title:{text:'x',font:{color:'#334155',size:8}}},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false,title:{text:'y',font:{color:'#334155',size:8}}},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false,title:{text:'z',font:{color:'#334155',size:8}}},
            camera:{eye:{x:1.4,y:1.4,z:1.0}},
        },
        showlegend:false,
    };

    const plotDiv = document.getElementById('daPlot');
    Plotly.newPlot(plotDiv, traces, layout, {
        responsive:true, displayModeBar:true,
        modeBarButtonsToRemove:['toImage','sendDataToCloud'], displaylogo:false,
    }).then(() => {
        document.getElementById('daSpinner').style.display = 'none';
        plotDiv.style.opacity = '1';
    });
}

// ── Bar chart (vertical, mirrors ChemBERT weight_a chart exactly) ─────────────
function _deepatomRenderBar(atoms) {
    const top50  = atoms.slice(0, 50);
    const barDiv = document.getElementById('daBar');
    if (!top50.length) { barDiv.innerHTML = ''; return; }

    const idxs  = top50.map((_,i) => i);
    const vals  = top50.map(a => a.importance_norm);
    const cols  = vals.map(v => v > 0.66 ? 'rgb(220,80,80)' : v > 0.33 ? 'rgb(0,200,180)' : 'rgb(60,80,160)');

    Plotly.newPlot(barDiv, [{
        type:'bar', orientation:'v',
        x:idxs, y:vals,
        marker:{ color:cols },
        text:top50.map(a => a.name),
        hovertemplate:'<b>%{text}</b><br>importance: %{y:.3f}<extra></extra>',
    }], {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:28,r:4,t:2,b:20},
        xaxis:{
            showgrid:false, zeroline:false,
            tickvals:idxs.filter((_,i) => i%5===0),
            ticktext:idxs.filter((_,i) => i%5===0).map(String),
            tickfont:{color:'#475569',size:7},
            title:{text:'Atom idx', font:{color:'#334155',size:8}},
        },
        yaxis:{ showgrid:true, gridcolor:'#1e293b', zeroline:true, zerolinecolor:'#334155',
                tickfont:{color:'#475569',size:7} },
        bargap:0.1, showlegend:false,
    }, {responsive:true, displayModeBar:false});
}

// ── Atom table — mirrors Vina atom list exactly ─────────────────────────────
// Shows: coloured dot · atom name · channel weight · importance score
// Click any row → pulse-highlight in 3D
function _deepatomRenderAtomTable(atoms) {
    const tableEl = document.getElementById('daTableA');
    tableEl.innerHTML =
        '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-2">' +
        'Atoms ↓ — most favourable first · click to locate in 3D</p>';

    atoms.slice(0, 80).forEach((a, i) => {
        // Importance-driven colour — relative to max in this compound
        const _maxImp = Math.max(...atoms.map(a => a.importance_norm)) || 1;
        const v   = a.importance_norm / _maxImp;  // 0→1 relative
        const col = v > 0.66 ? '#ef4444' : v > 0.33 ? '#94a3b8' : '#3b82f6';

        const row = document.createElement('div');
        row.style.cssText =
            'display:flex;align-items:center;gap:6px;padding:3px 2px;' +
            'border-bottom:1px solid rgba(30,41,59,0.5);cursor:pointer;' +
            'border-radius:3px;transition:background .1s;';
        row.onmouseover = () => { row.style.background = '#0f172a'; };
        row.onmouseout  = () => { row.style.background = ''; };
        row.onclick     = () => _deepatomHighlightAtom(i);

        // Active channel tags
        const chTags = (a.active_channels || [])
            .map(c => c.replace('lig_',''))
            .slice(0, 3)
            .join(' ');

        row.innerHTML =
            '<span style="width:8px;height:8px;border-radius:50%;background:' + col + ';flex-shrink:0;"></span>' +
            '<span style="font-family:Courier New,monospace;font-size:10px;color:' + col + ';min-width:36px;">' +
            a.name + '</span>' +
            '<span style="font-size:9px;color:#475569;flex:1;">' + chTags + '</span>' +
            '<span style="font-size:9px;color:#334155;min-width:24px;text-align:right;">' +
            (a.channel_weight||0).toFixed(0) + 'ch</span>' +
            '<span style="font-size:10px;font-weight:700;color:' + col + ';min-width:48px;text-align:right;">' +
            a.importance_norm.toFixed(4) + '</span>';
        tableEl.appendChild(row);
    });

    // Footer: total atoms + bonds
    const footer = document.createElement('div');
    footer.style.cssText = 'padding:4px 2px;font-size:9px;color:#334155;border-top:1px solid #1e293b;margin-top:4px;';
    footer.textContent = atoms.length + ' ligand heavy atoms';
    tableEl.appendChild(footer);
}

// ── Highlight atom in 3D (pulse on click, mirrors Vina atom-click) ────────────
function _deepatomHighlightAtom(idx) {
    const plotDiv = document.getElementById('daPlot');
    if (!plotDiv?.data?.length) return;
    const n      = _daAtoms.length;
    const defSz  = _daAtoms.map(a => 6 + a.importance_norm * 16);
    const sizes  = defSz.map((s,i) => i===idx ? s*2.2 : s*0.5);
    const opacs  = Array.from({length:n}, (_,i) => i===idx ? 1.0 : 0.2);
    // Atom trace is index 1 (bonds are index 0)
    const traceIdx = plotDiv.data.length > 1 ? 1 : 0;
    Plotly.restyle(plotDiv, {'marker.size':[sizes],'marker.opacity':[opacs]}, [traceIdx]);
    setTimeout(() => {
        Plotly.restyle(plotDiv, {'marker.size':[defSz],'marker.opacity':[Array(n).fill(0.95)]}, [traceIdx]);
    }, 2000);
}

// ── Route availability check on load ──────────────────────────────────────────
(function _deepatomRouteCheck() {
    // Check the datasets route (GET) — confirms the module loaded correctly
    fetch('/vina_visualization/deepatom_datasets')
        .then(r => console.log('[DeepAtom] routes:', r.ok ? '✓ loaded' : '✗ ' + r.status))
        .catch(() => console.warn('[DeepAtom] routes unreachable'));
})();

// ══════════════════════════════════════════════════════════════════════════════
// Make Atomtypes — generates .atomtypes for a single compound
// Calls POST /vina_visualization/deepatom_make_atomtypes
// Streams progress via GET /vina_visualization/deepatom_make_atomtypes_stream
// ══════════════════════════════════════════════════════════════════════════════

let _daAtSseSource = null;

async function _deepatomMakeAtomtypes() {
    const ligVal  = (document.getElementById('daLigandPath')?.value || '').trim();
    const dataDir = (document.getElementById('daDataDir')?.value   || '').trim();

    if (!ligVal) {
        document.getElementById('deepatomStatus').textContent =
            '⚠ Enter a compound ID first (e.g. BM-1-57).';
        document.getElementById('daLigandPath')?.focus();
        return;
    }

    const btn  = document.getElementById('daMakeAtBtn');
    const icon = document.getElementById('daMakeAtIcon');
    if (btn)  btn.disabled = true;
    if (icon) icon.textContent = '⏳';
    document.getElementById('deepatomStatus').textContent =
        'Generating .atomtypes for ' + ligVal + '…';

    // Open terminal and start SSE stream
    const terminal = document.getElementById('daTerminal');
    terminal.classList.remove('hidden');
    const logBody = document.getElementById('daLogBody');
    if (logBody) { logBody.innerHTML = ''; }
    const termCmd = document.getElementById('daTerminalCmd');
    if (termCmd) termCmd.textContent = '$ arpeggio → atomtypes/' + ligVal + '.atomtypes';

    _daLogLine('[deepatom] make_atomtypes: ' + ligVal, '#67e8f9');
    _daLogLine('[deepatom] data_dir: ' + (dataDir || '(default)'), '#475569');
    _daLogLine('', '');

    console.group('[DeepAtom] Make Atomtypes');
    console.log('Step 0 — Check cache: data_dir/atomtypes/' + ligVal + '.atomtypes');
    console.log('Step 1 — Copy from DEEP_MODEL_temp/* if present');
    console.log('Step 2 — Run arpeggio_mod2/arpeggio.py on _complex.pdb');
    console.log('Step 3 — Run pipeline_VS.py --stages 1,3 as last resort');
    console.log('POST body:', { compound_id: ligVal, data_dir: dataDir });

    // SSE for live logs
    if (_daAtSseSource) { _daAtSseSource.close(); _daAtSseSource = null; }
    _daAtSseSource = new EventSource('/vina_visualization/deepatom_make_atomtypes_stream');
    _daAtSseSource.onmessage = (e) => {
        const line = e.data;
        if (line === '__DONE__' || line.startsWith('__ERROR__')) {
            _daAtSseSource.close(); _daAtSseSource = null;
            return;
        }
        const col = line.includes('ERROR') || line.includes('ERR --') ? '#f87171'
                  : line.includes('OK')    || line.includes('Done')   ? '#6ee7b7'
                  : '#64748b';
        _daLogLine(line, col);
    };
    _daAtSseSource.onerror = () => {
        if (_daAtSseSource) { _daAtSseSource.close(); _daAtSseSource = null; }
    };

    // POST
    let data;
    try {
        console.log('POST → /vina_visualization/deepatom_make_atomtypes');
        const resp = await fetch('/vina_visualization/deepatom_make_atomtypes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ compound_id: ligVal, data_dir: dataDir }),
        });
        console.log('HTTP response:', resp.status, resp.statusText);
        if (!resp.ok) {
            const rawErr = await resp.text();
            console.error('Error body (500 chars):', rawErr.slice(0, 500));
            const msg = resp.status === 404
                ? 'make_atomtypes route not found (404) — deploy updated deepatom_routes.py and restart'
                : 'HTTP ' + resp.status + ': ' + rawErr.slice(0, 200);
            throw new Error(msg);
        }
        const rawText = await resp.text();
        console.log('Raw response:', rawText.slice(0, 500));
        try { data = JSON.parse(rawText); }
        catch(pe) { throw new Error('JSON parse error: ' + pe.message + ' | body: ' + rawText.slice(0, 200)); }
        console.log('Response:', data.status, '| method:', data.method, '| path:', data.atomtypes_path);
    } catch(err) {
        console.error('[DeepAtom] make_atomtypes error:', err);
        console.groupEnd();
        _daLogLine('⚠ ' + err.message, '#f87171');
        document.getElementById('deepatomStatus').textContent = 'Error: ' + err.message;
        if (btn)  btn.disabled = false;
        if (icon) icon.textContent = '🧬';
        return;
    } finally {
        if (_daAtSseSource) { _daAtSseSource.close(); _daAtSseSource = null; }
        if (btn)  btn.disabled = false;
        if (icon) icon.textContent = '🧬';
    }

    console.groupEnd();
    if (data.status === 'success') {
        const method = data.method || '';
        const methodLabel = { cached:'(cached)', temp_copy:'(from temp)', arpeggio:'(arpeggio)', pipeline:'(pipeline)' }[method] || '';
        _daLogLine('', '');
        _daLogLine('✓ ' + data.atomtypes_path, '#fbbf24');
        document.getElementById('deepatomStatus').textContent =
            '✓ Atomtypes ready ' + methodLabel + ' — click ⚡ Visualize to inspect ' + ligVal;

        // Pulse the Visualize button to indicate readiness
        const vizBtn = document.getElementById('daVizBtn');
        if (vizBtn) {
            vizBtn.classList.add('btn-pulse-wait');
            vizBtn.addEventListener('click', function once() {
                vizBtn.classList.remove('btn-pulse-wait');
                vizBtn.removeEventListener('click', once);
            });
        }

        // Auto-fill compound ID in case it was cleared
        const inp = document.getElementById('daLigandPath');
        if (inp && !inp.value.trim()) inp.value = ligVal;

    } else {
        _daLogLine('⚠ ' + (data.message || 'Failed'), '#f87171');
        document.getElementById('deepatomStatus').textContent =
            'Atomtypes error: ' + (data.message || 'Unknown error');
    }
}