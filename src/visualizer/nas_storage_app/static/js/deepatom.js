// ══ deepatom.js — DeepAtom CNN Saliency viewer
// Layout mirrors vina_modal.html exactly:
//   LEFT panel:  toggles between Results table ↔ 3D saliency plot
//   RIGHT sidebar: score card + bar chart + atom list (always visible)
//   Toggle row:  📋 Results | ⚛️ Saliency 3D  (appears after first run)
// ═══════════════════════════════════════════════════════════════════════════

// ── Mini-chat themes ──────────────────────────────────────────────────────────
function _miniChatSetContext(theme) {
    const iconEl  = document.querySelector('#miniChat [data-drag-handle] div');
    const titleEl = document.querySelector('#miniChat [data-drag-handle] span');
    const inputEl = document.getElementById('miniChatInput');
    const sendBtn = document.querySelector('#miniChat button[onclick="_miniChatSend()"]');
    if (iconEl)  { iconEl.style.background = theme.gradient; iconEl.textContent = theme.icon; }
    if (titleEl)  titleEl.textContent = theme.title;
    if (inputEl) {
        inputEl.placeholder = theme.placeholder;
        inputEl.onfocus = () => { inputEl.style.borderColor = theme.accentColor; };
        inputEl.onblur  = () => { inputEl.style.borderColor = '#1e293b'; };
    }
    if (sendBtn) sendBtn.style.background = theme.sendBg;
}
const _MINICHAT_VINA_THEME = {
    icon:'🔬', gradient:'linear-gradient(135deg,#22d3ee,#3b82f6)',
    title:'Elion · Vina Docking', placeholder:'Ask about docking…',
    accentColor:'#22d3ee', sendBg:'#0e7490'
};
const _MINICHAT_DEEPATOM_THEME = {
    icon:'⚛️', gradient:'linear-gradient(135deg,#f59e0b,#d97706)',
    title:'Elion · DeepAtom', placeholder:'Ask about properties…',
    accentColor:'#f59e0b', sendBg:'#b45309'
};

// ── Open / Close ──────────────────────────────────────────────────────────────
function _deepatomShow() {
    document.getElementById('deepatomModal').classList.remove('hidden');
    document.getElementById('deepatomModal').classList.add('flex');
    setTimeout(() => {
        _miniChatSetContext(_MINICHAT_DEEPATOM_THEME);
        _miniChatShow();
        const out = document.getElementById('miniChatOutput');
        if (out.children.length === 0 || out.innerHTML.includes('Vina Docking guide')) {
            _deepatomMiniWelcome();
        }
        const inp = document.getElementById('deepatomDataDir');
        if (inp) { inp.focus(); inp.select(); }
    }, 300);
}

function _deepatomHide() {
    document.getElementById('deepatomModal').classList.add('hidden');
    document.getElementById('deepatomModal').classList.remove('flex');
    _miniChatClose();
    _miniChatSetContext(_MINICHAT_VINA_THEME);
}

// ── Mini-chat welcome — Vina-style dataset selector ───────────────────────────
async function _deepatomMiniWelcome() {
    const out = document.getElementById('miniChatOutput');
    out.innerHTML = '';

    const greet = document.createElement('div');
    greet.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    greet.innerHTML =
        'Hi! I\'m your <strong style="color:#fbbf24">DeepAtom</strong> guide. ⚛️<br><br>' +
        'Which <strong style="color:#fbbf24">dataset</strong> would you like to screen?';
    out.appendChild(greet);

    let datasets = [], activeId = '';
    try {
        const resp = await fetch('/vina_visualization/deepatom_datasets');
        const d    = await resp.json();
        if (d.status === 'success') { datasets = d.datasets; activeId = d.active_dataset; }
    } catch(_) {}

    if (!datasets.length) {
        datasets = [{ id:'ZccE_VS', label:'ZccE · Virtual Screening',
                      description:'ZccE compound library — virtual screening mode',
                      test_type:'vs',
                      data_dir:'/blue/lic/huangzihang/repos/elion/src/elion/properties/deepatom/data/ZccE',
                      active: true }];
        activeId = 'ZccE_VS';
    }

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;flex-direction:column;gap:8px;align-self:flex-start;width:100%;margin-top:4px;';

    datasets.forEach(ds => {
        const isActive = ds.id === activeId || ds.active;
        const btn = document.createElement('button');
        btn.id = 'daDatasetBtn_' + ds.id;
        btn.style.cssText =
            'width:100%;text-align:left;padding:10px 12px;border-radius:12px;cursor:pointer;' +
            'font-family:inherit;transition:border-color .15s,background .15s;' +
            'background:' + (isActive ? '#1c1a00' : '#111827') + ';' +
            'border:1px solid ' + (isActive ? '#78350f' : '#1e293b') + ';' +
            'display:flex;align-items:flex-start;gap:10px;';
        const shortDir = ds.data_dir.split('/').slice(-2).join('/');
        btn.innerHTML =
            '<div style="width:28px;height:28px;border-radius:8px;flex-shrink:0;' +
            'background:' + (isActive?'#292000':'#1a1a2e') + ';' +
            'border:1px solid ' + (isActive?'#78350f':'#2d2d5e') + ';' +
            'display:flex;align-items:center;justify-content:center;font-size:14px;">⚛️</div>' +
            '<div style="flex:1;min-width:0;">' +
            '<div style="font-size:12px;font-weight:600;color:' + (isActive?'#fbbf24':'#94a3b8') + ';' +
            'display:flex;align-items:center;gap:6px;">' + ds.label +
            (isActive ? ' <span style="font-size:9px;padding:1px 6px;background:#292000;color:#f59e0b;border:1px solid #78350f;border-radius:6px;">active</span>' : '') +
            '</div>' +
            '<div style="font-size:10px;color:' + (isActive?'#a16207':'#475569') + ';margin-top:2px;">' + ds.description + '</div>' +
            '<div style="font-size:9px;color:#334155;margin-top:3px;font-family:Courier New,monospace;">' +
            '-t ' + ds.test_type + ' &nbsp;·&nbsp; -d \u2026' + shortDir + '</div>' +
            '</div>';
        btn.onmouseover = () => { btn.style.borderColor = '#f59e0b'; btn.style.background = '#1c1a00'; };
        btn.onmouseout  = () => {
            const act = window._daActiveDatasetId || activeId;
            btn.style.borderColor = (btn.id === 'daDatasetBtn_' + act) ? '#78350f' : '#1e293b';
            btn.style.background  = (btn.id === 'daDatasetBtn_' + act) ? '#1c1a00' : '#111827';
        };
        btn.onclick = () => _deepatomSelectDataset(ds, btnContainer);
        btnContainer.appendChild(btn);
    });

    out.appendChild(btnContainer);
    out.scrollTop = out.scrollHeight;

    const activeDs = datasets.find(d => d.id === activeId) || datasets.find(d => d.active) || datasets[0];
    if (activeDs) _deepatomSelectDataset(activeDs, btnContainer, true);
}

function _deepatomSelectDataset(ds, container, silent) {
    window._daTestType        = ds.test_type;
    window._daActiveDatasetId = ds.id;
    const dirInput = document.getElementById('deepatomDataDir');
    if (dirInput) dirInput.value = ds.data_dir;
    if (container) {
        container.querySelectorAll('button').forEach(b => {
            const active = b.id === 'daDatasetBtn_' + ds.id;
            b.style.background  = active ? '#1c1a00' : '#111827';
            b.style.borderColor = active ? '#78350f' : '#1e293b';
        });
    }
    if (!silent) {
        _miniChatAppend('ai',
            '\u29bf\ufe0f **' + ds.label + '** is now active \u2713\n\n' +
            'Running with:\n```\n-t ' + ds.test_type + '\n-d ' + ds.data_dir + '\n```\n\n' +
            'Click **\u26a1 Run Virtual Screening** to start.'
        );
    }
}

// ── View switcher ─────────────────────────────────────────────────────────────
let _daCurrentView = 'placeholder';  // 'placeholder' | 'results' | '3d'

function _deepatomSetView(view) {
    _daCurrentView = view;
    const placeholder = document.getElementById('daPlaceholder');
    const resultsView = document.getElementById('daResultsView');
    const view3D      = document.getElementById('da3DView');
    const toggleBtns  = document.getElementById('daViewToggle');

    placeholder.style.display = 'none';
    resultsView.classList.add('hidden');
    view3D.classList.add('hidden');

    if (view === 'results') {
        resultsView.classList.remove('hidden');
        document.getElementById('daViewResultsBtn').className =
            'px-4 py-1.5 border-r border-slate-700 bg-amber-900 text-amber-100';
        document.getElementById('daView3DBtn').className =
            'px-4 py-1.5 text-slate-400 hover:text-white hover:bg-slate-800';
    } else if (view === '3d') {
        view3D.classList.remove('hidden');
        document.getElementById('daView3DBtn').className =
            'px-4 py-1.5 bg-amber-900 text-amber-100';
        document.getElementById('daViewResultsBtn').className =
            'px-4 py-1.5 border-r border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800';
    } else {
        placeholder.style.display = 'flex';
    }
}

// ── State ─────────────────────────────────────────────────────────────────────
let _daAllRows  = [];
let _daSortKey  = 'pred';
let _daSortDir  = -1;
let _daAtoms    = [];
let _daActiveId = null;

// ── Clear results ─────────────────────────────────────────────────────────────
function _deepatomClearResults() {
    _daAllRows = []; _daAtoms = []; _daActiveId = null;
    document.getElementById('deepatomSummaryCards').classList.add('hidden');
    document.getElementById('daViewToggle').classList.add('hidden');
    document.getElementById('deepatomTableBody').innerHTML = '';
    document.getElementById('daScore').textContent  = '—';
    document.getElementById('daBarLabel').textContent = 'Importance \u00b7 top 50';
    document.getElementById('daBar').innerHTML = '';
    document.getElementById('daTableA').innerHTML =
        '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-2">Atoms \u2193 \u2014 click to locate in 3D</p>';
    document.getElementById('daStatusA').textContent = '\u2014';
    document.getElementById('daCompoundPill').textContent = '\u2014';
    document.getElementById('deepatomStatus').textContent =
        'Ready \u2014 confirm data dir and click \u26a1 Run Virtual Screening.';
    document.getElementById('daPlaceholderSpinner').classList.add('hidden');
    document.getElementById('daPlaceholderIcon').style.display = '';
    document.getElementById('daPlaceholderMsg').innerHTML =
        'Confirm the data directory above and click<br><strong class="text-amber-400">\u26a1 Run Virtual Screening</strong>';
    _deepatomSetView('placeholder');
}

// ── Run virtual screening ─────────────────────────────────────────────────────
async function _deepatomRun() {
    const dataDir = (document.getElementById('deepatomDataDir').value || '').trim();
    if (!dataDir) {
        document.getElementById('deepatomStatus').textContent = '\u26a0 Please enter a data directory path first.';
        return;
    }
    const btn  = document.getElementById('deepatomRunBtn');
    const icon = document.getElementById('deepatomRunIcon');
    btn.disabled = true; icon.textContent = '\u23f3';
    document.getElementById('daSpinnerBar').classList.remove('hidden');
    document.getElementById('daStatusSpinner').classList.remove('hidden');
    document.getElementById('deepatomStatus').textContent = 'Running DeepAtom virtual screening\u2026 (may take ~60 s)';

    // Show spinner in placeholder
    _deepatomSetView('placeholder');
    document.getElementById('daPlaceholderSpinner').classList.remove('hidden');
    document.getElementById('daPlaceholderIcon').style.display = 'none';
    document.getElementById('daPlaceholderMsg').textContent = 'Running DeepAtom 3D-CNN virtual screening\u2026';

    try {
        const resp = await fetch('/vina_visualization/deepatom_estimate', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ data_dir: dataDir, test_type: window._daTestType || 'vs' })
        });
        const data = await resp.json();

        if (!resp.ok || data.status === 'error') {
            document.getElementById('deepatomStatus').textContent = 'Error: ' + (data.message || 'Script error');
            document.getElementById('daPlaceholderSpinner').classList.add('hidden');
            document.getElementById('daPlaceholderIcon').style.display = '';
            document.getElementById('daPlaceholderMsg').innerHTML =
                '<span style="color:#f87171;">\u26a0\ufe0f ' + (data.message || 'Script error') + '</span>';
            return;
        }
        _deepatomRenderResults(data);
    } catch(err) {
        document.getElementById('deepatomStatus').textContent = 'Error: ' + err.message;
        document.getElementById('daPlaceholderMsg').innerHTML =
            '<span style="color:#f87171;">\u26a0\ufe0f ' + err.message + '</span>';
    } finally {
        btn.disabled = false; icon.textContent = '\u26a1';
        document.getElementById('daSpinnerBar').classList.add('hidden');
        document.getElementById('daStatusSpinner').classList.add('hidden');
    }
}

// ── Render results into the left panel table ──────────────────────────────────
function _deepatomRenderResults(data) {
    let rows = [];
    if (Array.isArray(data.compounds) && data.compounds.length) {
        rows = data.compounds.map(c => ({
            id:   String(c.id || c.name || c.complex_id || '\u2014'),
            pred: parseFloat(c.pred_pk ?? c.predicted ?? c.pred ?? NaN),
            exp:  parseFloat(c.exp_pk  ?? c.experimental ?? c.exp ?? NaN),
        })).filter(r => !isNaN(r.pred));
    }

    if (!rows.length) {
        document.getElementById('deepatomStatus').textContent =
            'Script ran \u2014 no scored compounds found. Check data directory.';
        document.getElementById('daPlaceholderMsg').innerHTML =
            '<span style="color:#f87171;">\u26a0\ufe0f No scored compounds found.</span>';
        return;
    }

    _daAllRows = rows;
    const predsDesc = [...rows.map(r => r.pred)].sort((a,b) => b-a);
    const best   = predsDesc[0];
    const mean   = predsDesc.reduce((s,v) => s+v, 0) / predsDesc.length;
    const top10v = predsDesc[Math.floor(predsDesc.length*0.10)] ?? best;

    document.getElementById('daStatBest').textContent  = best.toFixed(2);
    document.getElementById('daStatMean').textContent  = mean.toFixed(2);
    document.getElementById('daStatCount').textContent = rows.length.toLocaleString();
    document.getElementById('daStatTop10').textContent = top10v.toFixed(2);
    document.getElementById('deepatomSummaryCards').classList.remove('hidden');
    document.getElementById('daViewToggle').classList.remove('hidden');

    _daSortKey = 'pred'; _daSortDir = -1;
    _deepatomBuildTable();
    _deepatomSetView('results');

    document.getElementById('deepatomStatus').textContent =
        'DeepAtom complete \u2014 ' + rows.length + ' compounds \u00b7 best pK ' + best.toFixed(2) +
        ' \u00b7 click a row to view CNN saliency';
}

// ── Results table ─────────────────────────────────────────────────────────────
function _deepatomBuildTable() {
    const query    = (document.getElementById('deepatomSearch')?.value || '').toLowerCase();
    const filtered = _daAllRows.filter(r => !query || r.id.toLowerCase().includes(query));

    filtered.sort((a,b) => {
        if (_daSortKey === 'id')  return _daSortDir * a.id.localeCompare(b.id);
        const av = _daSortKey === 'exp' ? a.exp : a.pred;
        const bv = _daSortKey === 'exp' ? b.exp : b.pred;
        if (isNaN(av) && isNaN(bv)) return 0;
        if (isNaN(av)) return 1; if (isNaN(bv)) return -1;
        return _daSortDir * (bv - av);
    });

    ['Rank','Id','Pred','Exp'].forEach(k => {
        const el = document.getElementById('daSort' + k);
        const lk = {Rank:'rank', Id:'id', Pred:'pred', Exp:'exp'}[k];
        if (el) el.textContent = lk === _daSortKey ? (_daSortDir===-1?'\u2193':'\u2191') : '\u2195';
    });

    document.getElementById('deepatomRowCount').textContent =
        filtered.length < _daAllRows.length
            ? filtered.length + ' / ' + _daAllRows.length + ' compounds'
            : _daAllRows.length + ' compounds';

    const maxPK  = Math.max(..._daAllRows.map(r => r.pred));
    const minPK  = Math.min(..._daAllRows.map(r => r.pred));
    const range  = maxPK - minPK || 1;
    const pDesc  = [..._daAllRows.map(r => r.pred)].sort((a,b) => b-a);
    const top10v = pDesc[Math.floor(pDesc.length*0.10)] ?? maxPK;

    const tbody = document.getElementById('deepatomTableBody');
    tbody.innerHTML = '';
    filtered.forEach((r, i) => {
        const isTop    = r.pred >= top10v;
        const isActive = r.id === _daActiveId;
        const barPct   = Math.round(((r.pred-minPK)/range)*100);
        const barCol   = isTop ? '#fbbf24' : r.pred >= minPK+range*0.5 ? '#34d399' : '#475569';
        const expTxt   = isNaN(r.exp)
            ? '<span style="color:#334155;">\u2014</span>'
            : '<span style="color:#94a3b8;">' + r.exp.toFixed(2) + '</span>';
        const tr = document.createElement('tr');
        tr.style.cssText = 'border-bottom:1px solid #0a0f1a;cursor:pointer;' +
            (isTop   ? 'background:#1a1200;' : '') +
            (isActive ? 'outline:1px solid #f59e0b;outline-offset:-1px;' : '');
        tr.onclick     = () => _deepatomSaliency(r.id);
        tr.onmouseover = () => { tr.style.background = isTop?'#2d1f00':'#0f172a'; };
        tr.onmouseout  = () => { tr.style.background = isActive?'#1c1400':isTop?'#1a1200':''; };
        tr.innerHTML =
            '<td style="padding:6px 12px;color:#334155;font-size:10px;">' + (i+1) + '</td>' +
            '<td style="padding:6px 12px;font-family:Courier New,monospace;color:' + (isTop?'#fbbf24':'#94a3b8') + ';font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + r.id + '">' +
            (isTop?'\u2b50 ':'') + r.id + '</td>' +
            '<td style="padding:6px 20px 6px 12px;text-align:right;font-weight:700;font-family:Space Grotesk,sans-serif;color:' + (isTop?'#fbbf24':'#e2e8f0') + ';font-size:12px;">' + r.pred.toFixed(2) + '</td>' +
            '<td style="padding:6px 12px;text-align:right;">' + expTxt + '</td>' +
            '<td style="padding:6px 14px;min-width:110px;">' +
            '<div style="height:4px;background:#1e293b;border-radius:2px;">' +
            '<div style="height:4px;width:' + barPct + '%;background:' + barCol + ';border-radius:2px;"></div>' +
            '</div></td>';
        tbody.appendChild(tr);
    });
}

function _deepatomFilterTable() { _deepatomBuildTable(); }
function _deepatomSort(key) {
    if (_daSortKey === key) _daSortDir *= -1;
    else { _daSortKey = key; _daSortDir = key==='id' ? 1 : -1; }
    _deepatomBuildTable();
}

// ══════════════════════════════════════════════════════════════════════════════
// CNN Saliency — called on row click
// ══════════════════════════════════════════════════════════════════════════════
async function _deepatomSaliency(compoundId) {
    const dataDir = (document.getElementById('deepatomDataDir').value || '').trim();
    if (!dataDir) { alert('Run virtual screening first.'); return; }

    _daActiveId = compoundId;
    _deepatomBuildTable();   // highlight active row

    // Switch to 3D view, show spinner
    _deepatomSetView('3d');
    document.getElementById('daSpinner').style.display = 'flex';
    document.getElementById('daPlot').style.opacity    = '0';
    document.getElementById('daSpinnerMsg').textContent = 'Computing CNN saliency for ' + compoundId + '\u2026';
    document.getElementById('daStatusSpinner').classList.remove('hidden');
    document.getElementById('deepatomStatus').textContent = 'Computing saliency for ' + compoundId + '\u2026';

    document.getElementById('daCompoundPill').textContent = compoundId;
    document.getElementById('daCompoundPill').title       = compoundId;
    document.getElementById('daScore').textContent = '\u2014';
    document.getElementById('daBar').innerHTML = '';
    document.getElementById('daTableA').innerHTML =
        '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-2">Atoms \u2193 \u2014 click to locate in 3D</p>';
    document.getElementById('daStatusA').textContent = 'Loading\u2026';

    let data;
    try {
        const resp = await fetch('/vina_visualization/deepatom_saliency', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ compound_id: compoundId, data_dir: dataDir }),
        });
        data = await resp.json();
    } catch(err) {
        _deepatomSetView('results');   // fall back to results on error
        document.getElementById('deepatomStatus').textContent = 'Network error: ' + err.message;
        document.getElementById('daStatusSpinner').classList.add('hidden');
        return;
    }

    document.getElementById('daStatusSpinner').classList.add('hidden');

    if (data.status !== 'success') {
        _deepatomSetView('results');
        document.getElementById('deepatomStatus').textContent = 'Error: ' + (data.message || 'Unknown error');
        return;
    }

    _daAtoms = data.atoms;
    document.getElementById('daScore').textContent     = data.pred_pk != null ? data.pred_pk.toFixed(2) : '\u2014';
    document.getElementById('daScoreNote').textContent = 'pK units';

    _deepatomRenderSaliency3D(data.atoms);
    _deepatomRenderBar(data.atoms);
    _deepatomRenderAtomTable(data.atoms);

    document.getElementById('daStatusA').textContent =
        data.atoms.length + ' atoms \u00b7 pK ' + (data.pred_pk?.toFixed(2) ?? '\u2014');
    document.getElementById('deepatomStatus').textContent =
        'CNN saliency for ' + compoundId + ' \u00b7 ' + data.atoms.length +
        ' atoms \u00b7 pK ' + (data.pred_pk?.toFixed(2) ?? '\u2014') +
        '  \u2014  click 📋 Results to go back';
}

// ── 3D Plotly scatter (mirrors _vinaRenderMol) ────────────────────────────────
function _deepatomRenderSaliency3D(atoms) {
    const xs   = atoms.map(a => a.x);
    const ys   = atoms.map(a => a.y);
    const zs   = atoms.map(a => a.z);
    const imps = atoms.map(a => a.importance_norm);
    const tips = atoms.map(a =>
        a.name + ' (' + a.atom_type + ')<br>importance: ' + a.importance_norm.toFixed(3));

    const colors = imps.map(v => {
        if (v < 0.5) {
            const t = v * 2;
            return 'rgb(' + Math.round(30+t*(6-30)) + ',' + Math.round(41+t*(95-41)) + ',' + Math.round(59+t*(70-59)) + ')';
        } else {
            const t = (v - 0.5) * 2;
            return 'rgb(' + Math.round(6+t*(251-6)) + ',' + Math.round(95+t*(191-95)) + ',' + Math.round(70+t*(36-70)) + ')';
        }
    });
    const sizes = imps.map(v => 4 + v * 14);

    const layout = {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:0,r:0,t:0,b:0},
        scene:{
            bgcolor:'#05070f',
            xaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false,title:''},
        },
        showlegend:false,
    };

    const plotDiv = document.getElementById('daPlot');
    Plotly.newPlot(plotDiv, [{
        type:'scatter3d', mode:'markers',
        x:xs, y:ys, z:zs,
        marker:{ size:sizes, color:colors, opacity:0.92, line:{width:0} },
        text:tips,
        hovertemplate:'%{text}<extra></extra>',
    }], layout, {responsive:true, displayModeBar:false})
    .then(() => {
        document.getElementById('daSpinner').style.display = 'none';
        plotDiv.style.opacity = '1';
    });
}

// ── Bar chart (mirrors vBarA) ─────────────────────────────────────────────────
function _deepatomRenderBar(atoms) {
    const top50  = atoms.slice(0, 50);
    const barDiv = document.getElementById('daBar');
    if (!top50.length) { barDiv.innerHTML = ''; return; }

    const names = top50.map(a => a.name).reverse();
    const vals  = top50.map(a => a.importance_norm).reverse();
    const cols  = vals.map(v => v > 0.66 ? '#fbbf24' : v > 0.33 ? '#34d399' : '#475569');

    Plotly.newPlot(barDiv, [{
        type:'bar', orientation:'h',
        x:vals, y:names,
        marker:{ color:cols },
        hovertemplate:'%{y}: %{x:.3f}<extra></extra>',
    }], {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:40,r:4,t:2,b:16},
        xaxis:{ showgrid:true, gridcolor:'#1e293b', tickfont:{color:'#475569',size:7}, range:[0,1] },
        yaxis:{ tickfont:{color:'#475569',size:7}, gridcolor:'#1e293b' },
        showlegend:false,
    }, {responsive:true, displayModeBar:false});
}

// ── Atom list (mirrors vTableA) ───────────────────────────────────────────────
function _deepatomRenderAtomTable(atoms) {
    const tableEl = document.getElementById('daTableA');
    tableEl.innerHTML =
        '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-2">' +
        'Atoms \u2193 \u2014 click to locate in 3D</p>';

    atoms.slice(0, 80).forEach((a, i) => {
        const col = a.importance_norm > 0.66 ? '#fbbf24' : a.importance_norm > 0.33 ? '#34d399' : '#64748b';
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between py-0.5 px-0.5 border-b border-slate-800/40';
        row.style.cssText = 'cursor:pointer;border-radius:4px;transition:background .1s;';
        row.onmouseover = () => { row.style.background = '#0f172a'; };
        row.onmouseout  = () => { row.style.background = ''; };
        row.onclick     = () => _deepatomHighlightAtom(i);
        row.innerHTML =
            '<span style="font-family:Courier New,monospace;font-size:10px;color:' + col + ';">' +
            (a.importance_norm > 0.66 ? '\u2b50 ' : '') + a.name + '</span>' +
            '<span style="font-size:9px;color:#334155;">' + a.atom_type + '</span>' +
            '<span style="font-size:10px;font-weight:700;color:' + col + ';">' + a.importance_norm.toFixed(3) + '</span>';
        tableEl.appendChild(row);
    });
}

// ── Highlight atom in 3D plot (mirrors Vina atom-click) ──────────────────────
function _deepatomHighlightAtom(idx) {
    const plotDiv = document.getElementById('daPlot');
    if (!plotDiv || !plotDiv.data || !plotDiv.data.length) return;
    // Switch to 3D view if on results
    if (_daCurrentView !== '3d') _deepatomSetView('3d');
    const n       = _daAtoms.length;
    const defSz   = _daAtoms.map(a => 4 + a.importance_norm * 14);
    const sizes   = defSz.map((s,i) => i===idx ? s*2.2 : s*0.5);
    const opacs   = Array.from({length:n}, (_,i) => i===idx ? 1.0 : 0.2);
    Plotly.restyle(plotDiv, {'marker.size':[sizes], 'marker.opacity':[opacs]}, [0]);
    setTimeout(() => {
        Plotly.restyle(plotDiv, {'marker.size':[defSz], 'marker.opacity':[Array(n).fill(0.92)]}, [0]);
    }, 2000);
}