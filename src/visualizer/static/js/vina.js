// vina.js — Vina Docking visualizer
// Source of truth: vina_ajax.html + hub.html Vina-specific additions
// strict mode removed — uses cross-file var assignments


/* ════════════════════════════════════════════════════════════════════════
   Voxel Inspector — single permanent delegating listener, never stacks
   ════════════════════════════════════════════════════════════════════════ */

let _voxelPickMode = false;
let _voxelListenerAttached = false;

// Attach ONE listener after first Plotly render; flag gates behaviour
function _ensureVoxelListener() {
    if (_voxelListenerAttached) return;
    const plotEl = document.getElementById('vPlotA');
    if (!plotEl) return;
    plotEl.on('plotly_click', function(ev) {
        if (!_voxelPickMode) return;
        if (!ev.points || !ev.points.length) return;
        const cd = ev.points[0].customdata;
        if (!cd || cd.rec_idx === undefined) return;
        const recAtom = _recAtoms.find(a => a.idx === cd.rec_idx);
        if (!recAtom) return;
        _voxelPickMode = false;
        _deactivateVoxelPick();
        _showVoxelPanel(recAtom);
    });
    _voxelListenerAttached = true;
}

// ── box / grid constants (kept in sync with _GRID) ──────────────────────
function _getBoxParams() {
    const { cx, cy, cz, sx, sy, sz } = _GRID;
    return {
        begin: [cx - sx/2, cy - sy/2, cz - sz/2],
        end:   [cx + sx/2, cy + sy/2, cz + sz/2],
        range: [sx, sy, sz],
        n:     6,   // n_voxels per axis = floor(20/3) = 6
    };
}

// ── brick_distance_sqr (mirrors brick.h) ────────────────────────────────
function _closestBetween(begin, end, x) {
    if (x <= begin) return begin;
    if (x >= end)   return end;
    return x;
}
function _brickDistSqr(begin3, end3, v3) {
    let d2 = 0;
    for (let i = 0; i < 3; i++) {
        const c = _closestBetween(begin3[i], end3[i], v3[i]);
        d2 += (c - v3[i]) ** 2;
    }
    return d2;
}

// ── fl_to_sz (mirrors common.h) ─────────────────────────────────────────
function _flToSz(x, maxSz) {
    if (x <= 0)      return 0;
    if (x >= maxSz)  return maxSz;
    return Math.min(Math.floor(x), maxSz);
}

// ── index_to_coord (mirrors szv_grid.cpp) ───────────────────────────────
function _indexToCoord(ix, iy, iz, box) {
    return [
        box.begin[0] + box.range[0] * ix / box.n,
        box.begin[1] + box.range[1] * iy / box.n,
        box.begin[2] + box.range[2] * iz / box.n,
    ];
}

// ── compute full voxel trace for a receptor atom ─────────────────────────
function _computeVoxelInfo(recXyz) {
    const box = _getBoxParams();
    const [rx, ry, rz] = recXyz;

    // ── Filter 1: whole-box distance ────────────────────────────────────
    const f1distSqr = _brickDistSqr(box.begin, box.end, recXyz);
    const f1dist    = Math.sqrt(f1distSqr);
    const f1pass    = f1distSqr < 64;

    // Which axes clamp for Filter 1?
    const f1clamp = ['X','Y','Z'].map((ax,i) => {
        if (recXyz[i] < box.begin[i]) return `${ax}: ${recXyz[i].toFixed(3)} < ${box.begin[i].toFixed(3)} (Δ=${(box.begin[i]-recXyz[i]).toFixed(3)}Å)`;
        if (recXyz[i] > box.end[i])   return `${ax}: ${recXyz[i].toFixed(3)} > ${box.end[i].toFixed(3)} (Δ=${(recXyz[i]-box.end[i]).toFixed(3)}Å)`;
        return null;
    }).filter(Boolean);

    // ── Voxel index (from possibilities() formula) ──────────────────────
    const tmp = recXyz.map((c,i) => (c - box.begin[i]) * box.n / box.range[i]);
    const voxIdx = tmp.map(t => _flToSz(t, box.n - 1));
    const [vx, vy, vz] = voxIdx;

    // ── Voxel cell bounding box ─────────────────────────────────────────
    const vLow  = _indexToCoord(vx,   vy,   vz,   box);
    const vHigh = _indexToCoord(vx+1, vy+1, vz+1, box);

    // ── Filter 2: voxel-cell distance ───────────────────────────────────
    const f2distSqr = _brickDistSqr(vLow, vHigh, recXyz);
    const f2dist    = Math.sqrt(f2distSqr);
    const f2pass    = f2distSqr < 64;

    // Closest point on voxel
    const vClosest = recXyz.map((c,i) => _closestBetween(vLow[i], vHigh[i], c));
    // Per-axis clamping for voxel
    const f2clamp = ['X','Y','Z'].map((ax,i) => {
        if (recXyz[i] < vLow[i])  return `${ax}: ${recXyz[i].toFixed(3)} < vox_low ${vLow[i].toFixed(3)} (Δ=${(vLow[i]-recXyz[i]).toFixed(3)}Å)`;
        if (recXyz[i] > vHigh[i]) return `${ax}: ${recXyz[i].toFixed(3)} > vox_high ${vHigh[i].toFixed(3)} (Δ=${(recXyz[i]-vHigh[i]).toFixed(3)}Å)`;
        return null;
    }).filter(Boolean);

    return { f1dist, f1distSqr, f1pass, f1clamp, voxIdx, vLow, vHigh, f2dist, f2distSqr, f2pass, f2clamp, vClosest, box };
}

// ── render the voxel panel ───────────────────────────────────────────────
function _showVoxelPanel(recAtom) {
    const xyz = [recAtom.x, recAtom.y, recAtom.z];
    const info = _computeVoxelInfo(xyz);
    const { f1dist, f1distSqr, f1pass, f1clamp,
            voxIdx, vLow, vHigh,
            f2dist, f2distSqr, f2pass, f2clamp,
            vClosest, box } = info;

    const label = `${recAtom.name} ${recAtom.resname}${recAtom.resseq} [rec_atom=${recAtom.idx}]`;

    function _badge(pass) {
        return pass
            ? '<span class="text-emerald-400 font-semibold">✓ PASS</span>'
            : '<span class="text-red-400 font-semibold">✗ FAIL</span>';
    }
    function _row(label, val) {
        return `<div class="flex justify-between gap-2">
            <span class="text-slate-500 flex-shrink-0">${label}</span>
            <span class="text-slate-200 text-right">${val}</span>
        </div>`;
    }

    const clampHtml1 = f1clamp.length
        ? f1clamp.map(c=>`<div class="text-amber-400 pl-2 text-[10px]">⚠ ${c}</div>`).join('')
        : '<div class="text-slate-600 pl-2 text-[10px]">atom fully inside box</div>';

    const clampHtml2 = f2clamp.length
        ? f2clamp.map(c=>`<div class="text-amber-400 pl-2 text-[10px]">⚠ ${c}</div>`).join('')
        : '<div class="text-slate-600 pl-2 text-[10px]">atom inside voxel cell</div>';

    document.getElementById('voxelPanelTitle').textContent = '🔬 ' + label;
    document.getElementById('voxelPanelBody').innerHTML = `

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800">Atom</div>
<div class="space-y-0.5">
  ${_row('xyz', `(${xyz.map(v=>v.toFixed(3)).join(', ')})`)}
</div>

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800 mt-2">Grid Box</div>
<div class="space-y-0.5">
  ${_row('begin', `(${box.begin.map(v=>v.toFixed(3)).join(', ')})`)}
  ${_row('end',   `(${box.end.map(v=>v.toFixed(3)).join(', ')})`)}
  ${_row('n_voxels / axis', box.n)}
  ${_row('voxel cell size', `${(box.range[0]/box.n).toFixed(3)} Å`)}
</div>

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800 mt-2">
  Filter 1 — brick_dist(atom, whole_box) &lt; 8 Å ${_badge(f1pass)}
</div>
<div class="space-y-0.5">
  ${_row('closest pt on box', `(${[0,1,2].map(i=>_closestBetween(box.begin[i],box.end[i],xyz[i]).toFixed(3)).join(', ')})`)}
  ${_row('dist²', f1distSqr.toFixed(4))}
  ${_row('dist',  f1dist.toFixed(4) + ' Å')}
  ${_row('< 8 Å cutoff?', _badge(f1pass))}
  ${clampHtml1}
</div>

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800 mt-2">
  Voxel Assignment — possibilities()
</div>
<div class="space-y-0.5">
  ${['X','Y','Z'].map((ax,i)=>`
  <div class="text-slate-500 text-[10px]">${ax}: (${xyz[i].toFixed(3)} − ${box.begin[i].toFixed(3)}) × ${box.n} / ${box.range[i].toFixed(1)} = <span class="text-white">${((xyz[i]-box.begin[i])*box.n/box.range[i]).toFixed(4)}</span> → <span class="text-cyan-300">fl_to_sz → ${voxIdx[i]}</span></div>`).join('')}
  ${_row('voxel index', `(${voxIdx.join(', ')})`)}
</div>

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800 mt-2">
  Voxel Cell (${voxIdx.join(',')}) Bounding Box
</div>
<div class="space-y-0.5">
  ${_row('cell_low  = index_to_coord('+voxIdx.join(',')+')',  `(${vLow.map(v=>v.toFixed(3)).join(', ')})`)}
  ${_row('cell_high = index_to_coord('+(voxIdx[0]+1)+','+(voxIdx[1]+1)+','+(voxIdx[2]+1)+')', `(${vHigh.map(v=>v.toFixed(3)).join(', ')})`)}
</div>

<div class="text-cyan-400 text-[10px] font-semibold uppercase tracking-wider pb-1 border-b border-slate-800 mt-2">
  Filter 2 — brick_dist(atom, voxel_cell) &lt; 8 Å ${_badge(f2pass)}
</div>
<div class="space-y-0.5">
  ${_row('closest pt on voxel', `(${vClosest.map(v=>v.toFixed(3)).join(', ')})`)}
  ${_row('dist²', f2distSqr.toFixed(4))}
  ${_row('dist',  f2dist.toFixed(4) + ' Å')}
  ${_row('< 8 Å cutoff?', _badge(f2pass))}
  ${clampHtml2}
</div>

<div class="mt-3 pt-2 border-t border-slate-800 text-[10px] text-slate-600">
  stored in m_data(${voxIdx.join(',')}) — ligand atoms in this voxel will see this receptor atom as a candidate
</div>
`;

    document.getElementById('voxelPanel').classList.remove('hidden');

    // Draw the voxel box on the 3D plot
    _drawVoxelBox(vLow, vHigh, xyz);
}

// ── draw translucent voxel cell on the 3D plot ───────────────────────────
function _drawVoxelBox(vLow, vHigh, atomXyz) {
    const plotEl = document.getElementById('vPlotA');
    if (!plotEl || !plotEl.data || !plotEl.data.length) return;

    // Remove old voxel traces
    const toRemove = [];
    for (let i = plotEl.data.length-1; i >= 0; i--) {
        if ((plotEl.data[i].name||'').startsWith('VoxelCell')) toRemove.push(i);
    }
    if (toRemove.length) Plotly.deleteTraces('vPlotA', toRemove);

    const [x0,y0,z0] = vLow;
    const [x1,y1,z1] = vHigh;
    const vx=[x0,x1,x1,x0,x0,x1,x1,x0];
    const vy=[y0,y0,y1,y1,y0,y0,y1,y1];
    const vz=[z0,z0,z0,z0,z1,z1,z1,z1];
    const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const ex=[],ey=[],ez=[];
    edges.forEach(([a,b])=>{ ex.push(vx[a],vx[b],null); ey.push(vy[a],vy[b],null); ez.push(vz[a],vz[b],null); });

    const faceTrace = {
        type:'mesh3d', x:vx, y:vy, z:vz,
        i:[0,0,1,1,0,0,4,4,0,0,3,3], j:[1,2,2,3,1,5,5,6,4,5,7,6], k:[2,3,3,0,5,4,6,7,5,1,6,5],
        color:'#f59e0b', opacity:0.12, hoverinfo:'skip', name:'VoxelCell (face)', showlegend:false,
    };
    const edgeTrace = {
        type:'scatter3d', mode:'lines', x:ex, y:ey, z:ez,
        line:{color:'#f59e0b', width:2, dash:'dot'},
        hoverinfo:'skip', name:'VoxelCell (edges)', showlegend:false,
    };
    // Atom marker highlight
    const atomMarker = {
        type:'scatter3d', mode:'markers+text',
        x:[atomXyz[0]], y:[atomXyz[1]], z:[atomXyz[2]],
        marker:{size:8, color:'#f59e0b', symbol:'circle', opacity:1, line:{color:'#fff',width:1}},
        text:['◀ selected'], textfont:{size:9,color:'#f59e0b'}, textposition:'top center',
        hovertemplate:`Selected receptor atom<br>(${atomXyz.map(v=>v.toFixed(3)).join(', ')})<extra></extra>`,
        name:'VoxelCell (atom)', showlegend:false,
    };

    Plotly.addTraces('vPlotA', [faceTrace, edgeTrace, atomMarker]);
}

// ── toggle pick mode — just flip the flag; listener is permanent ─────────
function _toggleVoxelPick() {
    _voxelPickMode = !_voxelPickMode;
    if (_voxelPickMode) {
        document.getElementById('voxelInspectBtn').className =
            document.getElementById('voxelInspectBtn').className
            .replace('border-slate-600 text-slate-400','border-amber-500 text-amber-400');
        document.getElementById('voxelInspectIcon').textContent = '🟡';
        document.getElementById('voxelPickOverlay').classList.remove('hidden');
        document.getElementById('voxelPickOverlay').style.pointerEvents = 'none';
        _vinaStatus('🔬 Pick mode: click a protein atom in the 3D view');
        if (_currentView !== 'protein') _setView('protein');
        _ensureVoxelListener(); // attach once if not yet done
    } else {
        _deactivateVoxelPick();
    }
}

function _deactivateVoxelPick() {
    _voxelPickMode = false;
    document.getElementById('voxelInspectBtn').className =
        document.getElementById('voxelInspectBtn').className
        .replace('border-amber-500 text-amber-400','border-slate-600 text-slate-400');
    document.getElementById('voxelInspectIcon').textContent = '🔬';
    document.getElementById('voxelPickOverlay').classList.add('hidden');
    _vinaStatus('Voxel pick mode off');
}

function _closeVoxelPanel() {
    document.getElementById('voxelPanel').classList.add('hidden');
    // Remove voxel traces from plot
    const plotEl = document.getElementById('vPlotA');
    if (plotEl && plotEl.data) {
        const toRemove = [];
        for (let i = plotEl.data.length-1; i >= 0; i--) {
            if ((plotEl.data[i].name||'').startsWith('VoxelCell')) toRemove.push(i);
        }
        if (toRemove.length) Plotly.deleteTraces('vPlotA', toRemove);
    }
}


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
window._adjShow = function() { document.getElementById('adjModal').classList.remove('hidden'); }
function _adjHide() { document.getElementById('adjModal').classList.add('hidden'); _closeDrawer(); }
function _vinaHide() {
    document.getElementById('vinaModal').classList.add('hidden');
    document.getElementById('vinaModal').classList.remove('flex');
    _closeDrawer();
    if (typeof _miniChatClose === 'function') _miniChatClose();
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
function _vinaDock() {
    const btn = document.getElementById('vinaDockBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-ring" style="width:10px;height:10px;border-width:2px;display:inline-block"></span> Docking…';
    _vinaStatus('Starting Vina docking…'); _vinaBusy(true);

    // Show progress log panel
    _showProgressPanel();

    // Start SSE stream to tail the log file BEFORE posting the dock request
    let es = new EventSource('/vina_visualization/vina_dock_progress');
    let lineCount = 0;
    es.onmessage = ev => {
        if (ev.data === '__DONE__') {
            es.close();
            _vinaStatus('Reading results…');
            _vinaDockFetchResult(btn,
                document.getElementById('recPath').value.trim(),
                document.getElementById('ligPath').value.trim());
            return;
        }
        if (ev.data.startsWith(':')) return;
        if (/^[-|]{6,}/.test(ev.data.trim())) return;
        _appendProgress(ev.data);
        lineCount++;
        if (ev.data.startsWith('__BARCH__stars:')) {
            const stars = ev.data.slice(15).length;
            _vinaStatus(`Docking… ${Math.min(100,Math.round((stars/51)*100))}%`);
        } else if (ev.data.includes('mode |')) {
            _vinaStatus('Docking complete — reading results…');
        }
    };
    es.onerror = () => { es.close(); };

    // Fire-and-forget POST — result comes via SSE __DONE__ → _vinaDockFetchResult
    fetch('/vina_visualization/vina_dock', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
            receptor_path: document.getElementById('recPath').value.trim(),
            ligand_path:   document.getElementById('ligPath').value.trim(),
        })
    }).catch(e => {
        btn.disabled = false; btn.innerHTML = '⚗️ Vina Dock';
        _vinaBusy(false); _vinaStatus('Dock launch error: ' + e, true);
    });
}

function _vinaDockFetchResult(btn, recPath, ligPath) {
    fetch('/vina_visualization/vina_parse_log', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ receptor_path: recPath, ligand_path: ligPath })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false; btn.innerHTML = '⚗️ Vina Dock'; _vinaBusy(false);

        if (data.status !== 'success') {
            _vinaStatus('Parse error: ' + data.message, true);
            _appendProgress('ERROR: ' + data.message);
            return;
        }

        if (data.tail_block)      _appendProgress('__LOGBLOCK__' + data.tail_block);
        else if (data.mode_table) _appendProgress('__MODETABLE__' + data.mode_table);
        else                      _appendProgress(`✓ Best affinity = ${data.best_affinity?.toFixed(4)??'?'} kcal/mol`);
        _appendProgress(`✓ Log: ${data.log_file??'—'}`);
        _vinaStatus(`Done · mode 1 = ${data.best_affinity?.toFixed(3)??'?'} kcal/mol`);

        setTimeout(() => { _clearAllHighlights(); _highlightBtnUntilClick('vizBtn'); }, 500);

        const miniEl = document.getElementById('miniChat');
        if (miniEl && miniEl.style.display === 'flex') {
            _miniChatAppend('ai',
                `Docking complete! Best affinity: **${data.best_affinity?.toFixed(3)??'?'} kcal/mol**\n\n` +
                `Now click the flashing **Visualize** button to render the 3D per-atom energy decomposition.`
            );
        }

        if (data.atoms && data.atoms.length > 0) {
            _ligAtomsCache = data.atoms; _vinaAtoms = data.atoms; _selAtomIdx = null;
            _pairsByLig = data.pairs_by_lig_atom || {};
            const allE = Object.values(_pairsByLig).flatMap(ps => ps.map(p => p.pair_e));
            _globalPairEMin = allE.length ? Math.min(...allE) : -0.1;
            _globalPairEMax = allE.length ? Math.max(...allE) :  0.0;
            window._lastBonds = data.bonds || [];
            _render3D('vPlotA', data.atoms, data.bonds||[], false);
            _renderBar('vBarA', data.weight_vector, 'vPlotA', 'vTableA');
            if (typeof _renderTable === 'function')
                _renderTable('vTableA', data.atoms, 'vBarA', 'vPlotA', data.weight_vector);
            const aff = data.best_affinity, te = data.total_e;
            const scoreEl = document.getElementById('vScoreA');
            scoreEl.textContent = aff!=null ? aff.toFixed(3)+' kcal/mol' : '—';
            scoreEl.className   = 'score-val '+(aff<0?'score-better':'score-worse');
            document.getElementById('vScoreANote').textContent =
                `Vina affinity · mode 1 (lig_grids = ${te!=null?te.toFixed(3):'—'})`;
            document.getElementById('vBarLabel').textContent = 'this_e · all atoms';
            document.getElementById('vBarNote').innerHTML =
                '<code class="text-emerald-400">this_e</code> = Σ pair_e after curl';
            document.getElementById('vStatusA').textContent =
                `${data.atoms.length} atoms · Σ = ${te?.toFixed(4)??'?'}`;
            document.getElementById('vSpinnerA').style.display = 'none';
            document.getElementById('vPlotA').style.opacity = '1';
            _setView('ligand');
            setTimeout(() => { const vp=document.getElementById('vPlotA'); if(vp&&vp._fullLayout) Plotly.Plots.resize(vp); }, 80);
        }
    })
    .catch(e => {
        btn.disabled = false; btn.innerHTML = '⚗️ Vina Dock';
        _vinaBusy(false); _vinaStatus('Result fetch error: '+e, true);
    });

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
        _hss.textContent = '#progressLog::-webkit-scrollbar { display:none; }';
        document.head.appendChild(_hss);
        panel.appendChild(logArea);
        document.body.appendChild(panel);

        // Make draggable
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
        document.addEventListener('mouseup', ()=>{ drag=false; titleBar.style.cursor='grab'; });
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
    _appendProgress(`         --center_x -25.7 --center_y 0.22 --center_z 28.39`);
    _appendProgress(`         --size_x 20 --size_y 20 --size_z 20 --exhaustiveness 8`);
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

    // Mode result table only
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

    // Batched bar update — __BARCH__stars:<content> or __BARCH__sep:<content>
    if (line.startsWith('__BARCH__')) {
        const colon   = line.indexOf(':');
        const key     = line.slice(9, colon);
        const content = line.slice(colon + 1);
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
            barDiv.textContent = content;
        }
        log.scrollTop = log.scrollHeight;
        return;
    }

    // Percentage line (__BAR__pct...) — update in-place
    if (line.startsWith('__BAR__')) {
        const key     = line[7];
        const content = line.slice(8);
        const divId   = 'pbar_' + key;
        let barDiv = document.getElementById(divId);
        if (!barDiv) {
            barDiv = document.createElement('div');
            barDiv.id = divId;
            barDiv.style.cssText = `color:#64748b;font-size:11px;font-family:'Courier New',monospace;white-space:pre;`;
            log.appendChild(barDiv);
        }
        barDiv.textContent = content;
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


// ══ Qwen Chat (Claude-style centered) ════════════════════════════════════════

let _chatStream  = null;
let _msgCount    = 0;

// _chatToggle now just opens the visualizer (the chat IS the landing page)
function _chatToggle() {
    // Scroll the hub chatbox into view if not in a tool modal
    const inp = document.getElementById('chatInput');
    if (inp) inp.focus();
}

function _chatAppend(role, text) {
    const out = document.getElementById('chatOutput');
    // mini-chat has its own SSE stream — no mirroring needed
    // Hide empty state on first message
    if (_msgCount === 0) {
        const es = document.getElementById('chatEmptyState');
        if (es) es.style.display = 'none';
        const pills = document.getElementById('chatQuickPills');
        if (pills) pills.style.display = 'none';
    }
    _msgCount++;

    const wrap = document.createElement('div');
    wrap.style.cssText = 'width:100%;max-width:760px;margin:0 auto;padding:0 24px;box-sizing:border-box;';

    const div = document.createElement('div');
    div.className = role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai';

    if (role === 'ai') {
        // AI bubble: add small Elion avatar row
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
        const av = document.createElement('div');
        av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
        av.textContent = '🧬';
        row.appendChild(av);
        row.appendChild(div);
        wrap.appendChild(row);
    } else {
        wrap.style.display = 'flex';
        wrap.style.justifyContent = 'flex-end';
        wrap.appendChild(div);
    }

    div.innerHTML = text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
        .replace(/`([^`]+)`/g,'<code>$1</code>')
        .replace(/\n/g,'<br>');

    out.appendChild(wrap);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _chatShowTyping() {
    const out  = document.getElementById('chatOutput');
    const wrap = document.createElement('div');
    wrap.style.cssText = 'width:100%;max-width:760px;margin:0 auto;padding:0 24px;box-sizing:border-box;';
    wrap.id = 'chatTypingWrap';
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
    const av = document.createElement('div');
    av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
    av.textContent = '🧬';
    const dots = document.createElement('div');
    dots.className = 'chat-bubble-ai';
    dots.id = 'chatTyping';
    dots.style.padding = '10px 0';
    dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    row.appendChild(av); row.appendChild(dots);
    wrap.appendChild(row);
    out.appendChild(wrap);
    out.scrollTop = out.scrollHeight;
}
function _chatHideTyping() {
    const el = document.getElementById('chatTypingWrap');
    if (el) el.remove();
    const el2 = document.getElementById('chatTyping');
    if (el2) el2.remove();
}

function _chatGetContext() {
    const ctx = {};
    // SMILES
    const smilesEl = document.getElementById('smilesIn') || document.querySelector('.smiles-mono');
    if (smilesEl && smilesEl.value) ctx.smiles = smilesEl.value.trim();
    // Docking score
    const scoreEl = document.querySelector('.score-val');
    if (scoreEl) { const v = parseFloat(scoreEl.textContent); if (!isNaN(v)) ctx.score = v; }
    // Paths — from path-input fields in the modal
    document.querySelectorAll('.path-input').forEach(el => {
        const id = el.id || '';
        if (id.includes('lig') || el.placeholder.toLowerCase().includes('lig')) ctx.ligand_path = el.value;
        if (id.includes('rec') || el.placeholder.toLowerCase().includes('rec')) ctx.receptor_path = el.value;
    });
    return ctx;
}

function _chatSend() {
    const input = document.getElementById('chatInput');
    const text  = input.value.trim();
    if (!text || _chatStream === true || _chatStream === 'main') return;
    input.value = '';

    _chatAppend('user', text);
    _chatShowTyping();

    const ctx = _chatGetContext();
    _chatStream = 'main';

    // SSE streaming
    let aiDiv    = null;
    let fullText = '';
    let firstToken = true;

    fetch(_chatEndpoint(), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        function pump() {
            reader.read().then(({ done, value }) => {
                if (done) { _chatHideTyping(); _chatStream = null; return; }
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
                            _chatHideTyping();
                            firstToken = false;
                            aiDiv = _chatAppend('ai', '');
                            fullText = '';
                        }
                        fullText += payload;
                        // Re-render with basic markdown
                        aiDiv.innerHTML = fullText
                            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                            .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
                            .replace(/`([^`]+)`/g,'<code>$1</code>')
                            .replace(/\n/g,'<br>');
                        document.getElementById('chatOutput').scrollTop = 99999;
                    } else if (ev === 'ui_action') {
                        handleVinaUiAction(payload);
                    } else if (ev === 'done') {
                        _chatHideTyping(); _chatStream = null;
                    } else if (ev === 'error') {
                        _chatHideTyping();
                        _chatAppend('ai', '⚠️ ' + (payload || 'Server error'));
                        _chatStream = null;
                    }
                }
                pump();
            }).catch(err => {
                _chatHideTyping(); _chatStream = null;
                _chatAppend('ai', '⚠️ Stream error: ' + err.message);
            });
        }
        pump();
    }).catch(() => {
        _chatHideTyping(); _chatStream = null;
        _chatAppend('ai', '⚠️ Could not reach Qwen server. Run: nohup bash serve_qwen.sh &');
    });
}

function _qprompt(text) {
    const mini = document.getElementById('miniChat');
    if (mini && mini.style.display === 'flex') {
        // mini-chat is open — send through it
        document.getElementById('miniChatInput').value = text;
        _miniChatSend();
    } else {
        document.getElementById('chatInput').value = text;
        _chatSend();
    }
}

function _chatClear() {
    const out = document.getElementById('chatOutput');
    out.innerHTML = '';
    _msgCount = 0;
    // Restore empty state
    const es = document.getElementById('chatEmptyState');
    if (es) { out.appendChild(es); es.style.display = 'flex'; }
    const pills = document.getElementById('chatQuickPills');
    if (pills) pills.style.display = 'flex';
    fetch('/vina_visualization/chat/clear', { method: 'POST' });
}

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
                             'modeLigand','modeProtein'];

function _clearAllHighlights() {
    _ALL_HIGHLIGHTABLE.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('btn-pulse-wait','amber-pulse','btn-highlight','btn-glow');
    });
}

// Persistent gold pulse — stops when user clicks/interacts
function _highlightBtnUntilClick(btnId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.classList.remove('btn-pulse-wait', 'amber-pulse', 'btn-highlight', 'btn-glow');
    void btn.offsetWidth;
    btn.classList.add('btn-pulse-wait');

    const stop = () => {
        btn.classList.remove('btn-pulse-wait', 'btn-glow');
        btn.removeEventListener('click', stop);
    };
    btn.addEventListener('click', stop);
}

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
        // Gold-highlight path inputs so user verifies, then gold-pulse Vina Dock
        setTimeout(() => {
            _highlightAmberUntilAction(['recPath', 'recLoadBtn', 'ligPath', 'ligLoadBtn'], 'input');
            _highlightBtnUntilClick('vinaDockBtn');
        }, 400);
        return;
    }

    if (ui_action.action === 'run_visualization') {
        // Persistent gold pulse on Visualize until user clicks
        setTimeout(() => _highlightBtnUntilClick('vizBtn'), 400);
        return;
    }

    // All other actions: persistent gold pulse
    const entry = _VINA_ACTIONS[ui_action.action];
    if (!entry) return;
    setTimeout(() => _highlightBtnUntilClick(entry.btnId), 400);
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

    let aiDiv     = null;
    let fullText  = '';
    let firstToken = true;
    _chatStream = 'mini'; // block concurrent mini-chat sends

    ctx.guided_mode = true;  // enforce short step-by-step responses
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
window._adjShow = function() {
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



// ── Hub-only additions ────────────────────────────────────────────────────────

function _highlightEmptyPathFields() {
    const rec = document.getElementById('recPath');
    const lig = document.getElementById('ligPath');
    const missing = [];
    if (rec && !rec.value.trim()) missing.push('recPath');
    if (lig && !lig.value.trim()) missing.push('ligPath');
    if (missing.length) _highlightAmberUntilAction(missing, 'input');
}

window._vinaShow = function() {
    document.getElementById('vinaModal').classList.remove('hidden');
    document.getElementById('vinaModal').classList.add('flex');
    // Trigger Plotly resize after modal is visible so vPlotA gets correct dimensions
    setTimeout(() => {
        const vp = document.getElementById('vPlotA');
        if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
    }, 150);
    setTimeout(() => {
        _miniChatShow();
        if (document.getElementById('miniChatOutput').children.length === 0) {
            _miniChatWelcome();
        }
        // Highlight whichever path fields are empty on open
        _highlightEmptyPathFields();
    }, 400);
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

function _miniChatWelcome() {
    const out = document.getElementById('miniChatOutput');

    // Greeting bubble
    const greet = document.createElement('div');
    greet.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    greet.innerHTML =
        'Hi! I\'m your Vina Docking guide. 👋<br><br>' +
        'Do you already have your <strong style="color:#22d3ee">.pdbqt</strong> files ready, ' +
        'or do you need to convert a <strong style="color:#6ee7b7">.pdb</strong> file first?';
    out.appendChild(greet);

    // Action buttons row
    const row = document.createElement('div');
    row.id = 'miniWelcomeActions';
    row.style.cssText = 'display:flex;flex-direction:column;gap:6px;align-self:flex-start;width:100%;';

    const btnReady = document.createElement('button');
    btnReady.style.cssText = `
        padding:7px 12px;border-radius:10px;font-size:11px;font-weight:600;cursor:pointer;
        background:#083344;border:1px solid #0e7490;color:#67e8f9;font-family:inherit;
        text-align:left;transition:background .15s,border-color .15s;
    `;
    btnReady.innerHTML = '✅ &nbsp;I have .pdbqt files — show me where to load them';
    btnReady.onmouseover = () => { btnReady.style.background='#0e4f63'; btnReady.style.borderColor='#22d3ee'; };
    btnReady.onmouseout  = () => { btnReady.style.background='#083344'; btnReady.style.borderColor='#0e7490'; };
    btnReady.onclick = () => {
        row.remove();
        _miniChatAppend('user', 'I have .pdbqt files ready.');
        _miniChatAppend('ai',
            'Got it! Enter your file paths in the glowing **RECEPTOR** and **LIGAND** fields above, ' +
            'then click **Vina Dock** to run the docking.'
        );
        setTimeout(() => _highlightEmptyPathFields(), 300);
    };

    const btnConvert = document.createElement('button');
    btnConvert.style.cssText = `
        padding:7px 12px;border-radius:10px;font-size:11px;font-weight:600;cursor:pointer;
        background:#052e1c;border:1px solid #065f46;color:#6ee7b7;font-family:inherit;
        text-align:left;transition:background .15s,border-color .15s;
    `;
    btnConvert.innerHTML = '⚙️ &nbsp;I need to convert a .pdb file to .pdbqt first';
    btnConvert.onmouseover = () => { btnConvert.style.background='#064e3b'; btnConvert.style.borderColor='#34d399'; };
    btnConvert.onmouseout  = () => { btnConvert.style.background='#052e1c'; btnConvert.style.borderColor='#065f46'; };
    btnConvert.onclick = () => {
        row.remove();
        _miniChatAppend('user', 'I need to convert a .pdb file first.');
        _miniChatAppend('ai',
            'No problem! Opening the **PDB → PDBQT** converter now.\n\n' +
            'Choose **Ligand** or **Receptor**, drop your **.pdb** file in, then click **Convert**. ' +
            'Paste the output path into the field above when done.'
        );
        setTimeout(() => _sidebarOpenConverter(true), 400);
    };

    row.appendChild(btnReady);
    row.appendChild(btnConvert);
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
}

}