// hub-core.js — shared routing, highlights, drag, sidebar
// strict mode removed — uses cross-file var assignments

let _activeTool = null;

// ── Stub definitions (overridden by attn.js / vina.js after load) ────────────
function _clearAllHighlights() {
    const ALL = [
        'adjWeightBtn','vinaDockBtn','vizBtn','ligLoadBtn','recLoadBtn',
        'gridBoxBtn','voxelInspectBtn','modeLigand','modeProtein',
        'finetuneBtn','loadModelBtn','modeCompare','modeSingle','runSingle','runCompare',
        'vinaBtn','askElionBtn','sidebarPdb2PdbqtBtn',
    ];
    ALL.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('btn-pulse-wait','amber-pulse','btn-highlight','btn-glow');
    });
    ['smilesA','smilesCompA','smilesCompB','customModelPath','recPath','ligPath'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('input-pulse-wait','amber-pulse');
    });
}
// _adjShow, _vinaShow, _pdb2pdbqtOpen — safe early stubs.
// The real implementations in attn.js / vina.js / converter.js load after
// this file and REASSIGN these vars (not re-declare), so the last assignment wins.
// Using var (not let/const) so attn.js/vina.js can reassign without SyntaxError.
window._adjShow = function() { const m=document.getElementById('adjModal');  if(m){m.classList.remove('hidden');m.classList.add('flex');} };
window._vinaShow = function() { const m=document.getElementById('vinaModal'); if(m){m.classList.remove('hidden');m.classList.add('flex');} };
window._pdb2pdbqtOpen = function() {};

// ── Sidebar ──────────────────────────────────────────────────────────────────
function _toggleElionSidebar() {
    const sidebar  = document.getElementById('elionSidebar');
    const backdrop = document.getElementById('elionSidebarBackdrop');
    const isOpen   = sidebar.style.right === '0px';
    sidebar.style.right      = isOpen ? '-320px' : '0px';
    backdrop.style.display   = isOpen ? 'none'   : 'block';
    document.getElementById('askElionBtn').style.borderColor = isOpen ? '' : '#22d3ee';
}

function _sidebarOpenConverter(suppressMsg) {
    const sidebar  = document.getElementById('elionSidebar');
    const backdrop = document.getElementById('elionSidebarBackdrop');
    sidebar.style.right    = '-320px';
    backdrop.style.display = 'none';
    document.getElementById('askElionBtn').style.borderColor = '';
    _pdb2pdbqtOpen();
    if (!suppressMsg) {
        const mc = document.getElementById('miniChat');
        if (mc && mc.style.display === 'flex') {
            setTimeout(() => _miniChatAppend('ai',
                '⚙️ **Converter is open!**\n\nChoose **Ligand** or **Receptor**, ' +
                'drag your **.pdb** file into the drop zone, then click **Convert**. ' +
                'Once done, copy the output path and paste it into the **RECEPTOR** or **LIGAND** field above.'
            ), 300);
        }
    }
}

// ── Chat endpoints ───────────────────────────────────────────────────────────
function _chatEndpoint() {
    return _activeTool === 'vina'
        ? '/vina_visualization/chat/stream'
        : '/attention_visualization/chat/stream';
}
function _chatClearEndpoint() {
    return _activeTool === 'vina'
        ? '/vina_visualization/chat/clear'
        : '/attention_visualization/chat/clear';
}
function _fetchPresets() {
    const url = (_activeTool === 'vina')
        ? '/vina_visualization/chembert_models'
        : '/attention_visualization/chembert_models';
    $.getJSON(url, function(d) {
        if (d.status !== 'success') return;
        const container = $('#presetBtns').empty();
        d.presets.forEach(function(p) {
            const btn = $('<button>').addClass('preset-btn').text(p.label)
                .attr('data-path', p.path).attr('data-tag', p.tag)
                .on('click', function() { _selectPreset(p.path, p.tag, $(this)); });
            container.append(btn);
        });
    });
}

// ── Draggable panels ─────────────────────────────────────────────────────────
function _makeDraggable(panelId, handle) {
    const panel = document.getElementById(panelId);
    if (!panel || !handle) return;
    let ox=0, oy=0, sx=0, sy=0, drag=false;
    handle.addEventListener('mousedown', e => {
        if (e.target.tagName === 'BUTTON') return;
        drag = true;
        const r = panel.getBoundingClientRect();
        panel.style.cssText += ';left:'+r.left+'px;top:'+r.top+'px;bottom:auto;right:auto;';
        sx=e.clientX; sy=e.clientY; ox=r.left; oy=r.top;
        handle.style.cursor='grabbing'; e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
        if (!drag) return;
        panel.style.left = Math.max(0,Math.min(window.innerWidth-panel.offsetWidth, ox+e.clientX-sx))+'px';
        panel.style.top  = Math.max(0,Math.min(window.innerHeight-panel.offsetHeight,oy+e.clientY-sy))+'px';
    });
    document.addEventListener('mouseup', () => { drag=false; handle.style.cursor='grab'; });
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-drag-handle]').forEach(h => _makeDraggable(h.dataset.dragHandle, h));
    window.addEventListener('resize', () => {
        const vp = document.getElementById('vPlotA');
        if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
    });
});

// ── Highlight system ─────────────────────────────────────────────────────────
function _highlightBtnUntilClick(btnId) {
    const el = document.getElementById(btnId);
    if (!el) return;
    el.classList.add('btn-pulse-wait');
    el.addEventListener('click', function once() {
        el.classList.remove('btn-pulse-wait');
        el.removeEventListener('click', once);
    });
}
function _highlightAmberUntilAction(ids, eventType) {
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.add('amber-pulse');
        el.addEventListener(eventType, function once() {
            ids.forEach(i => { const e=document.getElementById(i); if(e) e.classList.remove('amber-pulse'); });
            el.removeEventListener(eventType, once);
        }, { once: true });
    });
}
function _highlightEmptyPathFields() {
    const missing = ['recPath','ligPath'].filter(id => {
        const el = document.getElementById(id);
        return el && !el.value.trim();
    });
    if (missing.length) _highlightAmberUntilAction(missing, 'input');
}