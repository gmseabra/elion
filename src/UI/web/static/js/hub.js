// =============================================================================
// hub.js — Elion platform hub controller
// Extracted from hub.html. Depends on: vina.js, elion_mini_chat.js (loaded first)
// Feature modules (ts/*, chembert/*, deepatom/*) are loaded on-demand below.
// =============================================================================

// ── Hub routing ───────────────────────────────────────────────────────────────
let _activeTool = null;

// ── Sidebar ───────────────────────────────────────────────────────────────────
function _toggleElionSidebar() {
    const sidebar  = document.getElementById('elionSidebar');
    const backdrop = document.getElementById('elionSidebarBackdrop');
    const isOpen   = sidebar.style.right === '0px';
    if (isOpen) {
        sidebar.style.right    = '-320px';
        backdrop.style.display = 'none';
        document.getElementById('askElionBtn').style.borderColor = '';
    } else {
        sidebar.style.right    = '0px';
        backdrop.style.display = 'block';
        document.getElementById('askElionBtn').style.borderColor = '#22d3ee';
    }
}

// Opens the PDB→PDBQT converter while keeping the Vina mini-chat alive.
function _sidebarOpenConverter(suppressMsg) {
    const sidebar  = document.getElementById('elionSidebar');
    const backdrop = document.getElementById('elionSidebarBackdrop');
    sidebar.style.right      = '-320px';
    backdrop.style.display   = 'none';
    document.getElementById('askElionBtn').style.borderColor = '';
    _pdb2pdbqtOpen();
    if (!suppressMsg) {
        const miniChat = document.getElementById('miniChat');
        if (miniChat && miniChat.style.display === 'flex') {
            setTimeout(() => {
                _miniChatAppend('ai',
                    '⚙️ **Converter is open!**\n\n' +
                    'Choose **Ligand** or **Receptor**, drag your **.pdb** file into the drop zone, ' +
                    'then click **Convert**. ' +
                    'Once done, copy the output path and paste it into the **RECEPTOR** or **LIGAND** field above.'
                );
            }, 300);
        }
    }
}

// ── Sidebar navigation dispatch (data-nav) ────────────────────────────────────
// Sidebar entries in hub.html carry data-nav="<key>" instead of inline onclicks;
// one delegated listener here runs the matching route. Cross-file openers
// (_adjShow, _vinaShow, PoseGen…) are resolved via window[...] at click time
// so their load order doesn't matter.
function _navRun(key) {
    const call = (name, arg) => { const f = window[name]; if (typeof f === 'function') f(arg); };
    switch (key) {
        case 'sidebar':   _toggleElionSidebar(); break;
        case 'chat':      _toggleElionSidebar();
                          { const inp = (typeof _chatActiveInput === 'function') && _chatActiveInput();
                            if (inp && inp.focus) inp.focus(); } break;
        case 'ts':        _toggleElionSidebar(); _activeTool = 'ts';        _clearAllHighlights(); call('_openFeature', 'ts'); break;
        case 'attn':      _toggleElionSidebar(); _activeTool = 'attn';      _clearAllHighlights(); call('_adjShow'); break;
        case 'vina':      _toggleElionSidebar(); _activeTool = 'vina';      _clearAllHighlights(); call('_vinaShow'); break;
        case 'deepatom':  _toggleElionSidebar(); _activeTool = 'deepatom';  _clearAllHighlights(); call('_deepatomShow'); break;
        case 'pose':      _toggleElionSidebar(); if (window.PoseGen && PoseGen.open) PoseGen.open(); break;
        case 'converter': _sidebarOpenConverter(); break;
        // The Reasoning and Get Ligand Center cases are gone. Reasoning is now a
        // plain external link that carries its own onclick and no data-nav, so it
        // never reaches this dispatcher; the Get Ligand Center tool was removed
        // outright. (pose.js's PG.init.ligCenter / PG._ligCenter are pose-frame
        // geometry — unrelated to that tool, and deliberately untouched.)
    }
}
if (!window.__navWired) {
    window.__navWired = true;
    document.addEventListener('click', function (e) {
        const t = (e.target && e.target.closest) ? e.target.closest('[data-nav]') : null;
        if (t) _navRun(t.getAttribute('data-nav'));
    }, false);
}

// ── Draggable mini-chat panels ────────────────────────────────────────────────
(function () {
    function makeDraggable(panelId, handle) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        let ox = 0, oy = 0, startX = 0, startY = 0, dragging = false;
        handle.addEventListener('mousedown', e => {
            if (e.target.tagName === 'BUTTON') return;
            dragging = true;
            const rect = panel.getBoundingClientRect();
            panel.style.left = rect.left + 'px'; panel.style.top = rect.top + 'px';
            panel.style.bottom = 'auto'; panel.style.right = 'auto';
            startX = e.clientX; startY = e.clientY;
            ox = rect.left; oy = rect.top;
            handle.style.cursor = 'grabbing';
            e.preventDefault();
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            const dx = e.clientX - startX, dy = e.clientY - startY;
            const nx = Math.max(0, Math.min(window.innerWidth  - panel.offsetWidth,  ox + dx));
            const ny = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, oy + dy));
            panel.style.left = nx + 'px'; panel.style.top = ny + 'px';
        });
        document.addEventListener('mouseup', () => { dragging = false; handle.style.cursor = 'grab'; });
    }

    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('[data-drag-handle]').forEach(handle => {
            makeDraggable(handle.getAttribute('data-drag-handle'), handle);
        });
        // Pre-fill receptor/ligand paths + docking box from input_TS.yml
        fetch('/vina_visualization/vina_defaults')
            .then(r => r.json())
            .then(data => {
                if (data.status !== 'success') return;
                const rec = document.getElementById('recPath');
                const lig = document.getElementById('ligPath');
                // title too: the fields are 360/380px and these paths run ~100 chars,
                // so the tail is clipped — the tooltip is the only way to read it all.
                if (rec && !rec.value && data.default_receptor) { rec.value = data.default_receptor; rec.placeholder = data.default_receptor; rec.title = data.default_receptor; }
                if (lig && !lig.value && data.default_ligand)   { lig.value = data.default_ligand;   lig.placeholder = data.default_ligand;   lig.title = data.default_ligand; }
                // Same source of truth for the box. /vina_defaults already returns
                // center_x/y/z + size_x/y/z for vina.active_protein; without this the
                // Box ctr / len inputs keep vina.js's hardcoded 8P0M literals, so any
                // other active_protein opens the page on the wrong site.
                if (typeof _applyGridBox === 'function') _applyGridBox(data, { quiet: true });
            })
            .catch(() => {
                const rec = document.getElementById('recPath');
                const lig = document.getElementById('ligPath');
                if (rec) rec.placeholder = '/path/to/receptor.pdbqt';
                if (lig) lig.placeholder = '/path/to/ligand.pdbqt';
            });
    });

    if (document.readyState !== 'loading') {
        document.querySelectorAll('[data-drag-handle]').forEach(handle => {
            makeDraggable(handle.getAttribute('data-drag-handle'), handle);
        });
    }

    window.addEventListener('resize', () => {
        const vp = document.getElementById('vPlotA');
        if (vp && vp._fullLayout) Plotly.Plots.resize(vp);
    });
})();

// ── Chat routing ──────────────────────────────────────────────────────────────
function _chatEndpoint() {
    if (_activeTool === 'vina') return '/vina_visualization/chat/stream';
    return '/attention_visualization/chat/stream';
}
function _chatClearEndpoint() {
    if (_activeTool === 'vina') return '/vina_visualization/chat/clear';
    return '/attention_visualization/chat/clear';
}
function _chatType() {
    if (_activeTool === 'vina')     return 'vina';
    if (_activeTool === 'deepatom') return 'deepatom';
    return 'attn';
}

// ── Highlight helpers ─────────────────────────────────────────────────────────
function _clearAllHighlights() {
    const ALL = [
        'adjWeightBtn','vinaDockBtn','vizBtn','ligLoadBtn','recLoadBtn',
        'gridBoxBtn','voxelInspectBtn','modeLigand','modeProtein',
        'finetuneBtn','loadModelBtn','modeCompare','modeSingle','runSingle','runCompare',
        'vinaBtn',
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

function _highlightBtnUntilClick(btnId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.classList.remove('btn-pulse-wait', 'input-pulse-wait');
    void btn.offsetWidth;
    btn.classList.add('btn-pulse-wait');
    const stop = () => { btn.classList.remove('btn-pulse-wait'); btn.removeEventListener('click', stop); };
    btn.addEventListener('click', stop);
}

function _highlightInputUntilType(inputId, stopBtnId) {
    const el = document.getElementById(inputId);
    if (!el) return;
    el.classList.remove('input-pulse-wait');
    void el.offsetWidth;
    el.classList.add('input-pulse-wait');
    const stop = () => {
        el.classList.remove('input-pulse-wait');
        el.removeEventListener('input', stop);
        if (stopBtnId) {
            const btn = document.getElementById(stopBtnId);
            if (btn) btn.removeEventListener('click', stop);
        }
    };
    el.addEventListener('input', stop);
    if (stopBtnId) {
        const btn = document.getElementById(stopBtnId);
        if (btn) btn.addEventListener('click', stop);
    }
}

// ── PDBQT file attachment ──────────────────────────────────────────────────────
let _attachedFiles   = [];
let _pendingFileObjs = [];

function _onPdbqtAttach(input) {
    _pendingFileObjs = Array.from(input.files);
    input.value = '';
    _updateAttachBadge();
}
function _updateAttachBadge(converting) {
    const n = _pendingFileObjs.length + _attachedFiles.length;
    ['attachBadge','attachBadge2'].forEach(id => {
        const b = document.getElementById(id);
        if (!b) return;
        if (converting) {
            b.textContent = '⟳ converting .pdb…'; b.style.display = 'inline-block';
            b.style.background = '#1e3a5f'; b.style.color = '#93c5fd';
        } else if (n > 0) {
            b.textContent = n + (n === 1 ? ' file ×' : ' files ×'); b.style.display = 'inline-block';
            b.style.background = '#164e63'; b.style.color = '#a5f3fc';
        } else {
            b.style.display = 'none';
        }
    });
    const lbl = document.getElementById('attachLabel');
    if (lbl) lbl.style.color = n > 0 ? '#22d3ee' : '#4b5563';
}
function _clearAttachedFiles() {
    _attachedFiles = []; _pendingFileObjs = [];
    _updateAttachBadge();
}
async function _uploadPdbqtFiles(fileObjs) {
    const form = new FormData();
    fileObjs.forEach(f => form.append('files[]', f));
    const resp = await fetch('/hub/upload_pdbqt', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.status !== 'success') throw new Error(data.message || 'Upload failed');
    return data.files;
}
function _handlePdbqtReady(u) {
    const lig = u.ligand_path || '', rec = u.receptor_path || '';
    _clearAllHighlights();
    const vb = document.getElementById('vinaBtn');
    if (vb) { vb.classList.add('btn-pulse-wait'); setTimeout(() => vb.classList.remove('btn-pulse-wait'), 900); }
    const mo = document.getElementById('miniChatOutput');
    if (mo) mo.innerHTML = '';
    setTimeout(() => {
        _activeTool = 'vina'; _vinaShow();
        const le = document.getElementById('ligPath'), re = document.getElementById('recPath');
        if (le && lig) le.value = lig;
        if (re && rec) re.value = rec;
        setTimeout(() => {
            _highlightAmberUntilAction(['recPath','ligPath'], 'input');
            _highlightBtnUntilClick('vinaDockBtn');
            const mo2 = document.getElementById('miniChatOutput');
            if (mo2) mo2.innerHTML = '';
            const anyConverted = u.files && u.files.some(f => f.converted);
            let msg = anyConverted ? '🔄 PDB converted to PDBQT!\n\n' : '📂 Files uploaded!\n\n';
            if (rec) msg += '**Receptor:**\n`' + rec + '`\n\n';
            if (lig) msg += '**Ligand:**\n`' + lig + '`\n\n';
            msg += 'Paths are pre-filled ✓ — verify, then click the flashing **Vina Dock** button.';
            _miniChatAppend('ai', msg);
        }, 500);
    }, 300);
}

// ── Qwen chat ─────────────────────────────────────────────────────────────────
let _chatStream = null;
let _msgCount   = 0;

function _chatActiveInput() {
    const sticky = document.getElementById('chatInputSticky');
    if (sticky && sticky.style.display !== 'none') return document.getElementById('chatInputSticky2');
    return document.getElementById('chatInput');
}

function _chatAppend(role, text) {
    const out = document.getElementById('chatOutput');
    if (_msgCount === 0) {
        const es = document.getElementById('chatEmptyState');
        if (es) es.style.display = 'none';
        out.style.display = 'flex';
        const sticky = document.getElementById('chatInputSticky');
        if (sticky) sticky.style.display = 'block';
        const centeredVal = (document.getElementById('chatInput') || {}).value || '';
        const stickyTA = document.getElementById('chatInputSticky2');
        if (stickyTA && centeredVal) stickyTA.value = centeredVal;
    }
    _msgCount++;
    const wrap = document.createElement('div');
    wrap.style.cssText = 'width:100%;max-width:760px;margin:0 auto;padding:0 24px;box-sizing:border-box;';
    const div = document.createElement('div');
    div.className = role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai';
    if (role === 'ai') {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
        const av = document.createElement('div');
        av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
        av.textContent = '🧬';
        row.appendChild(av); row.appendChild(div); wrap.appendChild(row);
    } else {
        wrap.style.display = 'flex'; wrap.style.justifyContent = 'flex-end';
        wrap.appendChild(div);
    }
    div.innerHTML = (text || '')
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
    const row  = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
    const av   = document.createElement('div');
    av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
    av.textContent = '🧬';
    const dots = document.createElement('div');
    dots.className = 'chat-bubble-ai'; dots.id = 'chatTyping'; dots.style.padding = '10px 0';
    dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    row.appendChild(av); row.appendChild(dots); wrap.appendChild(row);
    out.appendChild(wrap); out.scrollTop = out.scrollHeight;
}
function _chatHideTyping() {
    document.getElementById('chatTypingWrap')?.remove();
    document.getElementById('chatTyping')?.remove();
}

function _chatGetContext() {
    const ctx = {};
    const smilesEl = document.getElementById('smiles1') || document.querySelector('.smiles-mono');
    if (smilesEl && smilesEl.value) ctx.smiles = smilesEl.value.trim();
    const scoreEl = document.querySelector('.score-val') || document.getElementById('scoreVal');
    if (scoreEl) { const v = parseFloat(scoreEl.textContent); if (!isNaN(v)) ctx.score = v; }
    const activeModel = document.querySelector('.preset-btn.active, .preset-btn.pretrained-active');
    if (activeModel) ctx.model = activeModel.textContent.trim();
    const modeEl = document.querySelector('.mode-btn.active');
    if (modeEl) ctx.mode = modeEl.textContent.trim();
    return ctx;
}

function _chatSend() {
    const input = _chatActiveInput();
    const text  = input.value.trim();
    if (!text && !_pendingFileObjs.length) return;
    if (_chatStream) return;
    input.value = ''; input.style.height = 'auto';
    if (_pendingFileObjs.length) {
        const snap  = _pendingFileObjs.slice();
        _pendingFileObjs = [];
        const hasPdb = snap.some(f => f.name.toLowerCase().endsWith('.pdb'));
        _updateAttachBadge(hasPdb);
        const names = snap.map(f => f.name).join(', ');
        _chatAppend('user', text || (hasPdb ? 'Convert PDB→PDBQT and dock: ' : 'Run Vina docking with: ') + names);
        _chatShowTyping();
        _activeTool = 'vina';
        _uploadPdbqtFiles(snap).then(uploaded => {
            _attachedFiles = uploaded;
            const ctx = _chatGetContext();
            ctx.uploaded_pdbqt = uploaded;
            uploaded.forEach(f => {
                if (f.role === 'ligand'   && !ctx.ligand_path)   ctx.ligand_path   = f.path;
                if (f.role === 'receptor' && !ctx.receptor_path) ctx.receptor_path = f.path;
            });
            _doStreamSend(text || ('Run Vina docking with ' + names), ctx);
        }).catch(err => { _chatHideTyping(); _chatAppend('ai', '⚠️ Upload failed: ' + err.message); });
        return;
    }
    _chatAppend('user', text);
    _chatShowTyping();
    _doStreamSend(text, _chatGetContext());
}

function _doStreamSend(text, ctx) {
    let aiDiv = null, fullText = '', firstToken = true;
    _chatStream = 'main';
    fetch(_chatEndpoint(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        const reader = resp.body.getReader();
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
                        if (firstToken) { _chatHideTyping(); firstToken = false; aiDiv = _chatAppend('ai', ''); fullText = ''; }
                        fullText += payload;
                        if (aiDiv) {
                            aiDiv.innerHTML = fullText
                                .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                                .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
                                .replace(/`([^`]+)`/g,'<code>$1</code>')
                                .replace(/\n/g,'<br>');
                            _smartScroll('chatOutput');
                        }
                    } else if (ev === 'ui_action') {
                        if (payload) {
                            if (payload.action === 'pdbqt_ready') _handlePdbqtReady(payload);
                            else if (_activeTool === 'vina')      handleVinaUiAction(payload);
                            else                                   handleAttnUiAction(payload);
                        }
                    } else if (ev === 'done') {
                        _chatHideTyping(); _chatStream = null;
                        $('#streamBubble').removeAttr('id'); $('#streamContent').removeAttr('id');
                    } else if (ev === 'error') {
                        _chatHideTyping(); _chatStream = null;
                        _chatAppend('ai', '⚠️ ' + (payload || 'Server error'));
                    }
                }
                pump();
            }).catch(err => { _chatHideTyping(); _chatStream = null; _chatAppend('ai', '⚠️ Stream error: ' + err.message); });
        }
        pump();
    }).catch(err => {
        _chatHideTyping(); _chatStream = null;
        console.error('[Elion] fetch error:', err);
        _chatAppend('ai', '⚠️ Could not reach Qwen server. Is it running on port 8001?');
    });
}

function _qprompt(text) {
    _chatActiveInput().value = text;
    _chatSend();
}

function _chatClear() {
    const out = document.getElementById('chatOutput');
    out.innerHTML = ''; out.style.display = 'none';
    _msgCount = 0;
    const es = document.getElementById('chatEmptyState');
    if (es) es.style.display = 'flex';
    const sticky = document.getElementById('chatInputSticky');
    if (sticky) sticky.style.display = 'none';
    const pills = document.getElementById('chatQuickPills');
    if (pills) pills.style.display = 'none';
    ['chatInput','chatInputSticky2'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.value = ''; el.style.height = 'auto'; }
    });
    fetch(_chatClearEndpoint(), { method: 'POST' });
}

document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); _chatActiveInput().focus(); }
});

// ── Attn UI action handler ────────────────────────────────────────────────────
const _ATTN_ACTIONS = {
    open_visualizer: { btnId: 'adjWeightBtn' },
    run_finetune:    { btnId: 'finetuneBtn' },
    load_model:      { btnId: 'loadModelBtn' },
    compare_mode:    { btnId: 'modeCompare' },
    show_3d:         { btnId: 'runSingle' },
};
const _ATTN_ALL_BTNS   = ['adjWeightBtn','finetuneBtn','loadModelBtn','modeCompare','modeSingle','runSingle','runCompare'];
const _ATTN_ALL_INPUTS = ['smilesA','smilesCompA','smilesCompB','customModelPath'];

function _clearAttnHighlights() {
    _ATTN_ALL_BTNS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('btn-pulse-wait','btn-glow','btn-highlight');
    });
    _ATTN_ALL_INPUTS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('input-pulse-wait','amber-pulse');
    });
}

function handleAttnUiAction(ui_action) {
    if (!ui_action || ui_action.action === 'none') return;
    setTimeout(() => _clearAttnHighlights(), 0);
    if (ui_action.action === 'open_visualizer') {
        setTimeout(() => { _chatAppend('ai', 'Click the flashing **🧠 ChemBERT** button above ↑ to open the visualizer.'); _highlightBtnUntilClick('adjWeightBtn'); }, 400);
        return;
    }
    if (ui_action.action === 'open_vina') {
        setTimeout(() => { _chatAppend('ai', 'Click the flashing **🔬 Vina Docking** button above ↑ to open the docking visualizer.'); _highlightBtnUntilClick('vinaBtn'); }, 400);
        return;
    }
    if (ui_action.action === 'compare_mode') {
        setTimeout(() => { _highlightBtnUntilClick('modeCompare'); }, 400);
        return;
    }
    if (ui_action.action === 'show_3d') {
        setTimeout(() => { _highlightInputUntilType('smilesA', 'runSingle'); _highlightBtnUntilClick('runSingle'); }, 400);
        return;
    }
    const entry = _ATTN_ACTIONS[ui_action.action];
    if (!entry) return;
    setTimeout(() => _highlightBtnUntilClick(entry.btnId), 400);
}

// ── Feature loader (serial, lazy) ─────────────────────────────────────────────
// Scripts only load when a feature is first opened — not on page load.
const _FEATURES = {
    ts: {
        scripts: [
            'ts/ts_core.js', 'ts/ts_chart.js', 'ts/ts_ui.js',
            'ts/ts_warmup.js', 'ts/ts_worker_bridge.js', 'ts/ts_run.js',
            // ts_rl.js — the 🧪 RL diagnostics tab. MUST stay last: it reads
            // _TS_PANE / _TS_TAB_ACTIVE / _ts, all of which are top-level
            // `const` in ts_core.js. Top-level const does NOT become a window
            // property, so ts_rl.js reaches them as bare identifiers — which
            // only resolves if ts_core.js has already executed. _loadSerial is
            // serial, so array order IS execution order. Do not reorder.
            'ts/ts_rl.js',
        ],
        loaded: false,
        onload: () => {
            // Match the defensive opener pattern _navRun uses for every other feature
            // (a window[...] lookup): if the TS UI script defined _tsShow, call it. If
            // it didn't — renamed, or one of the ts/*.js above failed to load (look for
            // "[Hub] failed to load: …/ts/…" in the console) — fall back to opening
            // #tsModal directly rather than throwing "_tsShow is not defined". #tsModal
            // uses the same hidden/flex classes as the other feature modals.
            if (typeof window._tsShow === 'function') { window._tsShow(); return; }
            const m = document.getElementById('tsModal');
            if (m) { m.classList.remove('hidden'); m.classList.add('flex'); }
            else console.error('[Hub] TS open: _tsShow is undefined and #tsModal was not found');
        },
    },
};

function _loadSerial(urls, done, i = 0) {
    if (i >= urls.length) { done(); return; }
    const base = window._HUB_FEATURE_BASE || '/static/js/';
    const s = document.createElement('script');
    s.src = base + urls[i] + '?v=' + _HUB_VERSION;
    s.onload  = () => _loadSerial(urls, done, i + 1);
    s.onerror = () => { console.error('[Hub] failed to load:', s.src); _loadSerial(urls, done, i + 1); };
    document.head.appendChild(s);
}

function _openFeature(name) {
    const feat = _FEATURES[name];
    if (!feat) { console.warn('[Hub] unknown feature:', name); return; }
    if (feat.loaded) { feat.onload(); return; }
    _loadSerial(feat.scripts, () => { feat.loaded = true; feat.onload(); });
}

// Cache-bust version — update when deploying new JS
const _HUB_VERSION = '20260613';

// ── Eager loaders (non-TS features, loaded at startup) ────────────────────────
// url_for paths are injected by hub.html via window._HUB_EAGER_SCRIPTS before
// this file loads. Falls back to /static/js/<name> if not set.
(function () {
    var eager = window._HUB_EAGER_SCRIPTS || [
        { name: 'deepatom_chat.js',          url: '/static/js/deepatom_chat.js' },
        { name: 'deepatom.js',               url: '/static/js/deepatom.js' },
        { name: 'chembert_chat.js',          url: '/static/js/chembert_chat.js' },
        { name: 'vina_mini_chat_welcome.js', url: '/static/js/vina_mini_chat_welcome.js' },
    ];
    function next(i) {
        if (i >= eager.length) return;
        var f = eager[i];
        var s = document.createElement('script');
        s.src = f.url;
        s.onload  = function () { console.log('[Hub] Loaded ' + f.name); next(i + 1); };
        s.onerror = function () {
            console.error('[Hub] Failed to load ' + f.name + ' from ' + f.url);
            var s2 = document.createElement('script');
            s2.src = '/static/js/' + f.name;
            s2.onload  = function () { next(i + 1); };
            s2.onerror = function () { console.error('[Hub] All paths failed for ' + f.name); next(i + 1); };
            document.head.appendChild(s2);
        };
        document.head.appendChild(s);
    }
    next(0);
})();

// ── Model picker ──────────────────────────────────────────────────────────────
let _elionActiveModel = { key: 'gpt-oss-120b', label: 'GPT-OSS 120B', badge: 'GPT-OSS 120B' };
let _modelPickerOpen  = false;
let _sessionPanelOpen = false;

function _elionUpdateModelLabels(model) {
    _elionActiveModel = model;
    document.querySelectorAll('.elion-model-label').forEach(el => { el.textContent = model.badge || model.label; });
}

function _toggleModelPicker(evt) {
    if (evt) evt.stopPropagation();
    _modelPickerOpen = !_modelPickerOpen;
    const picker = document.getElementById('elionModelPicker');
    if (!picker) return;
    if (_modelPickerOpen) {
        const trigger = evt && evt.currentTarget
            ? evt.currentTarget
            : (document.getElementById('chatModelBadge1') || document.getElementById('chatModelBadge2'));
        if (trigger) {
            const rect = trigger.getBoundingClientRect();
            const pickerH = 260, pickerW = 280;
            let left = rect.left;
            if (left + pickerW > window.innerWidth - 8) left = window.innerWidth - pickerW - 8;
            left = Math.max(8, left);
            if (window.innerHeight - rect.bottom >= pickerH || window.innerHeight - rect.bottom >= rect.top) {
                picker.style.top = (rect.bottom + 6) + 'px'; picker.style.bottom = 'auto';
            } else {
                picker.style.bottom = (window.innerHeight - rect.top + 6) + 'px'; picker.style.top = 'auto';
            }
            picker.style.left = left + 'px'; picker.style.right = 'auto';
        } else {
            picker.style.bottom = '100px'; picker.style.right = '24px'; picker.style.top = 'auto'; picker.style.left = 'auto';
        }
        picker.style.display = 'block';
        _renderModelPicker();
    } else {
        picker.style.display = 'none';
    }
}

document.addEventListener('click', e => {
    if (!_modelPickerOpen) return;
    const picker = document.getElementById('elionModelPicker');
    const badge1 = document.getElementById('chatModelBadge1');
    const badge2 = document.getElementById('chatModelBadge2');
    const outside = el => !el || !el.contains(e.target);
    if (outside(picker) && outside(badge1) && outside(badge2)) {
        _modelPickerOpen = false;
        if (picker) picker.style.display = 'none';
    }
});

function _renderModelPicker() {
    const list = document.getElementById('elionModelList');
    if (!list) return;
    list.innerHTML = '<div style="padding:12px;color:#475569;font-size:12px;text-align:center;">Loading…</div>';
    fetch('/session/models')
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(data => {
            list.innerHTML = '';
            const activeKey = data.active_model ? data.active_model.key : _elionActiveModel.key;
            if (data.active_model) _elionUpdateModelLabels(data.active_model);
            data.models.forEach(m => {
                const isActive   = m.key === activeKey;
                const isDisabled = !m.active;
                const btn = document.createElement('button');
                btn.style.cssText = [
                    'width:100%;text-align:left;padding:10px 12px;border-radius:8px;',
                    'font-family:inherit;transition:background .12s;display:flex;align-items:center;gap:12px;',
                    isActive   ? 'background:rgba(34,211,238,0.08);border:1px solid #0e7490;cursor:default;'
                    : isDisabled ? 'background:transparent;border:1px solid transparent;cursor:not-allowed;opacity:0.5;'
                                 : 'background:transparent;border:1px solid transparent;cursor:pointer;',
                ].join('');
                const check = document.createElement('div');
                check.style.cssText = 'width:16px;flex-shrink:0;display:flex;align-items:center;justify-content:center;';
                check.innerHTML = isActive ? '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7L5.5 10.5L12 3.5" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>' : '';
                btn.appendChild(check);
                const text = document.createElement('div');
                text.style.cssText = 'flex:1;';
                text.innerHTML = `
                    <div style="font-size:13px;font-weight:600;color:${isActive?'#e2e8f0':'#94a3b8'};">
                        ${m.label}
                        ${isDisabled ? '<span style="margin-left:6px;font-size:10px;padding:1px 7px;border-radius:20px;background:#1e293b;color:#475569;font-weight:400;">'+m.env+' only</span>' : ''}
                    </div>
                    <div style="font-size:11px;color:#475569;margin-top:2px;">${m.description}</div>`;
                btn.appendChild(text);
                if (!isActive && !isDisabled) {
                    btn.onmouseover = () => btn.style.background = 'rgba(255,255,255,0.04)';
                    btn.onmouseout  = () => btn.style.background = 'transparent';
                    btn.onclick = () => _elionSetModel(m.key);
                } else if (isDisabled) {
                    btn.title = m.label + ' is only available on the ' + m.env + ' environment.';
                }
                list.appendChild(btn);
            });
        })
        .catch(err => {
            console.error('[Elion] /session/models error:', err);
            // Cached fallback: the models this NaviGator key actually allows.
            const cached = [
                { key: 'gpt-oss-120b',               label: 'GPT-OSS 120B',          description: 'OpenAI · 120B · NaviGator' },
                { key: 'nemotron-3-super-120b-a12b', label: 'Nemotron 3 Super 120B', description: 'NVIDIA · 120B · NaviGator' },
            ];
            const activeKey = _elionActiveModel.key;
            list.innerHTML = '';
            cached.forEach(m => {
                const isActive = m.key === activeKey;
                const btn = document.createElement('button');
                btn.style.cssText = [
                    'width:100%;text-align:left;padding:10px 12px;border-radius:8px;',
                    'font-family:inherit;transition:background .12s;display:flex;align-items:center;gap:12px;',
                    isActive ? 'background:rgba(34,211,238,0.08);border:1px solid #0e7490;cursor:default;'
                             : 'background:transparent;border:1px solid transparent;cursor:pointer;',
                ].join('');
                const check = document.createElement('div');
                check.style.cssText = 'width:16px;flex-shrink:0;display:flex;align-items:center;justify-content:center;';
                check.innerHTML = isActive ? '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7L5.5 10.5L12 3.5" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>' : '';
                btn.appendChild(check);
                const text = document.createElement('div');
                text.style.cssText = 'flex:1;';
                text.innerHTML = `
                    <div style="font-size:13px;font-weight:600;color:${isActive ? '#e2e8f0' : '#94a3b8'};">${m.label}</div>
                    <div style="font-size:11px;color:#475569;margin-top:2px;">${m.description}</div>`;
                btn.appendChild(text);
                if (!isActive) {
                    btn.onmouseover = () => btn.style.background = 'rgba(255,255,255,0.04)';
                    btn.onmouseout  = () => btn.style.background = 'transparent';
                    btn.onclick = () => _elionSetModel(m.key);
                }
                list.appendChild(btn);
            });
            const note = document.createElement('div');
            note.style.cssText = 'padding:4px 12px 10px;font-size:10px;color:#334155;';
            note.textContent = '(cached — /session/models unreachable)';
            list.appendChild(note);
        });
}

function _elionSetModel(key) {
    fetch('/session/set_model', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: key })
    })
    .then(r => r.json())
    .then(data => {
        if (!data.ok) { alert(data.error || 'Cannot switch model'); return; }
        _elionUpdateModelLabels(data.active_model);
        _modelPickerOpen = false;
        document.getElementById('elionModelPicker').style.display = 'none';
    })
    .catch(err => console.error('[Elion] set_model error:', err));
}

// ── Session panel ─────────────────────────────────────────────────────────────
function _toggleSessionPanel() {
    _sessionPanelOpen = !_sessionPanelOpen;
    const panel    = document.getElementById('elionSessionPanel');
    const backdrop = document.getElementById('elionSessionBackdrop');
    if (!panel) return;
    if (_sessionPanelOpen) {
        panel.style.display    = 'flex';
        backdrop.style.display = 'block';
        _loadSessionPanel();
    } else {
        panel.style.display    = 'none';
        backdrop.style.display = 'none';
    }
}

function _loadSessionPanel() {
    fetch('/session/sessions')
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(data => {
            const ipEl = document.getElementById('elionSessionIP');
            if (ipEl) ipEl.textContent = 'Your IP: ' + data.ip;
            const list = document.getElementById('elionSessionList');
            if (!list) return;
            list.innerHTML = '';
            if (!data.sessions || data.sessions.length === 0) {
                list.innerHTML = '<div style="color:#475569;font-size:12px;text-align:center;padding:30px 12px;line-height:1.7;">No sessions yet.<br>Start a conversation in the<br>Vina or ChemBERT chat to create one.</div>';
                return;
            }
            data.sessions.forEach(s => {
                const card = document.createElement('div');
                card.style.cssText = 'background:#111827;border:1px solid #1e293b;border-radius:12px;padding:12px;';
                const header = document.createElement('div');
                header.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:8px;';
                header.innerHTML = `
                    <span style="font-size:18px;">${s.icon}</span>
                    <div style="flex:1;">
                        <div style="font-size:13px;font-weight:600;color:#e2e8f0;">${s.label}</div>
                        <div style="font-size:10px;color:#475569;margin-top:1px;">${s.turn_count} turn${s.turn_count!==1?'s':''} · model: ${s.model}</div>
                    </div>
                    <button onclick="_elionClearSession('${s.chat_type}')" style="background:none;border:1px solid #1e293b;border-radius:6px;cursor:pointer;color:#475569;font-size:10px;padding:3px 8px;font-family:inherit;transition:all .15s;"
                        onmouseover="this.style.borderColor='#f87171';this.style.color='#f87171'"
                        onmouseout="this.style.borderColor='#1e293b';this.style.color='#475569'">Clear</button>`;
                card.appendChild(header);
                if (s.preview && s.preview.length) {
                    s.preview.forEach(t => {
                        const turn = document.createElement('div');
                        const isUser = t.role === 'user';
                        turn.style.cssText = [
                            'font-size:11px;padding:5px 8px;border-radius:6px;margin-bottom:3px;',
                            'border-left:2px solid ' + (isUser ? '#22d3ee' : '#334155') + ';',
                            'background:' + (isUser ? 'rgba(34,211,238,0.04)' : 'rgba(255,255,255,0.02)') + ';',
                            'color:' + (isUser ? '#e2e8f0' : '#94a3b8') + ';',
                            'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;',
                        ].join('');
                        turn.title = t.content;
                        turn.textContent = (isUser ? 'You: ' : 'Elion: ') + t.content;
                        card.appendChild(turn);
                    });
                }
                const ts = document.createElement('div');
                ts.style.cssText = 'font-size:10px;color:#334155;margin-top:8px;';
                ts.textContent = 'Last active ' + new Date(s.updated_at).toLocaleString();
                card.appendChild(ts);
                list.appendChild(card);
            });
        })
        .catch(err => {
            console.error('[Elion] sessions error:', err);
            const list = document.getElementById('elionSessionList');
            if (list) list.innerHTML = '<div style="color:#f87171;font-size:12px;padding:12px;">Could not load sessions.<br><span style="color:#334155;font-size:10px;">Is session_routes.py deployed?</span></div>';
        });
}

function _elionClearSession(chatType) {
    if (!confirm('Clear all ' + (chatType === 'vina' ? 'Vina Docking' : 'ChemBERT') + ' history for your IP?')) return;
    fetch('/session/clear', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat: chatType })
    }).then(() => _loadSessionPanel());
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    fetch('/session/models')
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(data => { if (data.active_model) _elionUpdateModelLabels(data.active_model); })
        .catch(() => { /* use defaults */ });
});