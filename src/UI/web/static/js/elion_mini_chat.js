// =============================================================================
// elion_mini_chat.js
// Base class for all Elion mini-chat panels.
//
// ARCHITECTURE
// ────────────
// This file owns:
//   1. Theme registry (_MINICHAT_*_THEME) — one entry per tool
//   2. _miniChatSetContext(theme) — applies a theme to the shared panel
//   3. _miniChatWelcomeBase(opts) — renders a generic welcome card
//   4. Per-tool welcome functions that extend the base:
//        _vinaMiniWelcome()        — Vina Docking
//        _attnMiniWelcome()        — ChemBERT Attention
//        _deepatomMiniWelcome()    — DeepAtom CNN Saliency
//        _tsMiniWelcome()          — Thompson Sampling  ← NEW
//
// Each tool's show/hide function (in vina.js, deepatom.js, ts.js, attn.js)
// just calls:
//   _miniChatSetContext(_MINICHAT_<TOOL>_THEME);
//   _miniChatShow();
//   _<tool>MiniWelcome();      // if output is empty or from a different tool
//
// Adding a new tool:
//   1. Add a _MINICHAT_NEWTOOL_THEME entry below
//   2. Add a _newtoolMiniWelcome() function below
//   3. Call both from the tool's show function
// =============================================================================


// =============================================================================
// ── 1. Theme registry ─────────────────────────────────────────────────────────
// Each theme drives the shared miniChat panel's visual identity.
// Fields: icon, gradient, title, placeholder, accentColor, sendBg
// =============================================================================

const _MINICHAT_VINA_THEME = {
    icon:        '🔬',
    gradient:    'linear-gradient(135deg,#22d3ee,#3b82f6)',
    title:       'Elion · Vina Docking',
    placeholder: 'Ask about docking, SMILES, binding affinity…',
    accentColor: '#22d3ee',
    sendBg:      '#0e7490',
    chatType:    'vina',
};

const _MINICHAT_ATTN_THEME = {
    icon:        '🧠',
    gradient:    'linear-gradient(135deg,#a78bfa,#7c3aed)',
    title:       'Elion · ChemBERT Attn',
    placeholder: 'Ask about attention weights, SMILES, binding…',
    accentColor: '#a78bfa',
    sendBg:      '#5b21b6',
    chatType:    'attn',
};

const _MINICHAT_DEEPATOM_THEME = {
    icon:        '⚛️',
    gradient:    'linear-gradient(135deg,#f59e0b,#d97706)',
    title:       'Elion · DeepAtom',
    placeholder: 'Ask about CNN saliency…',
    accentColor: '#f59e0b',
    sendBg:      '#b45309',
    chatType:    'deepatom',
};

const _MINICHAT_TS_THEME = {
    icon:        '🎲',
    gradient:    'linear-gradient(135deg,#7c3aed,#4c1d95)',
    title:       'Elion · Thompson Sampling',
    placeholder: 'Ask about TS runs, SMARTS, reagents, scores…',
    accentColor: '#7c3aed',
    sendBg:      '#4c1d95',
    chatType:    'ts',
};


// =============================================================================
// ── 2. _miniChatSetContext(theme) ─────────────────────────────────────────────
// Applies a theme to every visual element of the shared miniChat panel.
// Call this before _miniChatShow() and before the welcome function.
// =============================================================================

function _miniChatSetContext(theme) {
    // Header icon gradient + emoji
    const iconEl  = document.querySelector('#miniChat [data-drag-handle] div');
    // Header title text
    const titleEl = document.querySelector('#miniChat [data-drag-handle] span');
    // Input placeholder + focus ring
    const inputEl = document.getElementById('miniChatInput');
    // Send button background
    const sendBtn = document.querySelector('#miniChat button[onclick="_miniChatSend()"]');

    if (iconEl) {
        iconEl.style.background = theme.gradient;
        iconEl.textContent      = theme.icon;
    }
    if (titleEl) titleEl.textContent = theme.title;
    if (inputEl) {
        inputEl.placeholder = theme.placeholder;
        inputEl.style.setProperty('--accent', theme.accentColor);
        inputEl.onfocus = () => { inputEl.style.borderColor = theme.accentColor; };
        inputEl.onblur  = () => { inputEl.style.borderColor = '#1e293b'; };
    }
    if (sendBtn) {
        sendBtn.style.background = theme.sendBg;
        sendBtn.onmouseover = () => { sendBtn.style.filter = 'brightness(1.15)'; };
        sendBtn.onmouseout  = () => { sendBtn.style.filter = 'none'; };
    }

    // Store active chat_type so _attnMiniSend / vina send know which session to use
    window._elionActiveChatType = theme.chatType || 'vina';
}


// =============================================================================
// ── 3. _miniChatWelcomeBase(opts) ─────────────────────────────────────────────
// Renders a generic welcome card. All per-tool welcome functions call this first
// to clear the output and show a branded greeting, then append tool-specific UI.
//
// opts = {
//   icon:     string  — emoji or image
//   name:     string  — tool name in bold colour
//   color:    string  — CSS colour for name
//   intro:    string  — one-line description (HTML allowed)
//   workflow: string  — brief workflow hint (HTML allowed, optional)
// }
// Returns the output element so callers can append more children.
// =============================================================================

function _miniChatWelcomeBase(opts) {
    const out = document.getElementById('miniChatOutput');
    if (!out) return null;
    out.innerHTML = '';

    const greet = document.createElement('div');
    greet.style.cssText = [
        'align-self:flex-start;max-width:92%;',
        'color:#cbd5e1;font-size:12px;line-height:1.7;',
    ].join('');
    greet.innerHTML =
        `Hi! I'm your <strong style="color:${opts.color}">${opts.name}</strong> guide. ${opts.icon}<br><br>` +
        opts.intro +
        (opts.workflow ? `<br><br><span style="color:#475569;font-size:11px;">${opts.workflow}</span>` : '');
    out.appendChild(greet);
    out.scrollTop = out.scrollHeight;
    return out;
}


// =============================================================================
// ── 4a. Vina Docking welcome ──────────────────────────────────────────────────
// Tries the protein-selector flow first; falls back to .pdbqt / convert buttons.
// (Extracted from vina_mini_chat_welcome.js — same logic, now lives here.)
// =============================================================================

function _vinaMiniWelcome() {
    const out = _miniChatWelcomeBase({
        icon:     '👋',
        name:     'Vina Docking',
        color:    '#22d3ee',
        intro:    'Which <strong style="color:#22d3ee">protein target</strong> would you like to dock against?',
    });
    if (!out) return;

    const loadingRow = document.createElement('div');
    loadingRow.style.cssText = 'align-self:flex-start;color:#475569;font-size:11px;';
    loadingRow.textContent = 'Loading available proteins…';
    out.appendChild(loadingRow);
    out.scrollTop = out.scrollHeight;

    fetch('/vina_visualization/vina_proteins')
        .then(r => r.json())
        .then(data => {
            loadingRow.remove();
            if (data.status !== 'success' || !data.proteins?.length) {
                _vinaMiniWelcomeFallback(out);
                return;
            }
            const proteins  = data.proteins;
            const activeId  = data.active_protein || proteins[0].id;
            const row = document.createElement('div');
            row.id = 'miniWelcomeActions';
            row.style.cssText = 'display:flex;flex-direction:column;gap:6px;align-self:flex-start;width:100%;';

            proteins.forEach(protein => {
                const isActive = protein.id === activeId;
                const btn = document.createElement('button');
                btn.style.cssText = [
                    'padding:8px 12px;border-radius:10px;font-size:11px;font-weight:600;',
                    'cursor:pointer;font-family:inherit;text-align:left;transition:background .15s,border-color .15s;',
                    `background:${isActive ? '#0c3a5a' : '#0a1628'};`,
                    `border:1px solid ${isActive ? '#22d3ee' : '#1e3a5f'};`,
                    `color:${isActive ? '#e0f2fe' : '#94a3b8'};`,
                ].join('');

                const labelLine = document.createElement('div');
                labelLine.style.cssText = 'display:flex;align-items:center;gap:6px;';
                labelLine.innerHTML =
                    `<span style="font-size:13px;">🧬</span>` +
                    `<span style="color:${isActive ? '#67e8f9' : '#7dd3fc'};font-size:12px;">${protein.label}</span>` +
                    (isActive ? '<span style="font-size:9px;color:#22d3ee;background:#0e4f63;padding:1px 5px;border-radius:4px;margin-left:auto;">active</span>' : '');
                btn.appendChild(labelLine);

                if (protein.description) {
                    const desc = document.createElement('div');
                    desc.style.cssText = 'font-size:10px;color:#475569;margin-top:3px;font-weight:400;';
                    desc.textContent = protein.description;
                    btn.appendChild(desc);
                }
                btn.onmouseover = () => { btn.style.background='#0e3a5a'; btn.style.borderColor='#22d3ee'; };
                btn.onmouseout  = () => { btn.style.background=isActive?'#0c3a5a':'#0a1628'; btn.style.borderColor=isActive?'#22d3ee':'#1e3a5f'; };
                btn.onclick = () => _selectProtein(protein, row);
                row.appendChild(btn);
            });
            out.appendChild(row);
            out.scrollTop = out.scrollHeight;
        })
        .catch(() => { loadingRow.remove(); _vinaMiniWelcomeFallback(out); });
}

function _vinaMiniWelcomeFallback(out) {
    const note = document.createElement('div');
    note.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    note.innerHTML = 'Do you already have your <strong style="color:#22d3ee">.pdbqt</strong> files ready, ' +
        'or do you need to convert a <strong style="color:#6ee7b7">.pdb</strong> file first?';
    out.appendChild(note);

    const row = document.createElement('div');
    row.id = 'miniWelcomeActions';
    row.style.cssText = 'display:flex;flex-direction:column;gap:6px;align-self:flex-start;width:100%;';

    const mkBtn = (bg, border, color, html, over, out_, click) => {
        const b = document.createElement('button');
        b.style.cssText = `padding:7px 12px;border-radius:10px;font-size:11px;font-weight:600;cursor:pointer;background:${bg};border:1px solid ${border};color:${color};font-family:inherit;text-align:left;transition:background .15s,border-color .15s;`;
        b.innerHTML = html;
        b.onmouseover = over; b.onmouseout = out_; b.onclick = click;
        return b;
    };

    row.appendChild(mkBtn('#083344','#0e7490','#67e8f9',
        '✅ &nbsp;I have .pdbqt files — show me where to load them',
        () => { row.querySelectorAll('button')[0].style.cssText += 'background:#0e4f63;border-color:#22d3ee;'; },
        () => { row.querySelectorAll('button')[0].style.cssText += 'background:#083344;border-color:#0e7490;'; },
        () => { row.remove(); _miniChatAppend('user','I have .pdbqt files ready.'); _miniChatAppend('ai','Got it! Enter your paths in the **RECEPTOR** and **LIGAND** fields above, then click **Vina Dock**.'); setTimeout(_highlightEmptyPathFields, 300); }
    ));
    row.appendChild(mkBtn('#052e1c','#065f46','#6ee7b7',
        '⚙️ &nbsp;I need to convert a .pdb file to .pdbqt first',
        () => { row.querySelectorAll('button')[1].style.cssText += 'background:#064e3b;border-color:#34d399;'; },
        () => { row.querySelectorAll('button')[1].style.cssText += 'background:#052e1c;border-color:#065f46;'; },
        () => { row.remove(); _miniChatAppend('user','I need to convert a .pdb file first.'); _miniChatAppend('ai','Opening the **PDB → PDBQT** converter now.'); setTimeout(() => _sidebarOpenConverter(true), 400); }
    ));
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
}

function _selectProtein(protein, row) {
    row.remove();
    _miniChatAppend('user', `Select protein: ${protein.label}`);
    fetch('/vina_visualization/vina_select_protein', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({protein_id: protein.id}),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status !== 'success') { _miniChatAppend('ai', `⚠️ ${data.message || 'unknown error'}`); return; }
        // Everything the selected target needs — receptor, ligand and docking box —
        // comes out of the one response, which is this protein's input_TS.yml entry.
        const paths = _applyProteinPaths(data);
        const boxed = _applyProteinBox(data, protein);
        _miniChatAppend('ai', `**${protein.label}** is now active ✓\n\n`
            + _prefillSummary(paths, boxed, data)
            + `\nVerify the values above, then click **⚗️ Vina Dock** to run docking.`);
        setTimeout(() => _highlightBtnUntilClick('vinaDockBtn'), 400);
    })
    .catch(() => {
        // Server unreachable — the GET /vina_proteins payload we were rendered from
        // carries the same fields, so fall back to it for both paths and box.
        const paths = _applyProteinPaths(protein);
        const boxed = _applyProteinBox(protein, protein);
        _miniChatAppend('ai', `Pre-filled for **${protein.label}** (offline — server did not confirm)\n\n`
            + _prefillSummary(paths, boxed, protein)
            + `\nVerify them above, then click **Vina Dock**.`);
        setTimeout(() => _highlightBtnUntilClick('vinaDockBtn'), 400);
    });
}

// ── Fill the RECEPTOR / LIGAND path inputs from a protein payload ─────────────
// Takes anything carrying default_receptor / default_ligand — i.e. an entry of
// vina.proteins in input_TS.yml, as returned by vina_select_protein or
// vina_proteins — and writes it into the files bar above the viewport.
//
// Per field it sets three things, not just the value:
//   value       – what _vinaDock / _vinaVisualize read at click time
//   placeholder – so a field the user CLEARS falls back to hinting at this
//                 target's path instead of the previously-active protein's
//                 (that stale hint is indistinguishable from a real value at a
//                 glance, and made it look like selection had done nothing)
//   title       – the inputs are 360/380px wide and these paths run ~100 chars,
//                 so the tail is clipped; the tooltip makes the full path
//                 readable without clicking into the field and pressing End
// then dispatches 'input' so the amber "fill me in" pulse from
// _highlightEmptyPathFields / _highlightAmberUntilAction(…, 'input') clears —
// the old offline path skipped this and left both fields pulsing after a fill.
//
// Returns {receptor, ligand} with the applied string, or null where the YAML
// entry had no path, so the caller can say so instead of silently no-op'ing.
function _applyProteinPaths(payload) {
    const applied = { receptor: null, ligand: null };
    if (!payload) return applied;

    const fill = (inputId, path) => {
        const el = document.getElementById(inputId);
        if (!el || !path) return null;
        el.value       = path;
        el.placeholder = path;
        el.title       = path;
        el.dispatchEvent(new Event('input'));
        return path;
    };

    applied.receptor = fill('recPath', payload.default_receptor);
    applied.ligand   = fill('ligPath', payload.default_ligand);
    return applied;
}

// Hand a {center_x/y/z, size_x/y/z} payload to vina.js, which owns _GRID and the
// Box ctr / len inputs. Guarded because elion_mini_chat.js loads before vina.js
// and is also used on pages that do not host the Vina docking box at all.
function _applyProteinBox(payload, protein) {
    if (typeof _applyGridBox !== 'function') return false;
    return _applyGridBox(payload, { label: protein && protein.label });
}

// What actually got pre-filled, spelled out. The path inputs are too narrow to
// show a ~100-char path, so echoing them here is the only way to confirm the
// right receptor/ligand landed — and a missing YAML field becomes a visible
// warning rather than a field that quietly stayed on the previous target.
function _prefillSummary(paths, boxed, box) {
    const lines = [];
    lines.push(paths.receptor ? `**Receptor:**\n\`${paths.receptor}\`` : `⚠️ No \`default_receptor\` in \`input_TS.yml\` — enter the receptor path manually.`);
    lines.push(paths.ligand   ? `**Ligand:**\n\`${paths.ligand}\``     : `⚠️ No \`default_ligand\` in \`input_TS.yml\` — enter the ligand path manually.`);
    lines.push(boxed ? `**Docking box:**\n${_boxSummary(box)}` : `⚠️ No \`center_x/y/z\` in \`input_TS.yml\` — box left unchanged.`);
    return lines.join('\n\n') + '\n';
}

// One-line box recap for the chat bubble, e.g.
//   "center (-0.127, 2.204, -12.08) · 20×20×20 Å"
function _boxSummary(b) {
    const size = [b.size_x, b.size_y, b.size_z].filter(v => v !== null && v !== undefined && v !== '');
    return `center (${b.center_x}, ${b.center_y}, ${b.center_z})`
         + (size.length ? ` · ${size.join('×')} Å` : '');
}


// =============================================================================
// ── 4b. ChemBERT / Attn welcome ──────────────────────────────────────────────
// =============================================================================

function _attnMiniWelcome() {
    // ChemBERT now uses the shared miniChatOutput panel like all other tools
    const out = _miniChatWelcomeBase({
        icon:     '🧠',
        name:     'ChemBERT Attn',
        color:    '#a78bfa',
        intro:    'Enter a <strong style="color:#a78bfa">SMILES string</strong> in the field above, ' +
                  'select a model preset, then click ' +
                  '<strong style="color:#a78bfa">⚡ Visualize</strong> to see per-atom attention weights.',
        workflow: 'Enter SMILES → select model (Finetuned / Pretrained) → Visualize → optionally Compare',
    });
    if (!out) return;

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;flex-direction:column;gap:5px;align-self:flex-start;width:100%;margin-top:6px;';

    const mkBtn = (bg, border, color, label, click) => {
        const b = document.createElement('button');
        b.style.cssText = `padding:7px 10px;border-radius:9px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;text-align:left;background:${bg};border:1px solid ${border};color:${color};transition:filter .12s;`;
        b.textContent = label;
        b.onmouseover = () => b.style.filter = 'brightness(1.15)';
        b.onmouseout  = () => b.style.filter = 'none';
        b.onclick = click;
        return b;
    };

    row.appendChild(mkBtn('#1e1b4b','#4c1d95','#a78bfa',
        '🧬 I have a SMILES — show me how to visualize',
        () => { row.remove(); _miniChatAppend('user','I have a SMILES — how do I visualize?'); _miniChatAppend('ai','Paste your SMILES in the input field above, choose Finetuned or Pretrained model, then click ⚡ Visualize. The 3D view will show per-atom attention weights.'); }
    ));
    row.appendChild(mkBtn('#1e293b','#334155','#94a3b8',
        '📊 Show me Compare mode',
        () => { row.remove(); _miniChatAppend('user','How do I compare two compounds?'); _miniChatAppend('ai','Click the Compare button in the top bar, then enter two SMILES strings side by side. Both molecules render with attention weights for direct comparison.'); }
    ));
    row.appendChild(mkBtn('#1e293b','#334155','#64748b',
        "🎛️ What's the difference between Finetuned and Pretrained?",
        () => { row.remove(); _miniChatAppend('user','Finetuned vs Pretrained?'); _miniChatAppend('ai','Pretrained is the base ChemBERT model trained on general molecular data. Finetuned is further trained on your specific binding affinity dataset — more accurate for your target protein family.'); }
    ));
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
}


// =============================================================================
// ── 4c. DeepAtom CNN Saliency welcome ────────────────────────────────────────
// =============================================================================

function _deepatomMiniWelcome() {
    const out = _miniChatWelcomeBase({
        icon:     '⚛️',
        name:     'DeepAtom',
        color:    '#fbbf24',
        intro:    'Enter a <strong style="color:#fbbf24">compound ID</strong> (e.g. <code>BM-1-57</code>) and the <strong style="color:#fbbf24">data directory</strong>, then click <strong style="color:#fbbf24">⚡ Visualize</strong> to see which atoms the CNN thinks matter most for binding.',
        workflow: 'Enter compound ID → set data dir → Make Atomtypes → Visualize',
    });
    if (!out) return;

    // Quick-action button
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;flex-direction:column;gap:5px;align-self:flex-start;width:100%;margin-top:6px;';
    const btn = document.createElement('button');
    btn.style.cssText = 'padding:7px 10px;border-radius:9px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;text-align:left;background:#1c1008;border:1px solid #78350f;color:#fbbf24;transition:filter .12s;';
    btn.textContent = '⚛️ What is a compound ID?';
    btn.onmouseover = () => btn.style.filter = 'brightness(1.15)';
    btn.onmouseout  = () => btn.style.filter = 'none';
    btn.onclick = () => {
        row.remove();
        _miniChatAppend('user', 'What is a compound ID?');
        _miniChatAppend('ai', 'A compound ID (e.g. **BM-1-57**) is the filename stem of your `.atomtypes` file inside the data directory. If you only have a `.pdb` file, click **Make Atomtypes** first to generate it.');
    };
    row.appendChild(btn);
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
}


// =============================================================================
// ── 4d. Thompson Sampling welcome ─────────────────────────────────────────────
// Extends the base with TS-specific context:
//   - What TS does (reaction-based enumeration + belief-guided search)
//   - Key config fields (SMARTS, iterations)
//   - Warmup vs TS Belief phases explained
//   - Quick-action buttons for the two most common first questions
// =============================================================================

function _tsMiniWelcome() {
    const out = _miniChatWelcomeBase({
        icon:     '🎲',
        name:     'Thompson Sampling',
        color:    '#a78bfa',
        intro:
            '<strong style="color:#a78bfa">Reaction-based combinatorial molecule generator</strong>.<br>' +
            'TS scores reagent combinations using Elion estimators (SAScore · QED · ChemBERT), ' +
            'then iteratively focuses on the most promising chemical space.',
        workflow: 'Set SMARTS + iterations → Run TS → watch Warmup beliefs form → see TS Belief converge → Results',
    });
    if (!out) return;

    // Phase explainer cards
    const phases = document.createElement('div');
    phases.style.cssText = 'display:flex;flex-direction:column;gap:5px;align-self:flex-start;width:100%;margin-top:8px;';

    const mkPhase = (icon, label, color, border, desc) => {
        const card = document.createElement('div');
        card.style.cssText = `padding:7px 10px;border-radius:9px;font-size:11px;background:#0a0f1a;border:1px solid ${border};`;
        card.innerHTML =
            `<div style="color:${color};font-weight:700;margin-bottom:3px;">${icon} ${label}</div>` +
            `<div style="color:#64748b;font-size:10px;">${desc}</div>`;
        return card;
    };

    phases.appendChild(mkPhase('⚡','Warmup','#7c3aed','#3b0764',
        'Scores random building block pairs to form initial μ/σ beliefs about each reagent slot.'));
    phases.appendChild(mkPhase('🎲','TS Belief','#6d28d9','#2e1065',
        'Samples from the belief distribution to select the most promising combinations in each iteration.'));

    // Quick-action buttons
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;flex-direction:column;gap:5px;align-self:flex-start;width:100%;margin-top:6px;';

    const mkBtn = (bg, border, color, label, click) => {
        const b = document.createElement('button');
        b.style.cssText = `padding:7px 10px;border-radius:9px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;text-align:left;background:${bg};border:1px solid ${border};color:${color};transition:filter .12s;`;
        b.textContent = label;
        b.onmouseover = () => b.style.filter = 'brightness(1.15)';
        b.onmouseout  = () => b.style.filter = 'none';
        b.onclick = click;
        return b;
    };

    row.appendChild(mkBtn('#1e1b4b','#4c1d95','#a78bfa', '🧪 How do I write a reaction SMARTS?',
        () => {
            row.remove();
            _miniChatAppend('user', 'How do I write a reaction SMARTS?');
            _miniChatAppend('ai',
                'A reaction SMARTS maps reactant atom-map numbers to product atoms.\n\n' +
                'Example — N-alkylation:\n`[#6:1][Br].[#7:2]>>[#6:1][#7:2]`\n\n' +
                'Paste it in the **SMARTS** field above, set your iteration count, then click **🎲 Run TS**.'
            );
        }
    ));
    row.appendChild(mkBtn('#0a0f1a','#1e293b','#64748b', '📋 What reagent files should I use?',
        () => {
            row.remove();
            _miniChatAppend('user', 'What reagent files should I use?');
            _miniChatAppend('ai',
                'Reagent files are `.csv` files with SMILES in the first column — one reagent per row.\n\n' +
                'The TS config (`input_TS.yml`) lists them under `reagent_file_list`. ' +
                'Each slot corresponds to one reactant in your SMARTS.'
            );
        }
    ));

    out.appendChild(phases);
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
}