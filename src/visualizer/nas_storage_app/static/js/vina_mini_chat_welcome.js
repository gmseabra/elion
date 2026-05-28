// ============================================================
//  hub.html PATCH
//  Replace the existing _miniChatWelcome() function (lines ~3105-3165)
//  with this version.  Everything else in hub.html stays the same.
// ============================================================

function _miniChatWelcome() {
    const out = document.getElementById('miniChatOutput');

    // ── Greeting bubble ────────────────────────────────────────────────────────
    const greet = document.createElement('div');
    greet.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    greet.innerHTML =
        'Hi! I\'m your Vina Docking guide. 👋<br><br>' +
        'Which <strong style="color:#22d3ee">protein target</strong> would you like to dock against?';
    out.appendChild(greet);

    // ── Loading placeholder ────────────────────────────────────────────────────
    const loadingRow = document.createElement('div');
    loadingRow.style.cssText = 'align-self:flex-start;color:#475569;font-size:11px;';
    loadingRow.textContent = 'Loading available proteins…';
    out.appendChild(loadingRow);
    out.scrollTop = out.scrollHeight;

    // ── Fetch protein list from backend ───────────────────────────────────────
    fetch('/vina_visualization/vina_proteins')
        .then(r => r.json())
        .then(data => {
            loadingRow.remove();

            if (data.status !== 'success' || !data.proteins || !data.proteins.length) {
                // Fallback to the original pdbqt / convert flow
                _miniChatWelcomeFallback(out);
                return;
            }

            const proteins      = data.proteins;
            const activeId      = data.active_protein || proteins[0].id;

            // ── Button container ──────────────────────────────────────────────
            const row = document.createElement('div');
            row.id = 'miniWelcomeActions';
            row.style.cssText = 'display:flex;flex-direction:column;gap:6px;align-self:flex-start;width:100%;';

            proteins.forEach(protein => {
                const isActive = protein.id === activeId;

                const btn = document.createElement('button');
                btn.style.cssText = `
                    padding:8px 12px;border-radius:10px;font-size:11px;font-weight:600;
                    cursor:pointer;font-family:inherit;text-align:left;
                    transition:background .15s,border-color .15s;
                    background:${isActive ? '#0c3a5a' : '#0a1628'};
                    border:1px solid ${isActive ? '#22d3ee' : '#1e3a5f'};
                    color:${isActive ? '#e0f2fe' : '#94a3b8'};
                `;

                // Label line
                const labelLine = document.createElement('div');
                labelLine.style.cssText = 'display:flex;align-items:center;gap:6px;';
                labelLine.innerHTML =
                    `<span style="font-size:13px;">🧬</span>` +
                    `<span style="color:${isActive ? '#67e8f9' : '#7dd3fc'};font-size:12px;">${protein.label}</span>` +
                    (isActive ? ' <span style="font-size:9px;color:#22d3ee;background:#0e4f63;padding:1px 5px;border-radius:4px;margin-left:auto;">active</span>' : '');
                btn.appendChild(labelLine);

                // Description line (if present)
                if (protein.description) {
                    const desc = document.createElement('div');
                    desc.style.cssText = 'font-size:10px;color:#475569;margin-top:3px;font-weight:400;';
                    desc.textContent = protein.description;
                    btn.appendChild(desc);
                }

                btn.onmouseover = () => {
                    btn.style.background = '#0e3a5a';
                    btn.style.borderColor = '#22d3ee';
                    btn.style.color = '#e0f2fe';
                };
                btn.onmouseout = () => {
                    btn.style.background = isActive ? '#0c3a5a' : '#0a1628';
                    btn.style.borderColor = isActive ? '#22d3ee' : '#1e3a5f';
                    btn.style.color = isActive ? '#e0f2fe' : '#94a3b8';
                };

                btn.onclick = () => _selectProtein(protein, row);
                row.appendChild(btn);
            });

            out.appendChild(row);
            out.scrollTop = out.scrollHeight;
        })
        .catch(() => {
            loadingRow.remove();
            _miniChatWelcomeFallback(out);
        });
}


// ── Called when user clicks a protein button ──────────────────────────────────
function _selectProtein(protein, row) {
    row.remove();

    _miniChatAppend('user', `Select protein: ${protein.label}`);

    // Tell the backend to swap the active protein
    fetch('/vina_visualization/vina_select_protein', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ protein_id: protein.id }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status !== 'success') {
            _miniChatAppend('ai', `⚠️ Could not select protein: ${data.message || 'unknown error'}`);
            return;
        }

        // Pre-fill the RECEPTOR and LIGAND path fields from the chosen protein
        const recInput = document.getElementById('recPath');
        const ligInput = document.getElementById('ligPath');
        if (recInput && data.default_receptor) {
            recInput.value = data.default_receptor;
            recInput.dispatchEvent(new Event('input'));
        }
        if (ligInput && data.default_ligand) {
            ligInput.value = data.default_ligand;
            ligInput.dispatchEvent(new Event('input'));
        }

        // Confirm to the user and give next step
        _miniChatAppend('ai',
            `**${protein.label}** is now active ✓\n\n` +
            `Receptor and ligand paths have been pre-filled. ` +
            `Verify the paths in the fields above, then click **⚗️ Vina Dock** to run docking.`
        );

        // Pulse the dock button so it's obvious
        setTimeout(() => _highlightBtnUntilClick('vinaDockBtn'), 400);
    })
    .catch(() => {
        _miniChatAppend('ai',
            `Paths pre-filled for **${protein.label}** — verify them above, then click **Vina Dock**.`
        );
        // Still try to fill fields from the local protein object
        const recInput = document.getElementById('recPath');
        const ligInput = document.getElementById('ligPath');
        if (recInput && protein.default_receptor) recInput.value = protein.default_receptor;
        if (ligInput && protein.default_ligand)   ligInput.value = protein.default_ligand;
        setTimeout(() => _highlightBtnUntilClick('vinaDockBtn'), 400);
    });
}


// ── Fallback: show the original pdbqt / convert buttons ──────────────────────
// (kept so the UI still works if the YAML is misconfigured)
function _miniChatWelcomeFallback(out) {
    const fallbackNote = document.createElement('div');
    fallbackNote.style.cssText = 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.6;';
    fallbackNote.innerHTML =
        'Do you already have your <strong style="color:#22d3ee">.pdbqt</strong> files ready, ' +
        'or do you need to convert a <strong style="color:#6ee7b7">.pdb</strong> file first?';
    out.appendChild(fallbackNote);

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