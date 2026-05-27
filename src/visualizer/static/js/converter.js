// converter.js — PDB → PDBQT Converter modal
// strict mode removed — uses cross-file var assignments


// ── PDB → PDBQT converter state ───────────────────────────────────────────────
let _p2pFile    = null;
let _p2pType    = 'ligand';
let _p2pBusy    = false;
let _lastConversion = null;  // { mol_type, output_path, filename } — shared with mini-chat context

window._pdb2pdbqtOpen = function() {
    const m = document.getElementById('pdb2pdbqtModal');
    m.style.display = 'flex';
    // reset
    _p2pFile = null;
    _p2pBusy = false;
    document.getElementById('p2p_drop_icon').textContent  = '📂';
    document.getElementById('p2p_drop_label').style.display = '';
    document.getElementById('p2p_drop_file').style.display  = 'none';
    document.getElementById('p2p_drop_file').textContent    = '';
    document.getElementById('p2p_status').style.display     = 'none';
    document.getElementById('p2p_status').innerHTML         = '';
    document.getElementById('p2p_out_dir').value            = '';
    document.getElementById('p2p_btn_label').textContent    = 'Convert to PDBQT';
    document.getElementById('p2p_btn_icon').textContent     = '⚙️';
    document.getElementById('p2p_file_input').value         = '';
    _p2pSetType('ligand');
}
function _pdb2pdbqtClose() {
    document.getElementById('pdb2pdbqtModal').style.display = 'none';
}
// Close on backdrop click
// Converter closed only via ✕ button (no backdrop since it's transparent)

function _p2pSetType(t) {
    _p2pType = t;
    const lb = document.getElementById('p2p_ligand_btn');
    const rb = document.getElementById('p2p_receptor_btn');
    const hint = document.getElementById('p2p_type_hint');
    if (t === 'ligand') {
        lb.style.background = '#064e3b'; lb.style.color = '#6ee7b7';
        rb.style.background = '#111827'; rb.style.color = '#475569';
        hint.textContent = 'Ligand: adds hydrogens, assigns Gasteiger charges, writes AutoDock atom types.';
    } else {
        rb.style.background = '#064e3b'; rb.style.color = '#6ee7b7';
        lb.style.background = '#111827'; lb.style.color = '#475569';
        hint.textContent = 'Receptor: strips hydrogens, writes rigid PDBQT with zero partial charges.';
    }
}

function _p2pHandleDrop(e) {
    e.preventDefault();
    document.getElementById('p2p_dropzone').style.borderColor = '#1e3a2a';
    document.getElementById('p2p_dropzone').style.background  = '#060e08';
    const file = e.dataTransfer.files[0];
    if (file) _p2pHandleFile(file);
}
function _p2pHandleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdb')) {
        _p2pShowStatus('error', '⚠️ Please select a <strong>.pdb</strong> file.');
        return;
    }
    _p2pFile = file;
    document.getElementById('p2p_drop_icon').textContent       = '✅';
    document.getElementById('p2p_drop_label').style.display    = 'none';
    document.getElementById('p2p_drop_file').style.display     = '';
    document.getElementById('p2p_drop_file').textContent       = file.name;
    document.getElementById('p2p_status').style.display        = 'none';
}

async function _p2pConvert() {
    if (_p2pBusy) return;
    if (!_p2pFile) {
        _p2pShowStatus('error', '⚠️ Please select a <strong>.pdb</strong> file first.');
        return;
    }
    _p2pBusy = true;
    document.getElementById('p2p_btn_icon').textContent  = '⏳';
    document.getElementById('p2p_btn_label').textContent = 'Converting…';
    document.getElementById('p2p_convert_btn').style.opacity = '.6';
    document.getElementById('p2p_status').style.display = 'none';

    const fd = new FormData();
    fd.append('file',     _p2pFile);
    fd.append('mol_type', _p2pType);
    const outDir = document.getElementById('p2p_out_dir').value.trim();
    if (outDir) fd.append('out_dir', outDir);

    try {
        const resp = await fetch('/tools/pdb_to_pdbqt', { method: 'POST', body: fd });
        const data = await resp.json();

        if (data.status === 'success') {
            // Store conversion result globally for mini-chat context
            _lastConversion = {
                mol_type:    data.mol_type,
                output_path: data.output_path,
                filename:    data.filename,
            };

            // Auto-fill the matching Vina path field
            const fieldId = data.mol_type === 'receptor' ? 'recPath' : 'ligPath';
            const field   = document.getElementById(fieldId);
            if (field) {
                field.value = data.output_path;
                // Brief green flash to signal auto-fill
                field.style.borderColor = '#34d399';
                field.style.transition  = 'border-color .3s';
                setTimeout(() => { field.style.borderColor = ''; }, 2000);
            }

            // Show success in converter UI
            const fieldLabel = data.mol_type === 'receptor' ? 'Receptor' : 'Ligand';
            _p2pShowStatus('success',
                `✅ <strong>Conversion complete!</strong><br>` +
                `<span style="font-size:11px;color:#6ee7b7;">` +
                  `${fieldLabel} path auto-filled above ↑` +
                `</span><br><br>` +
                `<code style="font-size:11px;background:#0f172a;padding:4px 10px;border-radius:6px;` +
                        `color:#6ee7b7;word-break:break-all;display:block;line-height:1.6;">` +
                  `${data.output_path}` +
                `</code>`
            );

            // Only close converter if both fields are now filled;
            // if the other field is still empty, keep it open so user can convert again
            const recNow = document.getElementById('recPath')?.value?.trim();
            const ligNow = document.getElementById('ligPath')?.value?.trim();
            const bothFilled = recNow && ligNow;

            setTimeout(() => {
                _clearAllHighlights();  // remove any stale amber pulses

                if (bothFilled) {
                    _pdb2pdbqtClose();
                    _highlightBtnUntilClick('vinaDockBtn');
                } else {
                    // Keep converter open — reset dropzone and switch to the missing type
                    const missingType = !recNow ? 'receptor' : 'ligand';
                    _p2pFile = null;
                    _p2pBusy = false;
                    document.getElementById('p2p_drop_icon').textContent     = '📂';
                    document.getElementById('p2p_drop_label').style.display  = '';
                    document.getElementById('p2p_drop_file').style.display   = 'none';
                    document.getElementById('p2p_drop_file').textContent     = '';
                    document.getElementById('p2p_file_input').value          = '';
                    document.getElementById('p2p_btn_icon').textContent      = '⚙️';
                    document.getElementById('p2p_btn_label').textContent     = 'Convert to PDBQT';
                    document.getElementById('p2p_convert_btn').style.opacity = '1';
                    _p2pSetType(missingType);
                    // Highlight the missing path field behind the converter
                    _highlightAmberUntilAction([missingType === 'receptor' ? 'recPath' : 'ligPath'], 'input');
                }

                // Post mini-chat guidance
                const miniChat = document.getElementById('miniChat');
                if (miniChat && miniChat.style.display === 'flex') {
                    setTimeout(() => {
                        const otherLabel = data.mol_type === 'receptor' ? 'LIGAND' : 'RECEPTOR';
                        const otherId    = data.mol_type === 'receptor' ? 'ligPath' : 'recPath';
                        const otherVal   = document.getElementById(otherId)?.value?.trim();

                        if (otherVal) {
                            _miniChatAppend('ai',
                                `✅ **${data.filename}** saved — **${fieldLabel.toUpperCase()}** path auto-filled.\n\n` +
                                `Both paths are set. Click **Vina Dock** to start docking!`
                            );
                        } else {
                            const missingType = data.mol_type === 'receptor' ? 'ligand' : 'receptor';
                            _miniChatAppend('ai',
                                `✅ **${data.filename}** saved — **${fieldLabel.toUpperCase()}** done!\n\n` +
                                `The converter is ready for your **${otherLabel}** — ` +
                                `it's already switched to **${missingType}** mode. Drop your next **.pdb** file in.`
                            );
                        }
                    }, 200);
                }
            }, 800);
        } else {
            _p2pShowStatus('error', `⚠️ <strong>Error:</strong> ${data.message}`);
        }
    } catch(err) {
        _p2pShowStatus('error', `⚠️ Request failed: ${err.message}`);
    } finally {
        _p2pBusy = false;
        document.getElementById('p2p_btn_icon').textContent  = '⚙️';
        document.getElementById('p2p_btn_label').textContent = 'Convert to PDBQT';
        document.getElementById('p2p_convert_btn').style.opacity = '1';
    }
}

function _p2pShowStatus(type, html) {
    const el = document.getElementById('p2p_status');
    el.style.display = '';
    const isOk = type === 'success';
    el.innerHTML = `<div style="
        padding:14px 16px;border-radius:12px;font-size:13px;line-height:1.6;
        background:${isOk ? '#052e1c' : '#1f0a0a'};
        border:1px solid ${isOk ? '#065f46' : '#7f1d1d'};
        color:${isOk ? '#6ee7b7' : '#fca5a5'};
    ">${html}</div>`;
}