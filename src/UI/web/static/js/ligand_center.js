// ligand_center.js — Get Ligand Center tool
// Mirrors the pdb_converter.js open/close/status pattern.

// ── Open / Close ─────────────────────────────────────────────────────────────

window._ligandCenterOpen = function () {
    const m = document.getElementById('ligandCenterModal');
    if (!m) return;
    m.style.display = 'flex';
    // Close sidebar if open (same as _sidebarOpenConverter)
    const sidebar  = document.getElementById('elionSidebar');
    const backdrop = document.getElementById('elionSidebarBackdrop');
    if (sidebar)  sidebar.style.right    = '-320px';
    if (backdrop) backdrop.style.display = 'none';
    const askBtn = document.getElementById('askElionBtn');
    if (askBtn) askBtn.style.borderColor = '';
};

window._ligandCenterClose = function () {
    const m = document.getElementById('ligandCenterModal');
    if (m) m.style.display = 'none';
};

// Close on backdrop click
document.getElementById('ligandCenterModal').addEventListener('click', function (e) {
    if (e.target === this) _ligandCenterClose();
});

// ── Status helper ─────────────────────────────────────────────────────────────

function _lcStatus(msg, isError) {
    const el = document.getElementById('lc_status');
    el.style.display    = 'block';
    el.style.background = isError ? 'rgba(239,68,68,.1)'  : 'rgba(52,211,153,.08)';
    el.style.border     = isError ? '1px solid #7f1d1d'   : '1px solid #064e3b';
    el.style.color      = isError ? '#fca5a5'             : '#6ee7b7';
    el.textContent      = msg;
}

function _lcClearStatus() {
    const el = document.getElementById('lc_status');
    el.style.display = 'none';
    el.textContent   = '';
}

// ── Drop zone ────────────────────────────────────────────────────────────────

window._lcHandleDrop = function (e) {
    e.preventDefault();
    document.getElementById('lc_dropzone').style.borderColor = '#1e3a2a';
    document.getElementById('lc_dropzone').style.background  = '#060e08';
    const file = e.dataTransfer.files[0];
    if (file) _lcHandleFile(file);
};

window._lcHandleFile = function (file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdb')) {
        _lcStatus('Please upload a .pdb file.', true);
        return;
    }
    // Show filename in drop zone
    document.getElementById('lc_drop_icon').textContent  = '✅';
    document.getElementById('lc_drop_label').style.display = 'none';
    const fileLabel = document.getElementById('lc_drop_file');
    fileLabel.style.display = 'block';
    fileLabel.textContent   = file.name;

    document.getElementById('lc_result').style.display = 'none';
    _lcClearStatus();
    _lcCompute(file);
};

// ── Core API call ─────────────────────────────────────────────────────────────

async function _lcCompute(file) {
    _lcStatus('⏳ Computing center…');

    const fd = new FormData();
    fd.append('file', file);

    let data;
    try {
        const resp = await fetch('/tools/ligand_center', { method: 'POST', body: fd });
        data = await resp.json();
    } catch (err) {
        _lcStatus('Network error: ' + err.message, true);
        return;
    }

    if (data.status !== 'success') {
        _lcStatus('Error: ' + (data.message || 'Unknown error'), true);
        return;
    }

    // Populate result fields
    document.getElementById('lc_cx').value = data.center_x;
    document.getElementById('lc_cy').value = data.center_y;
    document.getElementById('lc_cz').value = data.center_z;
    document.getElementById('lc_sx').value = data.size_x;
    document.getElementById('lc_sy').value = data.size_y;
    document.getElementById('lc_sz').value = data.size_z;

    // Residue / atom summary
    const info = document.getElementById('lc_residue_info');
    if (data.residues && data.residues.length) {
        const names = data.residues.map(r => `${r.resname} ${r.chain}:${r.resseq}`).join(', ');
        info.textContent = `Residues: ${names} · ${data.n_atoms} heavy atoms`;
    } else {
        info.textContent = `${data.n_atoms} heavy atoms`;
    }

    document.getElementById('lc_result').style.display = 'block';
    _lcStatus(`✓ Center computed from ${data.n_atoms} atoms`);
}

// ── Copy button ───────────────────────────────────────────────────────────────

window._lcCopy = function (inputId, btn) {
    const val = document.getElementById(inputId).value;
    if (!val) return;
    navigator.clipboard.writeText(val).then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = orig; }, 1200);
    });
};

// ── Auto-fill Vina grid box ───────────────────────────────────────────────────
// Tries the input IDs used in vina.js / vina_modal.html first,
// then falls back to searching by label text.

window._lcAutoFill = function () {
    const map = {
        center_x: document.getElementById('lc_cx').value,
        center_y: document.getElementById('lc_cy').value,
        center_z: document.getElementById('lc_cz').value,
        size_x:   document.getElementById('lc_sx').value,
        size_y:   document.getElementById('lc_sy').value,
        size_z:   document.getElementById('lc_sz').value,
    };

    // Known Vina input IDs (from vina.js / vina_modal.html)
    const idMap = {
        center_x: ['vinaCenterX', 'vina_center_x', 'center_x'],
        center_y: ['vinaCenterY', 'vina_center_y', 'center_y'],
        center_z: ['vinaCenterZ', 'vina_center_z', 'center_z'],
        size_x:   ['vinaSizeX',   'vina_size_x',   'size_x'],
        size_y:   ['vinaSizeY',   'vina_size_y',   'size_y'],
        size_z:   ['vinaSizeZ',   'vina_size_z',   'size_z'],
    };

    let filled = 0;
    for (const [key, val] of Object.entries(map)) {
        let inp = null;
        // Try known IDs first
        for (const id of idMap[key]) {
            inp = document.getElementById(id);
            if (inp) break;
        }
        // Fallback: search label text
        if (!inp) {
            for (const lbl of document.querySelectorAll('label')) {
                if (lbl.textContent.trim().toLowerCase() === key) {
                    const forId = lbl.getAttribute('for');
                    inp = forId ? document.getElementById(forId) : lbl.nextElementSibling;
                    if (inp && inp.tagName === 'INPUT') break;
                }
            }
        }
        if (inp) {
            inp.value = val;
            inp.dispatchEvent(new Event('input',  { bubbles: true }));
            inp.dispatchEvent(new Event('change', { bubbles: true }));
            filled++;
        }
    }

    if (filled === 6) {
        _lcStatus('✓ All 6 Vina grid-box fields filled!');
    } else if (filled > 0) {
        _lcStatus(`✓ Filled ${filled}/6 fields — copy the rest manually.`);
    } else {
        _lcStatus(
            'Could not locate Vina input fields automatically. ' +
            'Upload vina.js input IDs or copy values above manually.',
            true
        );
    }
};