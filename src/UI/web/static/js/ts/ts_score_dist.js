// =============================================================================
// ts/ts_score_dist.js — 📈 Score distribution.
//
// Point it at a directory of TS results CSVs (recursively) and it pools every
// score column and shows you the distribution. Two charts, because there are
// two questions and they want different forms:
//
//   1. HERO — the pooled histogram of the REAL scores, with a fitted normal
//      drawn over it. The bell curve is the overlay, not the data. That is the
//      whole point: a filename that says `mean8.758_std0.475` does not make the
//      distribution Gaussian, and if it is skewed or bimodal you can see the
//      bars diverge from the curve instead of being told a mean and a sigma.
//
//   2. PER-RUN — each CSV's own mean with ±1σ whiskers, in chronological
//      order, against the pooled mean. This is the "did the distribution move
//      between rounds?" question — the one the RL loop exists to answer.
//
// WHY NOT 45 OVERLAID GAUSSIANS. That was the obvious build and it is a
// spaghetti chart: 45 curves in one hue is texture, not information, and no
// individual run is identifiable in it. Worse, on a shared count axis each run
// is ~1/45 the height of the pooled curve and simply vanishes. Two forms, two
// questions, both readable.
//
// ONE AXIS, ALWAYS. The fitted normal is converted to EXPECTED COUNTS
// (N · binWidth · pdf) so it shares the histogram's y-axis. A second y-scale
// for "density" would be a dual-axis chart, which is the one thing you never
// draw.
//
// Palette: #0891b2 (pooled) and #8b5cf6 (per-run) on surface #0b1120 — the
// house categorical trio already in ts_rl.js. Verified with the dataviz
// validator rather than assumed:
//   lightness band PASS · chroma floor PASS · CVD ΔE 12.6 deutan / 10.2 tritan
//   PASS · normal-vision ΔE 21.1 PASS · contrast vs surface PASS.
//
// The button (#_tsDistBtn) lives in #tsRlLoop, rendered by ts_rl_loop.js. This
// file owns everything behind it, including its own popover, which it builds on
// first use and appends to <body>. So it needs no markup from anyone and no
// template edits; load order does not matter, because the button's onclick
// resolves at click time rather than at parse time.
// =============================================================================
console.log('%c[ts-dist] LOADED', 'background:#3f1d5c;color:#e9d5ff;padding:2px 6px;border-radius:3px');

const _TS_D = {
    // Selected vs pinned. Validated as a categorical pair on surface #0b1120:
    // CVD ΔE 19.0 (protan) / 28.6 (tritan), normal-vision ΔE 27.2, contrast
    // PASS — comfortably above the ΔE 8 target, which matters here because the
    // two histograms OVERLAP rather than sit side by side.
    S1: '#0891b2', S1_INK: '#22d3ee',      // the current selection
    CMP: '#d97706', CMP_INK: '#f59e0b',    // the pinned comparison
    S2: '#8b5cf6', S2_INK: '#a78bfa',      // per-run means, single-selection mode
    INK: '#e2e8f0', INK2: '#94a3b8', INK3: '#64748b', INK4: '#475569',
    LINE: '#1e293b', SURF: '#0b1120', SURF2: '#0a0f1d',
    BAD: '#fb7185', WARN: '#f59e0b', GOOD: '#34d399',
};

const _tsDist = { data: null, busy: false };

/* ── the path popover, built on demand ──────────────────────────────────────
   This file owns its own panel rather than expecting one in someone else's
   markup. The button lives in #tsRlLoop (ts_rl_loop.js); everything behind it
   — panel, styles, charts, handlers — is here, so moving the button again is a
   one-line change and nothing is left orphaned in a template.

   It is appended to <body>, not next to the button: #tsRlLoop is inside the
   TS modal's scroll container, and a `position:fixed` child of a scrolled,
   transformed ancestor is positioned against THAT ancestor rather than the
   viewport, which puts the popover somewhere unrelated to the click. */
function _tsDistEnsurePanel() {
    let panel = document.getElementById('_tsDistPanel');
    if (panel) return panel;
    panel = document.createElement('div');
    panel.id = '_tsDistPanel';
    panel.style.cssText = 'display:none;position:fixed;z-index:99999;'
        + 'background:#0a0f1e;border:0.5px solid rgba(71,85,105,0.5);border-radius:6px;'
        + 'padding:10px 12px;width:min(560px,90vw);box-shadow:0 8px 24px rgba(0,0,0,0.6)';
    panel.innerHTML =
        '<div style="font-size:9px;font-weight:600;color:#475569;text-transform:uppercase;'
      + 'letter-spacing:0.05em;margin-bottom:6px">TS results directory</div>'
      + '<div style="display:flex;gap:6px;align-items:center">'
      + '<input id="_tsDistPathInput" class="path-input" spellcheck="false"'
      + ' placeholder="/path/to/RL_active_learning_loop/pre_train"'
      + ' style="flex:1;min-width:0;background:#0b1120;border:1px solid #1e293b;border-radius:6px;'
      + 'color:#cbd5e1;font-family:ui-monospace,monospace;font-size:11px;padding:5px 8px;outline:none"'
      + ' onkeydown="if(event.key===\'Enter\'){event.preventDefault();_tsDistScan();}">'
      + '<button id="_tsDistScanBtn" onclick="_tsDistScan()"'
      + ' style="flex-shrink:0;padding:4px 12px;font-size:10px;font-weight:600;color:#a78bfa;'
      + 'background:rgba(139,92,246,0.12);border:0.5px solid rgba(139,92,246,0.4);'
      + 'border-radius:4px;cursor:pointer">Scan</button></div>'
      + '<div id="_tsDistPanelMsg" style="margin-top:6px;font-size:9px;color:#475569;line-height:1.5">'
      + 'Every <code style="color:#64748b">.csv</code> under this directory (recursively) is read '
      + 'and its score column pooled. You get the <b style="color:#64748b">histogram of the real '
      + 'scores</b> with a fitted normal drawn over it — so "is it actually a bell curve?" is '
      + 'something you look at rather than assume.</div>';
    document.body.appendChild(panel);
    return panel;
}

/* ── panel open/close ───────────────────────────────────────────────────── */
function _tsDistToggle(e) {
    if (e) e.stopPropagation();
    const panel = _tsDistEnsurePanel();
    const btn = document.getElementById('_tsDistBtn');
    if (!panel || !btn) return;
    const open = panel.style.display !== 'none';
    if (open) { panel.style.display = 'none'; return; }

    // Close the sibling popover rather than stacking two fixed panels at the
    // same z-index — whichever painted last wins and the other is unreachable.
    const bb = document.getElementById('_tsBbPanel');
    if (bb) bb.style.display = 'none';

    const inp = document.getElementById('_tsDistPathInput');
    if (inp && !inp.value) {
        // Seed from the OUTPUT DIR box: the directory a run writes to is the
        // directory whose results you want to pool, nine times out of ten.
        const out = document.getElementById('tsOutputDirInput');
        if (out && out.value.trim()) inp.value = out.value.trim();
    }
    panel.style.display = 'block';
    const r = btn.getBoundingClientRect();
    panel.style.top = (r.bottom + 6) + 'px';
    // Clamp to the viewport so the panel cannot open off-screen when the button
    // sits near the right edge of a narrow window.
    const w = panel.offsetWidth || 560;
    panel.style.left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8)) + 'px';
    if (inp) inp.focus();

    if (!_tsDistToggle._bound) {
        _tsDistToggle._bound = true;
        document.addEventListener('click', ev => {
            const p = document.getElementById('_tsDistPanel');
            const b = document.getElementById('_tsDistBtn');
            if (!p || p.style.display === 'none') return;
            if (p.contains(ev.target) || (b && b.contains(ev.target))) return;
            p.style.display = 'none';
        });
    }
}

/* ── fetch ──────────────────────────────────────────────────────────────── */
function _tsDistScan() {
    if (_tsDist.busy) return;
    _tsDistEnsurePanel();                       // callable before the popover was ever opened
    const inp = document.getElementById('_tsDistPathInput');
    const msg = document.getElementById('_tsDistPanelMsg');
    const path = (inp && inp.value.trim()) || '';
    if (!path) { if (msg) msg.innerHTML = '<span style="color:#fb7185">Enter a directory first.</span>'; return; }

    _tsDist.busy = true;
    const btn = document.getElementById('_tsDistScanBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Scanning…'; }
    if (msg) msg.innerHTML = 'Reading every CSV under this directory…';

    fetch('/vina_visualization/ts_score_dist?bins=48&recursive=1&dir=' + encodeURIComponent(path))
        .then(r => r.json())
        .then(d => {
            if (!d || !d.ok) {
                if (msg) msg.innerHTML = '<span style="color:#fb7185">'
                    + _tsDEsc((d && d.err) || 'scan failed') + '</span>';
                return;
            }
            _tsDist.data = d;
            document.getElementById('_tsDistPanel').style.display = 'none';
            if (msg) msg.innerHTML = 'Pooled ' + d.pooled.n.toLocaleString() + ' scores from '
                + d.n_files_used + ' file(s).';
            _tsDistRender();
        })
        .catch(e => { if (msg) msg.innerHTML = '<span style="color:#fb7185">' + _tsDEsc(e.message) + '</span>'; })
        .finally(() => {
            _tsDist.busy = false;
            if (btn) { btn.disabled = false; btn.textContent = 'Scan'; }
        });
}

/* ── helpers ────────────────────────────────────────────────────────────── */
function _tsDEsc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
const _tsDPdf = (x, mu, sd) =>
    sd <= 0 ? 0 : Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

/* Ticks a human would have chosen: 1/2/5 × 10^k, then snapped to the range. */
function _tsDTicks(lo, hi, target) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
}

/* ═══ selection state ════════════════════════════════════════════════════════
   `sel` is what the charts show; `pin` is what they compare it against.

     sel  = {kind:'all'|'group'|'file', key}
     pin  = the same shape, or null
     open = the expanded folder names

   Pin one folder, click another, and the charts switch to A/B. Pin and click
   the SAME thing and there is nothing to compare, so it stays single. */
Object.assign(_tsDist, { sel: { kind: 'all', key: '' }, pin: null, open: new Set() });

const _tsDKey = s => (s ? s.kind + ':' + s.key : '');
const _tsDSame = (a, b) => _tsDKey(a) === _tsDKey(b);

/* Files belonging to a selection, in the order the server sorted them. */
function _tsDFiles(sel) {
    const all = (_tsDist.data && _tsDist.data.files) || [];
    if (!sel || sel.kind === 'all') return all;
    if (sel.kind === 'group') return all.filter(f => f.group === sel.key);
    return all.filter(f => f.rel === sel.key);
}

function _tsDLabel(sel) {
    if (!sel || sel.kind === 'all') return 'all files';
    if (sel.kind === 'group') return sel.key + '/';
    const f = _tsDFiles(sel)[0];
    return f ? f.rel : sel.key;
}

/* ── A selection's distribution, assembled from per-file parts ───────────────
   The histogram is the elementwise SUM of the member files' counts. That is
   exact and needs no re-read, because the server binned every file on ONE
   shared grid — which is also why two selections can be overlaid honestly:
   identical bin edges by construction, not by coincidence.

   Mean and variance come from the per-file moments, not from the bins:

       N   = Σ nᵢ                    μ = Σ nᵢμᵢ / N
       E[x²] = Σ nᵢ(σᵢ² + μᵢ²) / N   σ = √(E[x²] − μ²)

   Reading μ and σ back off a 48-bin histogram would quantise them to the bin
   width; these are the real numbers the files were scored with. */
function _tsDSeries(sel) {
    const d = _tsDist.data;
    if (!d) return null;
    const files = _tsDFiles(sel);
    if (!files.length) return null;
    const nb = d.pooled.counts.length;
    const counts = new Array(nb).fill(0);
    let N = 0, sw = 0, sq = 0, lo = Infinity, hi = -Infinity;
    files.forEach(f => {
        for (let i = 0; i < nb; i++) counts[i] += (f.counts && f.counts[i]) || 0;
        N += f.n;
        sw += f.n * f.mean;
        sq += f.n * (f.std * f.std + f.mean * f.mean);
        if (f.min < lo) lo = f.min;
        if (f.max > hi) hi = f.max;
    });
    const mean = N ? sw / N : 0;
    // Floating-point cancellation can push a genuinely-zero variance slightly
    // negative; sqrt of that is NaN and every downstream curve disappears.
    const varr = Math.max(0, (N ? sq / N : 0) - mean * mean);
    return {
        sel, files, counts, n: N, mean, std: Math.sqrt(varr),
        min: lo, max: hi, label: _tsDLabel(sel),
    };
}

function _tsDistPick(kind, key) {
    _tsDist.sel = { kind, key: key || '' };
    _tsDistPaint();
}

function _tsDistPin(kind, key, ev) {
    if (ev) ev.stopPropagation();          // the row underneath also selects
    const cand = { kind, key: key || '' };
    _tsDist.pin = _tsDSame(_tsDist.pin, cand) ? null : cand;
    _tsDistPaint();
}

function _tsDistGroupToggle(name, ev) {
    if (ev) ev.stopPropagation();
    if (_tsDist.open.has(name)) _tsDist.open.delete(name);
    else _tsDist.open.add(name);
    _tsDistTree();
}

function _tsDistExpandAll() {
    const groups = _tsDistGroups().map(g => g.name);
    if (_tsDist.open.size >= groups.length) _tsDist.open.clear();
    else groups.forEach(g => _tsDist.open.add(g));
    _tsDistTree();
}

/* Top-level folders, each with its own rolled-up stats. */
function _tsDistGroups() {
    const files = (_tsDist.data && _tsDist.data.files) || [];
    const order = [];
    const by = {};
    files.forEach(f => {
        if (!by[f.group]) { by[f.group] = []; order.push(f.group); }
        by[f.group].push(f);
    });
    return order.sort().map(name => {
        const s = _tsDSeries({ kind: 'group', key: name });
        return { name, files: by[name], stats: s };
    });
}

/* ═══ paint ══════════════════════════════════════════════════════════════ */
function _tsDistPaint() {
    _tsDistScopeBar();
    _tsDistTiles();
    _tsDistHist();
    _tsDistRuns();
    _tsDistTree();
    _tsDistDlSync();
}

function _tsDistClose() {
    const o = document.getElementById('_tsDistOverlay');
    if (o) o.remove();
}

function _tsDistRender() {
    const d = _tsDist.data;
    if (!d) return;
    _tsDistClose();
    _tsDist.sel = { kind: 'all', key: '' };
    _tsDist.pin = null;
    _tsDist.open = new Set();          // folders collapsed by default

    const capNote = [];
    if (d.caps && d.caps.files_truncated)
        capNote.push('only the first ' + d.caps.max_files + ' of ' + d.n_files_found + ' CSVs were read');
    if (d.caps && d.caps.rows_truncated)
        capNote.push('the ' + d.caps.max_rows.toLocaleString() + '-row cap was hit — this is a LOWER bound');
    const drift = (d.files || []).filter(f => f.name_drift != null && f.name_drift > 0.01);

    const ov = document.createElement('div');
    ov.id = '_tsDistOverlay';
    ov.style.cssText = 'position:fixed;inset:0;z-index:99998;background:rgba(2,4,10,.86);'
        + 'display:flex;align-items:center;justify-content:center;padding:24px';
    ov.addEventListener('click', ev => { if (ev.target === ov) _tsDistClose(); });
    ov.innerHTML = `
      <div style="background:${_TS_D.SURF};border:1px solid #334155;border-radius:16px;
                  width:min(1180px,96vw);max-height:92vh;overflow:auto;
                  font-family:Inter,system-ui,sans-serif;color:${_TS_D.INK}">
        <div style="position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;
                    padding:13px 22px;background:#0f172a;border-bottom:1px solid #334155">
          <b style="font-size:13.5px">📈 Score distribution</b>
          <span style="font-size:10.5px;color:${_TS_D.INK3};font-family:ui-monospace,monospace;
                       overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:46%"
                title="${_tsDEsc(d.dir)}">${_tsDEsc(d.dir)}</span>
          <button onclick="_tsDistExpandAll()" style="margin-left:auto;padding:5px 12px;border-radius:9px;
                  border:1px solid #334155;background:${_TS_D.SURF};color:${_TS_D.INK2};
                  font:inherit;font-size:11.5px;font-weight:600;cursor:pointer">⊞ Expand all</button>
          <button onclick="_tsDistClose()" style="padding:5px 12px;border-radius:9px;
                  border:1px solid #334155;background:${_TS_D.SURF};color:${_TS_D.INK2};
                  font:inherit;font-size:11.5px;font-weight:600;cursor:pointer">✕ Close</button>
        </div>
        <div style="padding:16px 22px 26px">
          <div id="_tsDistScope"></div>

          <div style="margin-top:12px">
            <div style="display:flex;align-items:flex-end;gap:14px;margin-bottom:10px">
              <div style="flex:1;min-width:0">
                <div style="font-size:12.5px;font-weight:700;margin-bottom:2px">Files</div>
                <div style="font-size:11px;color:${_TS_D.INK2}">
                  Folders are collapsed. Click a folder or a file to scope the chart to it;
                  📌 pins one so the next click renders an A/B comparison.
                </div>
              </div>
              <div style="flex-shrink:0;text-align:right">
                <button id="_tsDistDlBtn" onclick="_tsDistExport()"
                        title="Download the molecules currently in scope — the clicked selection plus the pinned one — as one CSV with exactly two columns, SMILES and Name."
                        style="padding:6px 13px;border-radius:9px;border:1px solid rgba(8,145,178,.45);
                               background:rgba(8,145,178,.14);color:${_TS_D.S1_INK};font:inherit;
                               font-size:11.5px;font-weight:600;cursor:pointer;white-space:nowrap">⬇ SMILES,Name .csv</button>
                <div id="_tsDistDlMsg" style="font-size:10px;color:${_TS_D.INK4};margin-top:4px;
                     font-family:ui-monospace,monospace"></div>
              </div>
            </div>
            <div id="_tsDistTree"></div>
          </div>

          ${capNote.length ? `<div style="margin:12px 0 0;padding:9px 12px;border-radius:10px;font-size:11px;
                background:rgba(245,158,11,.09);border:1px solid rgba(245,158,11,.35);color:${_TS_D.WARN}">
                <b>Capped:</b> ${capNote.join('; ')}.</div>` : ''}
          ${drift.length ? `<div style="margin:12px 0 0;padding:9px 12px;border-radius:10px;font-size:11px;
                background:rgba(251,113,133,.09);border:1px solid rgba(251,113,133,.35);color:${_TS_D.BAD}">
                <b>${drift.length} file(s) disagree with their own filename</b> by more than 0.01 —
                recomputing from the rows gives something other than the <code>mean…</code> in the name.
                Worst: <code>${_tsDEsc(drift[0].filename)}</code> (name ${drift[0].name_mean},
                actual ${drift[0].mean}).</div>` : ''}
          ${(d.skipped || []).length ? `<div style="margin:12px 0 0;font-size:10.5px;color:${_TS_D.INK4};
                font-family:ui-monospace,monospace">skipped ${d.skipped.length}:
                ${d.skipped.slice(0, 4).map(s => _tsDEsc(s.filename) + ' (' + _tsDEsc(s.reason) + ')').join(' · ')}
                ${d.skipped.length > 4 ? ' …' : ''}</div>` : ''}

          <div style="margin-top:18px">
            <div id="_tsDistHistTitle" style="font-size:12.5px;font-weight:700;margin-bottom:2px"></div>
            <div id="_tsDistHistSub" style="font-size:11px;color:${_TS_D.INK2};margin-bottom:10px"></div>
            <div id="_tsDistHist"></div>
          </div>

          <div style="margin-top:24px">
            <div style="font-size:12.5px;font-weight:700;margin-bottom:2px">Per-run mean ± 1σ, in order</div>
            <div style="font-size:11px;color:${_TS_D.INK2};margin-bottom:10px">
              One dot per CSV against its own group's mean. A run that starts from a biased prior
              should sit visibly off the line.
            </div>
            <div id="_tsDistRuns"></div>
          </div>

          <div id="_tsDistTiles" style="margin-top:24px"></div>
        </div>
      </div>`;
    document.body.appendChild(ov);
    _tsDistPaint();
}

/* ── scope bar ──────────────────────────────────────────────────────────── */
function _tsDistScopeBar() {
    const el = document.getElementById('_tsDistScope');
    if (!el) return;
    const a = _tsDSeries(_tsDist.sel);
    const cmp = _tsDist.pin && !_tsDSame(_tsDist.pin, _tsDist.sel) ? _tsDSeries(_tsDist.pin) : null;
    const chip = (txt, col) => `<span style="font-family:ui-monospace,monospace;font-size:10px;
        font-weight:700;padding:2px 8px;border-radius:6px;background:${col}22;
        border:1px solid ${col}66;color:${col}">${_tsDEsc(txt)}</span>`;
    el.innerHTML = `<div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;
         background:${_TS_D.SURF2};border:1px solid ${_TS_D.LINE};border-radius:12px;padding:9px 13px">
      <span style="font-size:11.5px;color:${_TS_D.INK2}">Showing</span>
      ${cmp ? chip(cmp.label, _TS_D.CMP) + `<span style="color:${_TS_D.INK3};font-size:11px">vs</span>`
                 + chip(a.label, _TS_D.S1) : chip(a ? a.label : '—', _TS_D.S1)}
      ${cmp ? `<span style="font-size:11px;color:${_TS_D.INK3}">Δmean
               <b style="color:${(a.mean - cmp.mean) >= 0 ? _TS_D.GOOD : _TS_D.BAD};
                  font-family:ui-monospace,monospace">${(a.mean - cmp.mean) >= 0 ? '+' : ''}${(a.mean - cmp.mean).toFixed(3)}</b></span>` : ''}
      ${(_tsDist.sel.kind !== 'all' || _tsDist.pin)
        ? `<button onclick="_tsDistReset()" style="margin-left:auto;padding:4px 11px;border-radius:8px;
             border:1px solid ${_TS_D.LINE};background:${_TS_D.SURF};color:${_TS_D.INK2};
             font:inherit;font-size:11px;font-weight:600;cursor:pointer">✕ all files</button>` : ''}
    </div>`;
}

function _tsDistReset() {
    _tsDist.sel = { kind: 'all', key: '' };
    _tsDist.pin = null;
    _tsDistPaint();
}

/* ── stat tiles ─────────────────────────────────────────────────────────── */
function _tsDistTiles() {
    const el = document.getElementById('_tsDistTiles');
    const d = _tsDist.data;
    if (!el || !d) return;
    const a = _tsDSeries(_tsDist.sel);
    if (!a) { el.innerHTML = ''; return; }
    const cmp = _tsDist.pin && !_tsDSame(_tsDist.pin, _tsDist.sel) ? _tsDSeries(_tsDist.pin) : null;
    const skew = (_tsDist.sel.kind === 'all') ? d.pooled.skew : null;
    const tile = (label, val, sub, col) => `
      <div style="background:${_TS_D.SURF2};border:1px solid ${_TS_D.LINE};border-radius:13px;padding:11px 13px">
        <div style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
                    color:${_TS_D.INK4};margin-bottom:6px">${label}</div>
        <div style="font-family:ui-monospace,monospace;font-size:19px;font-weight:700;
                    color:${col || _TS_D.INK};letter-spacing:-.02em">${val}</div>
        <div style="font-size:10px;color:${_TS_D.INK3};margin-top:3px">${sub}</div>
      </div>`;
    const tiles = [
        tile('Files', a.files.length, cmp ? cmp.files.length + ' pinned' : d.n_files_found + ' found'),
        tile('Molecules', a.n.toLocaleString(), cmp ? cmp.n.toLocaleString() + ' pinned' : 'score rows in scope'),
        tile('Mean', a.mean.toFixed(3), cmp ? 'pinned ' + cmp.mean.toFixed(3) : 'in scope', _TS_D.S1_INK),
        tile('σ', a.std.toFixed(3), cmp ? 'pinned ' + cmp.std.toFixed(3) : 'standard deviation'),
        tile('Range', a.min.toFixed(2) + ' – ' + a.max.toFixed(2), 'min to max'),
    ];
    if (cmp) {
        const dm = a.mean - cmp.mean;
        // Cohen's d on the pooled SD — "how far apart" in units of the spread,
        // which is the number that says whether a mean shift is meaningful.
        const sp = Math.sqrt(((a.n * a.std * a.std) + (cmp.n * cmp.std * cmp.std)) / Math.max(1, a.n + cmp.n));
        tiles.push(tile('Δ mean', (dm >= 0 ? '+' : '') + dm.toFixed(3),
            sp > 1e-9 ? ('d = ' + (dm / sp).toFixed(2) + ' pooled σ') : 'no spread to compare',
            dm >= 0 ? _TS_D.GOOD : _TS_D.BAD));
    } else if (skew != null) {
        const t = Math.abs(skew) < 0.25 ? 'near-symmetric' : (skew > 0 ? 'right-skewed' : 'left-skewed');
        const c = Math.abs(skew) < 0.25 ? _TS_D.GOOD : (Math.abs(skew) < 0.75 ? _TS_D.WARN : _TS_D.BAD);
        tiles.push(tile('Skew', (skew >= 0 ? '+' : '') + skew.toFixed(2), t, c));
    }
    el.innerHTML = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));gap:10px">'
                 + tiles.join('') + '</div>';
}

/* ═══ chart 1: histogram (+ fitted normal), single or A/B ════════════════════
   SINGLE mode plots COUNTS — "how many molecules landed here", the honest
   readout for one thing.

   COMPARISON mode plots DENSITY (countₖ / (n·w)). Two selections almost never
   have the same n — suzuki has 33 files here and snar has 12 — and overlaying
   raw counts would show you the sample sizes, not the distributions. Density
   integrates to 1 for both, so the shapes are what differ. The axis label and
   the subtitle both change, because a silent switch of what the y-axis means
   is worse than either choice. */
function _tsDistHist() {
    const host = document.getElementById('_tsDistHist');
    const d = _tsDist.data;
    if (!host || !d) return;
    const A = _tsDSeries(_tsDist.sel);
    if (!A) { host.innerHTML = ''; return; }
    const P = _tsDist.pin && !_tsDSame(_tsDist.pin, _tsDist.sel) ? _tsDSeries(_tsDist.pin) : null;
    const cmp = !!P;

    const title = document.getElementById('_tsDistHistTitle');
    const sub = document.getElementById('_tsDistHistSub');
    if (title) title.textContent = cmp
        ? 'Distribution comparison: ' + P.label + ' vs ' + A.label
        : 'Pooled score distribution — ' + A.label;
    if (sub) sub.innerHTML = cmp
        ? `Both normalised to <b>density</b>, because the two hold ${P.n.toLocaleString()} and
           ${A.n.toLocaleString()} molecules — overlaid counts would compare sample sizes rather
           than shapes. Identical bin edges, set once across every file.`
        : `Bars are the ${A.n.toLocaleString()} real scores. The line is a normal fitted to them,
           drawn as expected counts on the same axis, so the two are directly comparable.`;

    const W = 1080, H = 330, M = { t: 16, r: 18, b: 46, l: 60 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const nb = d.pooled.counts.length, w = d.pooled.bin_width, lo0 = d.pooled.bin_lo;

    // In comparison mode every value is a density; in single mode, a count.
    const val = (S, k) => cmp ? (S.counts[k] / (S.n * w)) : S.counts[k];
    const fit = (S, x) => cmp ? _tsDPdf(x, S.mean, S.std) : S.n * w * _tsDPdf(x, S.mean, S.std);

    let yMax = 0;
    [A, P].filter(Boolean).forEach(S => {
        for (let k = 0; k < nb; k++) yMax = Math.max(yMax, val(S, k));
        yMax = Math.max(yMax, fit(S, S.mean));           // the curve can top the tallest bar
    });
    yMax = (yMax || 1) * 1.08;

    const x = v => M.l + ((v - lo0) / (w * nb)) * iw;
    const y = c => M.t + ih - (c / yMax) * ih;
    const bw = iw / nb, gap = bw > 6 ? 2 : 0;
    const r = Math.min(4, Math.max(0, (bw - gap) / 3));

    const bars = (S, col, op) => {
        let out = '';
        for (let k = 0; k < nb; k++) {
            const c = val(S, k);
            if (!c) continue;
            const bx = M.l + k * bw + gap / 2, bwid = Math.max(0.5, bw - gap);
            const by = y(c), bh = M.t + ih - by, rr = Math.min(r, bh);
            out += `<path d="M${bx.toFixed(2)},${(M.t + ih).toFixed(2)} V${(by + rr).toFixed(2)}
                    Q${bx.toFixed(2)},${by.toFixed(2)} ${(bx + rr).toFixed(2)},${by.toFixed(2)}
                    H${(bx + bwid - rr).toFixed(2)} Q${(bx + bwid).toFixed(2)},${by.toFixed(2)}
                    ${(bx + bwid).toFixed(2)},${(by + rr).toFixed(2)} V${(M.t + ih).toFixed(2)} Z"
                    fill="${col}" opacity="${op}"></path>`;
        }
        return out;
    };
    const curve = (S, col) => {
        let dstr = '';
        for (let i = 0; i <= 240; i++) {
            const xv = lo0 + (i / 240) * w * nb;
            dstr += (i ? 'L' : 'M') + x(xv).toFixed(2) + ',' + y(fit(S, xv)).toFixed(2) + ' ';
        }
        return `<path d="${dstr}" fill="none" stroke="${col}" stroke-width="2"
                 stroke-linejoin="round" stroke-linecap="round"></path>`;
    };
    const meanLine = (S, col, dash) => `
        <line x1="${x(S.mean)}" y1="${M.t}" x2="${x(S.mean)}" y2="${M.t + ih}"
              stroke="${col}" stroke-width="1.6" stroke-dasharray="${dash}"></line>`;

    const yt = _tsDTicks(0, yMax, 5);
    const xt = _tsDTicks(lo0, lo0 + w * nb, 8);
    const fmtY = t => cmp ? t.toFixed(t < 1 ? 2 : 1) : (t >= 1000 ? (t / 1000) + 'k' : t);

    host.innerHTML = `
      <div style="position:relative">
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" id="_tsDistHistSvg">
        ${yt.map(t => `<line x1="${M.l}" y1="${y(t)}" x2="${W - M.r}" y2="${y(t)}"
              stroke="${_TS_D.LINE}" stroke-dasharray="3 4"></line>
            <text x="${M.l - 9}" y="${y(t) + 3.5}" text-anchor="end" font-family="ui-monospace"
              font-size="9.5" fill="${_TS_D.INK4}">${fmtY(t)}</text>`).join('')}
        ${P ? bars(P, _TS_D.CMP, .55) : ''}
        ${bars(A, _TS_D.S1, cmp ? .55 : .82)}
        ${P ? curve(P, _TS_D.CMP_INK) : ''}
        ${curve(A, _TS_D.S1_INK)}
        ${P ? meanLine(P, _TS_D.CMP_INK, '6 4') : ''}
        ${meanLine(A, cmp ? _TS_D.S1_INK : _TS_D.INK2, '4 3')}
        ${cmp ? '' : `<text x="${x(A.mean) + 6}" y="${M.t + 12}" font-family="ui-monospace"
              font-size="9.5" fill="${_TS_D.INK2}">mean ${A.mean.toFixed(3)}</text>`}
        ${cmp ? '' : [-1, 1].map(sg => {
            const v = A.mean + sg * A.std;
            if (v < lo0 || v > lo0 + w * nb) return '';
            return `<line x1="${x(v)}" y1="${M.t + ih}" x2="${x(v)}" y2="${M.t + ih + 5}"
                     stroke="${_TS_D.INK4}" stroke-width="1.2"></line>
                    <text x="${x(v)}" y="${M.t + ih + 16}" text-anchor="middle"
                     font-family="ui-monospace" font-size="9" fill="${_TS_D.INK4}">${sg > 0 ? '+' : '−'}1σ</text>`;
        }).join('')}
        <line x1="${M.l}" y1="${M.t + ih}" x2="${W - M.r}" y2="${M.t + ih}" stroke="#334155"></line>
        ${xt.map(t => `<text x="${x(t)}" y="${M.t + ih + 16}" text-anchor="middle"
              font-family="ui-monospace" font-size="9.5" fill="${_TS_D.INK4}">${t}</text>`).join('')}
        <text x="${M.l + iw / 2}" y="${H - 6}" text-anchor="middle" font-family="ui-monospace"
              font-size="10" fill="${_TS_D.INK3}">TS score (ChemBERT ΔG through the Elion reward)</text>
        <text x="14" y="${M.t + ih / 2}" text-anchor="middle" font-family="ui-monospace" font-size="10"
              fill="${_TS_D.INK3}" transform="rotate(-90 14 ${M.t + ih / 2})">${cmp ? 'density' : 'molecules per bin'}</text>
        <line class="_tsdx" x1="0" y1="${M.t}" x2="0" y2="${M.t + ih}" stroke="${_TS_D.S1_INK}"
              stroke-width="1" opacity="0"></line>
      </svg>
      <div id="_tsDistTip" style="position:absolute;display:none;pointer-events:none;z-index:3;
           background:#0f172a;border:1px solid #334155;border-radius:9px;padding:7px 10px;
           font-family:ui-monospace,monospace;font-size:10.5px;color:${_TS_D.INK};
           box-shadow:0 6px 18px rgba(0,0,0,.5);white-space:nowrap"></div>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:10.5px;color:${_TS_D.INK2};
                  margin-top:9px;font-family:ui-monospace,monospace">
        ${P ? `<span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;
              background:${_TS_D.CMP};margin-right:6px;vertical-align:-1px"></i>📌 ${_tsDEsc(P.label)}
              &nbsp;mean <b style="color:${_TS_D.CMP_INK}">${P.mean.toFixed(3)}</b> · σ ${P.std.toFixed(3)} · n ${P.n.toLocaleString()}</span>` : ''}
        <span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;
              background:${_TS_D.S1};margin-right:6px;vertical-align:-1px"></i>${_tsDEsc(A.label)}
              &nbsp;mean <b style="color:${_TS_D.S1_INK}">${A.mean.toFixed(3)}</b> · σ ${A.std.toFixed(3)} · n ${A.n.toLocaleString()}</span>
        <span style="color:${_TS_D.INK4}">bin width ${w.toFixed(4)} · ${nb} bins${cmp ? ' · shared edges' : ''}</span>
      </div>`;

    const svg = document.getElementById('_tsDistHistSvg');
    const tip = document.getElementById('_tsDistTip');
    const line = svg.querySelector('._tsdx');
    svg.addEventListener('mousemove', ev => {
        const r0 = svg.getBoundingClientRect();
        const k = Math.floor(((ev.clientX - r0.left) / r0.width * W - M.l) / bw);
        if (k < 0 || k >= nb) { tip.style.display = 'none'; line.setAttribute('opacity', '0'); return; }
        const bl = lo0 + k * w, bh = bl + w, cx = M.l + (k + 0.5) * bw;
        line.setAttribute('x1', cx); line.setAttribute('x2', cx); line.setAttribute('opacity', '.5');
        const row = (S, col, tag) => cmp
            ? `${tag} <b style="color:${col}">${(S.counts[k] / S.n * 100).toFixed(2)}%</b>`
              + ` <span style="color:${_TS_D.INK3}">(${S.counts[k].toLocaleString()})</span>`
            : `observed <b style="color:${col}">${S.counts[k].toLocaleString()}</b>`
              + ` (${(S.counts[k] / S.n * 100).toFixed(2)}%)<br>normal expects`
              + ` <span style="color:${_TS_D.INK2}">${fit(S, (bl + bh) / 2).toFixed(0)}</span>`;
        tip.innerHTML = `<b>${bl.toFixed(3)} – ${bh.toFixed(3)}</b><br>`
            + (P ? row(P, _TS_D.CMP_INK, '📌 ' + _tsDEsc(P.label)) + '<br>' : '')
            + row(A, _TS_D.S1_INK, _tsDEsc(A.label));
        tip.style.display = 'block';
        const hostR = svg.parentNode.getBoundingClientRect();
        let lx = (cx / W) * hostR.width + 12;
        if (lx + tip.offsetWidth > hostR.width) lx = (cx / W) * hostR.width - tip.offsetWidth - 12;
        tip.style.left = Math.max(0, lx) + 'px';
        tip.style.top = ((M.t + 8) / H) * hostR.height + 'px';
    });
    svg.addEventListener('mouseleave', () => {
        tip.style.display = 'none'; line.setAttribute('opacity', '0');
    });
}

/* ═══ chart 2: per-run mean ± 1σ ═════════════════════════════════════════ */
function _tsDistRuns() {
    const host = document.getElementById('_tsDistRuns');
    const d = _tsDist.data;
    if (!host || !d) return;
    const A = _tsDSeries(_tsDist.sel);
    if (!A) { host.innerHTML = ''; return; }
    const P = _tsDist.pin && !_tsDSame(_tsDist.pin, _tsDist.sel) ? _tsDSeries(_tsDist.pin) : null;

    // In A/B mode both sets are drawn, each against its OWN mean line. Pooling
    // them onto one reference would hide exactly the shift being compared.
    const sets = P ? [{ s: P, col: _TS_D.CMP, ink: _TS_D.CMP_INK },
                      { s: A, col: _TS_D.S1, ink: _TS_D.S1_INK }]
                   : [{ s: A, col: _TS_D.S2, ink: _TS_D.S2_INK }];
    // Sorted by TIMESTAMP across both sets, not concatenated set-by-set.
    // flatMap alone puts every pinned run before every selected one, which
    // draws two campaigns as if one finished before the other started — a
    // claim about chronology that the data does not make. Interleaved, you can
    // see the runs alternating, which is what actually happened.
    const files = sets.flatMap(g => g.s.files.map(f => ({ f, g })))
        .sort((p, q) => String(p.f.timestamp || p.f.filename)
            .localeCompare(String(q.f.timestamp || q.f.filename)));
    if (!files.length) { host.innerHTML = ''; return; }

    const W = 1080, H = 250, M = { t: 14, r: 18, b: 64, l: 60 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const lo = Math.min(...files.map(o => o.f.mean - o.f.std));
    const hi = Math.max(...files.map(o => o.f.mean + o.f.std));
    const pad = (hi - lo) * 0.12 || 0.5;
    const y = v => M.t + ih - ((v - (lo - pad)) / ((hi + pad) - (lo - pad))) * ih;
    const x = i => files.length === 1 ? M.l + iw / 2 : M.l + (i / (files.length - 1)) * iw;
    const yt = _tsDTicks(lo - pad, hi + pad, 5);
    const every = Math.max(1, Math.ceil(files.length / 8));

    host.innerHTML = `
      <div style="position:relative">
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" id="_tsDistRunsSvg">
        ${yt.map(t => `<line x1="${M.l}" y1="${y(t)}" x2="${W - M.r}" y2="${y(t)}"
              stroke="${_TS_D.LINE}" stroke-dasharray="3 4"></line>
            <text x="${M.l - 9}" y="${y(t) + 3.5}" text-anchor="end" font-family="ui-monospace"
              font-size="9.5" fill="${_TS_D.INK4}">${t.toFixed(2)}</text>`).join('')}
        ${sets.map(g => {
            // The label sits over the dots, so it gets an opaque chip behind
            // it. Placing it "somewhere free" does not generalise: with two
            // sets interleaved there is no side of the plot that is reliably
            // empty. ~6.1px per monospace char at 9.5px is close enough to
            // size the chip without measuring text.
            const txt = g.s.label + ' mean ' + g.s.mean.toFixed(3);
            const wpx = txt.length * 6.1 + 10;
            return `
          <line x1="${M.l}" y1="${y(g.s.mean)}" x2="${W - M.r}" y2="${y(g.s.mean)}"
                stroke="${g.ink}" stroke-width="1.5" stroke-dasharray="5 4"></line>
          <rect x="${M.l + 2}" y="${y(g.s.mean) - 18}" width="${wpx}" height="14" rx="4"
                fill="${_TS_D.SURF}" opacity=".92" stroke="${g.ink}" stroke-opacity=".35"></rect>
          <text x="${M.l + 7}" y="${y(g.s.mean) - 7.5}" text-anchor="start"
                font-family="ui-monospace" font-size="9.5" fill="${g.ink}">${_tsDEsc(txt)}</text>`;
        }).join('')}
        ${files.map((o, i) => `
          <line x1="${x(i)}" y1="${y(o.f.mean - o.f.std)}" x2="${x(i)}" y2="${y(o.f.mean + o.f.std)}"
                stroke="${o.g.col}" stroke-width="1.5" opacity=".55"></line>
          <circle cx="${x(i)}" cy="${y(o.f.mean)}" r="4" fill="${o.g.col}"
                  stroke="${_TS_D.SURF}" stroke-width="2" class="_tsdr" data-i="${i}"></circle>`).join('')}
        <line x1="${M.l}" y1="${M.t + ih}" x2="${W - M.r}" y2="${M.t + ih}" stroke="#334155"></line>
        ${files.map((o, i) => (i % every === 0 || i === files.length - 1)
            ? `<text x="${x(i) - 4}" y="${M.t + ih + 14}" text-anchor="end" font-family="ui-monospace"
                 font-size="8.5" fill="${_TS_D.INK4}"
                 transform="rotate(-38 ${x(i) - 4} ${M.t + ih + 14})">${_tsDEsc((o.f.timestamp || o.f.filename).slice(0, 13))}</text>`
            : '').join('')}
        <text x="14" y="${M.t + ih / 2}" text-anchor="middle" font-family="ui-monospace" font-size="10"
              fill="${_TS_D.INK3}" transform="rotate(-90 14 ${M.t + ih / 2})">score</text>
      </svg>
      <div id="_tsDistTip2" style="position:absolute;display:none;pointer-events:none;z-index:3;
           background:#0f172a;border:1px solid #334155;border-radius:9px;padding:7px 10px;
           font-family:ui-monospace,monospace;font-size:10.5px;color:${_TS_D.INK};
           box-shadow:0 6px 18px rgba(0,0,0,.5);white-space:nowrap"></div>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:10.5px;color:${_TS_D.INK2};
                  margin-top:9px;font-family:ui-monospace,monospace">
        ${sets.map(g => `<span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;
              background:${g.col};margin-right:6px;vertical-align:-1px"></i>${_tsDEsc(g.s.label)}
              — ${g.s.files.length} run${g.s.files.length === 1 ? '' : 's'}</span>`).join('')}
        <span style="color:${_TS_D.INK4}">whisker = ±1σ · oldest → newest</span>
      </div>`;

    const svg = document.getElementById('_tsDistRunsSvg');
    const tip = document.getElementById('_tsDistTip2');
    svg.querySelectorAll('._tsdr').forEach(c => {
        c.addEventListener('mouseenter', () => {
            const o = files[+c.dataset.i], f = o.f;
            tip.innerHTML = `<b>${_tsDEsc(f.rel || f.filename)}</b><br>`
                + `n ${f.n.toLocaleString()} &nbsp; mean <b style="color:${o.g.ink}">${f.mean.toFixed(3)}</b>`
                + ` &nbsp; σ ${f.std.toFixed(3)}<br>range ${f.min.toFixed(2)} – ${f.max.toFixed(2)}`
                + (f.name_drift != null && f.name_drift > 0.01
                    ? `<br><span style="color:${_TS_D.BAD}">filename says ${f.name_mean} — drift ${f.name_drift}</span>` : '');
            tip.style.display = 'block';
            const hostR = svg.parentNode.getBoundingClientRect();
            const cx = +c.getAttribute('cx'), cy = +c.getAttribute('cy');
            let lx = (cx / W) * hostR.width + 12;
            if (lx + tip.offsetWidth > hostR.width) lx = (cx / W) * hostR.width - tip.offsetWidth - 12;
            tip.style.left = Math.max(0, lx) + 'px';
            tip.style.top = Math.max(0, (cy / H) * hostR.height - 14) + 'px';
        });
        c.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    });
}

/* ═══ the folder tree ════════════════════════════════════════════════════════
   Top-level folders, collapsed by default — with this directory that is two
   rows, `snar` and `suzuki`, instead of 45. Each row is a control:

     click        scope every chart to it
     📌           pin it; the next click renders A/B against it
     ▸ / ▾        expand to the individual runs, which are the same two controls

   Rows are <div>s, not a <table>: a table row cannot hold a nested clickable
   region cleanly, and the pin has to swallow its own click (stopPropagation)
   or pinning would also select. */
function _tsDistTree() {
    const el = document.getElementById('_tsDistTree');
    const d = _tsDist.data;
    if (!el || !d) return;
    const groups = _tsDistGroups();
    const selK = _tsDKey(_tsDist.sel), pinK = _tsDKey(_tsDist.pin);

    const num = (v, dp) => (v == null ? '—' : v.toFixed(dp == null ? 3 : dp));
    const pinBtn = (kind, key) => {
        const on = pinK === kind + ':' + key;
        return `<button title="${on ? 'unpin' : 'pin for comparison'}"
            onclick="_tsDistPin('${kind}','${_tsDEsc(key).replace(/'/g, "\\'")}',event)"
            style="flex-shrink:0;width:24px;height:22px;border-radius:6px;cursor:pointer;
                   border:1px solid ${on ? _TS_D.CMP : _TS_D.LINE};
                   background:${on ? 'rgba(217,119,6,.18)' : 'transparent'};
                   color:${on ? _TS_D.CMP_INK : _TS_D.INK4};font-size:11px;line-height:1">📌</button>`;
    };
    const rowStyle = (k, indent) => {
        const sel = selK === k, pin = pinK === k;
        return `display:flex;align-items:center;gap:9px;padding:7px 11px 7px ${indent}px;cursor:pointer;
                border-bottom:1px solid rgba(30,41,59,.6);font-family:ui-monospace,monospace;
                font-size:10.5px;transition:background .12s;
                background:${sel ? 'rgba(8,145,178,.12)' : (pin ? 'rgba(217,119,6,.09)' : 'transparent')};
                border-left:3px solid ${sel ? _TS_D.S1 : (pin ? _TS_D.CMP : 'transparent')}`;
    };

    el.innerHTML = `<div style="border:1px solid ${_TS_D.LINE};border-radius:12px;overflow:hidden">`
      + groups.map(g => {
        const gk = 'group:' + g.name;
        const open = _tsDist.open.has(g.name);
        const st = g.stats;
        const head = `
          <div style="${rowStyle(gk, 11)}" onclick="_tsDistPick('group','${_tsDEsc(g.name)}')"
               onmouseover="this.style.filter='brightness(1.35)'" onmouseout="this.style.filter=''">
            <span onclick="_tsDistGroupToggle('${_tsDEsc(g.name)}',event)" title="${open ? 'collapse' : 'expand'}"
                  style="width:16px;text-align:center;color:${_TS_D.INK3};font-size:11px;flex-shrink:0">${open ? '▾' : '▸'}</span>
            <b style="font-family:Inter,system-ui,sans-serif;font-size:12.5px;color:${_TS_D.INK}">${_tsDEsc(g.name)}</b>
            <span style="font-size:9.5px;padding:1px 7px;border-radius:8px;background:rgba(8,145,178,.12);
                  border:1px solid rgba(8,145,178,.3);color:${_TS_D.S1_INK}">${g.files.length} files</span>
            <span style="margin-left:auto;color:${_TS_D.INK3}">n ${st.n.toLocaleString()}</span>
            <span style="color:${_TS_D.INK};min-width:56px;text-align:right">${num(st.mean)}</span>
            <span style="color:${_TS_D.INK2};min-width:52px;text-align:right">σ ${num(st.std)}</span>
            ${pinBtn('group', g.name)}
          </div>`;
        if (!open) return head;
        const kids = g.files.map(f => {
            const fk = 'file:' + f.rel;
            const drift = f.name_drift != null && f.name_drift > 0.01;
            return `
            <div style="${rowStyle(fk, 38)}" onclick="_tsDistPick('file','${_tsDEsc(f.rel).replace(/'/g, "\\'")}')"
                 onmouseover="this.style.filter='brightness(1.35)'" onmouseout="this.style.filter=''">
              <span style="color:${_TS_D.INK2};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                    flex:1;min-width:0" title="${_tsDEsc(f.rel)}">${_tsDEsc(f.filename)}</span>
              <span style="color:${_TS_D.INK3};min-width:46px;text-align:right">${f.n.toLocaleString()}</span>
              <span style="color:${_TS_D.INK};min-width:56px;text-align:right">${num(f.mean, 4)}</span>
              <span style="color:${_TS_D.INK2};min-width:52px;text-align:right">σ ${num(f.std, 4)}</span>
              <span style="color:${_TS_D.INK4};min-width:96px;text-align:right">${num(f.min, 2)} – ${num(f.max, 2)}</span>
              <span title="${drift ? 'the filename disagrees with the rows' : 'filename matches the rows'}"
                    style="color:${drift ? _TS_D.BAD : _TS_D.INK4};min-width:48px;text-align:right">${num(f.name_drift, 4)}</span>
              ${pinBtn('file', f.rel)}
            </div>`;
        }).join('');
        return head + kids;
      }).join('')
      + `</div>
      <div style="font-size:10px;color:${_TS_D.INK4};margin-top:8px;font-family:ui-monospace,monospace">
        ${groups.length} folder${groups.length === 1 ? '' : 's'} ·
        ${(d.files || []).length} files · columns: n · mean · σ · range · name drift
      </div>`;
}

/* ═══ export: SMILES,Name for what is clicked + what is pinned ═══════════════
   ONE FILE, both scopes. Click `snar`, pin `suzuki`, hit the button and you get
   a single CSV holding every molecule from both — not two downloads to
   concatenate by hand. The union is taken over FILES, keyed by their relative
   path, so a file reachable from both scopes is read once and its rows appear
   once. And a scope of `all` swallows the other: unioning "everything" with a
   subset of everything is still everything, and sending both would make the
   server read those files twice.

   TWO COLUMNS, no score. That is the requested shape and it is also the shape
   the downstream tools want — the results CSV is `score,SMILES,Name`, and what
   gets fed back into a generator or a docking run is the molecule, not the
   number that this overlay already draws.

   The bytes are built server-side (`/vina_visualization/ts_smiles_export`),
   because the scan response deliberately carries only histogram counts and
   moments — the SMILES were never shipped to the browser, and shipping 20k of
   them to every viewer just so the button can work would be the wrong trade.

   fetch → Blob rather than a plain <a download>: a failure here comes back as
   JSON, and a link would cheerfully save that error object to disk as a .csv.
   This way the failure lands in the status line under the button instead. */
const _tsDistDl = { busy: false };

/* The scopes to export: the selection, plus the pin when it differs. */
function _tsDistScopes() {
    const out = [_tsDist.sel];
    if (_tsDist.pin && !_tsDSame(_tsDist.pin, _tsDist.sel)) out.push(_tsDist.pin);
    return out.some(s => s && s.kind === 'all') ? [{ kind: 'all', key: '' }] : out;
}

/* Their files, deduplicated by rel — the same union the server will take. */
function _tsDistExportFiles() {
    const seen = new Set(), out = [];
    _tsDistScopes().forEach(s => _tsDFiles(s).forEach(f => {
        if (!seen.has(f.rel)) { seen.add(f.rel); out.push(f); }
    }));
    return out;
}

function _tsDistExportName() {
    const slug = s => (s.kind === 'all' ? 'all'
        : s.kind === 'group' ? s.key
        : (s.key.split('/').pop() || s.key).replace(/\.csv$/i, ''));
    return 'ts_smiles_' + _tsDistScopes()
        .map(s => slug(s).replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 60))
        .join('+') + '.csv';
}

/* Keep the caption honest about what the button would download right now. */
function _tsDistDlSync() {
    const btn = document.getElementById('_tsDistDlBtn');
    const msg = document.getElementById('_tsDistDlMsg');
    if (!btn || !msg || _tsDistDl.busy) return;      // never stomp "building…"
    const files = _tsDistExportFiles();
    const rows = files.reduce((a, f) => a + f.n, 0);
    btn.disabled = !files.length;
    btn.style.opacity = files.length ? '1' : '.45';
    msg.style.color = _TS_D.INK4;
    msg.innerHTML = files.length
        ? _tsDistScopes().map(s => _tsDEsc(_tsDLabel(s))).join(' + ')
          + ' · ' + files.length + ' file' + (files.length === 1 ? '' : 's')
          + ' · ' + rows.toLocaleString() + ' rows'
        : 'nothing in scope';
}

function _tsDistExport() {
    const d = _tsDist.data;
    const btn = document.getElementById('_tsDistDlBtn');
    const msg = document.getElementById('_tsDistDlMsg');
    if (!d || _tsDistDl.busy || !btn || !msg) return;
    const files = _tsDistExportFiles();
    if (!files.length) { msg.style.color = _TS_D.WARN; msg.textContent = 'nothing in scope'; return; }

    _tsDistDl.busy = true;
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳ building…';
    msg.style.color = _TS_D.INK3;
    msg.textContent = 'reading ' + files.length + ' file' + (files.length === 1 ? '' : 's') + '…';

    const qs = ['dir=' + encodeURIComponent(d.dir),
                'recursive=' + (d.recursive === false ? '0' : '1')]
        .concat(_tsDistScopes().map(s => 'scope=' + encodeURIComponent(s.kind + ':' + s.key)));

    fetch('/vina_visualization/ts_smiles_export?' + qs.join('&'))
        .then(r => {
            // The error path is JSON, the success path is text/csv. Sniff the
            // type rather than trusting the status: this app answers most
            // failures with 200 + {ok:false}.
            const ct = r.headers.get('content-type') || '';
            if (!r.ok || ct.indexOf('json') >= 0) {
                return r.json()
                    .catch(() => ({ err: 'HTTP ' + r.status }))
                    .then(j => { throw new Error((j && j.err) || ('HTTP ' + r.status)); });
            }
            const meta = {
                rows: +(r.headers.get('X-TS-Rows') || 0),
                files: +(r.headers.get('X-TS-Files') || 0),
                blank: +(r.headers.get('X-TS-Blank') || 0),
                trunc: r.headers.get('X-TS-Truncated') === '1',
            };
            return r.blob().then(b => ({ b: b, meta: meta }));
        })
        .then(res => {
            const url = URL.createObjectURL(res.b);
            const a = document.createElement('a');
            a.href = url;
            a.download = _tsDistExportName();
            document.body.appendChild(a);
            a.click();
            setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
            const m = res.meta;
            msg.style.color = m.trunc ? _TS_D.WARN : _TS_D.GOOD;
            msg.innerHTML = (m.trunc ? '⚠ capped — ' : '✓ ')
                + m.rows.toLocaleString() + ' rows from ' + m.files + ' file'
                + (m.files === 1 ? '' : 's') + ' → ' + _tsDEsc(a.download)
                + (m.blank ? ' · ' + m.blank.toLocaleString() + ' blank SMILES dropped' : '');
        })
        .catch(e => {
            msg.style.color = _TS_D.BAD;
            msg.textContent = '✕ ' + e.message;
        })
        .finally(() => {
            _tsDistDl.busy = false;
            btn.disabled = false;
            btn.textContent = label;
        });
}

window._tsDistExport      = _tsDistExport;
window._tsDistDlSync      = _tsDistDlSync;
window._tsDistToggle      = _tsDistToggle;
window._tsDistEnsurePanel = _tsDistEnsurePanel;
window._tsDistScan        = _tsDistScan;
window._tsDistRender      = _tsDistRender;
window._tsDistClose       = _tsDistClose;
window._tsDistPick        = _tsDistPick;
window._tsDistPin         = _tsDistPin;
window._tsDistGroupToggle = _tsDistGroupToggle;
window._tsDistExpandAll   = _tsDistExpandAll;
window._tsDistReset       = _tsDistReset;
window._tsDist            = _tsDist;   // debugging + harness access
