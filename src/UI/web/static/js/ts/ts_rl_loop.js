// =============================================================================
// ts/ts_rl_loop.js — the closed loop, inside the 🧪 RL tab.
//
//   TS run ends → top 30% by ChemBERT score → Vina search (in-browser, per
//   ligand) → DeepAtom / Yupu_GIGN pK → commit → bias_generator → warm-up
//   checkpoint → the NEXT Run TS starts from biased reagent priors.
//
// SELF-INSTALLING, like ts_rl.js. It needs no template edits: on load it
// injects its own <style>, finds the RL pane and splices one card in above
// the diagnostics grid. Load AFTER ts_rl.js (which creates #tsPaneRl) and
// after ts_run.js (whose _tsFinalise this file wraps).
//
// ── Three integration facts that shaped every line below ────────────────────
//
// 1. pose.js emits NO events. No CustomEvent, no callback argument, no
//    promise from vina.eng.run(). The single genuine promise in the whole
//    module is PG.build(). Everything else here is a bounded poll on the
//    documented state flags (_running / _modes / _dg), with a timeout, because
//    a poll with no deadline against a worker that can die silently is how you
//    get a loop that hangs on molecule 7 of 30 forever.
//
// 2. The receptor text lives in a module-private `let protein` with no
//    accessor (pose.js:599). The scoring endpoints need it as a string. So we
//    wrap PG._applyProtein to mirror `raw` somewhere reachable — the same
//    wrap-don't-fork trick ts_rl.js already uses on PG.open/PG.close.
//
// 3. We call /pose/deepatom_score and /pose/gign_score DIRECTLY rather than
//    going through PG.mc.scoreDeepAtom() / scoreGign(). Those two are
//    fire-and-forget (they return undefined and paint the DOM), and
//    scoreDeepAtom dereferences PG.mc.best with no fallback — it throws if the
//    MC stage was never entered. A loop needs the value back, in order, with
//    its errors attached; a fetch gives that and a DOM repaint does not.
// =============================================================================
console.log('%c[ts-rl-loop] LOADED', 'background:#0e7490;color:#cffafe;padding:2px 6px;border-radius:3px');

/* ─── Tunables. Every one of them is visible in the UI or explained here. ── */
const _TS_RL_L = {
    FRACTION      : 0.30,     // "top 30% of the ChemBERT estimated affinity"
    CAP           : 30,       // molecules per round. See the note by _tsRlLHarvest.
    POLL_MS       : 250,      // how often we look at pose.js's state flags
    VINA_TIMEOUT  : 420000,   // 7 min/ligand. Exhaustiveness 8 on 4 cores is ~10-60 s;
                              // this is a stuck-worker deadline, not a budget.
    SCORE_TIMEOUT : 1260000,  // 21 min. The server's own cap is 1200 s (input_routes.yml
                              // deepatom.timeout_seconds / pose.gign_timeout_seconds),
                              // so ours must be LONGER or we abandon a request the
                              // server is still going to answer.
    BUILD_TIMEOUT : 60000,
    HARVEST_TRIES : 5,        // the results CSV is renamed just before __DONE__;
    HARVEST_WAIT  : 2000,     // retry rather than race it.
};

const _tsRlL = {
    armed: false, running: false, cancelled: false,
    scorer: 'vina', frac: _TS_RL_L.FRACTION, cap: _TS_RL_L.CAP,
    mapping: 'zblend', strength: 0.5,
    cands: [], results: [], idx: 0,
    source: null, affinity: null, biasReport: null,
    rx: { raw: '', id: '' },      // mirrored receptor, see fact 2 above
    t0: 0, note: '',
};

/* ═══ style ══════════════════════════════════════════════════════════════ */
const _TS_RL_L_STYLE = `
  #tsRlLoop{background:var(--surface-1,#0b1120);border:1px solid var(--line,#1e293b);
       border-radius:15px;padding:15px 16px;margin:14px 24px 0}
  #tsRlLoop h3{margin:0;font-size:12.5px;font-weight:700;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  #tsRlLoop .lp-why{font-size:11.5px;color:var(--ink-2,#94a3b8);margin:7px 0 12px;line-height:1.55}
  #tsRlLoop .lp-why b{color:var(--ink,#e2e8f0)}
  #tsRlLoop .lp-bar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:11px}
  #tsRlLoop .lp-l{font-size:9.5px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em}
  #tsRlLoop select,#tsRlLoop input[type=number]{background:#0b1120;border:1px solid #1e293b;border-radius:8px;
       color:#cbd5e1;font-size:11px;padding:5px 8px;font-family:inherit;outline:none}
  #tsRlLoop input[type=number]{width:62px;font-family:ui-monospace,monospace;color:#22d3ee}
  #tsRlLoop .lp-b{padding:6px 13px;border-radius:9px;border:1px solid transparent;font:inherit;font-size:11.5px;
       font-weight:700;cursor:pointer;color:#fff;background:linear-gradient(135deg,#0891b2,#7c3aed)}
  #tsRlLoop .lp-b[disabled]{opacity:.4;cursor:not-allowed}
  #tsRlLoop .lp-b.g{background:#0b1120;border:1px solid #1e293b;color:#94a3b8;font-weight:600}
  #tsRlLoop .lp-b.r{background:rgba(127,29,58,.5);border:1px solid #7f1d3a;color:#fecdd3}
  #tsRlLoop .lp-b.ok{background:linear-gradient(135deg,#059669,#0891b2)}
  #tsRlLoop .lp-arm{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;cursor:pointer;
       padding:5px 10px;border:1px solid #1e293b;border-radius:9px;background:#0b1120}
  #tsRlLoop .lp-arm.on{border-color:#0e7490;background:#062a33;color:#67e8f9}
  #tsRlLoop .lp-prog{height:6px;border-radius:99px;background:#0f172a;overflow:hidden;margin:10px 0 8px}
  #tsRlLoop .lp-prog i{display:block;height:100%;background:linear-gradient(90deg,#0891b2,#7c3aed);
       width:0;transition:width .3s}
  #tsRlLoop .lp-st{font-family:ui-monospace,monospace;font-size:10.5px;color:#94a3b8;line-height:1.6;
       min-height:17px;word-break:break-word}
  #tsRlLoop .lp-warn{border-radius:10px;padding:9px 11px;font-size:11px;line-height:1.55;margin:10px 0 0;
       background:rgba(245,158,11,.09);border:1px solid rgba(245,158,11,.35);color:#fbbf24}
  #tsRlLoop .lp-err{background:rgba(251,113,133,.09);border-color:rgba(251,113,133,.35);color:#fda4af}
  #tsRlLoop .lp-good{background:rgba(52,211,153,.09);border-color:rgba(52,211,153,.35);color:#6ee7b7}
  #tsRlLoop table{width:100%;border-collapse:collapse;font-family:ui-monospace,monospace;font-size:10px;margin-top:11px}
  #tsRlLoop th{text-align:left;color:#475569;font-weight:600;padding:4px 6px;border-bottom:1px solid #1e293b;
       font-size:8.5px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0;background:#0b1120}
  #tsRlLoop td{padding:4px 6px;border-bottom:1px solid rgba(30,41,59,.6);color:#94a3b8;vertical-align:top}
  #tsRlLoop td.n{text-align:right;color:#e2e8f0}
  #tsRlLoop td.smi{max-width:270px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#64748b}
  #tsRlLoop tr.cur td{background:rgba(8,145,178,.09)}
  #tsRlLoop tr.bad td{color:#fb7185}
  #tsRlLoop .lp-wrap{max-height:340px;overflow:auto;border:1px solid #1e293b;border-radius:11px;margin-top:11px}
  #tsRlLoop .lp-wrap table{margin:0}
  #tsRlLoop .lp-k{display:inline-block;font-family:ui-monospace,monospace;font-size:9px;padding:1px 6px;
       border-radius:5px;background:rgba(8,145,178,.14);border:1px solid rgba(8,145,178,.4);color:#22d3ee}
`;

/* ═══ markup ═════════════════════════════════════════════════════════════ */
function _tsRlLBody() {
    return `
    <h3>🔁 Closed loop <span class="lp-k">post-run</span>
        <span class="lp-k" id="tsRlLImpl" style="background:rgba(139,92,246,.14);
              border-color:rgba(139,92,246,.4);color:#a78bfa">idle</span></h3>
    <div class="lp-why">
      Takes the top <b id="tsRlLFracEcho">30%</b> of the finished run by the score TS already
      computed (ChemBERT ΔG through the Elion reward), poses each one with the in-browser
      <b>Vina 1.2.7</b> search, re-scores the pose with the model you pick, and writes the result
      back as a reagent prior. <b>The next Run TS starts from it</b> — there is no channel into a
      run already in flight, so feedback lands on the following round by construction.
    </div>

    <div class="lp-bar">
      <span class="lp-l">Scorer</span>
      <select id="tsRlLScorer" onchange="_tsRlLSet('scorer',this.value)">
        <option value="vina">Vina ΔG only (in-browser)</option>
        <option value="deepatom">DeepAtom ΔG (server CNN)</option>
        <option value="yupu_gign">Yupu_GIGN · GIGN affinity</option>
      </select>
      <span class="lp-l">Top</span>
      <input id="tsRlLFrac" type="number" min="1" max="100" step="1" value="30"
             title="percent of the run's unique products, by TS score"
             onchange="_tsRlLSet('frac',this.value/100)"> <span class="lp-l">%</span>
      <span class="lp-l">Cap</span>
      <input id="tsRlLCap" type="number" min="1" max="500" step="1" value="30"
             title="hard limit on molecules per round — the in-browser search is 10-60 s each"
             onchange="_tsRlLSet('cap',this.value)">
      <span class="lp-arm" id="tsRlLArm" onclick="_tsRlLToggleArm()">
        <span id="tsRlLArmDot">○</span> auto-start when a run finishes</span>
      <button class="lp-b" id="tsRlLRun" onclick="_tsRlLStart()">▶ Run loop</button>
      <button class="lp-b r" id="tsRlLStop" onclick="_tsRlLCancel()" style="display:none">■ Cancel</button>
    </div>

    <div class="lp-prog"><i id="tsRlLProg"></i></div>
    <div class="lp-st" id="tsRlLStatus">Idle. Finish a TS run, then press <b>▶ Run loop</b> — or arm it above.</div>
    <div id="tsRlLNotes"></div>
    <div id="tsRlLTable"></div>

    <div class="lp-bar" id="tsRlLCommitBar" style="display:none;margin-top:12px">
      <span class="lp-l">Mapping</span>
      <select id="tsRlLMapping" onchange="_tsRlLSet('mapping',this.value)">
        <option value="zblend">z-blend (keep reward units)</option>
        <option value="rank">rank nudge (ignore magnitudes)</option>
        <option value="raw">raw pK (scale mismatch)</option>
      </select>
      <span class="lp-l">Strength</span>
      <input id="tsRlLStrength" type="number" min="0" max="2" step="0.1" value="0.5"
             title="a reagent one sigma above the batch moves this many reward-sigma"
             onchange="_tsRlLSet('strength',this.value)">
      <button class="lp-b ok" id="tsRlLCommit" onclick="_tsRlLCommit()">⇩ Commit → bias_generator</button>
      <button class="lp-b g" onclick="_tsRlLCommit(true)" title="compute the checkpoint and report it without writing">dry run</button>
    </div>`;
}

/* ═══ install ════════════════════════════════════════════════════════════ */
let _tsRlLInstalled = false;

function _tsRlLInstall() {
    if (_tsRlLInstalled && document.getElementById('tsRlLoop')) return true;
    const pane = document.getElementById('tsPaneRl');
    if (!pane) return false;

    if (!document.getElementById('tsRlLoopStyle')) {
        const st = document.createElement('style');
        st.id = 'tsRlLoopStyle';
        st.textContent = _TS_RL_L_STYLE;
        document.head.appendChild(st);
    }

    const card = document.createElement('div');
    card.id = 'tsRlLoop';
    card.innerHTML = _tsRlLBody();

    // Anchor, most-specific first, and SAY which one answered. Positional DOM
    // assumptions rot; ts_rl.js learned that with #poseLigGrid's sibling.
    const body = pane.querySelector('.body');
    const wrap = pane.querySelector('.wrap');
    if (body && body.parentNode) body.parentNode.insertBefore(card, body);
    else if (wrap) { wrap.insertBefore(card, wrap.firstChild); console.warn('[ts-rl-loop] .body not found — anchored to .wrap'); }
    else { pane.appendChild(card); console.warn('[ts-rl-loop] neither .body nor .wrap found — appended to the pane'); }

    _tsRlLInstalled = true;
    _tsRlLSyncScorerOptions();
    _tsRlLWrapPose();
    _tsRlLWrapFinalise();
    return true;
}

/* The two NN scorers are blank-by-default in config/input_routes.yml
   (pose.default_script and pose.gign_script both ""), and an unconfigured one
   answers HTTP 200 {ok:false} — which inside a 30-molecule loop is 30 silent
   nulls. Mirror the pose tool's own dropdown so an option that cannot work is
   labelled before the run, not discovered during it. */
function _tsRlLSyncScorerOptions() {
    const mine = document.getElementById('tsRlLScorer');
    const theirs = document.getElementById('poseMcScorer');
    if (!mine || !theirs) return;
    ['deepatom', 'yupu_gign'].forEach(v => {
        const has = !!theirs.querySelector(`option[value="${v}"]`);
        const opt = mine.querySelector(`option[value="${v}"]`);
        if (opt && !has) { opt.disabled = true; opt.textContent += ' — not on this page'; }
    });
}

/* ── fact 2: mirror the receptor text out of pose.js's private scope ────── */
function _tsRlLWrapPose() {
    const PG = window.PoseGen;
    if (!PG || typeof PG._applyProtein !== 'function' || PG._applyProtein.__rlLoop) return;
    const orig = PG._applyProtein.bind(PG);
    const w = function (p, id, raw, useOwnCenter) {
        try { if (typeof raw === 'string' && raw.length) { _tsRlL.rx.raw = raw; _tsRlL.rx.id = id || ''; } }
        catch (_) { /* mirroring must never break the pose tool */ }
        return orig(p, id, raw, useOwnCenter);
    };
    w.__rlLoop = true;
    PG._applyProtein = w;
}

/* If the user never loaded a receptor by hand, the tool's own default is the
   honest thing to score against — it is the one the pose panel is showing. */
function _tsRlLReceptor() {
    if (_tsRlL.rx.raw) return Promise.resolve(_tsRlL.rx.raw);
    return fetch('/pose/default_receptor').then(r => r.ok ? r.json() : null).then(d => {
        if (d && d.ok && d.pdb) { _tsRlL.rx.raw = d.pdb; _tsRlL.rx.id = d.name || 'default'; return d.pdb; }
        return '';
    }).catch(() => '');
}

/* ── auto-arm: wrap _tsFinalise, the one place a run is declared over ───── */
function _tsRlLWrapFinalise() {
    if (typeof window._tsFinalise !== 'function' || window._tsFinalise.__rlLoop) return;
    const orig = window._tsFinalise;
    const w = function (cancelled) {
        const r = orig.apply(this, arguments);
        try {
            if (!cancelled && _tsRlL.armed && !_tsRlL.running) {
                // _tsFinalise decides success/error inside a deferred callback
                // (ts_run.js:326) because the error line arrives ~500 ms late.
                // Reading _ts._sawError now would race it, so wait past that
                // window and then read the settled flag — the same reason the
                // status badge is written there and not here.
                setTimeout(() => {
                    const errored = (typeof _ts !== 'undefined' && _ts._sawError);
                    if (errored) { _tsRlLStatus('Run ended with an error — loop not started.', 'err'); return; }
                    _tsRlLStatus('Run finished. Auto-starting the loop…');
                    _tsRlLStart();
                }, 900);
            }
        } catch (e) { console.warn('[ts-rl-loop] finalise hook', e); }
        return r;
    };
    w.__rlLoop = true;
    window._tsFinalise = w;
    // ts_run.js calls _tsFinalise as a BARE identifier from _tsCancel and from
    // the __DONE__ handler, and a bare call resolves the top-level function
    // declaration, not window.*. Rebinding window._tsFinalise alone therefore
    // hooks nothing. There is no way to rebind a function declaration from
    // another script, so we also poll the run flag as a backstop.
    _tsRlLWatchRun();
}

/* Backstop for the bare-identifier problem above: watch _ts.running fall. */
let _tsRlLWatch = null, _tsRlLWasRunning = false;
function _tsRlLWatchRun() {
    if (_tsRlLWatch) return;
    _tsRlLWatch = setInterval(() => {
        if (typeof _ts === 'undefined') return;
        const now = !!_ts.running;
        if (_tsRlLWasRunning && !now && _tsRlL.armed && !_tsRlL.running) {
            _tsRlLWasRunning = now;
            setTimeout(() => {
                if (_ts._sawError) { _tsRlLStatus('Run ended with an error — loop not started.', 'err'); return; }
                if (!_tsRlL.running) { _tsRlLStatus('Run finished. Auto-starting the loop…'); _tsRlLStart(); }
            }, 900);
            return;
        }
        _tsRlLWasRunning = now;
    }, 1000);
}

/* ═══ small helpers ══════════════════════════════════════════════════════ */
function _tsRlLSet(k, v) {
    if (k === 'frac')     _tsRlL.frac = Math.max(0.01, Math.min(1, parseFloat(v) || 0.30));
    else if (k === 'cap') _tsRlL.cap = Math.max(1, parseInt(v, 10) || 30);
    else if (k === 'strength') _tsRlL.strength = Math.max(0, Math.min(2, parseFloat(v) || 0.5));
    else _tsRlL[k] = v;
    const e = document.getElementById('tsRlLFracEcho');
    if (e) e.textContent = Math.round(_tsRlL.frac * 100) + '%';
}

function _tsRlLToggleArm() {
    _tsRlL.armed = !_tsRlL.armed;
    const el = document.getElementById('tsRlLArm'), dot = document.getElementById('tsRlLArmDot');
    if (el)  el.classList.toggle('on', _tsRlL.armed);
    if (dot) dot.textContent = _tsRlL.armed ? '●' : '○';
    _tsRlLStatus(_tsRlL.armed
        ? 'Armed. The loop will start on its own when the next TS run finishes cleanly.'
        : 'Disarmed.');
}

function _tsRlLStatus(msg, kind) {
    const el = document.getElementById('tsRlLStatus');
    if (el) el.innerHTML = msg;
    if (kind) console.log('[ts-rl-loop] ' + String(msg).replace(/<[^>]+>/g, ''));
}

function _tsRlLNote(html, cls) {
    const el = document.getElementById('tsRlLNotes');
    if (el) el.innerHTML = html ? `<div class="lp-warn ${cls || ''}">${html}</div>` : '';
}

function _tsRlLProg(done, total) {
    const b = document.getElementById('tsRlLProg');
    if (b) b.style.width = (total ? Math.round(done / total * 100) : 0) + '%';
    const chip = document.getElementById('tsRlLImpl');
    if (chip) chip.textContent = total ? `${done}/${total}` : 'idle';
}

const _tsRlLSleep = ms => new Promise(r => setTimeout(r, ms));

/* Every string below reaches innerHTML. SMILES and product ids are engine
   data, not user prose, but a scorer's `err` is a server-formatted message
   that can carry a path or raw stderr — and G55 in the playbook is exactly
   this class of bug in this codebase. Escape, don't reason about the source. */
function _tsRlLEsc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/* Show/hide a control that may not exist if _tsRlLInstall never ran (the pane
   is created by ts_rl.js; if that file failed to load, every getElementById
   here is null and a bare .style would throw inside the loop). */
function _tsRlLShow(id, on) {
    const e = document.getElementById(id);
    if (e) e.style.display = on ? '' : 'none';
}

/* Bounded poll. Resolves with true, or false on timeout — never hangs.
   Everything in pose.js is observed this way because it publishes no events. */
async function _tsRlLUntil(test, timeoutMs, label) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        if (_tsRlL.cancelled) return false;
        let ok = false;
        try { ok = !!test(); } catch (_) { ok = false; }
        if (ok) return true;
        await _tsRlLSleep(_TS_RL_L.POLL_MS);
    }
    console.warn('[ts-rl-loop] timeout waiting for ' + label);
    return false;
}

/* ═══ 1. harvest — top 30% by the ChemBERT score TS already produced ══════ */
async function _tsRlLHarvest() {
    const rxn = (typeof _tsRlRxnKey === 'function' ? _tsRlRxnKey() : '') || '';
    const jobId = (typeof _ts !== 'undefined' && _ts.jobId) || '';
    const outDir = document.getElementById('tsOutputDirInput')?.value.trim() || '';
    const q = new URLSearchParams({
        rxn_key: rxn, job_id: jobId, output_dir: outDir,
        frac: String(_tsRlL.frac), cap: String(_tsRlL.cap),
    });
    // The engine renames <...>_tmp.csv to <...>_mean<X>_std<X>.csv just before
    // it pushes __DONE__. That ordering holds, but a retry costs nothing and
    // turns a lost race into a two-second delay instead of an empty round.
    for (let attempt = 1; attempt <= _TS_RL_L.HARVEST_TRIES; attempt++) {
        if (_tsRlL.cancelled) return null;
        let d = null;
        try { d = await fetch('/vina_visualization/ts_rl_harvest?' + q).then(r => r.json()); }
        catch (e) { d = { ok: false, err: String(e) }; }
        if (d && d.ok && (d.candidates || []).length) return d;
        if (attempt < _TS_RL_L.HARVEST_TRIES) {
            _tsRlLStatus(`Waiting for the results CSV… (${attempt}/${_TS_RL_L.HARVEST_TRIES})`);
            await _tsRlLSleep(_TS_RL_L.HARVEST_WAIT);
        } else {
            _tsRlLNote(`<b>No candidates.</b> ${(d && d.err) || 'harvest returned nothing'}`, 'lp-err');
            return null;
        }
    }
    return null;
}

/* ═══ 2. one molecule: build → Vina search → score ════════════════════════ */
async function _tsRlLOne(cand, receptorPdb) {
    const PG = window.PoseGen;
    const rec = {
        name: cand.name, reagents: cand.reagents || [], smiles: cand.smiles,
        ts_score: cand.ts_score, scorer: _tsRlL.scorer, ok: false,
        vina_dg: null, pk: null, dg: null, err: '',
    };
    if (!PG) { rec.err = 'PoseGen not loaded'; return rec; }

    // ── build the conformer. The ONE real promise pose.js offers. ─────────
    const box = document.getElementById('poseSmiles');
    if (box) { box.value = cand.smiles; box.dispatchEvent(new Event('input', { bubbles: true })); }
    try {
        await Promise.race([PG.build(), _tsRlLSleep(_TS_RL_L.BUILD_TIMEOUT).then(() => { throw new Error('build timeout'); })]);
    } catch (e) { rec.err = 'build failed: ' + e.message; return rec; }
    if (_tsRlL.cancelled) { rec.err = 'cancelled'; return rec; }

    // PG.build() resolves even when the server rejected the SMILES — it falls
    // back to the in-browser parser and, if that also fails, leaves the
    // previous ligand in place. Scoring then silently re-measures molecule
    // n-1 and attributes it to molecule n. eng.ready() is the guard.
    const eng = PG.vina && PG.vina.eng;
    if (!eng || !eng.available || !eng.available()) { rec.err = 'Vina engine unavailable'; return rec; }
    if (typeof eng.ready === 'function' && !eng.ready()) { rec.err = 'ligand did not load (SMILES rejected?)'; return rec; }

    // ── Vina search. _modes is NOT cleared between runs (pose.js _procReset
    //    leaves it), so a failed search would otherwise report the PREVIOUS
    //    ligand's affinity. Clear both before arming the poll.
    eng._modes = [];
    eng._dg = null;
    try { eng.run(); } catch (e) { rec.err = 'vina run threw: ' + e.message; return rec; }

    const done = await _tsRlLUntil(
        () => !eng._running && (eng._dg != null || (eng._modes && eng._modes.length)),
        _TS_RL_L.VINA_TIMEOUT, 'vina search');
    if (_tsRlL.cancelled) { try { eng.cancel(); } catch (_) {} rec.err = 'cancelled'; return rec; }
    if (!done) { try { eng.cancel(); } catch (_) {} rec.err = 'vina search timed out'; return rec; }

    const aff = (eng._dg != null) ? eng._dg
              : (eng._modes && eng._modes[0] ? eng._modes[0].affinity : null);
    rec.vina_dg = (typeof aff === 'number' && isFinite(aff)) ? +aff.toFixed(3) : null;

    if (_tsRlL.scorer === 'vina') {
        rec.ok = rec.vina_dg != null;
        if (!rec.ok) rec.err = 'vina produced no affinity';
        return rec;
    }

    // ── re-score the posed complex with the chosen NN ─────────────────────
    let ligPdb = '';
    try { ligPdb = PG.mc._ligandPdb(PG.vina._ligWorld()); }
    catch (e) { rec.err = 'could not export the posed ligand: ' + e.message; return rec; }
    if (!ligPdb || !receptorPdb) {
        rec.err = !receptorPdb ? 'no receptor loaded' : 'empty ligand PDB';
        return rec;
    }

    const url = _tsRlL.scorer === 'deepatom' ? '/pose/deepatom_score' : '/pose/gign_score';
    const safe = String(cand.name || ('rl' + cand.rank)).replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 40);
    let d = null;
    try {
        const ctl = new AbortController();
        const kill = setTimeout(() => ctl.abort(), _TS_RL_L.SCORE_TIMEOUT);
        d = await fetch(url, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctl.signal,
            body: JSON.stringify({ ligand_pdb: ligPdb, receptor_pdb: receptorPdb,
                                   name: 'rl_' + safe, smiles: cand.smiles, out_dir: '' }),
        }).then(r => r.json());
        clearTimeout(kill);
    } catch (e) { rec.err = 'scorer request failed: ' + e.message; return rec; }

    // Both scorers answer HTTP 200 with {ok:false, err} on every failure path,
    // including "script not configured". Surface err verbatim — it is the
    // difference between "install the model" and "the pose was garbage".
    if (!d || !d.ok) { rec.err = (d && d.err) || 'scorer returned not-ok'; return rec; }
    rec.pk = (typeof d.pred_pk === 'number') ? +d.pred_pk.toFixed(4) : null;
    rec.dg = (typeof d.deltaG === 'number') ? +d.deltaG.toFixed(4)
           : (rec.pk != null ? +(-rec.pk * 1.36).toFixed(4) : null);
    rec.ok = rec.pk != null;
    if (!rec.ok) rec.err = 'scorer returned no pK';
    return rec;
}

/* ═══ 3. the loop ════════════════════════════════════════════════════════ */
async function _tsRlLStart() {
    if (_tsRlL.running) return;
    _tsRlL.running = true; _tsRlL.cancelled = false;
    _tsRlL.results = []; _tsRlL.idx = 0; _tsRlL.affinity = null; _tsRlL.biasReport = null;
    _tsRlLShow('tsRlLRun', false);
    _tsRlLShow('tsRlLStop', true);
    _tsRlLShow('tsRlLCommitBar', false);
    _tsRlLNote('');
    _tsRlL.t0 = Date.now();

    try {
        _tsRlLStatus('Harvesting the top ' + Math.round(_tsRlL.frac * 100) + '% by ChemBERT score…');
        const h = await _tsRlLHarvest();
        if (!h) return;
        _tsRlL.source = h;
        _tsRlL.cands = h.candidates;

        // Say what was dropped. A capped list presented as "the top 30%" reads
        // as complete coverage when it is a sample of it.
        const s = h.selection || {};
        if (s.capped) {
            _tsRlLNote(`Posing <b>${s.n_selected}</b> of <b>${s.n_fraction}</b> molecules in the top `
                + `${Math.round((s.fraction || 0) * 100)}% (${s.n_unique} unique products in the run). `
                + `The cap is doing the work — raise it to widen the round, but the in-browser `
                + `search is 10-60 s per ligand.`);
        }

        const receptor = _tsRlL.scorer === 'vina' ? '' : await _tsRlLReceptor();
        if (_tsRlL.scorer !== 'vina' && !receptor) {
            _tsRlLNote('<b>No receptor.</b> Load one in the pose workspace (or set '
                + '<code>pose.default_receptor_pdb</code>) — DeepAtom and GIGN both score a complex.', 'lp-err');
            return;
        }

        for (let i = 0; i < _tsRlL.cands.length; i++) {
            if (_tsRlL.cancelled) break;
            _tsRlL.idx = i;
            const c = _tsRlL.cands[i];
            _tsRlLProg(i, _tsRlL.cands.length);
            _tsRlLStatus(`[${i + 1}/${_tsRlL.cands.length}] `
                + _tsRlLEsc(c.name || c.smiles.slice(0, 44)) + ' — posing…');
            _tsRlLTable();
            const rec = await _tsRlLOne(c, receptor);
            _tsRlL.results.push(rec);
            _tsRlLTable();
        }

        _tsRlLProg(_tsRlL.results.length, _tsRlL.cands.length);
        const good = _tsRlL.results.filter(r => r.ok).length;
        const secs = Math.round((Date.now() - _tsRlL.t0) / 1000);
        _tsRlLStatus(_tsRlL.cancelled
            ? `Cancelled after ${_tsRlL.results.length} of ${_tsRlL.cands.length} — ${good} scored, ${secs}s.`
            : `Done: <b>${good}</b> of ${_tsRlL.results.length} scored in ${secs}s. Review, then commit.`);
        if (good) _tsRlLShow('tsRlLCommitBar', true);
        else _tsRlLNote('<b>Nothing scored.</b> Every molecule failed — check the '
            + '<code>err</code> column; one repeated message usually means an unconfigured scorer '
            + 'rather than 30 bad poses.', 'lp-err');
    } catch (e) {
        console.error('[ts-rl-loop]', e);
        _tsRlLNote('<b>Loop crashed:</b> ' + e.message, 'lp-err');
    } finally {
        _tsRlL.running = false;
        _tsRlLShow('tsRlLRun', true);
        _tsRlLShow('tsRlLStop', false);
    }
}

function _tsRlLCancel() {
    _tsRlL.cancelled = true;
    try { window.PoseGen && PoseGen.vina.eng.cancel(); } catch (_) {}
    _tsRlLStatus('Cancelling after the current molecule…');
}

/* ═══ 4. results table ═══════════════════════════════════════════════════ */
function _tsRlLTable() {
    const el = document.getElementById('tsRlLTable');
    if (!el) return;
    if (!_tsRlL.cands.length) { el.innerHTML = ''; return; }
    const rows = _tsRlL.cands.map((c, i) => {
        const r = _tsRlL.results[i];
        const cur = (i === _tsRlL.idx && _tsRlL.running && !r);
        const cls = cur ? 'cur' : (r && !r.ok ? 'bad' : '');
        const use = r && r.ok;
        const smi = _tsRlLEsc(c.smiles);
        return `<tr class="${cls}">
          <td class="n">${i + 1}</td>
          <td><input type="checkbox" ${use ? 'checked' : ''} ${r ? '' : 'disabled'}
                     onchange="_tsRlLToggleRow(${i},this.checked)"></td>
          <td>${_tsRlLEsc(c.name) || '—'}</td>
          <td class="smi" title="${smi}">${smi}</td>
          <td class="n">${c.ts_score != null ? c.ts_score.toFixed(3) : '—'}</td>
          <td class="n">${r && r.vina_dg != null ? r.vina_dg.toFixed(2) : (cur ? '…' : '—')}</td>
          <td class="n">${r && r.pk != null ? r.pk.toFixed(2) : (r ? '—' : '')}</td>
          <td style="color:#fb7185">${r && r.err ? _tsRlLEsc(r.err.slice(0, 70)) : ''}</td>
        </tr>`;
    }).join('');
    el.innerHTML = `<div class="lp-wrap"><table>
        <thead><tr><th>#</th><th>use</th><th>product</th><th>SMILES</th>
        <th>TS score</th><th>Vina ΔG</th><th>pK</th><th>note</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`;
}

function _tsRlLToggleRow(i, on) {
    const r = _tsRlL.results[i];
    if (r) r._excluded = !on;
}

/* ═══ 5. commit — write the affinity file, then bias ══════════════════════ */
async function _tsRlLCommit(dryRun) {
    const use = _tsRlL.results.filter(r => r.ok && !r._excluded);
    if (!use.length) { _tsRlLNote('Nothing selected to commit.', 'lp-err'); return; }
    const btn = document.getElementById('tsRlLCommit');
    if (btn) btn.disabled = true;
    try {
        _tsRlLStatus(`Writing ${use.length} measurements…`);
        // The FULL result set goes to disk, excluded and failed rows included —
        // a round has to be auditable, and rl_bias skips what it cannot use and
        // reports the count. Only the selection is narrowed here.
        const fb = await fetch('/vina_visualization/ts_rl_feedback', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rxn_key: (typeof _tsRlRxnKey === 'function' ? _tsRlRxnKey() : ''),
                job_id: (typeof _ts !== 'undefined' && _ts.jobId) || '',
                scorer: _tsRlL.scorer, receptor: _tsRlL.rx.id,
                source_csv: (_tsRlL.source || {}).source_csv || '',
                selection: (_tsRlL.source || {}).selection || {},
                records: _tsRlL.results.map(r => Object.assign({}, r, { ok: r.ok && !r._excluded })),
            }),
        }).then(r => r.json());
        if (!fb.ok) { _tsRlLNote('<b>Could not write the affinity file:</b> ' + fb.err, 'lp-err'); return; }
        _tsRlL.affinity = fb.path;

        _tsRlLStatus('Biasing the reagent prior…');
        const rep = await fetch('/vina_visualization/ts_rl_bias', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                affinity_path: fb.path, mode: 'local', mapping: _tsRlL.mapping,
                strength: _tsRlL.strength, dry_run: !!dryRun,
            }),
        }).then(r => r.json());
        _tsRlL.biasReport = rep;
        _tsRlLRenderReport(rep, dryRun);
    } catch (e) {
        _tsRlLNote('<b>Commit failed:</b> ' + e.message, 'lp-err');
    } finally {
        if (btn) btn.disabled = false;
    }
}

function _tsRlLRenderReport(rep, dryRun) {
    if (!rep || !rep.ok) {
        _tsRlLNote('<b>bias_generator refused:</b> ' + _tsRlLEsc((rep && rep.err) || 'unknown'), 'lp-err');
        _tsRlLStatus('Not committed.');
        return;
    }
    const m = rep.mapping || {}, g = rep.merge || {}, ev = rep.evidence || {};
    const warn = (m.warnings || []).map(w => `<div style="margin-top:6px">⚠ ${_tsRlLEsc(w)}</div>`).join('');
    const added = g.n_added
        ? `<div style="margin-top:6px">⚠ ${g.n_added} measured reagents were not in the previous
             checkpoint — check the affinity file and the checkpoint came from the same run.</div>` : '';
    _tsRlLNote(
        `<b>${dryRun ? 'Dry run — nothing written.' : 'Committed.'}</b>
         ${ev.n_used}/${ev.n_records} measurements → <b>${ev.n_reagents}</b> reagents.
         pK ${m.pk_min}–${m.pk_max} (σ ${m.pk_std}), mapping <code>${m.mapping}</code>
         at strength ${m.strength} against prior σ ${g.prior_std}.<br>
         <b>${g.n_applied}</b> posteriors moved, <b>${g.n_added}</b> added,
         <b>${g.n_carried_unchanged}</b> carried through untouched.
         ${dryRun ? '' : `<br>Checkpoint: <code>${_tsRlLEsc(rep.checkpoint)}</code>`}
         ${warn}${added}`,
        dryRun ? '' : 'lp-good');
    _tsRlLStatus(dryRun
        ? 'Dry run complete — nothing was written.'
        : 'Biased. The next <b>🎲 Run TS</b> for this reaction will load this checkpoint and skip warm-up.');
}

/* ═══ boot ═══════════════════════════════════════════════════════════════ */
(function _tsRlLBoot(tries) {
    tries = tries || 0;
    if (_tsRlLInstall()) { console.log('[ts-rl-loop] installed'); return; }
    if (tries > 40) { console.warn('[ts-rl-loop] #tsPaneRl never appeared — loop panel not installed'); return; }
    setTimeout(() => _tsRlLBoot(tries + 1), 250);
})();

/* Injected markup uses inline onclick, so these must be reachable off window —
   the same reason ts_run.js re-exports its own handlers. */
window._tsRlLStart      = _tsRlLStart;
window._tsRlLCancel     = _tsRlLCancel;
window._tsRlLCommit     = _tsRlLCommit;
window._tsRlLSet        = _tsRlLSet;
window._tsRlLToggleArm  = _tsRlLToggleArm;
window._tsRlLToggleRow  = _tsRlLToggleRow;
window._tsRlLInstall    = _tsRlLInstall;
window._tsRlL           = _tsRlL;
