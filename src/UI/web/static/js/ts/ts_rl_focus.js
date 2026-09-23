// =============================================================================
// ts/ts_rl_focus.js — the verdict strip becomes a filter.
//
// The 🧪 RL tab ships six verdict tiles (Search · Synthesizability · DMTA cycle
// cost · Critic · Calibration · Net risk) and nine diagnostic cards. Until now
// the tiles were labels and the cards were a wall. This file makes each tile a
// toggle: click one and only the cards behind that verdict are shown, lifted
// directly ABOVE #tsRlLoop so the operations bar stays where your hands are.
//
// THE DEFAULT IS EMPTY, AND COSTS NO LAYOUT. With no tile selected the tab
// shows no diagnostics at all — not all nine — and #tsRlFocusHost is `hidden`,
// so it contributes zero height. #tsRlLoop therefore sits exactly where it did
// before this file existed: immediately under the verdict strip. Rendering
// anything in the host at rest, even one line of helper text, would push the
// operations bar down the page, which is the one thing it must not do.
//
// `.grid` is collapsed once at install and never used as a display surface
// again; cards are only ever seen inside the host.
//
// #tsRlLoop is never moved, hidden, reordered or re-rendered by this file.
// It is the control surface; everything here is reading material around it.
//
// ── Why the cards are MOVED, not cloned ─────────────────────────────────────
// Every diagnostic is an inline <svg>, and four of them reference shared
// markers by id: `marker-end="url(#ar)"`, `url(#arb)`, `url(#arg)`. Duplicating
// those nodes duplicates the ids, and `url(#ar)` then resolves to whichever
// copy the document happens to reach first — so a cloned card can silently
// steal the arrowheads from the original, or lose them entirely. appendChild
// relocates rather than copies, so there is exactly one of each id at all
// times. This is the same reasoning ts_rl.js uses to adopt the pose workspace
// rather than duplicate it (see _tsRlAdoptPose).
//
// ── Why restore replays a recorded order ────────────────────────────────────
// Restoring by a stored `nextSibling` breaks when several siblings have also
// moved — the reference node may itself be sitting in the focus host. So the
// original child order of .grid is captured ONCE at install and simply
// re-appended in that order. Deterministic, and idempotent if it runs twice.
//
// The category tags live in the MARKUP (data-rlcat on each card, data-rlv on
// each tile), not in a table here. Two reasons: a card and its category cannot
// drift apart when they are the same line of HTML, and adding a tenth card
// needs no edit to this file. _tsRlFocusAudit() reports any tile with no cards
// or any card with no tile, which is the one failure this split can still have.
//
// Load order: AFTER ts_rl.js (needs #tsPaneRl and its markup to exist).
// =============================================================================
console.log('%c[ts-rl-focus] LOADED', 'background:#3f1d5c;color:#e9d5ff;padding:2px 6px;border-radius:3px');

const _TS_RL_F_LABEL = {
    search: 'Search', synth: 'Synthesizability', dmta: 'DMTA cycle cost',
    critic: 'Critic', calibration: 'Calibration', netrisk: 'Net risk',
};

const _tsRlF = {
    on: new Set(),        // active category keys; empty === show NOTHING
    home: [],             // .grid's original child order, captured once
    installed: false,
};

const _TS_RL_F_STYLE = `
  /* the tiles are controls now — say so */
  #tsPaneRl .verdict .v[data-rlv]{cursor:pointer;user-select:none;
       transition:border-color .14s, background .14s, transform .08s}
  #tsPaneRl .verdict .v[data-rlv]:hover{border-color:var(--s1-ink);background:#0d1526}
  #tsPaneRl .verdict .v[data-rlv]:active{transform:translateY(1px)}
  #tsPaneRl .verdict .v[data-rlv]:focus-visible{outline:2px solid var(--s1-ink);outline-offset:2px}
  #tsPaneRl .verdict .v[data-rlv][aria-pressed="true"]{
       border-color:var(--s1-ink);background:#0b1f28;
       box-shadow:inset 0 0 0 1px rgba(34,211,238,.25)}
  #tsPaneRl .verdict .v[data-rlv] .rlf-n{float:right;font-family:var(--mono);font-size:9px;
       color:var(--ink-4);font-weight:700}
  #tsPaneRl .verdict .v[data-rlv][aria-pressed="true"] .rlf-n{color:var(--s1-ink)}

  #tsRlFocusHost{margin:14px 24px 0}
  #tsRlFocusHost[hidden]{display:none}
  #tsRlFocusHost .rlf-bar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
       background:var(--surface-1);border:1px solid var(--line);border-radius:13px;
       padding:9px 13px;margin-bottom:12px}
  #tsRlFocusHost .rlf-t{font-size:11.5px;color:var(--ink-2)}
  #tsRlFocusHost .rlf-t b{color:var(--ink)}
  #tsRlFocusHost .rlf-chip{font-family:var(--mono);font-size:9.5px;font-weight:700;padding:2px 8px;
       border-radius:6px;background:rgba(8,145,178,.14);border:1px solid rgba(8,145,178,.4);
       color:var(--s1-ink);cursor:pointer}
  #tsRlFocusHost .rlf-chip:hover{background:rgba(8,145,178,.24)}
  #tsRlFocusHost .rlf-clear{margin-left:auto;padding:5px 12px;border-radius:9px;
       border:1px solid var(--line);background:#0b1120;color:var(--ink-2);
       font:inherit;font-size:11.5px;font-weight:600;cursor:pointer}
  #tsRlFocusHost .rlf-clear:hover{border-color:var(--s1-ink);color:var(--s1-ink)}
  #tsRlFocusHost .rlf-none{background:var(--surface-1);border:1px dashed var(--line);
       border-radius:13px;padding:16px;text-align:center;font-size:11.5px;color:var(--ink-3)}

  /* NOTE: no backticks anywhere in this block. It lives inside a template
     literal, so one backtick in a CSS comment ends the string and the rest of
     the stylesheet is parsed as JavaScript. node --check still PASSES on the
     result -- it is valid syntax, just not a string -- and the only symptom is
     a runtime "X is not a function" the moment the file loads. See G82.

     What the two rules do: .grid is never a display surface once this file is
     installed, because cards are only ever seen lifted into #tsRlFocusHost.
     The class is added ONCE at install and never removed, so the default state
     of the tab is empty. Class-gated rather than unconditional on purpose: if
     this script fails to load, the cards stay visible where they always were
     instead of becoming permanently invisible with no way to reach them. */
  #tsPaneRl.rl-cardhost .body{grid-template-columns:1fr}
  #tsPaneRl.rl-cardhost .body > .grid{display:none}
`;

function _tsRlFocusInstall() {
    if (_tsRlF.installed && document.getElementById('tsRlFocusHost')) return true;
    const pane = document.getElementById('tsPaneRl');
    const loop = document.getElementById('tsRlLoop');
    const grid = pane && pane.querySelector('.body > .grid');
    if (!pane || !loop || !grid) return false;

    if (!document.getElementById('tsRlFocusStyle')) {
        const st = document.createElement('style');
        st.id = 'tsRlFocusStyle';
        st.textContent = _TS_RL_F_STYLE;
        document.head.appendChild(st);
    }

    // Capture the grid's order ONCE, before anything moves.
    _tsRlF.home = Array.from(grid.children);

    // Collapse the reference grid immediately and permanently. No flash of all
    // nine cards on first open, because #tsPaneRl is still `.hidden` at this
    // point — the tab has not been activated yet.
    pane.classList.add('rl-cardhost');

    const host = document.createElement('div');
    host.id = 'tsRlFocusHost';
    host.hidden = true;
    host.innerHTML = '<div class="rlf-bar"><span class="rlf-t" id="tsRlFocusSummary"></span>'
                   + '<span id="tsRlFocusChips" style="display:flex;gap:6px;flex-wrap:wrap"></span>'
                   + '<button class="rlf-clear" onclick="_tsRlFocusClear()">✕ Clear focus</button></div>'
                   + '<div class="grid" id="tsRlFocusGrid"></div>';
    // Directly above the operations card — the whole point of the feature.
    loop.parentNode.insertBefore(host, loop);

    // Delegated, so it survives the tiles being re-rendered by anything else.
    pane.addEventListener('click', _tsRlFocusClick);
    pane.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
        const tile = e.target.closest && e.target.closest('[data-rlv]');
        if (!tile || !pane.contains(tile)) return;
        e.preventDefault();
        _tsRlFocus(tile.getAttribute('data-rlv'));
    });

    _tsRlFocusTileCounts();
    _tsRlFocusRender();
    _tsRlF.installed = true;
    const audit = _tsRlFocusAudit();
    if (audit.emptyTiles.length || audit.orphanCards.length) console.warn('[ts-rl-focus]', audit);
    return true;
}

function _tsRlFocusClick(e) {
    const tile = e.target.closest && e.target.closest('[data-rlv]');
    if (!tile) return;
    _tsRlFocus(tile.getAttribute('data-rlv'));
}

function _tsRlFocusCards(cat) {
    // Query the union of BOTH containers: once a card has been moved into the
    // focus host it is no longer a child of .grid, so a query scoped to .grid
    // would stop finding it and the second click on a tile would do nothing.
    const pane = document.getElementById('tsPaneRl');
    if (!pane) return [];
    const all = Array.from(pane.querySelectorAll('[data-rlcat]'));
    if (!cat) return all;
    return all.filter(el => (el.getAttribute('data-rlcat') || '').split(/\s+/).indexOf(cat) >= 0);
}

/* Put the card count on each tile. A tile that filters to nothing is a wiring
   error, and the number is where you see it without opening the console. */
function _tsRlFocusTileCounts() {
    document.querySelectorAll('#tsPaneRl .verdict [data-rlv]').forEach(tile => {
        const cat = tile.getAttribute('data-rlv');
        const n = _tsRlFocusCards(cat).length;
        const t = tile.querySelector('.t');
        if (t && !t.querySelector('.rlf-n')) {
            const s = document.createElement('span');
            s.className = 'rlf-n';
            s.textContent = n;
            s.title = n + ' diagnostic card' + (n === 1 ? '' : 's');
            t.appendChild(s);
        }
    });
}

function _tsRlFocus(cat) {
    if (!cat) return;
    if (_tsRlF.on.has(cat)) _tsRlF.on.delete(cat);
    else _tsRlF.on.add(cat);
    _tsRlFocusRender();
}

function _tsRlFocusClear() {
    _tsRlF.on.clear();
    _tsRlFocusRender();
}

function _tsRlFocusRender() {
    const pane = document.getElementById('tsPaneRl');
    const host = document.getElementById('tsRlFocusHost');
    const fgrid = document.getElementById('tsRlFocusGrid');
    const grid = pane && pane.querySelector('.body > .grid');
    if (!pane || !host || !fgrid || !grid) return;

    document.querySelectorAll('#tsPaneRl .verdict [data-rlv]').forEach(t => {
        t.setAttribute('aria-pressed', _tsRlF.on.has(t.getAttribute('data-rlv')) ? 'true' : 'false');
    });

    // ── nothing selected: show NOTHING, and take up no space ─────────────
    // The host is HIDDEN, not merely emptied. It sits above #tsRlLoop, so a
    // host that renders anything at all — even one line of helper text — pushes
    // the operations bar down the page. In the default state the bar must sit
    // exactly where it did before this file existed: immediately under the
    // verdict strip. `hidden` + the [hidden]{display:none} rule means the host
    // contributes zero height, so it does.
    //
    // (An earlier build rendered a "pick a verdict above" hint here. It read
    //  well and displaced the controls, which is the wrong trade: the tiles
    //  already carry a pointer cursor, a hover border and a card count, so the
    //  affordance is on them, where it costs no layout.)
    if (!_tsRlF.on.size) {
        _tsRlF.home.forEach(el => grid.appendChild(el));
        fgrid.innerHTML = '';
        host.hidden = true;
        return;
    }

    const want = new Set();
    _tsRlF.on.forEach(cat => _tsRlFocusCards(cat).forEach(el => want.add(el)));

    // Send everything home first, then lift the wanted set in its ORIGINAL
    // order. Doing it in one direction avoids reasoning about which cards were
    // already moved, and keeps the reading order stable no matter which tiles
    // were clicked or in what sequence.
    _tsRlF.home.forEach(el => grid.appendChild(el));
    _tsRlF.home.forEach(el => { if (want.has(el)) fgrid.appendChild(el); });

    host.hidden = false;

    const active = Array.from(_tsRlF.on);          // one snapshot; key and label
    const sum = document.getElementById('tsRlFocusSummary');   // must not be zipped
    if (sum) {                                                  // from two arrays
        sum.innerHTML = want.size
            ? `Showing <b>${want.size}</b> of ${_tsRlF.home.filter(e => e.hasAttribute('data-rlcat')).length} diagnostics for`
            : 'No diagnostics tagged for';
    }
    const chips = document.getElementById('tsRlFocusChips');
    if (chips) {
        chips.innerHTML = active.map(k =>
            `<span class="rlf-chip" onclick="_tsRlFocus('${k}')"
                   title="remove this filter">${_TS_RL_F_LABEL[k] || k} ✕</span>`).join('');
    }
    if (!want.size) {
        fgrid.innerHTML = '';
        const d = document.createElement('div');
        d.className = 'rlf-none';
        d.textContent = 'No card carries data-rlcat="' + Array.from(_tsRlF.on).join('" or "')
                      + '". Tag one in ts_rl.js, or clear the filter.';
        fgrid.appendChild(d);
    } else {
        // Drop a previous empty-state placeholder without touching the cards.
        Array.from(fgrid.querySelectorAll('.rlf-none')).forEach(n => n.remove());
    }
}

/* Cross-reference the two halves of the contract. Every tile should filter to
   at least one card, and every card should be reachable from at least one
   tile — the same shape of test the playbook keeps asking for between the KB
   and the frontend action map (G58). Call it from the console. */
function _tsRlFocusAudit() {
    const tiles = Array.from(document.querySelectorAll('#tsPaneRl .verdict [data-rlv]'))
        .map(t => t.getAttribute('data-rlv'));
    const cards = _tsRlFocusCards(null);
    const covered = new Set();
    const emptyTiles = tiles.filter(cat => {
        const hit = _tsRlFocusCards(cat);
        hit.forEach(c => covered.add(c));
        return hit.length === 0;
    });
    const orphanCards = cards.filter(c => !covered.has(c))
        .map(c => c.getAttribute('data-rlid') || '(untagged)');
    const unknownCats = [];
    cards.forEach(c => (c.getAttribute('data-rlcat') || '').split(/\s+/).forEach(k => {
        if (k && tiles.indexOf(k) < 0 && unknownCats.indexOf(k) < 0) unknownCats.push(k);
    }));
    return { tiles, nCards: cards.length, emptyTiles, orphanCards, unknownCats };
}

(function _tsRlFocusBoot(tries) {
    tries = tries || 0;
    if (_tsRlFocusInstall()) { console.log('[ts-rl-focus] installed'); return; }
    if (tries > 40) { console.warn('[ts-rl-focus] #tsPaneRl / #tsRlLoop never appeared — filter not installed'); return; }
    setTimeout(() => _tsRlFocusBoot(tries + 1), 250);
})();

window._tsRlFocus        = _tsRlFocus;
window._tsRlFocusClear   = _tsRlFocusClear;
window._tsRlFocusAudit   = _tsRlFocusAudit;
window._tsRlFocusInstall = _tsRlFocusInstall;
