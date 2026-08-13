// =============================================================================
// pose_affinity_dock.js — draggable readout for the affinity predictors
// -----------------------------------------------------------------------------
// Grab a scorer card by its title bar, drop it on the pocket view, and it
// becomes a floating readout shaped exactly like #poseProcPanel: label, one big
// number, sparkline, sub-line. ✕ sends it back to the sidebar.
//
// This is the machinery only, shared by the two binding-affinity predictors:
//   pose_deepatom_dock.js  — DeepAtom / ShuffleNetV3
//   pose_gign_dock.js      — Yupu_GIGN
// They are identical in every respect
// that matters here — a card with a pK and a ΔG, a setter, a score action, a
// server-output-dir field — so they share one implementation and differ by a
// config object. Two copies of 350 lines would have drifted the first time
// either was touched.
//
//     PoseAffinityDock({ key, card, float, store, color, border, name, titles,
//                     pkId, outDirId, busyIds, scoreLabel,
//                     setter, score, lastPk, scorerVal })
//
// UNITS. Both scorers predict a pK; ΔG in kcal/mol is what this tool reports
// everywhere else, so the panel defaults to ΔG and the wide button at the
// bottom flips it. pK = −ΔG/1.36, so switching flips the sparkline vertically
// as well as relabelling it — deliberately, since lower is better in one and
// higher in the other and a curve that did not visibly change would be lying.
//
// Scoring lives in the ⚛ in the title bar; the busy state also drives the
// sidebar buttons named in cfg.busyIds.
//
// Everything it hooks (cfg.setter, cfg.score, PG.mc.setScorer) is a METHOD on
// PG.mc, resolved by property lookup at call time, so wrapping works. Contrast
// the top-level function declarations in ts_ui.js — see ts_rl_debug.js.
//
// Unit, position and docked state persist in localStorage per cfg.store.
// =============================================================================

(function () {
    'use strict';
    if (window.PoseAffinityDock) return;

    window.PoseAffinityDock = function (cfg) {
        if (window['__dock_' + cfg.key]) return;
        window['__dock_' + cfg.key] = true;
    var STORE = cfg.store;
    var W = 208;                       // matches #poseProcPanel exactly
    var VIOLET = cfg.color, BORDER = cfg.border;

    var state = read();                // {docked, x, y, unit}  x/y = px from top-left
    // Pairs, not a single series: the server hands back both pred_pk and deltaG,
    // and storing what it actually said beats re-deriving one from the other on
    // every repaint.
    var floatEl = null, hist = [], dragging = null;
    // Which staircase record the NEXT score belongs to, and which one the panel
    // is currently displaying.
    //
    // The trajectory scorer calls mark(i) immediately before the setter, so each
    // history entry can carry the record it came from. Deriving it from position
    // instead — "the i-th score is record i" — holds right up until someone
    // clicks "⚛ Score best pose" mid-run, and then every later record is off by
    // one with nothing on screen to say so.
    var pendingRec = null, selRec = null;
    var UNITS = {
        dg: { title: cfg.titles.dg, label: 'ΔG kcal/mol',
              y: 'ΔG', get: function (h) { return h.dg; }, dp: 2 },
        pk: { title: cfg.titles.pk,           label: 'pK',
              y: 'pK', get: function (h) { return h.pk; }, dp: 2 },
    };
    function unit() { return UNITS[state.unit] ? state.unit : 'dg'; }   // ΔG is the default
    function U()    { return UNITS[unit()]; }
    function other(){ return unit() === 'dg' ? 'pk' : 'dg'; }

    function read() {
        try { return JSON.parse(localStorage.getItem(STORE)) || {}; }
        catch (e) { return {}; }
    }
    function save() {
        try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {}
    }
    function $(id) { return document.getElementById(id); }

    // The 3-D viewport is the positioning context: #poseBox3D's parent is the
    // `flex-1 min-h-0 relative` wrapper every other overlay is absolute inside.
    function stage() {
        var box = $('poseBox3D');
        return (box && box.parentElement) || null;
    }

    // ── The float ────────────────────────────────────────────────────────────
    function build() {
        if (floatEl) return floatEl;
        var host = stage();
        if (!host) return null;

        var el = document.createElement('div');
        el.id = cfg.float;
        el.style.cssText =
            'display:none;position:absolute;top:8px;right:10px;width:' + W + 'px;z-index:7;' +
            'background:rgba(8,12,20,.85);border:1px solid ' + BORDER + ';border-radius:11px;' +
            'padding:10px 12px;pointer-events:auto;box-shadow:0 6px 22px rgba(0,0,0,.45);';
        el.innerHTML =
            '<div id="' + cfg.float + 'Bar" title="' + cfg.name +
                 ' affinity · ΔG lower = better, pK higher = better · drag to move, ✕ returns it to the sidebar" ' +
                 'style="display:flex;align-items:center;gap:6px;cursor:grab;user-select:none;">' +
              '<span style="color:#5b4b8a;font-size:10px;line-height:1;">⠿</span>' +
              '<div id="' + cfg.float + 'Title" style="flex:1;font-size:9.5px;font-weight:700;' +
                   'letter-spacing:.08em;text-transform:uppercase;color:#7c6bae;">' +
                   U().title + '</div>' +
              '<button id="' + cfg.float + 'Run" title="score the current best pose" ' +
                      'style="background:none;border:none;cursor:pointer;color:#c4b5fd;' +
                      'font-size:12px;line-height:1;padding:0 3px;">⚛</button>' +
              '<button id="' + cfg.float + 'Close" title="dock back into the sidebar" ' +
                      'style="background:none;border:none;cursor:pointer;color:#5b4b8a;' +
                      'font-size:13px;line-height:1;padding:0 1px;">✕</button>' +
            '</div>' +
            '<div id="' + cfg.float + 'Val" style="font-family:ui-monospace,monospace;font-size:19px;' +
                 'font-weight:700;color:' + VIOLET + ';line-height:1.1;margin-top:1px;">—</div>' +
            '<svg id="' + cfg.float + 'Chart" viewBox="0 0 184 44" ' +
                 'style="width:100%;height:44px;margin-top:5px;display:block;"></svg>' +
            '<div id="' + cfg.float + 'Sub" style="font-size:9.5px;color:#475569;' +
                 'font-family:ui-monospace,monospace;margin-top:2px;">ΔG — kcal/mol</div>' +
            '<div id="' + cfg.float + 'Unit" role="group" aria-label="units" ' +
                 'title="switch the readout between ΔG in kcal/mol and pK · pK = −ΔG/1.36" ' +
                 'style="display:flex;margin-top:7px;border:1px solid #6d28d9;border-radius:8px;' +
                 'overflow:hidden;font-family:inherit;">' +
              '<button data-unit="dg" style="flex:1;padding:5px 6px;border:none;cursor:pointer;' +
                      'font-size:10px;font-weight:600;font-family:inherit;">ΔG kcal/mol</button>' +
              '<button data-unit="pk" style="flex:1;padding:5px 6px;border:none;cursor:pointer;' +
                      'font-size:10px;font-weight:600;font-family:inherit;border-left:1px solid #6d28d9;">pK</button>' +
            '</div>';
        host.appendChild(el);
        floatEl = el;

        $(cfg.float + 'Close').addEventListener('click', function (e) {
            e.stopPropagation(); undock();
        });
        $(cfg.float + 'Run').addEventListener('click', function (e) {
            e.stopPropagation();
            if (window.PoseGen && PoseGen.mc && PoseGen.mc[cfg.score]) PoseGen.mc[cfg.score]();
        });
        Array.prototype.forEach.call(el.querySelectorAll('#' + cfg.float + 'Unit button'), function (b) {
            b.addEventListener('click', function (e) {
                e.stopPropagation();
                setUnit(b.getAttribute('data-unit'));
            });
        });
        paintUnit();
        grip($(cfg.float + 'Bar'), el, /* moveFloat */ true);
        return el;
    }

    // ── Sparkline ────────────────────────────────────────────────────────────
    // Rendered through PG._spark, the same function #poseProcChart and
    // #poseImpScoreChart use, rather than a private copy of its geometry. Two
    // reasons: the panels stay visually identical by construction, and
    // pose_spark_hover.js wraps _spark to add the crosshair readout — a
    // hand-rolled chart here would silently miss out on it.
    // Index into the plotted series for a record, or -1. The chart drops
    // non-finite entries, so a record's position on it is not its position in
    // `hist` and the ring would drift if we assumed otherwise.
    function plotIndexOf(rec) {
        var u = U(), j = -1;
        for (var i = 0; i < hist.length; i++) {
            var v = u.get(hist[i]);
            if (v == null || !isFinite(v)) continue;
            j++;
            if (hist[i].rec === rec) return j;
        }
        return -1;
    }

    // Ring the plotted point the stepper is sitting on. Geometry mirrors
    // PG._spark (W 184, H 44, pad 4) because it is the same chart; _spark
    // rewrites the SVG with innerHTML, so this is appended after every draw
    // rather than kept in sync with one.
    function markChart(vals) {
        var svg = $(cfg.float + 'Chart');
        if (!svg || selRec == null || vals.length < 2) return;
        var j = plotIndexOf(selRec);
        if (j < 0) return;
        var W = 184, H = 44, pad = 4;
        var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
        var rng = (mx - mn) || 1;
        var x = pad + (W - 2 * pad) * (vals.length === 1 ? 1 : j / (vals.length - 1));
        var y = H - pad - (H - 2 * pad) * ((vals[j] - mn) / rng);
        var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('id', cfg.float + 'Mark');
        g.innerHTML =
            '<line x1="' + x.toFixed(1) + '" y1="0" x2="' + x.toFixed(1) + '" y2="' + H + '" ' +
                'stroke="' + VIOLET + '" stroke-width="0.7" stroke-dasharray="2 2" opacity="0.5"/>' +
            '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.1" fill="none" ' +
                'stroke="' + VIOLET + '" stroke-width="1.4"/>';
        svg.appendChild(g);
    }

    function chart() {
        var svg = $(cfg.float + 'Chart');
        if (!svg) return;
        var u = U();
        var vals = hist.map(u.get).filter(function (v) { return v != null && isFinite(v); });
        // The hover layer reads these off the element rather than a registry, so
        // the tooltip's label follows the toggle with no extra plumbing.
        svg.dataset.ylabel = u.y;
        svg.dataset.xlabel = 'run';
        if (vals.length < 2) {
            svg.innerHTML = '<text x="92" y="26" text-anchor="middle" fill="#3f3a55" ' +
                            'font-family="ui-monospace,monospace" font-size="9">' +
                            (vals.length ? 'one score so far' : 'no scores yet') + '</text>';
            if (window._poseSparkSeries) window._poseSparkSeries(cfg.float + 'Chart', vals, VIOLET);
            return;
        }
        var PG = window.PoseGen;
        if (PG && typeof PG._spark === 'function') PG._spark(cfg.float + 'Chart', vals, VIOLET);
        markChart(vals);
    }

    // ── Unit toggle ──────────────────────────────────────────────────────────
    function paintUnit() {
        var box = $(cfg.float + 'Unit');
        if (!box) return;
        var cur = unit();
        Array.prototype.forEach.call(box.querySelectorAll('button'), function (b) {
            var on = b.getAttribute('data-unit') === cur;
            b.style.background = on ? 'rgba(124,58,237,.30)' : 'transparent';
            b.style.color      = on ? '#e9d5ff' : '#7c6bae';
            b.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        var t = $(cfg.float + 'Title');
        if (t) t.textContent = U().title;
    }

    function setUnit(u) {
        if (!UNITS[u]) return;
        state.unit = u; save();
        paintUnit();
        var last = hist.length ? hist[hist.length - 1] : null;
        paint(last ? last.pk : null, last ? last.dg : null, null, /* norecord */ true);
    }

    // ── Mirror whatever the card shows ───────────────────────────────────────
    // `norecord` repaints from existing history (a unit switch) instead of
    // appending — otherwise flipping the toggle would invent a data point.
    function paint(pk, dg, status, norecord) {
        var v = $(cfg.float + 'Val'), s = $(cfg.float + 'Sub');
        if (!v) return;
        if (pk == null && dg == null) {
            v.textContent = '—';
            v.style.color = '#5b4b8a';
            if (s) s.textContent = status ? String(status).slice(0, 46) : U().label;
            return;
        }
        // The server returns both; derive only what is genuinely missing.
        if (dg == null && pk != null) dg = -pk * 1.36;
        if (pk == null && dg != null) pk = -dg / 1.36;
        if (!norecord) {
            hist.push({ pk: pk, dg: dg, rec: pendingRec });
            pendingRec = null;                       // one mark, one score
            if (hist.length > 40) hist.shift();
            selRec = null;                           // a fresh score is now what is shown
        }

        var u = U(), shown = u.get({ pk: pk, dg: dg });
        v.textContent = (shown != null) ? shown.toFixed(u.dp) : '—';
        v.style.color = VIOLET;
        if (s) {
            var alt = UNITS[other()];
            var av = alt.get({ pk: pk, dg: dg });
            s.textContent = alt.y + ' ' + (av != null ? av.toFixed(alt.dp) : '—') +
                            (other() === 'dg' ? ' kcal/mol' : '') +
                            (selRec != null ? ' · pose ' + (selRec + 1) : ' · n=' + hist.length);
        }
        chart();
    }

    // ── Follow the staircase stepper ─────────────────────────────────────────
    // Show the score THIS scorer produced for record `rec`, without appending
    // anything. Both floats do it at once, so ◀ / ▶ moves the ligand and both
    // predictions together and you are always comparing two numbers about the
    // same geometry — which is the only comparison that means anything.
    function select(rec) {
        var v = $(cfg.float + 'Val'), s = $(cfg.float + 'Sub');
        if (!v) return false;
        selRec = rec;
        var h = null;
        for (var i = 0; i < hist.length; i++) if (hist[i].rec === rec) h = hist[i];
        if (!h) {
            // Not scored, or scored and failed. Say which pose is missing rather
            // than leaving the previous pose's number sitting under a new pose —
            // a stale-but-plausible reading is the failure mode worth avoiding.
            v.textContent = '—';
            v.style.color = '#5b4b8a';
            if (s) s.textContent = 'pose ' + (rec + 1) + ' · not scored';
            chart();
            return false;
        }
        paint(h.pk, h.dg, null, true);
        return true;
    }

    function busy(on) {
        cfg.busyIds.forEach(function (id) {
            var b = $(id);
            if (b) b.textContent = on ? '⏳ scoring…' : cfg.scoreLabel;
        });
        var f = $(cfg.float + 'Run');          // icon-sized: a sentence would not fit
        if (f) { f.textContent = on ? '⏳' : '⚛'; f.style.opacity = on ? '.6' : '1'; }
    }

    // ── Docking ──────────────────────────────────────────────────────────────
    function dock(x, y) {
        var el = build(), host = stage();
        if (!el || !host) return;
        var card = $(cfg.card);
        if (card) card.style.display = 'none';

        el.style.display = 'block';
        place(x, y);
        state.docked = true; save();
        placeAuto();                         // avoid sitting on top of #poseProcPanel
        paintUnit();
        var last = hist.length ? hist[hist.length - 1] : null;
        paint(last ? last.pk : (window.PoseGen && PoseGen.mc ? PoseGen.mc[cfg.lastPk] : null),
              last ? last.dg : null, null, true);
    }

    function undock() {
        if (floatEl) floatEl.style.display = 'none';
        state.docked = false; save();
        // Only restore the card if this scorer is still selected — otherwise
        // setScorer had it hidden for its own reasons and we must not override.
        var card = $(cfg.card);
        var gign = window.PoseGen && PoseGen.mc && PoseGen.mc.scorer === cfg.scorerVal;
        if (card && gign) card.style.display = 'flex';
    }

    /* Absolute placement, clamped so the panel can never be dragged out of
       reach. Stored as top/left in host-relative px. */
    function place(x, y) {
        var el = floatEl, host = stage();
        if (!el || !host) return;
        var r = host.getBoundingClientRect();
        var h = el.offsetHeight || 150;
        if (x == null || y == null) { x = state.x; y = state.y; }
        if (x == null || y == null) { el.style.left = ''; el.style.right = '10px'; el.style.top = '8px'; return; }
        x = Math.max(4, Math.min(x, r.width - W - 4));
        y = Math.max(4, Math.min(y, r.height - h - 4));
        el.style.right = 'auto';
        el.style.left = x + 'px';
        el.style.top = y + 'px';
        state.x = x; state.y = y; save();
    }

    /* Default corner only: #poseProcPanel and #poseImpScore both own top:8/right:10,
       so stack beneath whichever is showing — the same courtesy PG._impScoreShow
       already extends to them. Skipped once the user has placed it by hand. */
    function placeAuto() {
        if (!floatEl || state.x != null) return;
        var stacked = 8;
        ['poseProcPanel', 'poseImpScore'].forEach(function (id) {
            var p = $(id);
            if (p && p.style.display && p.style.display !== 'none') {
                stacked += (p.offsetHeight || 110) + 6;
            }
        });
        floatEl.style.top = stacked + 'px';
        floatEl.style.right = '10px';
        floatEl.style.left = 'auto';
    }

    // ── Dragging ─────────────────────────────────────────────────────────────
    // Pointer events, not HTML5 drag-and-drop: the drop target is a Plotly WebGL
    // canvas, and setPointerCapture is what keeps the orbit controls from
    // stealing the gesture halfway across the view.
    function grip(handle, panel, moveFloat) {
        if (!handle) return;
        handle.style.touchAction = 'none';
        handle.addEventListener('pointerdown', function (e) {
            if (e.button !== 0) return;
            if (e.target.closest('button,input,select,textarea,a')) return;
            e.preventDefault();
            var host = stage();
            if (!host) return;
            var pr = panel.getBoundingClientRect();
            dragging = {
                moveFloat: moveFloat,
                dx: e.clientX - pr.left,
                dy: e.clientY - pr.top,
                host: host,
                ghost: moveFloat ? null : ghostFor(panel, e),
            };
            handle.setPointerCapture(e.pointerId);
            handle.style.cursor = 'grabbing';
            if (!moveFloat) host.style.outline = '2px dashed ' + BORDER;
        });
        handle.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            var r = dragging.host.getBoundingClientRect();
            if (dragging.moveFloat) {
                place(e.clientX - r.left - dragging.dx, e.clientY - r.top - dragging.dy);
            } else if (dragging.ghost) {
                dragging.ghost.style.left = (e.clientX - dragging.dx) + 'px';
                dragging.ghost.style.top  = (e.clientY - dragging.dy) + 'px';
                var over = inside(e, r);
                dragging.host.style.outline = '2px dashed ' + (over ? VIOLET : BORDER);
                dragging.ghost.style.opacity = over ? '.95' : '.5';
            }
        });
        function end(e) {
            if (!dragging) return;
            var d = dragging; dragging = null;
            handle.style.cursor = 'grab';
            try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
            d.host.style.outline = '';
            if (d.ghost && d.ghost.parentNode) d.ghost.parentNode.removeChild(d.ghost);
            if (d.moveFloat) return;
            var r = d.host.getBoundingClientRect();
            if (inside(e, r)) {
                state.x = null; state.y = null;          // a fresh drop re-auto-places
                dock(e.clientX - r.left - d.dx, e.clientY - r.top - d.dy);
            }
        }
        handle.addEventListener('pointerup', end);
        handle.addEventListener('pointercancel', end);
    }

    function inside(e, r) {
        return e.clientX >= r.left && e.clientX <= r.right &&
               e.clientY >= r.top  && e.clientY <= r.bottom;
    }

    function ghostFor(panel, e) {
        var g = document.createElement('div');
        var pr = panel.getBoundingClientRect();
        g.style.cssText =
            'position:fixed;z-index:10500;pointer-events:none;width:' + W + 'px;' +
            'left:' + (e.clientX - (e.clientX - pr.left)) + 'px;top:' + pr.top + 'px;' +
            'background:rgba(8,12,20,.9);border:1px solid ' + VIOLET + ';border-radius:11px;' +
            'padding:10px 12px;opacity:.5;box-shadow:0 10px 30px rgba(0,0,0,.6);' +
            'font-family:ui-monospace,monospace;';
        g.innerHTML =
            '<div style="font-size:9.5px;font-weight:700;letter-spacing:.08em;' +
                 'text-transform:uppercase;color:#7c6bae;">' + cfg.name + ' ' + U().y + '</div>' +
            '<div style="font-size:19px;font-weight:700;color:' + VIOLET + ';line-height:1.1;">' +
                 (($(cfg.pkId) || {}).textContent || '—') + '</div>' +
            '<div style="font-size:9.5px;color:#475569;margin-top:3px;">drop on the pocket view</div>';
        document.body.appendChild(g);
        return g;
    }

    // ── Make the sidebar card's title bar a handle ───────────────────────────
    function arm(card) {
        var title = card.querySelector('p');
        // Per-scorer flag, not a shared one: two docks run over the same DOM,
        // and a card cloned or re-rendered with the attribute already set would
        // silently never get its listener.
        var flag = 'dockGrip' + cfg.key.charAt(0).toUpperCase() + cfg.key.slice(1);
        if (!title || title.dataset[flag]) return;
        title.dataset[flag] = '1';
        title.style.cursor = 'grab';
        title.title = 'drag onto the 3-D view to float this readout over the pocket';
        title.innerHTML = '<span style="color:#5b4b8a;margin-right:5px;">⠿</span>' + title.innerHTML +
                          '<span style="float:right;font-weight:400;letter-spacing:0;' +
                          'text-transform:none;color:#5b4b8a;font-size:9px;">drag me →</span>';
        grip(title, card, false);
    }

    // ── Hooks ────────────────────────────────────────────────────────────────
    function install() {
        var PG = window.PoseGen;
        if (!PG || !PG.mc || typeof PG.mc[cfg.setter] !== 'function') return false;
        var card = $(cfg.card);
        if (!card || !stage()) return false;

        arm(card);
        build();

        var origSet = PG.mc[cfg.setter];
        PG.mc[cfg.setter] = function (status, pk, dg) {
            origSet.apply(PG.mc, arguments);
            paint(pk, dg, status);
            if (status) busy(/scoring/i.test(String(status)));
        };

        var origScorer = PG.mc.setScorer;
        if (typeof origScorer === 'function') {
            PG.mc.setScorer = function () {
                origScorer.apply(PG.mc, arguments);
                // setScorer unconditionally shows the card for this scorer; while
                // docked the float IS the card, so put it back down.
                if (state.docked) { var c = $(cfg.card); if (c) c.style.display = 'none'; }
            };
        }

        var origScore = PG.mc[cfg.score];
        if (typeof origScore === 'function') {
            PG.mc[cfg.score] = function () { busy(true); return origScore.apply(PG.mc, arguments); };
        }

        if (state.docked) dock(state.x, state.y);
        window.addEventListener('resize', function () { if (state.docked) place(state.x, state.y); });
        return true;
    }

    // pose.js is loaded before this file, but the card only exists once the
    // pose modal's markup is in the DOM. Retry briefly rather than assume.

        var tries = 0;
        var t = setInterval(function () { if (install() || ++tries > 40) clearInterval(t); }, 250);

        return {
            dock: dock, undock: undock, unit: function (u) { if (u) setUnit(u); return unit(); },
            // mark(i) tags the NEXT score with the staircase record it came
            // from; select(i) shows the score already recorded for record i.
            mark: function (i) { pendingRec = (typeof i === 'number') ? i : null; },
            select: select,
            // A new search restarts the pose numbering at 1, so history from the
            // previous run must go with it. Keeping it would make select(0) find
            // the OLD run's pose 1 and display a number that belongs to a
            // geometry no longer on screen — right shape, right magnitude,
            // wrong pose, and nothing to give it away. Clearing also makes this
            // sparkline mean "this run's poses", which is what lets it be read
            // against #poseProcChart directly above it.
            reset: function () {
                hist = []; pendingRec = null; selRec = null;
                paint(null, null, 'no scores yet', true);
            },
            scoreAt: function (i) {
                for (var k = 0; k < hist.length; k++) if (hist[k].rec === i) return hist[k];
                return null;
            },
            state: function () {
                return { docked: !!state.docked, unit: unit(), n: hist.length,
                         selected: selRec,
                         records: hist.filter(function (h) { return h.rec != null; }).length };
            },
        };
    };
})();