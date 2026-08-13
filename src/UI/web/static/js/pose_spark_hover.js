// =============================================================================
// pose_spark_hover.js — hover any pose sparkline, read the score at that point
// -----------------------------------------------------------------------------
// The panels showed only the LATEST number; the line behind it was decoration.
// This gives every pose sparkline the crosshair-and-tooltip readout the TS batch
// chart already has (ts/ts_chart.js), so the history is legible instead of
// merely drawn:
//
//        ┊
//     ───┼──●───────       ┌──────────────┐
//        ┊                 │ step   228   │
//        ┊                 │ score 9.3616 │
//                          └──────────────┘
//
// Covers all three at once — #poseProcChart (minimise / Monte-Carlo / Vina),
// #poseImpScoreChart (imported batch) and #poseGignFloatChart (GIGN) — because
// it hooks the ONE renderer they share, PG._spark(id, vals, col), rather than
// each call site. Any sparkline added later through _spark inherits it.
//
// PG._spark is a METHOD on the PoseGen object, resolved by property lookup at
// call time, so wrapping it works. (Top-level function declarations elsewhere in
// this app cannot be wrapped that way — see the note atop ts_rl_debug.js.)
//
// Self-contained: one <script> tag in hub.html, no edit to pose.js. Delete the
// file and the tag and the charts go back to being static.
// =============================================================================

(function () {
    'use strict';
    if (window.__poseSparkHoverLoaded) return;
    window.__poseSparkHoverLoaded = true;

    // Must mirror PG._spark's geometry exactly — it is the inverse of that
    // mapping that turns a cursor position back into an array index.
    var W = 184, H = 44, PAD = 4;

    var SERIES = {};        // svg id -> { vals, col }

    // What the x axis counts, per chart. #poseProcChart is reused by three
    // stages that count different things, so it resolves at hover time.
    var LABELS = {
        poseImpScoreChart:  { x: 'file',  y: 'score' },
        poseGignFloatChart: { x: 'run',   y: 'pK' },
    };
    function labels(id, svg) {
        // An element may declare its own axis names. #poseGignFloatChart does,
        // because its y axis flips between ΔG and pK with the panel's unit
        // toggle — a label baked into the map here would go stale the moment
        // someone switched, and a tooltip confidently reading "pK 7.4500" over a
        // ΔG curve is worse than no tooltip.
        if (svg && svg.dataset && (svg.dataset.ylabel || svg.dataset.xlabel)) {
            return { x: svg.dataset.xlabel || 'n', y: svg.dataset.ylabel || 'score' };
        }
        if (LABELS[id]) return LABELS[id];
        if (id === 'poseProcChart') {
            var st = (window.PoseGen && PoseGen._stage) || '';
            if (st === 'min') return { x: 'iter', y: 'energy' };
            if (st === 'mc')  return { x: 'step', y: 'score' };
            return { x: 'point', y: 'score' };
        }
        return { x: 'n', y: 'score' };
    }

    function fmt(v) {
        if (v == null || !isFinite(v)) return '—';
        var a = Math.abs(v);
        return (a >= 1000) ? v.toFixed(1) : (a >= 100 ? v.toFixed(2) : v.toFixed(4));
    }

    // ── Tooltip, one per chart, parented to <body> ──────────────────────────
    // position:fixed on the body rather than absolute inside the panel: the
    // panels are 208 px wide and ~150 px tall, so a tooltip confined to one
    // collides with the sub-line and runs off the bottom edge. Fixed positioning
    // lets it sit outside the panel entirely and be clamped to the WINDOW, which
    // is the only box big enough to hold it comfortably.
    function tipFor(svg) {
        var id = svg.id + '__tip';
        var t = document.getElementById(id);
        if (t) return t;
        t = document.createElement('div');
        t.id = id;
        t.style.cssText =
            'display:none;position:fixed;pointer-events:none;z-index:10600;' +
            'background:rgba(5,7,15,.94);border:.5px solid rgba(34,211,238,.35);' +
            'border-radius:4px;padding:4px 7px;white-space:nowrap;' +
            'font-size:9px;font-family:ui-monospace,monospace;line-height:1.6;' +
            'box-shadow:0 2px 10px rgba(0,0,0,.6);';
        document.body.appendChild(t);
        return t;
    }

    function clear(svg) {
        var g = svg.querySelector('.pose-spark-hover');
        if (g && g.parentNode) g.parentNode.removeChild(g);
        var t = document.getElementById(svg.id + '__tip');
        if (t) t.style.display = 'none';
    }

    // ── Crosshair drawn into the svg itself, in viewBox units ────────────────
    function crosshair(svg, x, y, col) {
        clearOnly(svg);
        var NS = 'http://www.w3.org/2000/svg';
        var g = document.createElementNS(NS, 'g');
        g.setAttribute('class', 'pose-spark-hover');
        function line(x1, y1, x2, y2) {
            var l = document.createElementNS(NS, 'line');
            l.setAttribute('x1', x1); l.setAttribute('y1', y1);
            l.setAttribute('x2', x2); l.setAttribute('y2', y2);
            l.setAttribute('stroke', '#64748b');
            l.setAttribute('stroke-width', '0.6');
            l.setAttribute('stroke-dasharray', '2 2.5');
            l.setAttribute('opacity', '.75');
            g.appendChild(l);
        }
        line(x, 0, x, H);                       // vertical, full height
        line(0, y, W, y);                       // horizontal, through the point
        var c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', '2.6');
        c.setAttribute('fill', col || '#e2e8f0');
        c.setAttribute('stroke', '#05070f');
        c.setAttribute('stroke-width', '1');
        g.appendChild(c);
        svg.appendChild(g);
    }
    function clearOnly(svg) {
        var g = svg.querySelector('.pose-spark-hover');
        if (g && g.parentNode) g.parentNode.removeChild(g);
    }

    // ── Wiring, once per chart element ───────────────────────────────────────
    function attach(svg) {
        if (!svg || svg.dataset.sparkHover) return;
        svg.dataset.sparkHover = '1';
        // #poseProcPanel ships pointer-events:none; a child may re-enable it,
        // and 184×44 in a corner is a cheap thing to make interactive.
        svg.style.pointerEvents = 'auto';
        svg.style.cursor = 'crosshair';

        svg.addEventListener('mousemove', function (e) {
            var s = SERIES[svg.id];
            if (!s || !s.vals || s.vals.length < 1) { clear(svg); return; }
            var vals = s.vals, n = vals.length;
            var r = svg.getBoundingClientRect();
            if (!r.width) return;

            // px → viewBox → index. Same mapping PG._spark used to draw it.
            var vbX = (e.clientX - r.left) / r.width * W;
            var i = (n === 1) ? 0
                  : Math.round((vbX - PAD) / (W - 2 * PAD) * (n - 1));
            i = Math.max(0, Math.min(n - 1, i));

            var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
            var rng = (mx - mn) || 1;
            var px = PAD + (W - 2 * PAD) * (n === 1 ? 1 : i / (n - 1));
            var py = H - PAD - (H - 2 * PAD) * ((vals[i] - mn) / rng);
            crosshair(svg, px, py, s.col);

            var L = labels(svg.id, svg);
            var t = tipFor(svg);
            if (!t) return;
            t.innerHTML =
                '<span style="color:#475569">' + L.x + '&thinsp;</span>' +
                '<span style="color:#e2e8f0;font-weight:700">' + (i + 1) + '</span><br>' +
                '<span style="color:#475569">' + L.y + '&thinsp;</span>' +
                '<span style="color:' + (s.col || '#34d399') + ';font-weight:600">' +
                fmt(vals[i]) + '</span>' +
                (n > 1 ? '<br><span style="color:#334155">of ' + n + '</span>' : '');
            t.style.display = 'block';

            // Clamp to the viewport and flip when the cursor is near an edge —
            // the value is the part a clipped tooltip loses first.
            t.style.left = '0px'; t.style.top = '0px';       // measure unclamped
            var tw = t.offsetWidth, th = t.offsetHeight;
            var tx = e.clientX + 12, ty = e.clientY - th - 8;
            if (tx + tw > window.innerWidth - 6)  tx = e.clientX - tw - 12;
            if (ty < 6)                            ty = e.clientY + 14;
            if (ty + th > window.innerHeight - 6)  ty = window.innerHeight - th - 6;
            t.style.left = Math.max(6, tx) + 'px';
            t.style.top  = Math.max(6, ty) + 'px';
        });

        svg.addEventListener('mouseleave', function () { clear(svg); });
    }

    // ── Hook the shared renderer ─────────────────────────────────────────────
    function install() {
        var PG = window.PoseGen;
        if (!PG || typeof PG._spark !== 'function') return false;
        if (PG._spark.__hoverWrapped) return true;

        var orig = PG._spark;
        var wrapped = function (id, vals, col) {
            var out = orig.apply(PG, arguments);
            SERIES[id] = { vals: (vals || []).slice(), col: col };
            var el = document.getElementById(id);
            if (el) attach(el);                 // idempotent
            return out;
        };
        wrapped.__hoverWrapped = true;
        PG._spark = wrapped;

        // Charts already drawn before this file loaded still need listeners;
        // their series arrive on the next _spark call.
        ['poseProcChart', 'poseImpScoreChart', 'poseGignFloatChart']
            .forEach(function (id) { attach(document.getElementById(id)); });
        return true;
    }

    var tries = 0;
    var t = setInterval(function () { if (install() || ++tries > 40) clearInterval(t); }, 250);

    // Let another module hand over a series it renders itself.
    window._poseSparkSeries = function (id, vals, col) {
        SERIES[id] = { vals: (vals || []).slice(), col: col };
        attach(document.getElementById(id));
    };
})();