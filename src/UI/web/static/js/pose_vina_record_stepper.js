// =============================================================================
// pose_vina_record_stepper.js — ◀ ▶ ▶Play through the poses BEHIND the staircase
// -----------------------------------------------------------------------------
// #poseProcPanel plots the Vina search's running minimum: eight steps down means
// eight poses were found, each strictly better than the last. Until now the
// chart was the only trace of seven of them — the ligand you were left looking
// at was the eighth. This makes the other seven reachable.
//
// Same three controls as the import gallery's stepper (PG.importStep /
// importPlay, #poseImpCount / #poseImpPlay), injected under #poseProcSub, and
// deliberately the same shape so the two read as one idiom:
//
//     ◀   3 / 8   ▶                                             ▶ Play
//
// ONE ENTRY PER CHART POINT, which is the property that makes the count
// trustworthy. #poseProcChart is built from eng._bestHist, and that array gets
// its points from two places: _procBest pushes one per new running minimum, and
// _procFinal pushes ONE MORE — the refined mode-1 affinity, which is a different
// and better computation of the same quantity, not another search step. So the
// stepper captures from both, and the last entry is labelled `refined` rather
// than given a record number. A stepper whose N/M disagreed with the visible
// number of steps would be worse than no stepper.
//
// CONFS ARE DEEP-COPIED ON CAPTURE. `conf` is {pos, q: Float64Array, tors:
// Float64Array}; keeping the reference would leave the stepper holding whatever
// the engine last wrote into those buffers, so every entry would replay the same
// pose and it would look like the search never moved. Copy on the way in.
//
// Nothing here re-scores anything. Stepping calls eng._applyMode, exactly what
// clicking a row of #poseVinaModeList does, so ①/②/③, the interaction overlay
// and the ΔG readout all follow through the normal path.
//
//   window._recStep(±1) / _recGo(i)     move
//   window._recPlay()                   toggle playback (dwell ~450 ms)
//   window._recState()                  what was captured
//
// Load after pose.js. Order against pose_affinity_trajectory.js does not matter
// — both wrap _procBest by property lookup and both call through.
// =============================================================================

(function () {
    'use strict';
    if (window.__poseVinaRecStepper) return;
    window.__poseVinaRecStepper = true;

    var DWELL = 450;             // ms per pose during playback
    var recs = [];               // [{conf, dg, kind:'record'|'refined', n}]
    var idx = 0, playing = false, timer = null, built = false;

    // Subscribers. This file is the single owner of "what the staircase poses
    // are and which one we are looking at"; the trajectory scorers consume that
    // rather than each re-deriving it from _procBest.
    //
    // They used to. Two modules independently counting records is the drift bug
    // in G58/G76 shape: the trajectory skipped clashing poses and the stepper
    // did not, so #poseGignFloat said n=5 under a 7-step chart and "pose 3" in
    // one panel was a different pose from "pose 3" in the other. One list, one
    // index, everything downstream agrees by construction.
    var subsRecord = [], subsChange = [], subsReset = [];
    function fire(list, a, b, c, d) {
        for (var i = 0; i < list.length; i++) {
            // A throwing subscriber must not stop the search or the other
            // subscribers; this runs inside the engine's hot path.
            try { list[i](a, b, c, d); } catch (e) {
                if (window.console) console.warn('[rec-stepper] subscriber failed:', e);
            }
        }
    }

    function $(id) { return document.getElementById(id); }
    function PGof() { return window.PoseGen; }

    // ── Capture ──────────────────────────────────────────────────────────────
    function clone(conf) {
        try {
            return {
                pos:  Array.prototype.slice.call(conf.pos),
                q:    new Float64Array(conf.q),
                tors: new Float64Array(conf.tors),
            };
        } catch (e) { return null; }
    }

    function push(conf, dg, kind) {
        var c = clone(conf);
        if (!c) return;
        recs.push({ conf: c, dg: dg, kind: kind,
                    n: kind === 'refined' ? null : recs.length + 1 });
        idx = recs.length - 1;
        render();
        fire(subsRecord, idx, c, dg, kind);
    }

    // ── UI ───────────────────────────────────────────────────────────────────
    var BTN = 'padding:1px 7px;border-radius:5px;border:1px solid #1e293b;' +
              'background:rgba(11,17,32,.7);color:#94a3b8;font-size:11px;' +
              'cursor:pointer;line-height:1.15;';

    function build() {
        if (built) return true;
        var sub = $('poseProcSub');
        if (!sub || !sub.parentNode) return false;
        var row = document.createElement('div');
        row.id = 'poseRecRow';
        // #poseProcPanel is pointer-events:none so it never eats a drag on the
        // 3-D scene behind it. Buttons inside an unclickable panel are furniture,
        // so this row — and only this row — takes its events back.
        row.style.cssText = 'display:none;align-items:center;gap:4px;margin-top:4px;' +
                            'pointer-events:auto;';
        row.innerHTML =
            '<button id="poseRecPrev" title="previous pose on the staircase" style="' + BTN + '">&#9664;</button>' +
            '<span id="poseRecCount" style="font-family:ui-monospace,monospace;font-size:10px;' +
                'color:#67e8f9;min-width:46px;text-align:center;">0 / 0</span>' +
            '<button id="poseRecNext" title="next pose on the staircase" style="' + BTN + '">&#9654;</button>' +
            '<button id="poseRecPlay" title="play every pose the search recorded, first &#8594; last" ' +
                'style="margin-left:auto;padding:2px 8px;border-radius:6px;border:1px solid #14532d;' +
                'background:rgba(11,17,32,.7);color:#34d399;font-family:ui-monospace,monospace;' +
                'font-size:9.5px;font-weight:700;cursor:pointer;line-height:1.3;">&#9654; Play</button>';
        sub.parentNode.insertBefore(row, sub.nextSibling);
        $('poseRecPrev').onclick = function () { step(-1); };
        $('poseRecNext').onclick = function () { step(1); };
        $('poseRecPlay').onclick = function () { play(); };
        built = true;
        return true;
    }

    function playLabel() {
        var b = $('poseRecPlay');
        if (!b) return;
        b.innerHTML = playing ? '&#10074;&#10074; Stop' : '&#9654; Play';
        b.style.color = playing ? '#fbbf24' : '#34d399';
        b.style.borderColor = playing ? '#78350f' : '#14532d';
    }

    function render() {
        if (!build()) return;
        var row = $('poseRecRow');
        // One point is not a staircase; showing "1 / 1" and two dead arrows over
        // a flat chart just asks a question the panel cannot answer.
        if (row) row.style.display = recs.length > 1 ? 'flex' : 'none';
        var lab = $('poseRecCount');
        if (lab) lab.textContent = recs.length ? (idx + 1) + ' / ' + recs.length : '0 / 0';
        playLabel();
        mark();
    }

    // Put a ring on the chart point the stepper is sitting on.
    //
    // PG._spark writes the SVG with innerHTML, so this marker is destroyed by
    // every redraw — which is correct: redraws only happen while the search is
    // running, and the stepper is for afterwards. Re-marking on each move is
    // enough, and it means nothing has to coordinate with the renderer.
    // Geometry mirrors _spark exactly (W 184, H 44, pad 4); if that changes,
    // the ring drifts and this comment is where to look.
    function mark() {
        var PG = PGof();
        var svg = $('poseProcChart');
        var vals = PG && PG.vina && PG.vina.eng && PG.vina.eng._bestHist;
        if (!svg || !vals || !vals.length || idx >= vals.length) return;
        var old = svg.querySelector('#poseRecMark');
        if (old) old.parentNode.removeChild(old);
        var W = 184, H = 44, pad = 4;
        var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
        var rng = (mx - mn) || 1;
        var x = pad + (W - 2 * pad) * (vals.length === 1 ? 1 : idx / (vals.length - 1));
        var y = H - pad - (H - 2 * pad) * ((vals[idx] - mn) / rng);
        var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('id', 'poseRecMark');
        g.innerHTML =
            '<line x1="' + x.toFixed(1) + '" y1="0" x2="' + x.toFixed(1) + '" y2="' + H + '" ' +
                'stroke="#67e8f9" stroke-width="0.7" stroke-dasharray="2 2" opacity="0.55"/>' +
            '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.1" fill="none" ' +
                'stroke="#67e8f9" stroke-width="1.4"/>';
        svg.appendChild(g);
    }

    // ── Move ─────────────────────────────────────────────────────────────────
    function go(i) {
        var PG = PGof();
        if (!recs.length || !PG || !PG.vina || !PG.vina.eng) return false;
        idx = Math.max(0, Math.min(recs.length - 1, i | 0));
        var r = recs[idx];
        try {
            // The same call #poseVinaModeList's rows make. `live` omitted, so
            // this counts as a discrete pose change and _bump() fires — stages
            // ①/②/③ re-derive instead of showing a stale derived view.
            PG.vina.eng._applyMode({ conf: r.conf });
        } catch (e) {
            if (PG._miniLog) PG._miniLog('could not load that pose: ' + e.message, '#fb7185');
            return false;
        }
        var s = $('poseProcSub');
        if (s) {
            s.textContent = (r.kind === 'refined' ? 'refined' : 'record ' + r.n) +
                            ' · ΔG ' + (r.dg != null ? r.dg.toFixed(2) : '—') +
                            ' · ' + (idx + 1) + '/' + recs.length;
        }
        var v = $('poseProcVal');
        if (v && r.dg != null) {
            v.textContent = r.dg.toFixed(2);
            v.style.color = r.dg < 0 ? '#34d399' : '#fb7185';
        }
        render();
        // The affinity floats follow: each shows the score IT produced for this
        // exact pose, so ◀ / ▶ moves the ligand, the Vina ΔG, the GIGN number
        // and the DeepAtom number as one.
        fire(subsChange, idx, r);
        return true;
    }

    function step(d) {
        stop();
        go(idx + (d || 0));
    }

    function visible() {
        var p = $('poseProcPanel');
        return !!(p && p.offsetParent !== null);
    }

    function play() {
        if (playing) { stop(); return 'stopped'; }
        if (recs.length < 2) return 'nothing to play — run a Vina search first';
        playing = true; playLabel();
        var i = 0;
        var next = function () {
            // Closing the modal hides the panel; without this the timer would
            // keep stepping poses nobody is looking at for the life of the tab.
            if (!playing || !visible()) { stop(); return; }
            if (i >= recs.length) { stop(); return; }
            go(i); i++;
            timer = setTimeout(next, DWELL);
        };
        next();
        return 'playing ' + recs.length + ' poses';
    }

    function stop() {
        playing = false;
        if (timer) { clearTimeout(timer); timer = null; }
        playLabel();
    }

    // ── Install ──────────────────────────────────────────────────────────────
    function install() {
        var PG = PGof();
        var eng = PG && PG.vina && PG.vina.eng;
        if (!eng || typeof eng._procBest !== 'function') return false;
        build();

        if (!eng._procBest.__recStep) {
            var origBest = eng._procBest;
            var wBest = function (best, frac, conf) {
                var before = (eng._bestHist || []).length;
                var out = origBest.apply(eng, arguments);
                var after = (eng._bestHist || []).length;
                // Watch the array rather than re-deriving the "is this a record?"
                // test. The two could drift; the chart cannot drift from itself.
                if (after > before) push(conf, eng._bestHist[after - 1], 'record');
                return out;
            };
            wBest.__recStep = true;
            eng._procBest = wBest;
        }

        if (typeof eng._procFinal === 'function' && !eng._procFinal.__recStep) {
            var origFinal = eng._procFinal;
            var wFinal = function (mode, n) {
                var out = origFinal.apply(eng, arguments);
                // _procFinal pushes the refined affinity onto _bestHist itself,
                // so capturing here keeps entries and chart points in lockstep.
                if (mode && mode.conf) push(mode.conf, mode.affinity, 'refined');
                return out;
            };
            wFinal.__recStep = true;
            eng._procFinal = wFinal;
        }

        if (typeof eng._procReset === 'function' && !eng._procReset.__recStep) {
            var origReset = eng._procReset;
            var wReset = function () {
                stop(); recs = []; idx = 0; render();
                fire(subsReset);
                return origReset.apply(eng, arguments);
            };
            wReset.__recStep = true;
            eng._procReset = wReset;
        }
        return true;
    }

    var tries = 0;
    var t = setInterval(function () { if (install() || ++tries > 60) clearInterval(t); }, 250);

    window.PoseVinaRecords = {
        list:  function () { return recs.map(function (r, i) {
                   return { i: i, kind: r.kind, n: r.n, dg: r.dg }; }); },
        count: function () { return recs.length; },
        index: function () { return idx; },
        go: go, step: step, play: play, stop: stop,
        // onRecord(i, conf, dg, kind) — a new staircase pose was captured.
        // onChange(i, rec)            — the displayed pose moved.
        // onReset()                   — a new search started; drop everything.
        onRecord: function (cb) { if (typeof cb === 'function') subsRecord.push(cb); },
        onChange: function (cb) { if (typeof cb === 'function') subsChange.push(cb); },
        onReset:  function (cb) { if (typeof cb === 'function') subsReset.push(cb); },
        state: function () {
            return { count: recs.length, index: idx, playing: playing,
                     current: recs[idx] ? { kind: recs[idx].kind, n: recs[idx].n, dg: recs[idx].dg } : null };
        },
    };
    window._recStep  = function (d) { step(d); return window.PoseVinaRecords.state(); };
    window._recGo    = function (i) { go(i);   return window.PoseVinaRecords.state(); };
    window._recPlay  = function ()  { return play(); };
    window._recState = function ()  { return window.PoseVinaRecords.state(); };
})();