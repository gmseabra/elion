// =============================================================================
// ts/ts_chart.js — sparkline chart: draw, hover tooltip, push helper
// Depends on: ts_core.js (_ts)
// =============================================================================

// ── Canvas panel injection ────────────────────────────────────────────────
function _tsPatchChartPanel() {
    if (document.getElementById('tsBatchCanvas')) return;

    const barsEl = document.getElementById('tsTsBars');
    if (!barsEl) return;

    const wrapper = document.createElement('div');
    wrapper.id = '_tsBarsChartRow';
    // Fill the whole left column. The raw-log section that used to take the
    // lower ~45% is now hidden (the 💬 panel shows the log), so the bars+chart
    // row expands to occupy the entire tsPaneTs height.
    wrapper.style.cssText = 'display:flex;flex-direction:row;gap:0;min-height:0;overflow:hidden;flex:1;';

    barsEl.parentNode.insertBefore(wrapper, barsEl);
    barsEl.style.flex = '0 0 55%';
    barsEl.style.minWidth = '0';
    wrapper.appendChild(barsEl);

    const chartPanel = document.createElement('div');
    chartPanel.id = '_tsBatchChartPanel';
    chartPanel.style.cssText = [
        'flex:0 0 calc(45% - 8px)',
        'min-width:0',
        'display:flex',
        'flex-direction:column',
        'background:rgba(5,7,15,0.95)',
        'border-left:0.5px solid rgba(148,163,184,0.08)',
        'border-radius:4px',
        'overflow:hidden',
        'margin-left:8px',
        'padding:8px 10px 6px 4px',
    ].join(';');

    chartPanel.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;flex-shrink:0">
        <span style="font-size:9px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.05em">
          Batch mean · last <span id="tsBatchN2">—</span> iters
        </span>
        <span style="font-size:9px;color:#64748b">
          μ&thinsp;<span id="tsBatchMean2" style="color:#22d3ee;font-family:monospace;font-weight:600">—</span>
          &thinsp;±&thinsp;<span id="tsBatchStd2" style="color:#fbbf24;font-family:monospace;font-weight:600">—</span>
        </span>
      </div>
      <div id="_tsBatchCanvasWrap" style="position:relative;flex:1;min-height:0;display:flex;">
        <canvas id="tsBatchCanvas"
          style="width:100%;flex:1;min-height:0;display:block;border-radius:3px;background:rgba(15,23,42,0.7)">
        </canvas>
        <div id="_tsSparkTooltip" style="
          display:none;position:absolute;pointer-events:none;
          background:rgba(5,7,15,0.92);border:0.5px solid rgba(34,211,238,0.35);
          border-radius:4px;padding:4px 7px;white-space:nowrap;
          font-size:9px;font-family:monospace;line-height:1.6;
          box-shadow:0 2px 8px rgba(0,0,0,0.5);z-index:10;
        "></div>
      </div>`;

    wrapper.appendChild(chartPanel);
    setTimeout(_tsSparkHoverSetup, 0);
}

// ── Spark history ─────────────────────────────────────────────────────────
const _TS_SPARK_MAX = 2000;
// Hard cap on the retained full-resolution buffer (bounds memory on very long
// runs). Display always shows at most _TS_SPARK_MAX points via stride decimation.
const _TS_SPARK_FULL_MAX = 60000;

function _tsSparkReset() {
    _ts._sparkHistory = [];
    _ts._sparkRafPending = false;
    _ts._batchBest = undefined;
    _ts._batchWorst = undefined;
    _ts._jobSparkHistory = {};
    _ts._jobSparkFull    = {};   // full-resolution per-job buffer (pre-decimation)
    _ts._jobBatchBest    = {};
    _ts._jobBatchWorst   = {};
}

// Deterministic decimation: keep every stride-th point (+ always the last),
// where stride = ceil(n / _TS_SPARK_MAX). Because it depends only on the point's
// index in the full buffer, live and replay produce IDENTICAL display sets —
// fixing the "spiky before, smooth after refresh" mismatch that came from the
// old incremental halving (which dropped different points live vs on replay).
function _tsStrideDecimate(full) {
    const n = full.length;
    if (n <= _TS_SPARK_MAX) return full.slice();
    const stride = Math.ceil(n / _TS_SPARK_MAX);
    const kept = [];
    for (let i = 0; i < n; i++) {
        if (i % stride === 0 || i === n - 1) kept.push(full[i]);
    }
    return kept;
}

function _tsSparkPush(iter, mean, std, ji, score) {
    if (!_ts._jobSparkFull) _ts._jobSparkFull = {};
    if (!_ts._jobSparkHistory) _ts._jobSparkHistory = {};
    if (!_ts._jobSparkFull[ji]) _ts._jobSparkFull[ji] = [];
    const full = _ts._jobSparkFull[ji];
    // Enforce a STRICTLY-INCREASING iteration axis. batch_stats can arrive out of
    // order or from more than one iteration counter — e.g. two concurrent jobs
    // (Suzuki + SnAr) whose points collide in one buffer, or reconnect seeding —
    // which made the chart's x-axis (and the hover tooltip's "iter N") jump
    // backwards (849 → 52). Accept a point only if its iter advances past the last
    // one; if the SAME iter re-reports, refresh its mean/std in place; drop any
    // stale/backwards iter so the displayed sequence is monotonic.
    const lastIter = full.length ? full[full.length - 1].iter : -Infinity;
    if (Number.isFinite(iter) && Number.isFinite(lastIter) && iter <= lastIter) {
        if (iter === lastIter) {
            const lp = full[full.length - 1];
            lp.mean = mean; lp.std = std; if (score != null) lp.score = score;
        }
        // strictly-less → out-of-order/foreign point: do not append
    } else {
        full.push({ iter, mean, std, score });
    }
    // Bound memory: once the full buffer hits the hard cap, drop every other
    // point (rare — only on 60k+ iteration runs).
    if (full.length > _TS_SPARK_FULL_MAX) {
        const kept = [];
        for (let i = 0; i < full.length; i += 2) kept.push(full[i]);
        _ts._jobSparkFull[ji] = kept;
    }
    // Derive the display (decimated) history deterministically.
    _ts._jobSparkHistory[ji] = _tsStrideDecimate(_ts._jobSparkFull[ji]);
    if (ji === (_ts._activeJobIdx ?? 0)) {
        _ts._sparkHistory = _ts._jobSparkHistory[ji];
    }
}

// ── Layout cache shared between draw and hover ────────────────────────────
const _tsSparkLayout = {};

// ── Draw ──────────────────────────────────────────────────────────────────
function _tsDrawSparkline(hoverIdx) {
    const canvas = document.getElementById('tsBatchCanvas');
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width  || 200;
    const H = rect.height || 90;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width  = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const hist = _ts._sparkHistory;
    if (!hist || hist.length < 2) {
        ctx.fillStyle = 'rgba(100,116,139,0.4)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('accumulating data…', W / 2, H / 2);
        return;
    }

    const WMAX        = _ts.evalBatchSize || 256;
    const TRANS_START = WMAX;
    const TRANS_END   = WMAX * 2;

    const PAD = { top: 8, right: 6, bottom: 16, left: 38 };
    const cW  = W - PAD.left - PAD.right;
    const cH  = H - PAD.top  - PAD.bottom;

    const totalPts = hist.length;

    // Single O(n) pass for all range values — avoids 4 ephemeral arrays and
    // 5 Math.max/min spread calls that allocate and GC on every draw frame.
    let rawMin = Infinity, rawMax = -Infinity, maxStd = 0,
        bandMin = Infinity, bandMax = -Infinity;
    for (const p of hist) {
        if (p.mean < rawMin) rawMin = p.mean;
        if (p.mean > rawMax) rawMax = p.mean;
        if (p.std  > maxStd) maxStd  = p.std;
        const lo = p.mean - p.std;
        const hi = p.mean + p.std;
        if (lo < bandMin) bandMin = lo;
        if (hi > bandMax) bandMax = hi;
    }
    bandMin -= maxStd * 0.1;
    bandMax += maxStd * 0.1;
    const blend   = Math.min(1, Math.max(0, (totalPts - TRANS_START) / (TRANS_END - TRANS_START)));

    const minScore   = rawMin * (1 - blend) + bandMin * blend;
    const maxScore   = rawMax * (1 - blend) + bandMax * blend;
    const scoreRange = maxScore - minScore || 1;

    const xOf = i => PAD.left + (i / (hist.length - 1)) * cW;
    const yOf = v => PAD.top  + (1 - (v - minScore) / scoreRange) * cH;

    Object.assign(_tsSparkLayout, { PAD, cW, cH, W, H, hist, xOf, yOf, minScore, maxScore, scoreRange });

    // ── σ band (solid fill + solid border lines, no dashes — dashes cause bar artifacts) ──
    if (blend > 0 && maxStd > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(xOf(0), yOf(hist[0].mean + hist[0].std));
        for (let i = 1; i < hist.length; i++)
            ctx.lineTo(xOf(i), yOf(hist[i].mean + hist[i].std));
        for (let i = hist.length - 1; i >= 0; i--)
            ctx.lineTo(xOf(i), yOf(hist[i].mean - hist[i].std));
        ctx.closePath();
        ctx.fillStyle = `rgba(251,191,36,${blend * 0.12})`;
        ctx.fill();

        ctx.lineWidth   = 0.6;
        ctx.strokeStyle = `rgba(251,191,36,${blend * 0.25})`;
        ctx.setLineDash([]);
        for (const sign of [1, -1]) {
            ctx.beginPath();
            ctx.moveTo(xOf(0), yOf(hist[0].mean + sign * hist[0].std));
            for (let i = 1; i < hist.length; i++)
                ctx.lineTo(xOf(i), yOf(hist[i].mean + sign * hist[i].std));
            ctx.stroke();
        }
        ctx.restore();
    }

    // ── Raw score dots (early phase, fading out) ──────────────────────────
    if (blend < 1) {
        ctx.save();
        for (let i = 0; i < hist.length; i++) {
            const ptBlend = Math.min(1, Math.max(0, (i - (totalPts - TRANS_END)) / TRANS_START));
            const a = Math.max(0, (1 - ptBlend) * 0.45 * (1 - blend));
            if (a < 0.02) continue;
            ctx.globalAlpha = a;
            ctx.fillStyle = '#22d3ee';
            ctx.beginPath();
            ctx.arc(xOf(i), yOf(hist[i].mean), 1.2, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.restore();
    }

    // ── Mean line ─────────────────────────────────────────────────────────
    ctx.save();
    ctx.setLineDash([]);
    ctx.lineWidth   = 1.5;
    ctx.strokeStyle = `rgba(34,211,238,${0.4 + blend * 0.6})`;
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(hist[0].mean));
    for (let i = 1; i < hist.length; i++)
        ctx.lineTo(xOf(i), yOf(hist[i].mean));
    ctx.stroke();
    ctx.restore();

    // ── Live dot ──────────────────────────────────────────────────────────
    const livePt = hist[hist.length - 1];
    ctx.beginPath();
    ctx.arc(xOf(hist.length - 1), yOf(livePt.mean), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#22d3ee';
    ctx.fill();

    // ── Hover crosshair ───────────────────────────────────────────────────
    if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < hist.length) {
        const hx = xOf(hoverIdx);
        const hy = yOf(hist[hoverIdx].mean);
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.lineWidth   = 0.8;
        ctx.strokeStyle = 'rgba(34,211,238,0.35)';
        ctx.beginPath(); ctx.moveTo(hx, PAD.top); ctx.lineTo(hx, PAD.top + cH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD.left, hy); ctx.lineTo(PAD.left + cW, hy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        ctx.beginPath();
        ctx.arc(hx, hy, 3, 0, Math.PI * 2);
        ctx.fillStyle   = '#22d3ee';
        ctx.strokeStyle = 'rgba(5,7,15,0.9)';
        ctx.lineWidth   = 1.5;
        ctx.fill();
        ctx.stroke();
    }

    // ── Axis labels ───────────────────────────────────────────────────────
    ctx.fillStyle = '#334155';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('iter ' + hist[0].iter, PAD.left, H - 2);
    ctx.textAlign = 'right';
    ctx.fillText(hist[hist.length - 1].iter, W - PAD.right, H - 2);

    ctx.fillStyle = '#475569';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(maxScore.toFixed(2), PAD.left - 3, PAD.top + 6);
    ctx.fillText(minScore.toFixed(2), PAD.left - 3, PAD.top + cH);

    // ── Stats readout ─────────────────────────────────────────────────────
    const last  = hist[hist.length - 1];
    const setEl = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    setEl('tsBatchMean',  last.mean.toFixed(4));
    setEl('tsBatchMean2', last.mean.toFixed(4));
    setEl('tsBatchStd',   last.std.toFixed(4));
    setEl('tsBatchStd2',  last.std.toFixed(4));
}

// ── Hover setup ───────────────────────────────────────────────────────────
function _tsSparkHoverSetup() {
    const canvas  = document.getElementById('tsBatchCanvas');
    const tooltip = document.getElementById('_tsSparkTooltip');
    if (!canvas || !tooltip) return;

    let _hoverIdx = null;
    canvas.style.cursor = 'crosshair';

    canvas.addEventListener('mousemove', (e) => {
        const L = _tsSparkLayout;
        if (!L.hist || L.hist.length < 2) return;

        const rect = canvas.getBoundingClientRect();
        const mx   = e.clientX - rect.left;
        const my   = e.clientY - rect.top;

        if (mx < L.PAD.left || mx > L.PAD.left + L.cW ||
            my < L.PAD.top  || my > L.PAD.top  + L.cH) {
            _tsSparkHoverClear(canvas, tooltip);
            _hoverIdx = null;
            return;
        }

        const frac    = (mx - L.PAD.left) / L.cW;
        const clamped = Math.max(0, Math.min(L.hist.length - 1, Math.round(frac * (L.hist.length - 1))));
        if (clamped === _hoverIdx) return;
        _hoverIdx = clamped;

        const pt       = L.hist[clamped];
        const WMAX     = _ts.evalBatchSize || 256;
        const isPhase1 = clamped < WMAX;

        // Phase 1 (iter < WMAX, blend=0): raw score dots only — show score, nothing else.
        //   pt.score is present for live-streamed points; null for old session files that
        //   predate the score-in-points change. In that case show iter only — never show
        //   mean/std in phase 1, they aren't meaningful with a tiny rolling window.
        // Phase 2+ (iter >= WMAX): rolling μ ± σ is the signal — show both.
        let valueHtml;
        if (isPhase1) {
            valueHtml = pt.score != null
                ? `<span style="color:#475569">score&thinsp;</span>` +
                  `<span style="color:#34d399;font-weight:600">${pt.score.toFixed(4)}</span>`
                : '';   // old session file — no score stored, show iter only
        } else {
            valueHtml = `<span style="color:#475569">μ&thinsp;</span><span style="color:#22d3ee">${pt.mean.toFixed(4)}</span><br>` +
                        `<span style="color:#475569">σ&thinsp;</span><span style="color:#fbbf24">${pt.std.toFixed(4)}</span>`;
        }

        tooltip.innerHTML =
            `<span style="color:#475569">iter&thinsp;</span><span style="color:#e2e8f0;font-weight:700">${pt.iter}</span>` +
            (valueHtml ? `<br>${valueHtml}` : '');

        const wrap  = document.getElementById('_tsBatchCanvasWrap');
        const wRect = wrap ? wrap.getBoundingClientRect() : rect;
        const tipX  = e.clientX - wRect.left + 10;
        const tipY  = e.clientY - wRect.top  - 10;
        tooltip.style.left    = (tipX > wRect.width * 0.65 ? tipX - tooltip.offsetWidth - 20 : tipX) + 'px';
        tooltip.style.top     = Math.max(0, tipY) + 'px';
        tooltip.style.display = 'block';

        _tsDrawSparkline(clamped);
    });

    canvas.addEventListener('mouseleave', () => {
        _tsSparkHoverClear(canvas, tooltip);
        _hoverIdx = null;
    });
}

function _tsSparkHoverClear(canvas, tooltip) {
    if (!tooltip) tooltip = document.getElementById('_tsSparkTooltip');
    if (tooltip)  tooltip.style.display = 'none';
    _tsDrawSparkline();
}

function _tsQueueSparkline() {
    if (_ts._sparkRafPending) return;
    _ts._sparkRafPending = true;
    requestAnimationFrame(() => { _ts._sparkRafPending = false; _tsDrawSparkline(); });
}

// ── History replay (called by _tsReconnectActive in ts_warmup.js) ─────────────
// Owns all score-backfill logic so ts_warmup.js stays free of chart concerns.
//
// @param hist  — job.history from /ts_active: { points, scores, reagents }
// @param ji    — job index
//
// Returns the populated spark history array (also written into _ts._jobSparkHistory[ji]).
function _tsSparkReplayHistory(hist, ji) {
    if (!_ts._jobSparkHistory) _ts._jobSparkHistory = {};
    _ts._jobSparkHistory[ji] = [];

    const pts    = hist.points || [];
    const scores = hist.scores || [];   // rolling buffer: last ≤256 raw scores, oldest→newest

    // ── Score backfill for old session files ──────────────────────────────────
    // New ts_routes.py stores score in each point for phase 1 (iter < BATCH_MAX).
    // Old session files have no score field at all (pts[i].score === undefined).
    // In that case, reconstruct from hist.scores: they align with the last
    // min(256, pts.length) points in order.
    //
    // Example — iter 121 run (all fit in 256):
    //   pts.length=121, scores.length=121, offset=0  → pts[0..120] all backfilled
    //
    // Example — iter 400 run (scores capped at 256):
    //   pts.length=400, scores.length=256, offset=144
    //   pts[0..143]   → score stays null (outside rolling window, phase 2+ anyway)
    //   pts[144..399] → backfilled from scores[0..255]
    const needsBackfill = pts.length > 0 && pts[0].score === undefined;
    if (needsBackfill && scores.length > 0) {
        const offset = pts.length - scores.length;
        scores.forEach((s, si) => {
            const pi = offset + si;
            if (pi >= 0 && pi < pts.length) pts[pi] = { ...pts[pi], score: s };
        });
    }

    if (!_ts._jobSparkFull) _ts._jobSparkFull = {};
    const fullBuf = [];
    pts.forEach((p, idx) => {
        fullBuf.push({
            iter:  idx + 1,
            mean:  p.mean,
            std:   p.std,
            score: p.score ?? null,
        });
    });
    _ts._jobSparkFull[ji] = fullBuf;
    // Derive the display history via the SAME deterministic decimation the live
    // path uses, so the chart is identical before and after a refresh.
    _ts._jobSparkHistory[ji] = _tsStrideDecimate(fullBuf);

    // Keep active job's reference in sync
    if (ji === (_ts._activeJobIdx ?? 0)) {
        _ts._sparkHistory = _ts._jobSparkHistory[ji];
    }

    return _ts._jobSparkHistory[ji];
}