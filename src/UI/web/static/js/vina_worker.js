/* =============================================================================
 * vina_worker.js — runs the Vina 1.2.7 Monte-Carlo/BFGS search off the main
 * thread. At Vina's real step counts (globalSteps ~= 26,000 x 8 tasks) a search
 * takes tens of seconds; running it on the UI thread would freeze the tab.
 *
 * Protocol
 *   -> {cmd:'dock',   spec:{...}}   run the full search
 *   -> {cmd:'score',  spec:{...}}   score a single pose (--score_only)
 *   -> {cmd:'cancel'}               stop after the current task
 *   <- {type:'progress', frac, best, task, tasks, conf?}   conf = the running best pose
 *   <- {type:'done',     modes, numTors, params, elapsed}
 *   <- {type:'scored',   breakdown}
 *   <- {type:'error',    message}
 *
 * `spec.lig`, `spec.recAtoms`, `spec.box`, `spec.weights` are plain JSON, so
 * this works with or without SharedArrayBuffer / cross-origin isolation.
 * ============================================================================= */
/* global importScripts, VinaEngine */
'use strict';

// importScripts resolves relative to THIS worker's URL, so vina_engine.js is
// picked up from the same static/js/ directory with no configuration.
try { importScripts('vina_engine.js'); }
catch (e) { self.postMessage({ type: 'error',
  message: 'could not load vina_engine.js next to vina_worker.js: ' + e.message }); }

var V = self.VinaEngine;
var cancelled = false;

function buildModel(spec) {
  var rec = V.typeReceptor(spec.recAtoms);
  var weights = spec.weights || V.DEFAULT_WEIGHTS;
  var prec = new V.Precalculate(weights, spec.precalcFactor || 32);
  var slope = (spec.slope !== undefined) ? spec.slope : 1e6;
  var model = new V.Model(spec.lig, rec, {
    weights: weights, box: spec.box, precalculate: prec, slope: slope
  });
  if (spec.useCache !== false) {
    self.postMessage({ type: 'progress', frac: 0, best: null, task: 0,
                       tasks: spec.exhaustiveness || 8, phase: 'building affinity maps' });
    model.cache = new V.Cache(rec, prec, spec.box, spec.granularity || 0.375,
                              model.neededTypes, slope,
      function (f) { self.postMessage({ type: 'progress', frac: 0, best: null,
        task: 0, tasks: spec.exhaustiveness || 8,
        phase: 'building affinity maps ' + Math.round(f*100) + '%' }); });
  }
  return { model: model, rec: rec, weights: weights, prec: prec };
}

/**
 * Vina::global_search, task by task, with progress reporting between tasks and
 * every few hundred MC steps. Kept here rather than calling VinaEngine.dock()
 * so the worker can interleave progress messages and honour cancellation.
 */
function runDock(spec) {
  var t0 = Date.now();
  var built = buildModel(spec);
  var model = built.model, weights = built.weights;

  var ndof = model.ndof;
  var heuristic = model.nHeavy + 10 * ndof;
  // vina.cpp:897-899
  var globalSteps = spec.globalSteps || Math.floor(70 * 3 * (50 + heuristic) / 2);
  var localSteps  = spec.localSteps  || Math.floor((25 + model.nHeavy) / 3);
  var exhaustiveness = spec.exhaustiveness || 8;
  var numModes = spec.numModes || 9;

  var p = {
    globalSteps: globalSteps,
    localSteps: localSteps,
    temperature: (spec.temperature !== undefined) ? spec.temperature : 1.2,
    mutationAmplitude: (spec.mutationAmplitude !== undefined) ? spec.mutationAmplitude : 2.0,
    huntCap: (spec.huntCap !== undefined) ? spec.huntCap : 10,
    authenticV: 1000,
    numSavedMins: numModes,
    minRmsd: (spec.minRmsd !== undefined) ? spec.minRmsd : 1.0
  };

  var seed = (spec.seed !== undefined && spec.seed !== null) ? spec.seed : 1;
  var taskFrom = spec.taskFrom || 0;
  var taskTo   = (spec.taskTo !== undefined) ? spec.taskTo : exhaustiveness;
  var all = [], task;
  var globalBest = Infinity;
  var lastPost = 0;

  for (task = taskFrom; task < taskTo; task++) {
    if (cancelled) break;
    var rng = new V.MT19937((seed + task * 7919) >>> 0);
    var thisTask = task;
    // When the running best improves, remember the conformation that did it and
    // ship it with the NEXT progress message rather than posting immediately —
    // early in a run records arrive far faster than 5 Hz, and the main thread
    // redraws a molecular scene for each one. Holding it in `pendingConf` means
    // the throttle still governs the message rate while no record is ever lost:
    // whatever the current best is, that is the pose that goes out.
    var pendingConf = null;
    var res = V.monteCarlo(model, p, rng, function (step, best, bestConf) {
      if (best < globalBest) { globalBest = best; if (bestConf) pendingConf = bestConf; }
      var now = Date.now();
      if (now - lastPost < 200) return;
      lastPost = now;
      var msg = {
        type: 'progress',
        frac: (thisTask - taskFrom + step / p.globalSteps) / Math.max(1, taskTo - taskFrom),
        best: isFinite(globalBest) ? globalBest : null,
        task: thisTask + 1, tasks: exhaustiveness
      };
      if (pendingConf) {                       // plain arrays: structured-clone friendly
        msg.conf = { pos:  Array.prototype.slice.call(pendingConf.pos),
                     q:    Array.prototype.slice.call(pendingConf.q),
                     tors: Array.prototype.slice.call(pendingConf.tors) };
        pendingConf = null;
      }
      self.postMessage(msg);
    });
    for (var i = 0; i < res.length; i++) all.push(res[i]);
  }

  // merge_output_containers uses min_rmsd = 2 across tasks (parallel_mc.cpp:57)
  all.sort(function (a, b) { return a.e - b.e; });
  var merged = [];
  for (var m = 0; m < all.length; m++) {
    (function addTo(out, cand) {
      for (var k = 0; k < out.length; k++) {
        if (V.rmsdLB(out[k].coords, cand.coords, model.nHeavy) < 2.0) {
          if (cand.e < out[k].e) { out[k] = cand; out.sort(function(a,b){return a.e-b.e;}); }
          return;
        }
      }
      out.push(cand); out.sort(function (a, b) { return a.e - b.e; });
      if (out.length > numModes) out.length = numModes;
    })(merged, all[m]);
  }

  if (spec.rawOnly) {
    self.postMessage({ type: 'raw', minima: merged.map(function (m) {
      return { e: m.e,
               conf: { pos: Array.prototype.slice.call(m.conf.pos),
                       q:   Array.prototype.slice.call(m.conf.q),
                       tors:Array.prototype.slice.call(m.conf.tors) },
               coords: Array.prototype.slice.call(m.coords) };
    }), taskFrom: taskFrom, taskTo: taskTo, elapsed: Date.now() - t0 });
    return;
  }
  var modes = merged.length
    ? V.refineAndRescore(model, merged, localSteps, weights,
        (spec.energyRange !== undefined ? spec.energyRange : 3), numModes, spec.numTors)
        .map(function (m) {
          return { affinity: m.affinity, inter: m.inter, intra: m.intra, raw: m.raw,
                   confIndependent: m.confIndependent, rmsdLB: m.rmsdLB,
                   conf: { pos: Array.prototype.slice.call(m.conf.pos),
                           q:   Array.prototype.slice.call(m.conf.q),
                           tors:Array.prototype.slice.call(m.conf.tors) },
                   coords: Array.prototype.slice.call(m.coords) };
        })
    : [];

  self.postMessage({
    type: 'done', modes: modes, numTors: model.numTors,
    cancelled: cancelled,
    params: {
      globalSteps: globalSteps, localSteps: localSteps,
      exhaustiveness: exhaustiveness, seed: seed, ndof: ndof,
      nHeavy: model.nHeavy, nReceptorAtoms: built.rec.n,
      intraPairs: model.intraPairs.length / 2,
      temperature: p.temperature, huntCap: p.huntCap,
      mutationAmplitude: p.mutationAmplitude, minRmsd: p.minRmsd
    },
    elapsed: Date.now() - t0
  });
}

function runRefine(spec) {
  var built = buildModel(spec);          // useCache:false -- refinement is exact pairwise
  var model = built.model;
  var localSteps = spec.localSteps || Math.floor((25 + model.nHeavy) / 3);
  var merged = spec.minima.map(function (m) {
    var c = V.makeConf(model.tors.length);
    c.pos = m.conf.pos.slice(); c.q.set(m.conf.q); c.tors.set(m.conf.tors);
    return { e: m.e, conf: c, coords: Float64Array.from(m.coords) };
  });
  var modes = V.refineAndRescore(model, merged, localSteps, built.weights,
    (spec.energyRange !== undefined ? spec.energyRange : 3), spec.numModes || 9, spec.numTors);
  self.postMessage({ type: 'refined', modes: modes.map(function (m) {
    return { affinity: m.affinity, inter: m.inter, intra: m.intra, raw: m.raw,
             confIndependent: m.confIndependent, rmsdLB: m.rmsdLB,
             conf: { pos: Array.prototype.slice.call(m.conf.pos),
                     q:   Array.prototype.slice.call(m.conf.q),
                     tors:Array.prototype.slice.call(m.conf.tors) },
             coords: Array.prototype.slice.call(m.coords) };
  }), numTors: model.numTors, nHeavy: model.nHeavy, nReceptorAtoms: built.rec.n });
}

function runScore(spec) {
  var built = buildModel(spec);
  var model = built.model;
  var conf = V.makeConf(model.tors.length);
  if (spec.conf) {
    conf.pos = spec.conf.pos.slice();
    conf.q.set(spec.conf.q);
    conf.tors.set(spec.conf.tors);
  }
  var b = V.scorePose(model, conf, built.weights);
  b.nReceptorAtoms = built.rec.n;
  b.nHeavy = model.nHeavy;
  self.postMessage({ type: 'scored', breakdown: b });
}

self.onmessage = function (ev) {
  var d = ev.data || {};
  try {
    // Note: runDock is synchronous, so this message is only delivered once a
    // dock has already finished. Real cancellation is worker.terminate() from
    // the main thread (see PoseGen.vina.eng.cancel).
    if (d.cmd === 'cancel') { cancelled = true; return; }
    if (d.cmd === 'dock')   { cancelled = false; runDock(d.spec); return; }
    if (d.cmd === 'refine') { runRefine(d.spec); return; }
    if (d.cmd === 'score')  { runScore(d.spec); return; }
  } catch (e) {
    self.postMessage({ type: 'error', message: (e && e.message) || String(e),
                       stack: e && e.stack });
  }
};