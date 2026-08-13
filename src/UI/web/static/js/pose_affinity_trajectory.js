// =============================================================================
// pose_affinity_trajectory.js — every staircase pose, scored by EVERY predictor
// -----------------------------------------------------------------------------
// This is the trajectory twin of pose_affinity_dock.js: one engine, N scorers
// registered against it. pose_gign_trajectory.js and pose_deepatom_trajectory.js
// are ~30-line configs, exactly as pose_gign_dock.js / pose_deepatom_dock.js are
// configs for the dock factory.
//
// IT DOES NOT DECIDE WHAT A POSE IS. pose_vina_record_stepper.js owns the
// staircase — it captures from _procBest and _procFinal, deep-copies each conf,
// and hands out an index. This file subscribes.
//
// That indirection is the fix for a bug you could see on screen: this module
// used to wrap _procBest itself and skip records whose Vina ΔG was positive, so
// #poseGignFloat read n=5 under a 7-step chart, and "pose 3" in the float and
// "3 / 7" in the panel were different poses. Two modules independently deriving
// "which pose is this" is the G58/G76 drift shape. Now there is one list, one
// index, and every panel agrees by construction.
//
//     for each staircase pose i:
//         serialise it ONCE
//         save it to <trajectory_dir>/<name>.pdb
//         enqueue one job per armed scorer, tagged with i
//     when a score returns:
//         dock.mark(i) then the setter -> the history entry carries its pose
//     when the stepper moves to i:
//         dock.select(i) on every scorer -> both floats show THAT pose
//
// ONE SERIALISATION, FANNED OUT. The pose is converted to PDB once and the same
// text goes to every scorer, so a disagreement between the two curves is a
// disagreement between the models — never between two captures of a moving
// ligand. That is the entire point of comparing them.
//
// ONE SHARED QUEUE, STRICTLY SERIAL. Not one queue per scorer. Every job here
// spawns a `conda activate` + torch import + model load, and GIGN and DeepAtom
// land on the same GPU; two in flight is slower than two in order and risks OOM.
// The queue also respects the manual "⚛ Score best pose" buttons: if any
// scorer's busy flag is set by a hand-clicked score, the pump waits.
//
// Consequence worth knowing: DeepAtom is seconds-to-minutes per pose against
// GIGN's ~1 s, so on a 12-pose run the GIGN curve completes long before the
// DeepAtom one. That is the cost showing through, not a stall. _trajState()
// reports the queue depth, and a pose whose score has not landed yet reads
// "not scored" in that float rather than showing the previous pose's number.
//
// PER-SCORER FAILURE ISOLATION. Two consecutive failures disable that scorer and
// drain only its jobs. A missing arpeggio in the DeepAtom env must not stop GIGN
// from finishing its curve.
//
//   window._trajOn(key?) / _trajOff(key?)   arm / disarm (persisted per scorer)
//   window._trajMinDg(x)                    scoring gate; default Infinity
//   window._trajState()                     queue depth, per-scorer counters
//   window._trajDir()                       where the poses are being written
//
// Load after pose.js, the dock files, and pose_vina_record_stepper.js.
// =============================================================================

(function () {
    'use strict';
    if (window.PoseAffinityTrajectory) return;

    var MAX_QUEUE = 120;         // shared across scorers; a runaway search must not queue unboundedly

    // SCORE EVERYTHING BY DEFAULT.
    //
    // This was 0 — skip any pose whose Vina ΔG is positive — and the reasoning
    // still holds: a positive ΔG is net repulsion, atoms overlapping, and
    // neither model can say so. Both were trained on crystal-like complexes,
    // neither has a repulsion term, and both will return something in the 4–9 pK
    // band for a pose that is not a binding mode at all.
    //
    // It is still the wrong default, because the panel's job is to let you
    // compare predictors on the poses the search actually visited, and a gate
    // that silently drops the first two makes the two panels disagree with the
    // chart above them. An in-range number you can see next to a ΔG of +74 is
    // more useful than a gap: it shows you exactly how uninformative these
    // models are outside their training distribution.
    //
    // _trajMinDg(0) restores the old behaviour if you want the queue shorter.
    var maxDg = Infinity;

    var scorers = [];            // registration order == enqueue order per pose
    var queue = [], busy = false, nPoses = 0;
    var trajDir = '';            // pose.trajectory_dir, fetched once
    var wired = false;

    function $(id) { return document.getElementById(id); }
    function PGof() { return window.PoseGen; }
    function log(m, c) {
        var PG = PGof();
        if (PG && PG._miniLog) PG._miniLog(m, c || '#94a3b8');
    }
    function dockOf(S) { return S.dock ? window[S.dock] : null; }

    // ── Where the poses go ───────────────────────────────────────────────────
    // pose.trajectory_dir, served by /pose/default_receptor as `traj_dir`. The
    // panel's own output-dir box is the fallback, so a deployment that has not
    // set the new key keeps the old behaviour instead of silently writing
    // nowhere.
    function outRoot() {
        if (trajDir) return trajDir;
        var el = $('poseGignOutDir') || $('poseDaOutDir');
        return el ? (el.value || '').trim() : '';
    }
    (function fetchDir() {
        try {
            fetch('/pose/default_receptor')
                .then(function (r) { return r.ok ? r.json() : null; })
                .then(function (d) { if (d && d.traj_dir) trajDir = String(d.traj_dir).trim(); })
                .catch(function () { /* fallback covers it */ });
        } catch (e) { /* ditto */ }
    })();

    // ── Serialise an arbitrary engine conf without disturbing the view ───────
    // Mirrors PG.vina.eng._applyMode's conversion, minus the _draw/evaluate it
    // ends with. Going through _applyMode itself would move the on-screen ligand
    // to every pose and re-run the interaction scan, which is exactly the cost
    // _procPose is rate-limited to avoid.
    function pdbForConf(conf) {
        var PG = PGof(), E = window.VinaEngine;
        if (!PG || !E || !conf) return null;
        var eng = PG.vina && PG.vina.eng;
        if (!eng || typeof eng._build !== 'function') return null;
        var m;
        try { m = eng._build(); } catch (e) { return null; }
        if (!m) return null;
        var saved = PG.init ? PG.init.cur : null;
        try {
            var c = E.makeConf(m.tors.length);
            c.pos = conf.pos.slice(); c.q.set(conf.q); c.tors.set(conf.tors);
            var ui = E.confToUI(m, c, eng._box());
            // _ligWorld reads PG.init.cur, so borrow it for one synchronous call
            // and hand it straight back. Nothing renders in between.
            PG.init.cur = { off: ui.off, quat: ui.quat, tors: ui.tors, seed: 0 };
            var world = PG.vina._ligWorld();
            return (world && world.length) ? PG.mc._ligandPdb(world) : null;
        } catch (e) {
            return null;
        } finally {
            if (saved) PG.init.cur = saved;
        }
    }

    // pose.js keeps the receptor in a module-private `let protein` with no
    // accessor, so mirror the raw text off PG._applyProtein — the same
    // wrap-don't-fork trick ts_rl_loop.js uses for the same reason.
    var receptor = { raw: null, id: '' };
    function wrapProtein() {
        var PG = PGof();
        if (!PG || typeof PG._applyProtein !== 'function' || PG._applyProtein.__traj) return;
        var orig = PG._applyProtein.bind(PG);
        var w = function (pp, id, raw, useOwnCenter) {
            try { if (typeof raw === 'string' && raw.length) { receptor.raw = raw; receptor.id = id || ''; } }
            catch (_) { /* mirroring must never break the pose tool */ }
            return orig(pp, id, raw, useOwnCenter);
        };
        w.__traj = true;
        PG._applyProtein = w;
    }

    function smiles() {
        var el = $('poseSmiles');
        return el ? (el.value || '').trim() : '';
    }

    function tag(i, dg, kind) {
        // Sorts lexically in capture order and carries the ΔG that earned each
        // pose — so `ls` on the output dir reads as the staircase without
        // opening anything. `i` is the stepper's index, so the filename and the
        // panel's "N / M" name the same thing.
        var d = (dg == null) ? 'na' : dg.toFixed(3).replace('-', 'm').replace('.', 'p');
        return (kind === 'refined' ? 'fin' : 'rec') +
               String(i + 1).padStart(3, '0') + '_dg' + d;
    }

    // ── Queue ────────────────────────────────────────────────────────────────
    function anyBusy() {
        var PG = PGof();
        if (!PG || !PG.mc) return true;
        for (var i = 0; i < scorers.length; i++) {
            if (PG.mc[scorers[i].busyFlag]) return true;
        }
        return false;
    }

    function enqueue(job) {
        if (queue.length >= MAX_QUEUE) {
            // Say so rather than silently dropping: a truncated curve that looks
            // complete is worse than a short one that admits it.
            if (queue.length === MAX_QUEUE) {
                log('trajectory queue full at ' + MAX_QUEUE + ' — later poses will not be ' +
                    'scored. _trajOff() to stop, or _trajMinDg(0) to skip clashing poses.', '#fbbf24');
            }
            return;
        }
        queue.push(job);
        pump();
    }

    function pump() {
        var PG = PGof();
        if (busy || !queue.length || !PG || !PG.mc) return;
        if (anyBusy()) { setTimeout(pump, 800); return; }   // a hand-clicked score is running
        var job = queue.shift(), S = job.scorer;
        busy = true; PG.mc[S.busyFlag] = true;
        note(S);

        fetch(S.endpoint, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(job.body),
        })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (res) {
                busy = false; PG.mc[S.busyFlag] = false;
                if (res && res.ok) {
                    S.done++; S.fail = 0;          // consecutive failures, not cumulative
                    var pk = (res.pred_pk != null) ? res.pred_pk : null;
                    var dg = (res.deltaG != null) ? res.deltaG : (pk != null ? -pk * 1.36 : null);
                    // Tag first, THEN set. The dock records the pending index on
                    // the history entry it is about to push, which is what lets
                    // select(i) find this score again after the user steps away
                    // and back — and what keeps a hand-clicked score, which
                    // carries no index, from being mistaken for a pose.
                    var api = dockOf(S);
                    if (api && api.mark) api.mark(job.rec);
                    PG.mc[S.setter](S.label + ' ' + job.label + ' · vina ' +
                                    (job.dg != null ? job.dg.toFixed(2) : '—'), pk, dg);
                } else {
                    S.fail++;
                    var err = (res && res.err) || 'no response';
                    log(S.label + ' ' + job.label + ' failed: ' + err, '#fb7185');
                    if (res && res.log) log('📝 ' + res.log, '#a78bfa');
                    if (S.fail === 1 && res && res.stdout) {
                        log('── ' + S.label + ' output ──\n' + res.stdout.slice(-1200), '#94a3b8');
                    }
                    // One failure is almost always all of them (bad env, missing
                    // module). Drop THIS scorer's jobs and leave the others
                    // running — a broken DeepAtom env must not truncate the GIGN
                    // curve, which is the pair-wise comparison this whole module
                    // exists to produce.
                    if (S.fail >= 2) {
                        S.armed = false; write(S, false);
                        var before = queue.length;
                        queue = queue.filter(function (j) { return j.scorer !== S; });
                        log(S.label + ' trajectory scoring stopped after 2 failures (' +
                            (before - queue.length) + ' queued job(s) dropped). Other scorers ' +
                            'continue. Fix the error above, then _trajOn("' + S.key + '").', '#fbbf24');
                    }
                }
                note(S); pump();
            })
            .catch(function (e) {
                busy = false; PG.mc[S.busyFlag] = false; S.fail++;
                log(S.label + ' ' + job.label + ' request failed: ' + e, '#fb7185');
                note(S); pump();
            });
    }

    function note(S) {
        var s = $(S.subId);
        if (!s) return;
        var mine = queue.filter(function (j) { return j.scorer === S; }).length;
        if (mine || busy) {
            s.textContent = S.label + ' ' + S.done + '/' + nPoses +
                            (mine ? ' · ' + mine + ' queued' : ' · scoring…');
        }
    }

    // ── A new staircase pose arrived ─────────────────────────────────────────
    function onRecord(i, conf, dg, kind) {
        var armed = scorers.filter(function (s) { return s.armed; });
        var root = outRoot();

        if (dg != null && isFinite(maxDg) && dg >= maxDg) {
            log('pose ' + (i + 1) + ' skipped — vina ΔG ' + dg.toFixed(2) + ' ≥ ' + maxDg +
                '. _trajMinDg(Infinity) scores these too.', '#64748b');
            return;
        }
        var pdb = pdbForConf(conf);
        if (!pdb) { log('could not serialise pose ' + (i + 1) + ' — skipped', '#fbbf24'); return; }
        if (!receptor.raw) {
            if (i === 0) log('no receptor mirrored yet — load a target .pdb before the ' +
                             'search to score its poses.', '#fbbf24');
            return;
        }

        nPoses = Math.max(nPoses, i + 1);
        var label = tag(i, dg, kind);

        // Save the pose itself, independently of any scorer. A file write, not a
        // GPU job, so it does not go through the serial queue — and it means the
        // run leaves a complete set of staircase poses on disk even if every
        // scorer is disarmed or every score fails.
        if (root) {
            try {
                fetch('/pose/save_pdb', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pdb: pdb, out_dir: root, name: label }),
                }).catch(function () {});
            } catch (e) { /* never let bookkeeping break the search */ }
        }

        for (var k = 0; k < armed.length; k++) {
            var S = armed[k];
            enqueue({
                scorer: S, label: label, dg: dg, rec: i,
                body: {
                    ligand_pdb: pdb, receptor_pdb: receptor.raw,
                    name: label, smiles: smiles(),
                    // DeepAtom's script takes `-d <root>` and atom-types EVERY
                    // complex under <root>/Dataset_VS, so a shared root makes
                    // pose 12 re-process poses 1..11 — 78 complexes for a
                    // 12-pose run instead of 12, growing quadratically. One root
                    // per pose keeps each call at exactly one complex. GIGN
                    // already stages into its own <root>/gign_<name>/.
                    out_dir: root ? (S.perRecordDir ? (root.replace(/\/+$/, '') + '/' + label) : root) : '',
                },
            });
        }
    }

    // ── The stepper moved ────────────────────────────────────────────────────
    function onChange(i) {
        for (var k = 0; k < scorers.length; k++) {
            var api = dockOf(scorers[k]);
            if (api && api.select) api.select(i);
        }
    }

    function onReset() {
        queue = []; busy = false; nPoses = 0;
        scorers.forEach(function (s) {
            s.done = 0; s.fail = 0;
            // Drop each float's history too. Pose numbering restarts at 1, so a
            // surviving entry tagged rec 0 would be found by select(0) on the
            // NEW run's first pose — a plausible number for the wrong geometry,
            // with nothing on screen to give it away.
            var api = dockOf(s);
            if (api && api.reset) api.reset();
        });
    }

    // ── Wiring ───────────────────────────────────────────────────────────────
    function wire() {
        if (wired) return true;
        var R = window.PoseVinaRecords;
        if (!R || !R.onRecord) return false;
        wrapProtein();
        R.onRecord(onRecord);
        R.onChange(onChange);
        R.onReset(onReset);
        wired = true;
        return true;
    }

    var tries = 0;
    var t = setInterval(function () {
        wrapProtein();                       // cheap and idempotent; pose.js may load late
        if (wire() || ++tries > 60) {
            if (!wired && tries > 60) {
                console.warn('[affinity-traj] pose_vina_record_stepper.js never appeared — ' +
                             'no poses will be scored automatically.');
            }
            clearInterval(t);
        }
    }, 250);

    // ── Persistence ──────────────────────────────────────────────────────────
    function read(S) {
        try { var v = localStorage.getItem(S.store); return v === null ? true : v === '1'; }
        catch (e) { return true; }
    }
    function write(S, v) { try { localStorage.setItem(S.store, v ? '1' : '0'); } catch (e) {} }

    function find(key) {
        for (var i = 0; i < scorers.length; i++) if (scorers[i].key === key) return scorers[i];
        return null;
    }

    // ── Public ───────────────────────────────────────────────────────────────
    window.PoseAffinityTrajectory = {
        register: function (cfg) {
            if (find(cfg.key)) return find(cfg.key);
            var S = {
                key: cfg.key, label: cfg.label, endpoint: cfg.endpoint,
                setter: cfg.setter, busyFlag: cfg.busyFlag, subId: cfg.subId,
                store: cfg.store, perRecordDir: !!cfg.perRecordDir, dock: cfg.dock,
                done: 0, fail: 0, armed: false,
            };
            S.armed = read(S);
            scorers.push(S);
            wire();
            return S;
        },
        on: function (key) {
            scorers.forEach(function (s) {
                if (!key || s.key === key) { s.armed = true; s.fail = 0; write(s, true); }
            });
            pump();
            return 'armed: ' + scorers.filter(function (s) { return s.armed; })
                                      .map(function (s) { return s.key; }).join(', ');
        },
        off: function (key) {
            scorers.forEach(function (s) { if (!key || s.key === key) { s.armed = false; write(s, false); } });
            queue = queue.filter(function (j) { return j.scorer.armed; });
            return 'armed: ' + (scorers.filter(function (s) { return s.armed; })
                                       .map(function (s) { return s.key; }).join(', ') || '(none)');
        },
        minDg: function (v) {
            if (typeof v === 'number') maxDg = v;
            return 'scoring poses with vina ΔG < ' + maxDg;
        },
        dir: function (v) {
            if (typeof v === 'string') trajDir = v.trim();
            return outRoot();
        },
        state: function () {
            return {
                poses: nPoses, queued: queue.length, inFlight: busy,
                maxDg: maxDg, dir: outRoot(), wired: wired,
                scorers: scorers.map(function (s) {
                    return { key: s.key, armed: s.armed, scored: s.done, failStreak: s.fail,
                             queued: queue.filter(function (j) { return j.scorer === s; }).length };
                }),
            };
        },
    };

    window._trajOn    = function (k) { return window.PoseAffinityTrajectory.on(k); };
    window._trajOff   = function (k) { return window.PoseAffinityTrajectory.off(k); };
    window._trajMinDg = function (v) { return window.PoseAffinityTrajectory.minDg(v); };
    window._trajDir   = function (v) { return window.PoseAffinityTrajectory.dir(v); };
    window._trajState = function ()  { return window.PoseAffinityTrajectory.state(); };
})();