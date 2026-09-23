// =============================================================================
// pose_results_card.js — the "Vina Poses" sidebar card
// -----------------------------------------------------------------------------
// Adds ONE entry point, PoseGen.openResults(), which hub.js's data-nav="poses"
// case calls. It opens the ordinary pose workspace (#poseModal — the same single
// copy the Pose Generation card and the TS RL tab share), types the configured
// results folder into the Import box, and runs the scan, so the card lands on a
// real docked pose instead of on an empty form the user then has to fill in.
//
// Everything it drives already existed in pose.js:
//     PoseGen.open()            open the workspace (and take it back from the
//                               RL tab, via that file's own wrapper)
//     PoseGen.reactToggle(true) show the ⌬ React popover, which is where the
//                               Import-from-disk controls live
//     #poseImpPath              the Folder path box
//     PoseGen.importConfirm()   POST /pose/scan_pdbqt, filter, load file 1
//
// So this file is additive: delete it and the <script> tag in hub.html, drop the
// 'poses' case from hub.js and the card from the template, and the workspace
// behaves exactly as it did before — nothing else reads anything defined here.
//
// The folder itself is server-side configuration (pose.results_dir in
// config/input_routes.yml, served by GET /pose/results_dir), not a constant
// here, so repointing it at another run's output is a config edit and not a
// frontend change.
//
// Load AFTER pose.js: the three wrappers below need PG._setLigand, PG._impLoad
// and PG.build to exist. Order against the other pose_*.js addons is free —
// none of them touch these three, and PoseGen.open() is called by property
// lookup at click time, so whatever wrappers ts_rl.js added by then still run.
// =============================================================================

(function () {
    'use strict';

    var PG = window.PoseGen;
    if (!PG) { console.warn('[pose-results] pose.js not loaded — Vina Poses card disabled'); return; }

    var RESULTS_API   = '/pose/results_dir';
    var POLL_MS         = 60;     // how often the waiters below re-check
    var MOUNT_WAIT_MS   = 4000;   // cap on waiting for the popover to mount
    var BUILD_WAIT_MS   = 2500;   // cap on waiting for a start-up build to START
    var BUILD_SETTLE_MS = 15000;  // cap on waiting for one that started to FINISH

    var _dir  = null;            // cached {path, exists, source} — asked once per page
    var _busy = false;           // a click while a scan is in flight is a no-op

    function $(id) { return document.getElementById(id); }

    function status(msg, color) {
        try { PG._importStatus(msg, color); } catch (e) {}
    }

    /* Poll until read() returns something truthy, or give up after ms.
       Resolves with that value, or null on timeout. */
    function waitFor(read, ms) {
        return new Promise(function (resolve) {
            var got = null;
            try { got = read(); } catch (e) { got = null; }
            if (got) { resolve(got); return; }
            var t0 = Date.now();
            var iv = setInterval(function () {
                var v = null;
                try { v = read(); } catch (e) { v = null; }
                if (v || (Date.now() - t0) > ms) { clearInterval(iv); resolve(v || null); }
            }, POLL_MS);
        });
    }

    // ── Shadowing pose.js's private `L` ──────────────────────────────────────
    // PG._loadConfig() — which PG.open() calls on every open — builds the
    // default SMILES ligand, but ONLY when its own closure-private `L` is still
    // null, and that build is a POST that resolves whenever it resolves. Our
    // scan is also a POST (two, in fact: scan_pdbqt then read_pdbqt). Run them
    // concurrently and the slower one wins: on a fast scan the imported pose
    // renders first and is then silently replaced by the default molecule a
    // moment later, which reads as "the card opened the wrong ligand" and is
    // maddening to reproduce, because it is a race and usually the scan is the
    // slower of the two.
    //
    // `L` is not reachable from here, so instead we watch the only two functions
    // that ever assign it, and hold the scan until any start-up build is done.
    // Then the import lands last and stays.
    var _hasLigand = false;      // does the workspace already hold a ligand?
    var _build     = null;       // the in-flight build's promise, if any

    (function wrap() {
        var setLigand = PG._setLigand;
        if (typeof setLigand === 'function') {
            PG._setLigand = function (res) {
                if (res && res.ok) _hasLigand = true;          // a failed parse leaves L untouched
                return setLigand.apply(this, arguments);
            };
        }

        // Imported poses assign L directly rather than going through _setLigand,
        // and the same function also handles receptor files, which set no ligand
        // at all — hence the _poseSrc check rather than an unconditional true.
        var impLoad = PG._impLoad;
        if (typeof impLoad === 'function') {
            PG._impLoad = function () {
                return Promise.resolve(impLoad.apply(this, arguments)).then(function (r) {
                    if (PG._poseSrc === 'uploaded') _hasLigand = true;
                    return r;
                });
            };
        }

        // PG.build() returns its fetch chain; _loadConfig discards it. Keep it.
        var build = PG.build;
        if (typeof build === 'function') {
            PG.build = function () {
                var p = build.apply(this, arguments);
                _build = Promise.resolve(p).catch(function () {});   // never reject the wait
                return p;
            };
        }
    })();

    /* Resolve once the workspace's start-up build is finished — or once we know
       there will not be one. Two distinct cases, and guessing with a fixed sleep
       gets both wrong: a cold open must WAIT (the build is coming), a re-open
       must NOT (nothing is coming, and a 2.5 s stall before every scan is the
       card feeling broken). */
    function settleAutoBuild() {
        if (_hasLigand && !_build) return Promise.resolve();           // re-open: L is set, _loadConfig will skip its build
        return waitFor(function () { return _build; }, BUILD_WAIT_MS).then(function (p) {
            if (!p) return null;                                       // never started; stop waiting
            // Cap the wait on the build itself too. PG.build() resolves through
            // its own .catch, so a hang should be unreachable — but "should" is
            // carrying a lot of weight when the failure mode is a card that
            // opens, shows an empty Folder path and then silently never scans,
            // with no status line to say why. Time out and scan anyway.
            return Promise.race([p, new Promise(function (r) { setTimeout(r, BUILD_SETTLE_MS); })]);
        });
    }

    /* GET the configured folder once per page load. */
    function fetchDir() {
        if (_dir) return Promise.resolve(_dir);
        return fetch(RESULTS_API)
            .then(function (r) { return r.json(); })
            .then(function (res) {
                if (!res || !res.ok) {
                    return Promise.reject(new Error((res && res.err) || 'no results folder configured'));
                }
                _dir = res;
                return _dir;
            });
    }

    // ── The card ─────────────────────────────────────────────────────────────
    PG.openResults = function () {
        if (_busy) return Promise.resolve();
        _busy = true;
        var done = function () { _busy = false; };

        // Clear the build latch BEFORE opening, so settleAutoBuild() waits on
        // this open's build and not on a stale promise from a previous one.
        _build = null;

        try { PG.open(); } catch (e) {                  // property lookup: picks up pose.js's
            _busy = false;                              // _reactInject wrapper and ts_rl.js's
            console.warn('[pose-results] PoseGen.open failed:', e);   // release-from-the-RL-tab wrapper
            return Promise.resolve();
        }

        // The Import controls are injected into the ⌬ React popover by
        // PG._reactInject(), which PG.open() calls. Static markup in hub.html
        // means it is normally already mounted, but wait rather than assume.
        return waitFor(function () { return $('poseImpPath'); }, MOUNT_WAIT_MS)
            .then(function (inp) {
                if (!inp) throw new Error('the Import panel did not mount — is #poseBox3D in the page?');

                try { PG.reactToggle(true); } catch (e) {}
                // The popover is scrollable and Import sits below the fold.
                try { inp.scrollIntoView({ block: 'nearest' }); } catch (e) {}

                return fetchDir().then(function (d) {
                    // Always overwrite. The card's whole contract is "show me the
                    // results folder"; leaving a previously-typed path in place
                    // would make it open something else.
                    inp.value = d.path;
                    if (d.exists === false) {
                        // Not fatal — scan_pdbqt gives the authoritative error, and
                        // the path is now visible and editable either way.
                        status('⚠ ' + d.path + ' does not exist on the server — scanning anyway', '#fbbf24');
                    }
                    return settleAutoBuild().then(function () { return PG.importConfirm(); });
                });
            })
            .catch(function (e) {
                // Land the user somewhere they can act: workspace open, popover
                // up, Folder path focused, and the reason on the status line.
                try { PG.reactToggle(true); } catch (_e) {}
                var inp = $('poseImpPath');
                if (inp) { try { inp.focus(); } catch (_e) {} }
                status('✗ ' + (e && e.message ? e.message : 'could not open the results folder') +
                       ' — type a folder above and press Import', '#fb7185');
                console.warn('[pose-results]', e);
            })
            .then(done, done);
    };

    /* Small convenience for the console / other modules: what the card would
       open, without opening it. */
    PG.resultsDir = function () { return _dir ? _dir.path : null; };
})();
