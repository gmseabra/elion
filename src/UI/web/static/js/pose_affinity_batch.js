// =============================================================================
// pose_affinity_batch.js — the "Predict Affinity Batch" sidebar card
// -----------------------------------------------------------------------------
// One entry point, PoseGen.openAffinityBatch(), which hub.js's
// data-nav="affinity_batch" case calls.
//
// IT OPENS THE POSE GENERATION WORKSPACE. Not a variant of it, not a trimmed
// one: the same thing #sidebarPoseGenBtn opens, in the same stage, with the
// same sidebar, the same header and the same search objective. An earlier
// version of this card hid most of the right column and retitled the modal, and
// that was wrong — this card is not a different tool, it is the same tool with
// extra panels available.
//
// THE THREE PANELS
// ----------------
// #poseGignBatch  under #poseGignCard,  shown when the scorer is Yupu_GIGN
// #poseDaBatch    under #poseDaCard,    shown when the scorer is DeepAtom
// #poseGatorBatch under #poseGatorCard, shown when the scorer is GatorAffinity
//
// Each is the folder-wide version of the button on the card above it: score
// every .pdbqt under a folder with the SAME call that card's single-pose button
// makes, one file at a time. All three are built by makeBatch() below from a
// config of six ids and an endpoint — the scorers take an identical payload
// ({ligand_pdb, receptor_pdb, name, out_dir}) and answer with an identical
// shape ({ok, pred_pk, deltaG}), so there is one implementation, not three.
//
// Only one runs at a time (see _active): both shell out to a model subprocess,
// typically on the same GPU.
//
// Everything they drive already existed in pose.js:
//     PoseGen.open()          open the workspace (and take it back from the RL
//                             tab, via that file's own wrapper)
//     PG.mc.setScorer         the select that reveals each card
//     PG._applyProtein        where the loaded receptor's raw .pdb comes from
//
// WHERE THE RUN LIVES
// -------------------
// On the server. POST /pose/batch_start scans the folder and keeps the whole
// row set in the process; the browser holds a ten-row window — the last five
// finished and the next five queued — and nothing else. Before that, a folder
// of ~5000 poses meant 5000 rows rebuilt after every score: O(n²) DOM work that
// cost more wall-clock than the GPU did. Same shape as the TS tool's live view
// in ts/ts_worker_bridge.js, which keeps history server-side and trims the
// client, and for the same reason.
//
// The client still makes the scoring call itself, because every endpoint needs
// the receptor .pdb that is loaded in this browser and nowhere else. It just
// stopped being the thing that remembers.
//
// ⬇ .csv comes from GET /pose/batch_csv — the complete run, every row, not the
// ten on screen. The run records which scorer produced it, so the file is named
// gign_batch.csv, deepatom_batch.csv or gatoraffinity_batch.csv and its columns
// say which.
//
// So this file is additive: delete it and the <script> tag in hub.html, drop the
// 'affinity_batch' case from hub.js and the card from the template, and the
// workspace behaves exactly as it did before. Nothing else reads anything
// defined here, and every wrapper below calls through.
//
// The folder is server-side configuration (pose.results_dir in
// config/input_routes.yml, served by GET /pose/results_dir), not a constant
// here, so repointing it at another run's output is a config edit.
//
// Load AFTER pose.js: the wrappers need PG.open, PG._applyProtein,
// PG.mc.setScorer and PG.close to exist. Order against the other pose_*.js
// addons is free — none of them touch those — and PoseGen.open() is called by
// property lookup at click time, so whatever wrappers ts_rl.js added by then
// still run.
// =============================================================================

(function () {
    'use strict';

    var PG = window.PoseGen;
    if (!PG) { console.warn('[affinity-batch] pose.js not loaded — card disabled'); return; }

    var RESULTS_API = '/pose/results_dir';
    var READ_API    = '/pose/read_pdbqt';
    var START_API   = '/pose/batch_start';
    var RESULT_API  = '/pose/batch_result';
    var STOP_API    = '/pose/batch_stop';
    var RESUME_API  = '/pose/batch_resume';
    var RUNS_API    = '/pose/batch_runs';
    var CSV_API     = '/pose/batch_csv';
    var RETRIES     = 3;          // attempts per request before a run pauses
    var RETRY_MS    = 800;        // first backoff; doubles

    var _dir  = null;             // cached {path, exists, source} — asked once per page
    var _busy = false;

    function $(id) { return document.getElementById(id); }
    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    /* Two kinds of failure, and conflating them is what turns a blip into a
       ruined run:

       · TRANSPORT — the request never got an answer (server down, connection
         dropped, 500, unparseable body). Nothing is known about that pose. It
         is marked `.transport` so the caller retries, and if it still fails,
         PAUSES the run rather than recording a result it never received.
       · MODEL — a 200 with {ok:false}. That is an answer: this pose failed.
         Recorded as a failed row, and the run carries on.

       Before this, a server restart mid-run marked every remaining pose
       "failed" in a few seconds and called the batch done. */
    function postJson(url, body) {
        function transport(msg) { var e = new Error(msg); e.transport = true; return e; }
        return fetch(url, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }).then(function (r) {
            if (!r.ok) throw transport('the server answered ' + r.status);
            return r.json().then(null, function () { throw transport('unreadable reply from ' + url); });
        }, function () { throw transport('cannot reach the server'); });
    }

    /* Retry transport failures only, with a doubling backoff. A model failure
       is an answer and is never retried — re-running a GPU job that already
       told us it failed just costs minutes. */
    function postRetry(url, body, tries, wait) {
        tries = tries || RETRIES;
        wait  = wait  || RETRY_MS;
        return postJson(url, body).then(null, function (e) {
            if (!e || !e.transport || tries <= 1) throw e;
            return new Promise(function (r) { setTimeout(r, wait); })
                .then(function () { return postRetry(url, body, tries - 1, wait * 2); });
        });
    }

    function fmtDur(ms) {
        var s = Math.round(ms / 1000);
        return s < 60 ? (s + 's') : (Math.floor(s / 60) + 'm' + String(s % 60).padStart(2, '0') + 's');
    }

    // =========================================================================
    // Shadowing pose.js's private state
    // -------------------------------------------------------------------------
    // The receptor's raw .pdb text is closure-private to pose.js (`protein.raw`)
    // and both scoring endpoints require it. Rather than edit that file, wrap
    // the function that assigns it. Every wrapper calls through unchanged.
    // =========================================================================
    var _recPdb   = null;        // raw receptor .pdb text — what the scorers want
    var _recName  = '';
    var _fromCard = false;       // did the Predict Affinity Batch card open this?
    var _active   = null;        // the batch currently stepping through files

    (function wrap() {
        // Whose workspace is this? The panels are mounted into the ③ Monte-Carlo
        // sidebar and stay in the DOM once mounted, so without this flag the
        // plain Pose Generation card would grow a batch panel the moment you
        // picked a scorer — this module editing a tool it does not own. Every
        // opener clears it; openAffinityBatch sets it again after calling
        // through. Pose Generation stays byte-identical to what it was.
        var open = PG.open;
        if (typeof open === 'function') {
            PG.open = function () {
                _fromCard = false;
                var r = open.apply(this, arguments);
                try { reflectAll(); } catch (e) {}
                return r;
            };
        }

        // _applyProtein(parsed, name, rawPdbText, …) — arg 3 is the text both
        // scorers send as receptor_pdb. Cache it so a batch sends exactly the
        // same receptor the single-pose button would.
        var applyProtein = PG._applyProtein;
        if (typeof applyProtein === 'function') {
            PG._applyProtein = function (parsed, name, raw) {
                if (typeof raw === 'string' && raw) { _recPdb = raw; _recName = name || 'receptor'; }
                try { syncRecAll(); } catch (e) {}
                return applyProtein.apply(this, arguments);
            };
        }

        // Each panel is shown and hidden by the same condition its card is:
        // pose.js's own setScorer sets #poseGignCard / #poseDaCard display, and
        // this mirrors it onto the panel beneath. Nothing else in pose.js
        // touches either element's visibility, so there is no second code path.
        var setScorer = PG.mc && PG.mc.setScorer;
        if (typeof setScorer === 'function') {
            PG.mc.setScorer = function (v) {
                var r = setScorer.apply(this, arguments);
                try { reflectAll(v); } catch (e) {}
                return r;
            };
        }

        // Leaving the workspace stops a run that is still stepping through files.
        var close = PG.close;
        if (typeof close === 'function') {
            PG.close = function () {
                try { if (_active) _active.stop(); } catch (e) {}
                return close.apply(this, arguments);
            };
        }
    })();

    function fetchDir() {
        if (_dir) return Promise.resolve(_dir);
        return fetch(RESULTS_API).then(function (r) { return r.json(); }).then(function (res) {
            if (!res || !res.ok) {
                return Promise.reject(new Error((res && res.err) || 'no results folder configured'));
            }
            _dir = res;
            return _dir;
        });
    }

    // =========================================================================
    // Do the poses and the receptor even occupy the same space?
    // -------------------------------------------------------------------------
    // Both scorers cut a pocket around the ligand. If the poses were docked
    // against a different structure — or the same protein in another crystal
    // frame, which is what a Glide grid or a re-deposited PDB gives you — the
    // cut finds nothing and the run fails on every file with an error that
    // names neither cause:
    //
    //     pocket: CUT to 0 residues / 0 atoms within 5.0 A (drop_water=True)
    //     RuntimeError: RDKit could not parse pocket/receptor PDB: …_pocket.pdb
    //
    // RDKit is not the problem; it was handed an empty file. Checking the first
    // pose costs one read and turns 67,000 identical failures into one sentence
    // before anything is scored.
    // =========================================================================
    var _recXYZ = null, _recXYZFor = null;

    function receptorAtoms() {
        if (_recXYZ && _recXYZFor === _recPdb) return _recXYZ;
        var out = [], lines = String(_recPdb || '').split('\n');
        for (var i = 0; i < lines.length; i++) {
            var h = lines[i].slice(0, 6);
            if (h !== 'ATOM  ' && h !== 'HETATM') continue;
            var x = parseFloat(lines[i].slice(30, 38)),
                y = parseFloat(lines[i].slice(38, 46)),
                z = parseFloat(lines[i].slice(46, 54));
            if (isFinite(x) && isFinite(y) && isFinite(z)) out.push(x, y, z);
        }
        _recXYZ = out; _recXYZFor = _recPdb;
        return out;
    }

    /* Returns a message when the ligand is nowhere near the receptor, else ''.
       The 8 Å probe is deliberately looser than GIGN's 5 Å pocket: this decides
       whether to REFUSE, so it should only fire when there is no doubt. */
    function frameMismatch(ligPdb, ligName, scorerName) {
        var rec = receptorAtoms();
        if (!rec.length) return '';                       // no receptor parsed: not our call
        var lig = [], lines = ligPdb.split('\n');
        for (var i = 0; i < lines.length; i++) {
            if (lines[i].slice(0, 6) !== 'HETATM') continue;
            lig.push(parseFloat(lines[i].slice(30, 38)),
                     parseFloat(lines[i].slice(38, 46)),
                     parseFloat(lines[i].slice(46, 54)));
        }
        if (!lig.length) return '';
        var PROBE = 8.0, P2 = PROBE * PROBE, best = Infinity;
        for (var a = 0; a < lig.length; a += 3) {
            for (var b = 0; b < rec.length; b += 3) {
                var dx = lig[a] - rec[b], dy = lig[a + 1] - rec[b + 1], dz = lig[a + 2] - rec[b + 2];
                var d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < P2) return '';                   // they touch — nothing to say
                if (d2 < best) best = d2;
            }
        }
        var mid = function (arr) {
            var s = [0, 0, 0], n = arr.length / 3;
            for (var k = 0; k < arr.length; k += 3) { s[0] += arr[k]; s[1] += arr[k + 1]; s[2] += arr[k + 2]; }
            return s.map(function (v) { return (v / n).toFixed(1); }).join(', ');
        };
        return 'the poses are not in the same coordinate frame as the loaded receptor — ' +
               'no atom of ' + ligName + ' comes within ' + PROBE + ' Å of any of the ' +
               (rec.length / 3) + ' atoms in ' + (_recName || 'the receptor') +
               ' (closest approach ' + Math.sqrt(best).toFixed(1) + ' Å; ligand centre ' +
               mid(lig) + ', receptor centre ' + mid(rec) + '). ' +
               scorerName + ' would cut a 0-atom pocket and fail on every file. ' +
               'Load the receptor these poses were docked against, or re-dock them against this one.';
    }

    /* One ligand .pdb in EXACTLY the shape PG.mc._ligandPdb produces, built from
       a file on disk instead of from the in-memory pose.

       Matching that format is the whole point. Both single-pose buttons hand
       their script HETATM / resName LIG / chain A / resSeq 1 records; a batch
       that fed them the raw ATOM/UNL records straight out of the .pdbqt would be
       running a different input through the same model, and its numbers would
       not be comparable with the ones the button produces.

       Atom selection mirrors ligandFromPDB: HETATM if the file has any, else
       every ATOM. Only the first MODEL is read — Vina writes its best pose
       first, and /pose/read_pdbqt already returns just that one. */
    function ligandPdbFromText(text) {
        var lines = String(text || '').split('\n');
        var het = [], atom = [];
        for (var i = 0; i < lines.length; i++) {
            var ln = lines[i], head = ln.slice(0, 6);
            if (head.trim() === 'ENDMDL') break;                   // best pose only
            if (head === 'HETATM') het.push(ln);
            else if (head === 'ATOM  ') atom.push(ln);
        }
        var src = het.length ? het : atom;
        var pad = function (s, n) { return String(s).padStart(n); };
        var out = '', n = 0;
        for (var k = 0; k < src.length; k++) {
            var l = src[k];
            var x = parseFloat(l.slice(30, 38)), y = parseFloat(l.slice(38, 46)), z = parseFloat(l.slice(46, 54));
            if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
            var el = (l.slice(76, 78) || '').trim();
            if (!el) el = (l.slice(12, 16).replace(/[^A-Za-z]/g, '') || 'C').slice(0, 2);
            el = el.charAt(0).toUpperCase() + el.slice(1).toLowerCase();
            n++;
            var name = ((el.length === 1 ? ' ' + el : el) + '  ').slice(0, 4);
            out += 'HETATM' + pad(n, 5) + ' ' + name + ' LIG A' + pad(1, 4) + '    ' +
                   pad(x.toFixed(3), 8) + pad(y.toFixed(3), 8) + pad(z.toFixed(3), 8) +
                   '  1.00  0.00          ' + pad(el.toUpperCase(), 2) + '\n';
        }
        return n ? (out + 'END\n') : '';
    }

    // =========================================================================
    // makeBatch — one folder-batch panel
    // -------------------------------------------------------------------------
    // cfg: { key, card, scorerVal, api, outDirId, setter, label, name, blurb }
    // Element ids all derive from `key`: #pose<Key>Batch / Path / Run / Stop /
    // Csv / Progress / List / Rec.
    // =========================================================================
    var VIO = { bg: '#140d24', line: '#3b2a6b', text: '#c4b5fd', dim: '#7c6bae', deep: '#5b4b8a' };

    function makeBatch(cfg) {
        var ID = {
            panel: cfg.key + 'Batch', path: cfg.key + 'BatchPath', run: cfg.key + 'BatchRun',
            stop: cfg.key + 'BatchStop', csv: cfg.key + 'BatchCsv', resume: cfg.key + 'BatchResume',
            prog: cfg.key + 'BatchProgress', list: cfg.key + 'BatchList', rec: cfg.key + 'BatchRec',
            jdir: cfg.key + 'BatchJournalDir'
        };

        /* Everything the client remembers about a run. Ten rows and six numbers
           — the row set itself is the server's, fetched a window at a time. */
        function fresh() {
            return {
                running: false, stop: false, runId: null, dir: '', t0: 0,
                total: 0, done: 0, ok: 0, fail: 0, notRun: 0,
                view: { recent: [], queued: [], hidden_done: 0, hidden_queued: 0 },
                best: null, cur: null, msg: 'idle', color: '#64748b',
                diagShown: false, failStreak: 0, recNoteShown: false, frameChecked: false
            };
        }
        var B = fresh();
        var _resumable = null;        // {run_id, done, total, not_run} on the server, if any

        function mount() {
            var card = $(cfg.card);
            if (!card || $(ID.panel)) return;

            var inputCss = 'width:100%;box-sizing:border-box;font-family:ui-monospace,monospace;font-size:10.5px;' +
                           'background:#0a0716;border:1px solid ' + VIO.line + ';border-radius:6px;padding:4px 7px;' +
                           'color:' + VIO.text + ';outline:none;';
            var btnCss   = 'padding:6px 12px;border-radius:9px;border:1px solid #6d28d9;background:rgba(124,58,237,.16);' +
                           'color:' + VIO.text + ';font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;';
            var subBtn   = 'padding:4px 10px;border-radius:8px;border:1px solid ' + VIO.line + ';background:rgba(11,17,32,.7);' +
                           'color:#94a3b8;font-size:10px;font-weight:600;cursor:pointer;font-family:ui-monospace,monospace;';
            var resumeCss = 'padding:6px 12px;border-radius:9px;border:1px solid #047857;background:rgba(5,150,105,.16);' +
                            'color:#6ee7b7;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;';

            var el = document.createElement('div');
            el.id = ID.panel;
            el.style.cssText = 'display:none;flex-direction:column;gap:0;background:' + VIO.bg +
                               ';border:1px solid ' + VIO.line + ';border-radius:13px;padding:13px;';
            el.innerHTML =
                '<p style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#a78bfa;margin:0 0 8px;">' +
                  '⚛ ' + esc(cfg.name) + ' batch <span style="color:' + VIO.deep + ';font-weight:400;letter-spacing:0;text-transform:none;font-size:9px;">· every .pdbqt in a folder</span>' +
                '</p>' +

                '<div style="display:flex;flex-direction:column;gap:4px;padding:0 0 9px;margin-bottom:5px;border-bottom:1px solid rgba(59,42,107,.5);">' +
                  '<label for="' + ID.path + '" style="font-size:9.5px;color:' + VIO.dim + ';font-family:ui-monospace,monospace;letter-spacing:.03em;">' +
                    'folder of docked poses · scanned recursively for <b style="color:' + VIO.text + ';">.pdbqt</b></label>' +
                  '<input id="' + ID.path + '" type="text" spellcheck="false" autocomplete="off" ' +
                    'placeholder="/&#8230;/vina_poses/results" ' +
                    'title="Every .pdbqt / .pdb / .ent under this folder is scored, subfolders included. Prepared receptors (more than 300 atoms) are skipped — this scores ligand poses." ' +
                    'style="' + inputCss + '" ' +
                    'onfocus="this.style.borderColor=\'#7c5cff\'" onblur="this.style.borderColor=\'' + VIO.line + '\'"></div>' +

                '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">' +
                  '<button id="' + ID.run + '" title="Runs the same call ' + esc(cfg.label) + ' makes — best pose (MODEL 1) of each file, against the loaded receptor — once per file, in order. Serial on purpose: each score is its own server subprocess." style="' + btnCss + '">⚛ Score all with ' + esc(cfg.name) + '</button>' +
                  '<button id="' + ID.resume + '" title="Continue the interrupted run of this folder. Everything already scored is kept — it picks up at the first pose with no result." style="' + resumeCss + 'display:none;">▶ Resume</button>' +
                  '<button id="' + ID.stop + '" title="Stop after the file being scored right now. The running subprocess is not killed, and the run stays resumable." style="' + subBtn + 'display:none;">■ Stop</button>' +
                  '<button id="' + ID.csv + '" title="Download every finished row as .csv — rank, file, the Vina affinity from the file\'s own REMARK line, predicted pK and ΔG. The whole run, not the rows on screen." style="' + subBtn + 'margin-left:auto;display:none;">⬇ .csv</button>' +
                '</div>' +

                '<div id="' + ID.prog + '" style="margin-top:7px;font-size:9.5px;color:#64748b;font-family:ui-monospace,monospace;line-height:1.5;min-height:13px;">idle</div>' +
                '<div id="' + ID.list + '" class="pose-scroll" style="margin-top:7px;max-height:236px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:' + VIO.line + ' transparent;"></div>' +

                '<p style="font-size:10px;color:' + VIO.deep + ';font-family:ui-monospace,monospace;line-height:1.5;margin:8px 0 0;">' +
                  'Live view only — the <b>last 5 finished</b> and the <b>next 5 queued</b>. The full run is kept on the server; ' +
                  '<b style="color:#94a3b8;">⬇ .csv</b> downloads all of it, ranked by <b>pK</b> highest first ' +
                  '(<b>higher pK = stronger</b>, the opposite sign convention to the Vina column beside it). ' +
                  esc(cfg.blurb || '') + ' ' +
                  'Scored against whichever target is loaded (<b id="' + ID.rec + '" style="color:#94a3b8;">—</b>); staging goes to the ' +
                  '<b style="color:#94a3b8;">server output dir</b> above, one subdirectory per pose.' +
                  '<br>Progress is journalled to <b id="' + ID.jdir + '" style="color:#94a3b8;" title="pose.batch_dir. Every run keeps three files here — the scan, and one appended line per finished pose — which is what ▶ Resume reads.">—</b>' +
                  ' so a closed tab or a restarted server does not lose it.' +
                '</p>';

            card.parentNode.insertBefore(el, card.nextSibling);

            $(ID.run).onclick    = run;
            $(ID.stop).onclick   = stop;
            $(ID.csv).onclick    = csv;
            $(ID.resume).onclick = resume;
            $(ID.path).addEventListener('keydown', function (e) {
                if (e.key === 'Enter') { e.preventDefault(); run(); }
            });
            // Point the box at a different folder and the offer must follow it —
            // resuming a run of some other folder because the button was still
            // on screen is the one way this feature could do damage.
            $(ID.path).addEventListener('change', checkResumable);
            syncPath();
            syncRec();
            reflect();
            checkResumable();
        }

        // ── resuming ────────────────────────────────────────────────────────
        /* Ask the server whether there is an unfinished run to offer, and offer
           it. Cheap: one small GET, on mount, on open, when the folder box
           changes and after a run ends — not on a timer.

           With a folder in the box the question is "is THIS folder unfinished".
           With the box empty it is the broader "is anything unfinished", and
           the answer fills the box in. That case is the one that matters: a
           reload wipes the box, and seeding it from config first would point the
           panel at the default folder and hide the 4,300-pose run you were
           actually in the middle of. An interrupted run beats a default. */
        function checkResumable() {
            var b = $(ID.path);
            var folder = b ? b.value.trim() : '';
            var q = RUNS_API + '?scorer=' + encodeURIComponent(cfg.scorer);
            if (folder) q += '&dir=' + encodeURIComponent(folder);
            return fetch(q)
                .then(function (r) { return r.json(); })
                .then(function (res) {
                    // Where the journal actually is. The panel says it out loud
                    // because the answer depends on server config and, before
                    // this, on TMPDIR — which is not something you should have
                    // to go and find out when a run goes missing.
                    var je = $(ID.jdir);
                    if (je && res && res.dir) { je.textContent = res.dir; }
                    var runs = (res && res.runs) || [];
                    var hit = null;
                    for (var i = 0; i < runs.length; i++) {
                        if (runs[i].resumable) { hit = runs[i]; break; }   // newest first
                    }
                    if (hit && !folder && b && hit.dir) b.value = hit.dir;
                    showResume(hit);
                    return hit;
                })
                .then(null, function () { showResume(null); return null; });
        }

        function ago(unixSeconds) {
            var s = Math.max(0, Math.round(Date.now() / 1000 - (unixSeconds || 0)));
            if (s < 90) return 'just now';
            if (s < 5400) return Math.round(s / 60) + ' min ago';
            if (s < 172800) return Math.round(s / 3600) + ' h ago';
            return Math.round(s / 86400) + ' days ago';
        }

        /* The offer is always the NEWEST unfinished run of this folder and
           scorer — which is usually the one the panel is showing, but after a
           completed run it may be an older one that was stopped or interrupted
           and never picked up. That is worth offering, and it is worth being
           unambiguous about: the title carries the id and the age, so a stale
           1 / 4 from an hour ago cannot be mistaken for the run just finished. */
        function showResume(hit) {
            _resumable = hit;
            var btn = $(ID.resume); if (!btn) return;
            if (!hit || B.running) { btn.style.display = 'none'; return; }
            btn.style.display = 'inline-block';
            btn.textContent = '▶ Resume ' + hit.done + ' / ' + hit.total;
            btn.title = 'Continue an unfinished run of this folder — ' + hit.not_run +
                        ' pose(s) still to score, started ' + ago(hit.created) + ' (' + hit.run_id + ').' +
                        ' Everything already scored is kept; it picks up at the first pose with no result.';
        }

        function resume() {
            if (B.running || !_resumable) return Promise.resolve();
            if (_active && _active.busy()) {
                progress('the ' + _active.name + ' batch is still running — stop it first', '#fbbf24');
                return Promise.resolve();
            }
            // The same guard ⚛ Score all has, and it matters more here: the
            // usual way to reach Resume is a page reload, which throws the
            // loaded receptor away. Without this, resuming a minute too early
            // would post a null receptor and mark every remaining pose failed.
            if (!_recPdb) {
                progress('load a target protein (.pdb) first — ' + cfg.name +
                         ' scores a complex, not a ligand', '#fb7185');
                return Promise.resolve();
            }
            var rid = _resumable.run_id;
            var runBtn = $(ID.run), stopBtn = $(ID.stop), csvBtn = $(ID.csv), resBtn = $(ID.resume);

            B = fresh();
            B.running = true; B.runId = rid; B.t0 = Date.now();
            _active = api;
            if (runBtn) { runBtn.disabled = true; runBtn.textContent = '⏳ scoring…'; }
            if (stopBtn) stopBtn.style.display = 'inline-block';
            if (resBtn) resBtn.style.display = 'none';
            if (csvBtn) csvBtn.style.display = 'none';
            progress('resuming ' + esc(rid) + '…', '#94a3b8');

            var outDir = (($(cfg.outDirId) || {}).value || '').trim();
            return postRetry(RESUME_API, { run_id: rid })
                .then(function (res) {
                    if (!res || !res.ok) throw new Error((res && res.err) || 'could not resume');
                    B.dir = res.dir || B.dir;
                    var next = absorb(res);
                    progress('resumed · ' + B.done + ' of ' + B.total + ' already scored' +
                             (res.requeued ? (' · ' + res.requeued + ' requeued from mid-flight') : ''),
                             '#94a3b8');
                    return step(next, outDir);
                })
                .then(null, function (e) { fail(e); })
                .then(function () { settle(); });
        }

        /* Show the panel exactly when pose.js is showing the card above it — and
           only while the Predict Affinity Batch card owns the workspace. */
        function reflect(v) {
            var el = $(ID.panel); if (!el) return;
            if (v === undefined) v = ($('poseMcScorer') || {}).value;
            el.style.display = (_fromCard && v === cfg.scorerVal) ? 'flex' : 'none';
        }

        /* Seed the folder box: whatever the Import box points at, else config. */
        function syncPath() {
            var b = $(ID.path); if (!b || b.value.trim()) return;
            var imp = $('poseImpPath');
            var v = (imp && imp.value.trim()) || (_dir && _dir.path) || '';
            if (v) b.value = v;
        }

        function syncRec() {
            var e = $(ID.rec); if (!e) return;
            e.textContent = _recName || '— none loaded —';
        }

        // ── the ten-row live view ────────────────────────────────────────────
        // Never more than twelve nodes, and rebuilt on a rAF so a fast run
        // cannot queue renders faster than the browser can paint them — the
        // same throttle ts_worker_bridge.js puts in front of its bar redraws.
        var _renderQueued = false;
        function queueRender() {
            if (_renderQueued) return;
            _renderQueued = true;
            requestAnimationFrame(function () { _renderQueued = false; render(); });
        }

        function rowHtml(r, running) {
            var right;
            if (r.state === 'ok') {
                right = '<span style="color:' + VIO.text + ';font-weight:700;">pK ' +
                        (r.pk != null ? r.pk.toFixed(2) : '—') + '</span>' +
                        '<span style="color:#64748b;"> · ΔG ' + (r.dg != null ? r.dg.toFixed(2) : '—') + '</span>';
            } else if (running || r.state === 'running') {
                right = '<span style="color:#a78bfa;">⏳ scoring…</span>';
            } else if (r.state === 'fail') {
                right = '<span style="color:#fb7185;" title="' + esc(r.err) + '">✗ failed</span>';
            } else {
                right = '<span style="color:#475569;">queued</span>';
            }
            var vina = (typeof r.vina === 'number' && isFinite(r.vina))
                ? '<span style="color:#38bdf8;" title="best Vina affinity, from the file\'s own REMARK line">vina ' + r.vina.toFixed(1) + '</span>'
                : '<span style="color:#334155;">vina —</span>';
            return '<div title="' + esc(r.path) + '" style="display:flex;align-items:center;gap:6px;' +
                   'font-family:ui-monospace,monospace;font-size:9.5px;padding:3px 0;' +
                   'border-bottom:1px solid rgba(59,42,107,.4);">' +
                     '<span style="color:#94a3b8;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
                       esc(r.name) + '</span>' + vina +
                     '<span style="flex-shrink:0;">' + right + '</span>' +
                   '</div>';
        }

        function ellipsisHtml(text) {
            return '<div style="font-family:ui-monospace,monospace;font-size:9px;color:' + VIO.deep +
                   ';padding:3px 0;text-align:center;letter-spacing:.04em;">' + esc(text) + '</div>';
        }

        function render() {
            var prog = $(ID.prog);
            if (prog) { prog.innerHTML = B.msg; prog.style.color = B.color; }
            var host = $(ID.list);
            if (!host) return;

            var v = B.view || {}, h = '';
            var rec = v.recent || [], q = v.queued || [];
            if (v.hidden_done)   h += ellipsisHtml('⋯ ' + v.hidden_done + ' earlier · in the .csv');
            for (var i = 0; i < rec.length; i++) h += rowHtml(rec[i]);
            for (var j = 0; j < q.length; j++)   h += rowHtml(q[j], B.running && j === 0 && !B.stop);
            if (v.hidden_queued) h += ellipsisHtml('⋯ ' + v.hidden_queued + ' more queued');
            host.innerHTML = h;
        }

        function progress(msg, color) {
            B.msg = msg;
            B.color = color || '#64748b';
            queueRender();
        }

        /* Fold one server reply — counts, best, window, next — into local state. */
        function absorb(res) {
            var c = res.counts || {};
            B.total  = c.total   != null ? c.total   : B.total;
            B.done   = c.done    != null ? c.done    : B.done;
            B.ok     = c.ok      != null ? c.ok      : B.ok;
            B.fail   = c.fail    != null ? c.fail    : B.fail;
            B.notRun = c.not_run != null ? c.not_run : B.notRun;
            if (res.view) B.view = res.view;
            if (res.best !== undefined) B.best = res.best;
            return res.next || null;
        }

        function stop() {
            if (!B.running) return;
            B.stop = true;
            progress('stopping after the current file…', '#fbbf24');
        }

        /* The scoring endpoints answer a failure with everything you need to
           chase it — the exact command, the conda env, the cwd, a debug log
           path and the script's stdout. The single-pose buttons put all of that
           in the activity panel; the batch used to keep only `err`, so "✗
           failed" on 67,000 rows told you nothing. Show the FIRST failure in
           full, once per run: the second one is almost never a different story,
           and 67,000 of them is the uncapped log all over again. */
        function diagnose(row, res) {
            if (B.diagShown) return;
            B.diagShown = true;
            var say = function (m, c, mono) { try { PG._miniLog(m, c || '#94a3b8', mono); } catch (e) {} };
            say('── ' + cfg.name + ' batch · first failure: ' + row.name + ' ──', '#fb7185');
            if (res && res.err)        say('error:  ' + res.err, '#fb7185');
            if (res && res.cmd)        say('▶ command:  ' + res.cmd, '#7dd3fc', true);
            if (res && res.conda_env)  say('🅒 conda env:  ' + res.conda_env, '#7dd3fc', true);
            if (res && res.cwd)        say('📂 cwd:  ' + res.cwd, '#7dd3fc', true);
            if (res && res.log)        say('📝 debug log → ' + res.log, '#a78bfa', true);
            if (res && res.produced)   say('produced:  ' + (res.produced.atomtypes || []).length +
                                           ' .atomtypes, ' + (res.produced.npz || []).length + ' .npz', '#94a3b8', true);
            if (res && res.stdout)     say('── script output (tail) ──\n' +
                                           String(res.stdout).slice(-4000), '#94a3b8', true);
            say('Later failures are not logged — each row carries its reason in its tooltip, ' +
                'and every reason is in the ⬇ .csv.', '#64748b');
        }

        function finish() {
            var el = Date.now() - B.t0;
            var best = B.best;
            progress((B.stop ? '■ stopped' : '✓ done') +
                     ' · ' + B.ok + ' scored, ' + B.fail + ' failed, ' +
                     B.notRun + ' not run · ' + fmtDur(el) +
                     (best && best.pk != null
                        ? ('<br>best <b style="color:' + VIO.text + ';">' + esc(best.name) +
                           '</b> · pK ' + best.pk.toFixed(2))
                        : ''),
                     B.stop ? '#fbbf24' : '#34d399');
            try {
                PG._miniLog(cfg.name + ' batch ' + (B.stop ? 'stopped' : 'finished') + ' · ' +
                            B.ok + ' scored, ' + B.fail + ' failed of ' + B.total,
                            B.stop ? '#fbbf24' : '#34d399');
            } catch (e) {}
        }

        /* A run that ended because the world went away, not because it finished.
           The distinction the message has to carry: nothing was lost, and the
           button below will pick it up. */
        function fail(e) {
            var msg = (e && e.message) ? e.message : 'batch failed';
            progress('✗ ' + esc(msg) +
                     (B.runId ? '<br>' + B.done + ' of ' + B.total +
                                ' scored and saved — press ▶ Resume to carry on' : ''),
                     '#fb7185');
        }

        /* The one exit path, whether the run finished, stopped, or broke. */
        function settle() {
            B.running = false;
            if (_active === api) _active = null;
            var runBtn = $(ID.run), stopBtn = $(ID.stop), csvBtn = $(ID.csv);
            if (runBtn) { runBtn.disabled = false; runBtn.textContent = '⚛ Score all with ' + cfg.name; }
            if (stopBtn) stopBtn.style.display = 'none';
            if (csvBtn && B.runId && B.done) csvBtn.style.display = 'inline-block';
            queueRender();
            checkResumable();          // anything left unfinished is offered back
        }

        function run() {
            if (B.running) return;
            // One at a time across BOTH panels: these shell out to model
            // subprocesses, typically on the same GPU. Two concurrent folder
            // runs is how a batch becomes an out-of-memory report.
            if (_active && _active.busy()) {
                progress('the ' + _active.name + ' batch is still running — stop it first', '#fbbf24');
                return;
            }

            var pathEl = $(ID.path);
            var folder = pathEl ? pathEl.value.trim() : '';
            if (!folder) {
                progress('enter a folder above', '#fb7185');
                if (pathEl) pathEl.focus();
                return;
            }
            if (!_recPdb) {
                progress('load a target protein (.pdb) first — ' + cfg.name +
                         ' scores a complex, not a ligand', '#fb7185');
                return;
            }

            var runBtn = $(ID.run), stopBtn = $(ID.stop), csvBtn = $(ID.csv), resBtn = $(ID.resume);
            B = fresh();
            B.running = true; B.dir = folder; B.t0 = Date.now();
            _active = api;
            if (runBtn) { runBtn.disabled = true; runBtn.textContent = '⏳ scoring…'; }
            if (stopBtn) stopBtn.style.display = 'inline-block';
            if (resBtn) resBtn.style.display = 'none';
            if (csvBtn) csvBtn.style.display = 'none';
            progress('scanning ' + esc(folder) + '… (no file limit — a big folder takes a moment)', '#94a3b8');

            var outDir = (($(cfg.outDirId) || {}).value || '').trim();

            return postRetry(START_API, { path: folder, scorer: cfg.scorer })
                .then(function (res) {
                    if (!res || !res.ok) throw new Error((res && res.err) || 'scan failed');
                    B.runId = res.run_id;
                    B.dir   = res.dir || folder;
                    var next = absorb(res);
                    progress('scanning ' + esc(B.dir) + ' → ' + res.total + ' pose file(s)' +
                             (res.skipped ? (' · ' + res.skipped + ' receptor(s) skipped') : '') +
                             (res.resumable === false ? ' · not journalled, so this run will not survive a restart' : ''),
                             res.resumable === false ? '#fbbf24' : '#94a3b8');
                    return step(next, outDir);
                })
                .then(null, function (e) { fail(e); })
                .then(function () { settle(); });
        }

        /* The output dir this pose is scored in.

           DeepAtom needs its OWN root per pose and it is not a preference.
           `/pose/deepatom_score` runs the pipeline as `-d <root>`, and that
           script is a virtual-screening driver: it atom-types, voxelises and
           predicts EVERY compound under <root>/Dataset_VS, not the one just
           written. Point every pose at the card's single output dir and pose N
           re-processes 1..N-1 — n²/2 work — and, worse, the route can only
           attribute a score when there is exactly one compound or the CSV id
           matches the name, so a root that already holds earlier compounds
           makes the parse fail and EVERY pose come back "failed". That is the
           `0 scored, 2 failed of 2` on an output dir that had been used before.
           pose_routes.py logs "Use one root per pose" when it sees it, and the
           trajectory modules already do exactly this.

           The row index prefixes the folder so it is unique even if two files
           in different subfolders share a stem, stays stable across a resume
           (the index is the server's row id), and sorts in run order.

           GIGN needs none of this: /pose/gign_score stages into
           <root>/gign_<name>/ and parses its own stdout, so one shared root is
           correct there — which is exactly why GIGN worked and DeepAtom did
           not. A blank out_dir is already per-call (the route makes a temp
           dir), so it is left alone. */
        function poseOutDir(base, row, stem) {
            if (!cfg.perPoseOutDir || !base) return base;
            var safe = String(stem).replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 60);
            return base.replace(/\/+$/, '') + '/' + row.i + '_' + safe;
        }

        /* One file at a time, on purpose: the scoring endpoint shells out to a
           model subprocess (and typically a GPU). Firing N of those concurrently
           is how a batch turns into an out-of-memory report instead of a
           ranking.

           Recursion is on the server's `next`, not on an index into a local
           array: the client never holds the file list. Written as a loop over a
           promise rather than a self-call so a 5000-file run cannot build a
           5000-deep chain of pending .then frames. */
        function step(first, outDir) {
            var next = first;

            function once(row) {
                B.cur = row;
                progress('scoring ' + (B.done + 1) + ' / ' + B.total + ' · ' + esc(row.name) +
                         ' · ' + fmtDur(Date.now() - B.t0) + ' elapsed', '#a78bfa');
                queueRender();

                var stem = row.name.replace(/\.(pdbqt|pdb|ent)$/i, '');
                return postRetry(READ_API, { path: row.path })
                    .then(function (res) {
                        if (!res || !res.ok || !res.pdb) throw new Error((res && res.err) || 'could not read the file');
                        var lig = ligandPdbFromText(res.pdb);
                        if (!lig) throw new Error('no atom records in the file');
                        // Once per run, on the first pose only: refuse a folder
                        // that cannot possibly score against this receptor,
                        // before spending hours proving it one file at a time.
                        // The row is left un-posted, so the run stays exactly
                        // where it is — fix the receptor and press ▶ Resume.
                        if (!B.frameChecked) {
                            B.frameChecked = true;
                            var bad = frameMismatch(lig, row.name, cfg.name);
                            if (bad) { var fe = new Error(bad); fe.fatal = true; throw fe; }
                        }
                        return postRetry(cfg.api, {
                            ligand_pdb: lig,
                            receptor_pdb: _recPdb,
                            name: stem,
                            smiles: '',
                            out_dir: poseOutDir(outDir, row, stem)
                        });
                    })
                    .then(function (res) {
                        if (!res || !res.ok) { diagnose(row, res); throw new Error((res && res.err) || (cfg.name + ' failed')); }
                        var pk = (res.pred_pk != null) ? +res.pred_pk : null;
                        var dg = (res.deltaG != null) ? +res.deltaG : (pk != null ? -pk * 1.36 : null);
                        if (pk == null && dg == null) throw new Error('no prediction in the model output');
                        // Feed the single-pose readout so the card and its
                        // floating panel track the batch. Empty status on
                        // purpose: the setter's status line appends a node to
                        // the activity mini-chat, which is uncapped — 5000 of
                        // those is its own slow leak. One summary line is
                        // logged when the run ends.
                        try { PG.mc[cfg.setter]('', pk, dg); } catch (e) {}
                        // A repaired receptor is worth saying once. Silent
                        // repair is how you end up trusting a structure that
                        // has two atoms called H in the same residue.
                        if (res.receptor_note && !B.recNoteShown) {
                            B.recNoteShown = true;
                            try {
                                PG._miniLog('⚠ ' + res.receptor_note + ' — the complex handed to ' +
                                            cfg.name + ' was repaired so it parses. Worth checking the ' +
                                            'uploaded receptor.', '#fbbf24');
                            } catch (e) {}
                        }
                        return { pk: pk, dg: dg, err: '' };
                    })
                    .then(null, function (e) {
                        // A transport failure that outlived its retries is not a
                        // verdict on this pose — the request never landed. Let it
                        // out of the loop so the run PAUSES here, resumable, rather
                        // than marking every remaining pose "failed" at network
                        // speed and calling the batch complete. `fatal` is the
                        // pre-flight refusal and travels the same road.
                        if (e && (e.transport || e.fatal)) throw e;
                        return { pk: null, dg: null, err: (e && e.message) ? e.message : 'failed' };
                    })
                    .then(function (out) {
                        // A folder where every pose fails is a setup problem, not
                        // 67,000 individual problems. Say so once rather than let
                        // it run for hours producing nothing.
                        if (out.err) {
                            B.failStreak++;
                            if (B.failStreak === 5) {
                                try {
                                    PG._miniLog('⚠ the first 5 poses of this ' + cfg.name +
                                                ' batch all failed — this looks like a setup problem, ' +
                                                'not the poses. The first failure is logged above; ' +
                                                '■ Stop, fix it, then ▶ Resume — nothing scored is lost.',
                                                '#fbbf24');
                                } catch (e) {}
                            }
                        } else {
                            B.failStreak = 0;
                        }
                        return postRetry(RESULT_API, {
                            run_id: B.runId, i: row.i, pk: out.pk, dg: out.dg, err: out.err
                        });
                    })
                    .then(function (res) {
                        if (!res || !res.ok) throw new Error((res && res.err) || 'the run was lost on the server');
                        next = absorb(res);
                        queueRender();
                    });
            }

            function loop() {
                if (!next || B.stop) {
                    if (!B.stop) { finish(); return Promise.resolve(); }
                    return postJson(STOP_API, { run_id: B.runId })
                        .then(function (res) { if (res && res.ok) absorb(res); })
                        .catch(function () {})
                        .then(finish);
                }
                var row = next;
                next = null;
                return once(row).then(loop);
            }

            return loop();
        }

        function csv() {
            if (!B.runId) return;
            var a = document.createElement('a');
            a.href = CSV_API + '?run_id=' + encodeURIComponent(B.runId);
            a.download = cfg.scorer + '_batch.csv';
            document.body.appendChild(a); a.click();
            setTimeout(function () { a.remove(); }, 0);
        }

        var api = {
            name: cfg.name, key: cfg.key, scorer: cfg.scorer, scorerVal: cfg.scorerVal,
            mount: mount, reflect: reflect, syncRec: syncRec, syncPath: syncPath,
            run: run, stop: stop, csv: csv, resume: resume,
            check: checkResumable,
            resumable: function () { return _resumable; },
            busy: function () { return !!B.running; },
            runId: function () { return B.runId; },
            /* the ten rows on screen, in display order */
            view: function () {
                var v = B.view || {};
                return { recent: (v.recent || []).slice(), queued: (v.queued || []).slice(),
                         hidden_done: v.hidden_done || 0, hidden_queued: v.hidden_queued || 0,
                         total: B.total, done: B.done, ok: B.ok, fail: B.fail,
                         best: B.best, runId: B.runId };
            },
            /* the whole run, from the server — not what the panel holds */
            all: function () {
                if (!B.runId) return Promise.resolve([]);
                return fetch('/pose/batch_state?rows=1&run_id=' + encodeURIComponent(B.runId))
                    .then(function (r) { return r.json(); })
                    .then(function (res) { return (res && res.rows) || []; });
            }
        };
        return api;
    }

    // =========================================================================
    // The three panels
    // -------------------------------------------------------------------------
    // /pose/gign_score, /pose/deepatom_score and /pose/gatoraffinity_score take
    // the same payload ({ligand_pdb, receptor_pdb, name, out_dir}) and answer
    // with the same shape ({ok, pred_pk, deltaG, elapsed}) — the three configs
    // below are the whole difference between them. Adding a fourth scorer is
    // this object and nothing else in this file.
    // =========================================================================
    var BATCHES = [
        makeBatch({
            key: 'poseGign', card: 'poseGignCard', scorerVal: 'yupu_gign',
            api: '/pose/gign_score', outDirId: 'poseGignOutDir', setter: '_gignSet',
            scorer: 'gign', name: 'GIGN', label: '⚛ Score (GIGN)',
            blurb: 'The server cuts the 5 Å training pocket per pose, as the single-pose button does.'
        }),
        makeBatch({
            key: 'poseDa', card: 'poseDaCard', scorerVal: 'deepatom',
            api: '/pose/deepatom_score', outDirId: 'poseDaOutDir', setter: '_daSet',
            scorer: 'deepatom', name: 'DeepAtom', label: '⚛ Score best pose',
            perPoseOutDir: true,
            blurb: 'Each pose runs the full prep → grid → inference pipeline in its own ' +
                   'numbered subfolder of the output dir — the pipeline scores everything under the root ' +
                   'it is handed, so one shared root would rescore earlier poses and break the score ' +
                   'attribution. Budget seconds–minutes per file.'
        }),
        makeBatch({
            key: 'poseGator', card: 'poseGatorCard', scorerVal: 'gatoraffinity',
            api: '/pose/gatoraffinity_score', outDirId: 'poseGatorOutDir', setter: '_gatorSet',
            scorer: 'gatoraffinity', name: 'GatorAffinity', label: '⚛ Score (GatorAffinity)',
            // GIGN's setting, not DeepAtom's: gatoraffinity_pose.py reads the one
            // complex named on its command line, so a shared root neither
            // rescores earlier poses nor confuses the attribution.
            blurb: 'The server cuts the 5 Å pocket per pose and runs the ATOMICA block pipeline, ' +
                   'as the single-pose button does. Between GIGN and DeepAtom on cost — a graph ' +
                   'model, but with a fragmentation step and the ATOMICA backbone in front of it.'
        })
    ];

    function eachBatch(fn) { for (var i = 0; i < BATCHES.length; i++) { try { fn(BATCHES[i]); } catch (e) {} } }
    function reflectAll(v) { eachBatch(function (b) { b.reflect(v); }); }
    function syncRecAll()  { eachBatch(function (b) { b.syncRec(); }); }

    /* The one visible right now, else the first. What the flat PG.affinityBatch
       helpers below act on, so a console call does what the panel on screen
       would do. */
    function current() {
        var v = ($('poseMcScorer') || {}).value;
        for (var i = 0; i < BATCHES.length; i++) if (BATCHES[i].scorerVal === v) return BATCHES[i];
        return BATCHES[0];
    }

    var CSS = BATCHES.map(function (b) {
        return '#' + b.key + 'BatchList::-webkit-scrollbar { width: 7px; }\n' +
               '#' + b.key + 'BatchList::-webkit-scrollbar-thumb { background: #3b2a6b; border-radius: 4px; }';
    }).join('\n');

    function ensureCss() {
        if ($('poseAffinityBatchCss')) return;
        var st = document.createElement('style');
        st.id = 'poseAffinityBatchCss';
        st.textContent = CSS;
        document.head.appendChild(st);
    }

    // =========================================================================
    // The card
    // -------------------------------------------------------------------------
    // The same workspace #sidebarPoseGenBtn opens. PG.open() and nothing else —
    // no stage change, no sidebar surgery, no retitle, no objective switch. The
    // panels are mounted (idempotently) and shown by the scorer, like the cards
    // above them.
    // =========================================================================
    PG.openAffinityBatch = function () {
        if (_busy) return Promise.resolve();
        _busy = true;

        try { PG.open(); } catch (e) {                  // property lookup: picks up pose.js's
            _busy = false;                              // _reactInject wrapper and ts_rl.js's
            console.warn('[affinity-batch] PoseGen.open failed:', e);   // release-from-the-RL-tab wrapper
            return Promise.resolve();
        }

        _fromCard = true;
        ensureCss();
        eachBatch(function (b) { b.mount(); });
        reflectAll();
        syncRecAll();

        // Order matters. Ask about interrupted runs FIRST: if one exists it puts
        // its own folder in the box, which is how a user who closed the tab
        // yesterday opens the card and finds their 4,312 scored poses waiting
        // behind a ▶ Resume button. Only a panel with nothing to resume falls
        // back to the configured results folder.
        //
        // Only the panels' own boxes — the Import box belongs to the workspace
        // and is left exactly as Pose Generation leaves it.
        return Promise.all(BATCHES.map(function (b) {
                return b.check().then(null, function () { return null; });
            }))
            .then(function () { return fetchDir(); })
            .then(function (d) {
                eachBatch(function (b) {
                    var el = $(b.key + 'BatchPath');
                    if (el && !el.value.trim()) el.value = d.path;
                });
            })
            .catch(function (e) {
                console.warn('[affinity-batch]', e);
            })
            .then(function () { _busy = false; }, function () { _busy = false; });
    };

    /* Small conveniences for the console and for tests. The named handles are
       the stable API; the flat ones act on whichever panel the scorer is
       showing. */
    PG.resultsDir = function () { return _dir ? _dir.path : null; };
    PG.affinityBatch = {
        gign: BATCHES[0], deepatom: BATCHES[1], gator: BATCHES[2], all_: BATCHES,
        current: current,
        running: function () { return _active ? _active.name : null; },
        run:    function () { return current().run(); },
        stop:   function () { return current().stop(); },
        csv:    function () { return current().csv(); },
        resume: function () { return current().resume(); },
        check:  function () { return current().check(); },
        runId:  function () { return current().runId(); },
        view:   function () { return current().view(); },
        all:    function () { return current().all(); },
        /* every journalled run on the server, newest first */
        runs:   function () {
            return fetch(RUNS_API + '?all=1').then(function (r) { return r.json(); })
                .then(function (res) { return (res && res.runs) || []; });
        }
    };
})();
