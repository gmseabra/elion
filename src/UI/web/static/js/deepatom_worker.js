// =============================================================================
// deepatom_worker.js — Web Worker for DeepAtom log parsing
// Runs off the main thread. Receives raw SSE lines, emits structured updates.
//
// Messages TO worker:   { type: 'line', data: '<raw log line>' }
//                       { type: 'reset' }
//
// Messages FROM worker: { type: 'progress',  current, total, name }
//                       { type: 'phase',     phase: 'init'|'preprocessing'|'grid'|'inference'|'complete' }
//                       { type: 'compound',  id, pred_pk }
//                       { type: 'summary',   best, mean, worst, count, top5: [{id,pred_pk}] }
//                       { type: 'error',     text }
//                       { type: 'config',    key, value }
//                       { type: 'log',       html }
//                       { type: 'status',    text }
//                       { type: 'done' }
//                       { type: 'debug',     msg }
// =============================================================================

// ── Parser state ──────────────────────────────────────────────────────────────
let _phase            = 'init';
let _complexTotal     = 0;
let _complexCurrent   = 0;
let _inTraceback      = false;
let _tracebackLines   = [];
let _inArgDict        = false;
let _compounds        = [];   // [{id, pred_pk}] accumulated from [deepatom] lines
let _inSummary        = false;

// ── Phase emit ────────────────────────────────────────────────────────────────
function setPhase(phase) {
    if (_phase === phase) return;
    _phase = phase;
    self.postMessage({ type: 'phase', phase });
    self.postMessage({ type: 'debug', msg: '[worker] phase → ' + phase });
}

// ── Log helpers ───────────────────────────────────────────────────────────────
function logHtml(html) {
    self.postMessage({ type: 'log', html });
}
function status(text) {
    self.postMessage({ type: 'status', text });
}

// ── Parse one line ────────────────────────────────────────────────────────────
function parseLine(rawLine) {
    const line = rawLine.trim();
    if (!line) return;

    // ── [deepatom] prefixed lines (our own injected messages) ────────────────
    if (line.startsWith('[deepatom]')) {
        const body = line.slice('[deepatom]'.length).trim();

        // Section separators — skip silently
        if (/^[=─\-]{10,}$/.test(body)) return;

        // Run started
        const startM = body.match(/^Run started\s+(.+)/);
        if (startM) {
            setPhase('init');
            logHtml(
                `<span style="color:#f59e0b;font-weight:600">⚛️ Run started</span> ` +
                `<span style="color:#475569">${startM[1]}</span>`
            );
            status('DeepAtom started — ' + startM[1]);
            return;
        }

        // Config lines: test_type, data_dir, script
        const cfgM = body.match(/^(test_type|data_dir|script)\s+(.+)/);
        if (cfgM) {
            self.postMessage({ type: 'config', key: cfgM[1].trim(), value: cfgM[2].trim() });
            const keyCol = cfgM[1] === 'test_type' ? '#67e8f9'
                         : cfgM[1] === 'data_dir'  ? '#6ee7b7'
                         :                           '#94a3b8';
            logHtml(
                `<span style="color:#334155">  ${cfgM[1].padEnd(12)}</span>` +
                `<span style="color:${keyCol}">${cfgM[2].trim()}</span>`
            );
            return;
        }

        // Script finished
        const finM = body.match(/^Script finished.*exit code\s*(\d+)/i);
        if (finM) {
            const ok = finM[1] === '0';
            logHtml(
                `<span style="color:${ok ? '#fbbf24' : '#f87171'};font-weight:600">` +
                `Script finished — exit code ${finM[1]}</span>`
            );
            status('Script finished — exit code ' + finM[1]);
            if (ok) setPhase('complete');
            return;
        }

        // Stdout lines captured
        const capM = body.match(/^Stdout lines captured:\s*(\d+)/i);
        if (capM) {
            logHtml(`<span style="color:#64748b">  Stdout lines captured: ${capM[1]}</span>`);
            return;
        }

        // Parsed N compounds
        const parsedM = body.match(/^Parsed\s+(\d+)\s+compounds?/i);
        if (parsedM) {
            _inSummary = true;
            logHtml(
                `<span style="color:#fbbf24;font-weight:600">⚛️ Parsed ${parsedM[1]} compounds</span>`
            );
            return;
        }

        // Summary stats: best/mean/worst pK
        if (_inSummary) {
            const statM = body.match(/^\s*(best|mean|worst)\s+pK\s*=\s*([\d.eE+\-]+)/i);
            if (statM) {
                const col = statM[1] === 'best' ? '#fbbf24'
                          : statM[1] === 'mean' ? '#34d399'
                          :                       '#94a3b8';
                logHtml(
                    `<span style="color:#334155">  ${statM[1].padEnd(6)} pK = </span>` +
                    `<span style="color:${col};font-weight:600">${statM[2]}</span>`
                );
                return;
            }

            // Top N hits header
            const topHdrM = body.match(/^Top\s+(\d+)\s+hits?/i);
            if (topHdrM) {
                logHtml(`<span style="color:#f59e0b;font-weight:600">Top ${topHdrM[1]} hits:</span>`);
                return;
            }

            // Individual hit: #N  ID   pred=X
            const hitM = body.match(/^#\s*(\d+)\s+(\S+)\s+pred=([\d.eE+\-]+)/);
            if (hitM) {
                const rank = parseInt(hitM[1]);
                const col  = rank === 1 ? '#fbbf24' : rank <= 3 ? '#f59e0b' : '#6ee7b7';
                logHtml(
                    `<span style="color:#334155">  #${hitM[1].padStart(2)} </span>` +
                    `<span style="color:${col};font-family:monospace">${hitM[2].padEnd(30)}</span>` +
                    `<span style="color:#94a3b8">pred=</span>` +
                    `<span style="color:${col};font-weight:600">${hitM[3]}</span>`
                );
                // Emit compound
                const cmp = { id: hitM[2], pred_pk: parseFloat(hitM[3]) };
                _compounds.push(cmp);
                self.postMessage({ type: 'compound', id: cmp.id, pred_pk: cmp.pred_pk });
                return;
            }
        }

        // Top-line fallback: show other [deepatom] lines as-is
        logHtml(`<span style="color:#64748b">${body}</span>`);
        return;
    }

    // ── Traceback detection ───────────────────────────────────────────────────
    if (line.startsWith('Traceback (most recent call last):')) {
        _inTraceback   = true;
        _tracebackLines = [line];
        logHtml(`<span style="color:#f87171;font-weight:600">⚠ Traceback detected</span>`);
        return;
    }
    if (_inTraceback) {
        _tracebackLines.push(line);
        // End of traceback = line starting with error type (e.g. TypeError:, OSError:)
        if (/^[A-Za-z][A-Za-z0-9_]*Error[:\s]/.test(line) ||
            /^[A-Za-z][A-Za-z0-9_]*Exception[:\s]/.test(line) ||
            /^SyntaxError[:\s]/.test(line)) {
            _inTraceback = false;
            self.postMessage({ type: 'error', text: line });
            logHtml(`<span style="color:#f87171">  ${escHtml(line)}</span>`);
            return;
        }
        // Show file/line refs in dim colour
        if (line.startsWith('  File ')) {
            logHtml(`<span style="color:#334155">  ${escHtml(line)}</span>`);
        }
        return;
    }

    // ── SyntaxError (no traceback preamble) ──────────────────────────────────
    const synM = line.match(/^SyntaxError:\s+(.+)/);
    if (synM) {
        self.postMessage({ type: 'error', text: line });
        logHtml(`<span style="color:#f87171;font-weight:600">⚠ SyntaxError: ${escHtml(synM[1])}</span>`);
        return;
    }

    // ── TypeError / OSError inline ────────────────────────────────────────────
    const errM = line.match(/^(TypeError|OSError|RuntimeError|ValueError)[:\s](.+)/);
    if (errM) {
        self.postMessage({ type: 'error', text: line });
        logHtml(`<span style="color:#f87171">⚠ ${escHtml(errM[1])}: ${escHtml(errM[2])}</span>`);
        return;
    }

    // ── [N/M] OK -- filename.pdb  (pipeline_VS.py augmented file progress) ──────
    const augProgressM = line.match(/^\[(\d+)\/(\d+)\]\s+OK\s+--\s+(\S+)/);
    if (augProgressM) {
        const cur  = parseInt(augProgressM[1]);
        const tot  = parseInt(augProgressM[2]);
        const name = augProgressM[3];
        self.postMessage({ type: 'progress', current: cur, total: tot, name });
        // Show every 10th line to avoid flooding; always show last
        if (cur % 10 === 0 || cur === tot) {
            logHtml(
                `<span style="color:#475569">[${cur}/${tot}]</span> ` +
                `<span style="color:#64748b">OK -- </span>` +
                `<span style="color:#94a3b8;font-family:monospace">${name}</span>`
            );
        }
        self.postMessage({ type: 'status', text: `[${cur}/${tot}] ${name}` });
        return;
    }

    // ── Complex progress: "BM-2-31:   complex 9 (out of 15)" ─────────────────
    const complexM = line.match(/^(\S+):\s+complex\s+(\d+)\s+\(out of\s+(\d+)\)/i);
    if (complexM) {
        _complexCurrent = parseInt(complexM[2]);
        _complexTotal   = parseInt(complexM[3]);
        const name      = complexM[1];
        setPhase('preprocessing');
        self.postMessage({ type: 'progress', current: _complexCurrent, total: _complexTotal, name });
        logHtml(
            `<span style="color:#0e7490">complex</span> ` +
            `<span style="color:#67e8f9;font-weight:600;font-family:monospace">${name}</span> ` +
            `<span style="color:#334155">${_complexCurrent}/${_complexTotal}</span>`
        );
        status(`Processing complex ${name} — ${_complexCurrent}/${_complexTotal}`);
        return;
    }

    // ── === separator ─────────────────────────────────────────────────────────
    if (/^={20,}$/.test(line)) {
        logHtml(`<span style="color:#1e3a5f">${line.slice(0,40)}</span>`);
        return;
    }

    // ── "All the augmented data: N" ───────────────────────────────────────────
    const augM = line.match(/^All the augmented data:\s*(\d+)/i);
    if (augM) {
        setPhase('grid');
        logHtml(
            `<span style="color:#34d399">All augmented data: ` +
            `<span style="font-weight:600">${augM[1]}</span></span>`
        );
        status('Grid generation — ' + augM[1] + ' augmented complexes');
        return;
    }

    // ── "The pdb complexes which need to be generated: N" ────────────────────
    const needM = line.match(/^The pdb complexes which need to be generated:\s*(\d+)/i);
    if (needM) {
        setPhase('grid');
        logHtml(
            `<span style="color:#6ee7b7">Grid generation: ` +
            `<span style="font-weight:600">${needM[1]}</span> complexes</span>`
        );
        status('Generating 3D grids for ' + needM[1] + ' complexes…');
        return;
    }

    // ── Inference stage: "Process Id: N" ─────────────────────────────────────
    const pidM = line.match(/^Process Id:\s*(\d+)/);
    if (pidM) {
        setPhase('inference');
        logHtml(`<span style="color:#475569">inference PID: ${pidM[1]}</span>`);
        status('CNN inference running — PID ' + pidM[1]);
        return;
    }

    // ── Runtime summary ───────────────────────────────────────────────────────
    const rtM = line.match(/^Runtime\s+\((?:Batch|Total)\)\s*=\s*(.+)/i);
    if (rtM) {
        logHtml(`<span style="color:#64748b">Runtime: ${rtM[1]}</span>`);
        return;
    }

    // ── args_dict line — skip (pure noise, very long Python dict) ─────────────
    if (line.startsWith('args_dict:')) return;
    if (line.startsWith('args.test_type:')) {
        const val = line.split(':').slice(1).join(':').trim();
        logHtml(`<span style="color:#475569">test_type: <span style="color:#67e8f9">${escHtml(val)}</span></span>`);
        return;
    }

    // ── debug6 lines — show very dimly (they carry useful path info) ──────────
    if (line.includes('-----------------debug6')) {
        // Only show os.getcwd and cmplx_pdb lines; skip the rest
        if (line.includes('cmplx_pdb:') || line.includes('os.getcwd():')) {
            const val = line.split(':').slice(1).join(':').trim();
            logHtml(`<span style="color:#334155;font-size:9px;font-family:monospace">${escHtml(val)}</span>`);
        }
        return;
    }

    // ── Generic config key: value lines (test_type, batch_size, etc.) ─────────
    const kvM = line.match(/^([a-z_]+):\s+(.{1,80})$/);
    if (kvM && !line.startsWith('Traceback')) {
        logHtml(
            `<span style="color:#475569">  ${escHtml(kvM[1])}: </span>` +
            `<span style="color:#64748b">${escHtml(kvM[2])}</span>`
        );
        return;
    }

    // ── Default: show non-empty, non-debug lines up to 200 chars ─────────────
    if (line.length > 0 && line.length < 200) {
        logHtml(`<span style="color:#64748b">${escHtml(line)}</span>`);
    }
}

// ── Tiny HTML escaper (no DOM access in worker) ───────────────────────────────
function escHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Message handler ───────────────────────────────────────────────────────────
console.log('[deepatom_worker.js] worker started, version:', Date.now());

self.onmessage = (evt) => {
    const msg = evt.data;

    if (msg.type === 'reset') {
        _phase          = 'init';
        _complexTotal   = 0;
        _complexCurrent = 0;
        _inTraceback    = false;
        _tracebackLines = [];
        _inArgDict      = false;
        _compounds      = [];
        _inSummary      = false;
        return;
    }

    if (msg.type === 'line') {
        parseLine(msg.data);
    }
};