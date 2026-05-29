// =============================================================================
// ts.js — Thompson Sampling modal controller (main thread only)
// All log parsing + Welford math runs in ts_worker.js (Web Worker).
// This file: UI, SSE stream, Worker bridge, renders.
// =============================================================================

/* ─── State ──────────────────────────────────────────────────────────────── */
console.log('[ts.js] loaded, version:', Date.now());
const _ts = {
    speed: 3, jobId: null, wuTimer: null, tsTimer: null,
    es: null, worker: null, running: false, activeTab: 'warmup',
    wuState: null, tsState: null,
    // Reagent map maintained from worker messages (main thread copy for rendering)
    reagents: {},
    _barRaf: null, _pendingBarText: '',
    _top5Cache: null,   // {ids: Set, min: number} — recomputed when dirty
};

function _tsInitWu() { return { reagents:{}, evals:0, best:0, phase:'—', events:[], idx:0 }; }
function _tsInitTs() { return { reagents:{}, winners:[], events:[], idx:0, masked:0 }; }

/* ─── Speed ──────────────────────────────────────────────────────────────── */
function _tsWuInterval() { return [1200,800,500,300,150][_ts.speed-1]; }
function _tsTsInterval() { return [1400,950,600,350,180][_ts.speed-1]; }
function _tsSetSpeed(v) {
    _ts.speed = +v;
    if (_ts.wuTimer) { clearInterval(_ts.wuTimer); _ts.wuTimer = setInterval(_tsWuStep, _tsWuInterval()); }
    if (_ts.tsTimer) { clearInterval(_ts.tsTimer); _ts.tsTimer = setInterval(_tsTsStep, _tsTsInterval()); }
}

/* ─── Tab switching ──────────────────────────────────────────────────────── */
// Active colours per tab: warmup=violet, ts=cyan, results=emerald
const _TS_TAB_ACTIVE = {
    warmup:  'bg-violet-900 text-violet-200',
    ts:      'bg-cyan-900   text-cyan-200',
    results: 'bg-emerald-900 text-emerald-200',
};
const _TS_TAB_INACTIVE = 'bg-transparent text-slate-400 hover:bg-slate-800 hover:text-white';
const _TS_PANE = { warmup:'tsPaneWarmup', ts:'tsPaneTs', results:'tsPaneResults' };

function _tsTab(tab) {
    _ts.activeTab = tab;
    // Update all tab buttons
    document.querySelectorAll('[data-tab]').forEach(btn => {
        const t = btn.dataset.tab;
        if (t === tab) {
            btn.className = `px-4 py-1.5 ${t !== 'results' ? 'border-r border-slate-700 ' : ''}${_TS_TAB_ACTIVE[t]}`;
        } else {
            btn.className = `px-4 py-1.5 ${t !== 'results' ? 'border-r border-slate-700 ' : ''}${_TS_TAB_INACTIVE}`;
        }
    });
    // Show/hide panes
    Object.entries(_TS_PANE).forEach(([k, id]) => {
        const pane = document.getElementById(id);
        if (!pane) return;
        if (k === tab) pane.classList.remove('hidden');
        else           pane.classList.add('hidden');
    });
}

/* ─── Show / hide ────────────────────────────────────────────────────────── */
function _tsShow() {
    const modal=document.getElementById('tsModal');
    if (modal) modal.classList.remove('hidden');
    const h=document.querySelector('[data-drag-handle="miniChat"] span');
    if (h) h.textContent='Elion · Thompson Sampling';
    if (typeof _miniChatShow==='function') _miniChatShow();
    _tsTab('warmup');
    _tsLoadConfig();
    // Welcome message
    setTimeout(() => {
        const mo=document.getElementById('miniChatOutput');
        if (mo && mo.innerHTML.trim()==='') {
            if (typeof _miniChatAppend==='function') {
                _miniChatAppend('ai',
                    '🎲 **Thompson Sampling** generator ready.\n\n' +
                    'Configure your reaction SMARTS and iterations, then click **Run TS** to start.\n\n' +
                    '**Warmup** phase scores random building block pairs to form initial beliefs.\n' +
                    '**TS Belief** phase iteratively selects the most promising combinations using Thompson Sampling.');
            }
        }
    }, 200);
}

function _tsLoadConfig() {
    fetch('/vina_visualization/ts_config')
        .then(r=>r.json())
        .then(d=>{
            if (d.status!=='ok') return;
            const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
            set('tsCfgLogLevel', d.log_level||'—');
            set('tsCfgMode',     d.ts_mode||'—');
            set('tsCfgIter',     d.iterations!==undefined?d.iterations.toLocaleString():'—');
            set('tsCfgWarmup',   d.warmup!==undefined?d.warmup:'—');
            set('tsCfgBatch',    d.batch!==undefined?d.batch:'—');
            set('tsCfgSmarts',   d.smarts||'—');
            set('tsCfgResults',  d.results||'—');
            const reagEl=document.getElementById('tsCfgReagents');
            if (reagEl && Array.isArray(d.reagents)) {
                reagEl.innerHTML=d.reagents.map(r=>{
                    // Show just the filename, full path on hover
                    const fname=r.split('/').pop();
                    return `<div title="${r}" style="color:#475569;cursor:default">· ${fname}</div>`;
                }).join('');
            }
        })
        .catch(()=>{});
}
function _tsHide() {
    const modal=document.getElementById('tsModal');
    if (modal) modal.classList.add('hidden');
}

/* ─── Log buffer (DOM writes batched via rAF) ────────────────────────────── */
const _tsLogBuf=[], _TS_LOG_MAX=150;
let _tsLogRaf=null;
function _tsWuLog(html) {
    _tsLogBuf.push(html);
    if (!_tsLogRaf) _tsLogRaf=requestAnimationFrame(_tsFlushLog);
}
function _tsFlushLog() {
    _tsLogRaf=null;
    if (!_tsLogBuf.length) return;
    const el=document.getElementById('tsWuLog');
    if (!el) { _tsLogBuf.length=0; return; }
    const frag=document.createDocumentFragment();
    _tsLogBuf.splice(0).forEach(html => { const d=document.createElement('div'); d.innerHTML=html; frag.appendChild(d); });
    el.appendChild(frag);
    while (el.children.length>_TS_LOG_MAX) el.removeChild(el.firstChild);
    el.scrollTop=el.scrollHeight;
}

/* ─── Top-5 helper (main thread reagent map) ─────────────────────────────── */
function _tsGetTop5() {
    const entries=Object.entries(_ts.reagents).sort((a,b)=>b[1].best-a[1].best).slice(0,5);
    return new Set(entries.map(([id])=>id));
}

/* ─── Bar renderers ──────────────────────────────────────────────────────── */
function _tsRenderWuBars() {
    const sorted=Object.entries(_ts.reagents).sort((a,b)=>b[1].best-a[1].best).slice(0,5);
    const maxMu=Math.max(...sorted.map(([,r])=>r.mu),0.001);
    const el=document.getElementById('tsWuBars');
    if (!el) return;
    el.innerHTML=sorted.map(([id,r])=>{
        const muPct=(r.mu/maxMu*100).toFixed(1);
        const stdPct=Math.min((r.std/maxMu)*100,18).toFixed(1);
        const stdLeft=Math.max(0,+muPct - +stdPct/2).toFixed(1);
        const stdNorm=Math.min(r.std/(r.mu||1),1);
        const stdColor=stdNorm<0.15?'#34d399':stdNorm<0.4?'#fbbf24':'#94a3b8';
        const best=(r.best??r.mu).toFixed(4);
        const partner=r.bestPartner||'—';
        return `<div style="padding:5px 0;border-bottom:0.5px solid rgba(148,163,184,0.07)">
          <div style="display:grid;grid-template-columns:130px 1fr auto;align-items:center;gap:8px;margin-bottom:3px">
            <span style="font-family:monospace;font-size:10px;color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${id}">${id}</span>
            <div style="position:relative;height:12px;background:rgba(30,41,59,0.8);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${muPct}%;background:#7c3aed;border-radius:3px;transition:width 0.55s ease"></div>
              <div style="position:absolute;top:2px;height:8px;left:${stdLeft}%;width:${stdPct}%;background:rgba(180,180,80,0.38);border-radius:2px"></div>
            </div>
            <span style="font-size:11px;font-weight:500;color:#34d399;white-space:nowrap">${best}</span>
          </div>
          <div style="display:grid;grid-template-columns:130px 1fr;gap:8px">
            <span></span>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
              <span style="font-size:9px;padding:1px 5px;border-radius:3px;background:rgba(124,58,237,0.15);color:#a78bfa">n=${r.count}</span>
              <span style="font-size:9px;color:#94a3b8">μ=${r.mu.toFixed(4)}</span>
              ${r.count>=2?`<span style="font-size:9px;color:${stdColor}">σ=${r.std.toFixed(4)}</span>`:`<span style="font-size:9px;color:#1e3a5f">σ=—</span>`}
              <span style="font-size:9px;color:#1e3a5f">⊕</span>
              <span style="font-family:monospace;font-size:9px;color:#0f766e;white-space:nowrap">${partner}</span>
            </div>
          </div>
        </div>`;
    }).join('');
    // Update sidebar
    const n=Object.keys(_ts.reagents).length;
    const best=sorted.length?sorted[0][1].best:0;
    const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
    set('tsWuReagents',n);
    set('tsWuBest',best>0?best.toFixed(3):'—');
}

function _tsRenderTsBars() {
    const reagents=_ts.tsState.reagents;
    const sorted=Object.entries(reagents).sort((a,b)=>b[1].mu-a[1].mu).slice(0,5);
    const maxMu=Math.max(...sorted.map(([,r])=>r.mu),0.001);
    const el=document.getElementById('tsTsBars');
    if (!el) return;
    el.innerHTML=sorted.map(([id,r])=>{
        const muPct=(r.mu/maxMu*100).toFixed(1);
        const std=r.std||0;
        const stdPct=Math.min((std/maxMu)*100,18).toFixed(1);
        const stdLeft=Math.max(0,+muPct-+stdPct/2).toFixed(1);
        const delta=r.delta_mu!==undefined
            ?`<span style="font-size:9px;color:${r.delta_mu>=0?'#34d399':'#f87171'}">${r.delta_mu>=0?'+':''}${r.delta_mu.toFixed(3)}</span>`:'';
        const stdNorm=Math.min(std/(r.mu||1),1);
        const stdColor=stdNorm<0.15?'#34d399':stdNorm<0.4?'#fbbf24':'#94a3b8';
        const sc=r.sc||0;
        const partner=r.bestPartner||'—';
        return `<div style="padding:5px 0;border-bottom:0.5px solid rgba(148,163,184,0.07)">
          <div style="display:grid;grid-template-columns:130px 1fr auto;align-items:center;gap:8px;margin-bottom:3px">
            <span style="font-family:monospace;font-size:10px;color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${id}">${r.name||id}</span>
            <div style="position:relative;height:12px;background:rgba(30,41,59,0.8);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${muPct}%;background:#0891b2;border-radius:3px;transition:width 0.5s ease"></div>
              <div style="position:absolute;top:2px;height:8px;left:${stdLeft}%;width:${stdPct}%;background:rgba(180,180,80,0.38);border-radius:2px"></div>
            </div>
            <span style="font-size:11px;font-weight:500;color:#e2e8f0;white-space:nowrap">${r.mu.toFixed(4)} ${delta}</span>
          </div>
          <div style="display:grid;grid-template-columns:130px 1fr;gap:8px">
            <span></span>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
              <span style="font-size:9px;padding:1px 5px;border-radius:3px;background:rgba(8,145,178,0.15);color:#67e8f9">n=${sc}</span>
              <span style="font-size:9px;color:#94a3b8">μ=${r.mu.toFixed(4)}</span>
              ${sc>=2?`<span style="font-size:9px;color:${stdColor}">σ=${std.toFixed(4)}</span>`:`<span style="font-size:9px;color:#1e3a5f">σ=—</span>`}
              <span style="font-size:9px;color:#1e3a5f">⊕</span>
              <span style="font-family:monospace;font-size:9px;color:#0f766e;white-space:nowrap">${partner}</span>
            </div>
          </div>
        </div>`;
    }).join('');
}


function _tsRenderTsWinners() {
    const el=document.getElementById('tsTsWinners');
    if (!el) return;
    el.innerHTML=_ts.tsState.winners.slice(-5).reverse().map(w=>
        `<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:6px 8px;margin-bottom:4px">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span style="font-size:9px;color:#64748b">iter ${w.iter}</span>
            <span style="font-size:11px;font-weight:500;color:#34d399">${w.score.toFixed(4)}</span>
          </div>
          <div style="font-family:monospace;font-size:9px;color:#67e8f9;word-break:break-all">${w.smiles||'—'}</div>
        </div>`
    ).join('');
}

/* ─── Animation steps (timer-driven, consume event queues) ──────────────── */
function _tsWuStep() {
    const wu=_ts.wuState;
    if (!wu||wu.idx>=wu.events.length) return;
    const ev=wu.events[wu.idx++];
    wu.evals++;
    if (ev.score>wu.best) wu.best=ev.score;
    // Phase: always use the live-parsed phase, not the stale event phase
    const livePhase = _ts._currentPhase || ev.phase || wu.phase || '—';
    const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
    set('tsWuPhase', livePhase);
    set('tsWuEvals', wu.evals);
}

function _tsTsStep() {
    const ts=_ts.tsState;
    if (!ts||ts.idx>=ts.events.length) return;
    const ev=ts.events[ts.idx++];
    (ev.updates||[]).forEach(u=>{
        const prev=ts.reagents[u.id]?.mu??u.mu_before;
        ts.reagents[u.id]={name:u.name,mu:u.mu_after,std:u.std_after,sc:u.sc,delta_mu:u.mu_after-prev};
    });
    if (ev.score!==undefined&&ev.smiles) ts.winners.push({iter:ev.iter,score:ev.score,smiles:ev.smiles});
    if (ev.masked!==undefined) ts.masked=ev.masked;
    const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
    set('tsTsIter',    ev.iter??'—');
    set('tsTsScore',   ev.score!==undefined?ev.score.toFixed(4):'—');
    set('tsTsWinner',  ev.winner_idx??'—');
    set('tsTsWinnerMu',ev.winner_mu!==undefined?ev.winner_mu.toFixed(4):'—');
    set('tsTsMasked',  ts.masked||'—');
    _tsRenderTsBars();
    _tsRenderTsWinners();
    const bar=document.getElementById('tsStatusBar');
    if (bar&&ev.iter!==undefined) bar.textContent=`Iteration ${ev.iter} · score ${(ev.score||0).toFixed(4)} · winner ${ev.winner_idx??'?'}`;
}

/* ─── Worker bridge ──────────────────────────────────────────────────────── */
// Throttled bar render (avoid hammering DOM with every reagent update)
let _tsRenderQueued=false;
function _tsQueueRender() {
    if (_tsRenderQueued) return;
    _tsRenderQueued=true;
    requestAnimationFrame(()=>{_tsRenderQueued=false; _tsRenderWuBars();});
}

let _tsRenderCount=0;
function _tsHandleWorkerMsg(msg) {
    switch(msg.type) {
        case 'reagent_update':
            _ts.reagents[msg.id]={mu:msg.mu,std:msg.std,count:msg.count,best:msg.best,bestPartner:msg.bestPartner};
            // Re-render bars every 50 reagent updates
            if ((++_tsRenderCount)%50===0) _tsQueueRender();
            break;
        case 'wu_event':
            _ts.wuState.events.push(msg.event);
            break;
        case 'ts_event':
            _ts.tsState.events.push(msg.event);
            if (_ts.activeTab==='warmup') _tsTab('ts');
            if (!_ts._tsDrainQueued) {
                _ts._tsDrainQueued = true;
                requestAnimationFrame(() => {
                    _ts._tsDrainQueued = false;
                    while (_ts.tsState && _ts.tsState.idx < _ts.tsState.events.length) _tsTsStep();
                });
            }
            break;
        case 'ts_reagent':
            // Direct reagent update from post-update line — immediately renders TS bars
            if (_ts.tsState) {
                const prev = _ts.tsState.reagents[msg.id]?.mu ?? msg.mu;
                _ts.tsState.reagents[msg.id] = {
                    name: msg.name, mu: msg.mu, std: msg.std, sc: msg.sc,
                    delta_mu: msg.mu - prev,
                    bestPartner: msg.bestPartner || _ts.tsState.reagents[msg.id]?.bestPartner || '—',
                };
                if (!_ts._tsBarsQueued) {
                    _ts._tsBarsQueued = true;
                    requestAnimationFrame(() => { _ts._tsBarsQueued = false; _tsRenderTsBars(); });
                }
            }
            break;
        case 'debug':
            console.log(msg.msg);
            break;
        case 'ts_log': {
            const logEl = document.getElementById('tsTsLog');
            if (logEl) {
                const d = document.createElement('div');
                d.innerHTML = msg.html;
                logEl.appendChild(d);
                if (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
                logEl.scrollTop = logEl.scrollHeight;
            }
            break;
        }
        case 'phase': {
            _ts._currentPhase=msg.phase;
            const phaseEl=document.getElementById('tsWuPhase');
            if (phaseEl) phaseEl.textContent=msg.phase;
            const bar=document.getElementById('tsStatusBar');
            if (bar) bar.textContent=msg.phase;
            break;
        }
        case 'switch_tab':
            // Worker signals warmup complete — switch to TS Belief tab
            _tsTab(msg.tab);
            // Mark warmup live badge as done
            ['tsWuLiveBadge'].forEach(id=>{const e=document.getElementById(id);if(e){e.textContent='done';e.classList.remove('animate-pulse');}});
            ['tsTsLiveBadge'].forEach(id=>{const e=document.getElementById(id);if(e){e.textContent='live';e.classList.add('animate-pulse');}});
            break;
        case 'log':
            _tsWuLog(msg.html);
            break;
        case 'status':
            // Timestamp-based throttle: update DOM at most once per second
            _ts._pendingBarText=msg.text;
            {
                const now=Date.now();
                if (!_ts._lastBarUpdate || now-_ts._lastBarUpdate >= 1000) {
                    _ts._lastBarUpdate=now;
                    const bar=document.getElementById('tsStatusBar');
                    if (bar) bar.textContent=msg.text;
                }
            }
            break;
        case 'done':
            _tsFinalise();
            break;
    }
}

/* ─── Reset ──────────────────────────────────────────────────────────────── */
function _tsResetState() {
    _ts.wuState=_tsInitWu(); _ts.tsState=_tsInitTs();
    _ts.reagents={}; _ts._currentPhase='Warmup 1';
    _tsRenderCount=0; _tsRenderQueued=false;
    _ts._barRaf=null; _ts._pendingBarText='';
    _ts._lastBarUpdate=0;
    _ts._tsDrainQueued=false;
    _ts._tsBarsQueued=false;
}
function _tsResetUI() {
    _tsLogBuf.length=0;
    if (_tsLogRaf){cancelAnimationFrame(_tsLogRaf);_tsLogRaf=null;}
    ['tsWuLog','tsTsBars','tsWuBars','tsTsWinners','tsResultsBody'].forEach(id=>{const e=document.getElementById(id);if(e)e.innerHTML='';});
    ['tsWuPhase','tsWuEvals','tsWuReagents','tsWuBest','tsTsIter','tsTsScore',
     'tsTsWinner','tsTsWinnerMu','tsTsMasked','tsStatusBadge','tsMolCount',
     'tsBestScore','tsIterDone','tsElapsed'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent='—';});
    document.getElementById('tsResultsSpinner')?.classList.remove('hidden');
    document.getElementById('tsResultsWrap')?.classList.add('hidden');
}

/* ─── Main run ───────────────────────────────────────────────────────────── */
function _tsRun() {
    if (_ts.running) return;
    if (_ts.es) { _ts.es.close(); _ts.es=null; }
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);

    // Terminate old worker, start fresh
    if (_ts.worker) { _ts.worker.terminate(); _ts.worker=null; }
    const _tsBase = document.querySelector('script[src*="ts.js"]')?.src?.replace('ts.js','') || '/static/js/';
    const workerUrl = _tsBase + 'ts_worker.js?v=' + Date.now();
    console.log('[main] Loading worker from:', workerUrl);
    _ts.worker=new Worker(workerUrl);
    _ts.worker.onmessage=(evt)=>_tsHandleWorkerMsg(evt.data);
    _ts.worker.onerror=(e)=>console.error('[TS worker]',e);

    _ts.running=true;
    _tsResetState(); _tsResetUI();

    const smarts=document.getElementById('tsSmartsInput')?.value.trim()||'';
    const iters=parseInt(document.getElementById('tsIterInput')?.value)||5000;

    const btn=document.getElementById('tsRunBtn'), icon=document.getElementById('tsRunIcon');
    const sBar=document.getElementById('tsSpinnerBar'), bar=document.getElementById('tsStatusBar');
    if (btn){btn.disabled=true;btn.title='Running…';btn.onclick=_tsCancel;}
    if (icon) icon.textContent='⏳';
    if (sBar) sBar.classList.remove('hidden');
    if (bar)  bar.textContent='Launching python elion.py -i input_TS.yml …';

    ['tsWuLiveBadge','tsTsLiveBadge'].forEach(id=>{const e=document.getElementById(id);if(e){e.textContent='live';e.classList.add('animate-pulse');}});
    _tsTab('warmup');

    _ts.wuTimer=setInterval(_tsWuStep, _tsWuInterval());
    _ts.tsTimer=setInterval(_tsTsStep, _tsTsInterval());

    fetch('/vina_visualization/ts_run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({reaction_smarts:smarts,num_ts_iterations:iters})})
    .then(r=>r.json())
    .then(data=>{
        if (data.status!=='started'){_tsError(data.message||'Failed to start');return;}
        _ts.jobId=data.job_id;
        if (bar) bar.textContent=`Job started — streaming output`;
        _tsStream(data.job_id);
    })
    .catch(e=>_tsError(e.message));
}

/* ─── SSE stream ─────────────────────────────────────────────────────────── */
function _tsStream(job_id) {
    const es=new EventSource(`/vina_visualization/ts_status/${job_id}`);
    _ts.es=es;
    es.onmessage=(evt)=>{
        const line=evt.data;
        if (line==='__DONE__'){es.close();_ts.es=null;_tsFinalise();return;}
        // Send raw line to worker — zero main-thread parse cost
        if (_ts.worker) _ts.worker.postMessage({type:'line',data:line});
    };
    es.onerror=()=>{es.close();_ts.es=null;if(_ts.running)_tsFinalise();};
}

/* ─── Cancel ─────────────────────────────────────────────────────────────── */
function _tsCancel() {
    if (_ts.es){_ts.es.close();_ts.es=null;}
    if (_ts.worker){_ts.worker.terminate();_ts.worker=null;}
    if (_ts.jobId) fetch(`/vina_visualization/ts_kill/${_ts.jobId}`,{method:'POST'});
    _tsFinalise(true);
}

/* ─── Finalise ───────────────────────────────────────────────────────────── */
function _tsFinalise(cancelled=false) {
    _ts.running=false;
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);
    _ts.wuTimer=null; _ts.tsTimer=null;

    const btn=document.getElementById('tsRunBtn'), icon=document.getElementById('tsRunIcon');
    const sBar=document.getElementById('tsSpinnerBar'), bar=document.getElementById('tsStatusBar');
    if (btn){btn.disabled=false;btn.title='';btn.onclick=_tsRun;}
    if (icon) icon.textContent='🎲';
    if (sBar) sBar.classList.add('hidden');

    ['tsWuLiveBadge','tsTsLiveBadge'].forEach(id=>{const e=document.getElementById(id);if(e){e.textContent=cancelled?'cancelled':'done';e.classList.remove('animate-pulse');}});

    // Final render
    _tsRenderWuBars();
    if (_ts.wuState) while(_ts.wuState.idx<_ts.wuState.events.length)_tsWuStep();
    if (_ts.tsState) while(_ts.tsState.idx<_ts.tsState.events.length)_tsTsStep();

    const winners=[...(_ts.tsState?.winners||[])].sort((a,b)=>b.score-a.score);
    const best=winners.length?winners[0].score:null;
    const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
    set('tsStatusBadge', cancelled?'Cancelled':'✓ Done');
    set('tsMolCount',    winners.length||'—');
    set('tsBestScore',   best!==null?best.toFixed(4):'—');
    set('tsIterDone',    _ts.tsState?.winners.length||'—');
    if (bar&&!cancelled) bar.textContent=`✓ Complete — ${winners.length} molecules, best ${best!==null?best.toFixed(4):'?'}`;

    const spinner=document.getElementById('tsResultsSpinner'), wrap=document.getElementById('tsResultsWrap'), tbody=document.getElementById('tsResultsBody');
    if (spinner) spinner.classList.add('hidden');
    if (wrap&&tbody&&winners.length){
        wrap.classList.remove('hidden');
        tbody.innerHTML=winners.map((m,i)=>
            `<tr class="hover:bg-slate-800/40 transition-colors">
              <td class="py-2 pr-4 text-slate-500">${i+1}</td>
              <td class="py-2 pr-4 font-mono text-[10px] text-cyan-300 max-w-[260px] truncate" title="${m.smiles||''}">${m.smiles||'—'}</td>
              <td class="py-2 pr-4 text-right ${m.score>=0?'text-emerald-400':'text-red-400'}">${m.score.toFixed(4)}</td>
              <td class="py-2 text-right text-violet-400">${m.reward??'—'}</td>
            </tr>`
        ).join('');
    }
    if (!cancelled) _tsTab('results');
}

function _tsError(msg) {
    _ts.running=false;
    clearInterval(_ts.wuTimer); clearInterval(_ts.tsTimer);
    const btn=document.getElementById('tsRunBtn'), icon=document.getElementById('tsRunIcon');
    const sBar=document.getElementById('tsSpinnerBar');
    if (btn){btn.disabled=false;btn.onclick=_tsRun;} if (icon) icon.textContent='🎲'; if (sBar) sBar.classList.add('hidden');
    const bar=document.getElementById('tsStatusBar'); if (bar) bar.textContent='❌ '+(msg||'Error');
    const badge=document.getElementById('tsStatusBadge'); if (badge) badge.textContent='❌ Error';
}