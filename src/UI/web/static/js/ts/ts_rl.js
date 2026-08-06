// =============================================================================
// ts/ts_rl.js — 🧪 RL tab: the loop, annotated with where it breaks.
//
// SELF-INSTALLING. Drop this file in web/static/js/ts/ and add ONE line after
// the other ts_*.js scripts:
//     <script src="{{ url_for('static', filename='js/ts/ts_rl.js') }}"></script>
//
// It needs no template edits. On load it:
//   1. injects a scoped <style> (every rule is prefixed #tsPaneRl — nothing leaks)
//   2. inserts the "🧪 RL" button between #tsTabTs and #tsTabResults
//   3. inserts <div id="tsPaneRl"> before #tsPaneResults
//   4. registers 'rl' in _TS_PANE / _TS_TAB_ACTIVE  (const objects are mutable)
//   5. renders the top-5 TS products from live state on every tab activation
//
// Load order: AFTER ts_core.js (needs _ts, _TS_PANE, _TS_TAB_ACTIVE, _tsTab).
// =============================================================================
console.log('%c[ts-rl] LOADED', 'background:#7f1d3a;color:#fecdd3;padding:2px 6px;border-radius:3px');

const _TS_RL_STYLE = `
  #tsPaneRl{
    --surface-0:#05070f; --surface-1:#0b1120; --surface-2:#111827; --surface-3:#0f172a;
    --line:#1e293b; --line-soft:rgba(30,41,59,.6);
    --ink:#e2e8f0; --ink-2:#94a3b8; --ink-3:#64748b; --ink-4:#475569;
    /* validated categorical trio (dark, surface #0b1120) */
    --s1:#0891b2; --s2:#d97706; --s3:#8b5cf6;
    /* brighter house tints for text/marks on dark, used as ink not as series fill */
    --s1-ink:#22d3ee; --s2-ink:#f59e0b; --s3-ink:#a78bfa;
    --good:#34d399; --warn:#f59e0b; --bad:#fb7185; --none:#64748b;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  #tsPaneRl *{box-sizing:border-box}

  #tsPaneRl{background:var(--surface-0);color:var(--ink);font-family:Inter,system-ui,-apple-system,sans-serif;font-size:13px;line-height:1.5;overflow-y:auto;width:100%}
  #tsPaneRl .wrap{max-width:1500px;margin:0 auto;padding:0 0 60px}

  /* ── modal chrome replica ─────────────────────────────────────────── */
  #tsPaneRl .hdr{padding:12px 24px;border-bottom:1px solid #334155;background:#0f172a;
       display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
  #tsPaneRl .hdr h2{font-size:13.5px;font-weight:600;margin:0;letter-spacing:-.01em}
  #tsPaneRl .hdr p{font-size:10px;color:var(--ink-2);margin:2px 0 0}
  #tsPaneRl .tabs{display:flex;border:1px solid #334155;border-radius:12px;overflow:hidden;font-size:12px;font-weight:500}
  #tsPaneRl .tabs button{padding:6px 16px;background:transparent;color:var(--ink-2);border:0;
               border-right:1px solid #334155;font:inherit;cursor:pointer}
  #tsPaneRl .tabs button:last-child{border-right:0}
  #tsPaneRl .tabs button.on{background:#7f1d3a;color:#fecdd3}

  /* ── config bar ───────────────────────────────────────────────────── */
  #tsPaneRl .cfg{padding:10px 24px;background:var(--surface-0);border-bottom:1px solid var(--line);
       display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  #tsPaneRl .lbl{font-size:10px;font-weight:600;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;flex-shrink:0}
  #tsPaneRl .inp{background:#0b1120;border:1px solid var(--line);border-radius:8px;padding:6px 10px;
       color:var(--s1-ink);font-family:var(--mono);font-size:11.5px;outline:none;min-width:230px;flex:1}
  #tsPaneRl .btn{padding:7px 15px;border-radius:9px;border:1px solid transparent;font:inherit;font-size:12px;
       font-weight:700;cursor:pointer;color:#fff;background:linear-gradient(135deg,#0891b2,#7c3aed)}
  #tsPaneRl .btn-ghost{background:#0b1120;border:1px solid var(--line);color:var(--ink-2);font-weight:600}

  /* ── verdict strip ────────────────────────────────────────────────── */
  #tsPaneRl .verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;padding:16px 24px 4px}
  #tsPaneRl .v{background:var(--surface-1);border:1px solid var(--line);border-radius:13px;padding:11px 13px}
  #tsPaneRl .v .t{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-4);margin-bottom:6px}
  #tsPaneRl .v .m{display:flex;align-items:center;gap:7px}
  #tsPaneRl .v .m b{font-size:15px;font-weight:700;letter-spacing:-.01em}
  #tsPaneRl .v .s{font-size:10px;color:var(--ink-3);margin-top:4px;line-height:1.45}
  #tsPaneRl .pill{display:inline-flex;align-items:center;gap:4px;font-size:9.5px;font-weight:700;
        padding:2px 7px;border-radius:6px;font-family:var(--mono)}
  #tsPaneRl .pill.ok{background:rgba(52,211,153,.13);border:1px solid rgba(52,211,153,.4);color:var(--good)}
  #tsPaneRl .pill.no{background:rgba(251,113,133,.13);border:1px solid rgba(251,113,133,.4);color:var(--bad)}
  #tsPaneRl .pill.wr{background:rgba(245,158,11,.13);border:1px solid rgba(245,158,11,.4);color:var(--warn)}

  /* ── body split ───────────────────────────────────────────────────── */
  #tsPaneRl .body{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:14px 24px 0;align-items:start}
  @media(max-width:1180px){.body{grid-template-columns:1fr}}
  #tsPaneRl .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}

  #tsPaneRl .card{background:var(--surface-1);border:1px solid var(--line);border-radius:15px;padding:15px 16px;min-width:0}
  #tsPaneRl .card.span{grid-column:1/-1}
  #tsPaneRl .card h3{margin:0;font-size:12.5px;font-weight:700;letter-spacing:-.01em;display:flex;
           align-items:center;gap:8px;flex-wrap:wrap}
  #tsPaneRl .tag{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 6px;border-radius:5px;
       background:rgba(139,92,246,.14);border:1px solid rgba(139,92,246,.4);color:var(--s3-ink)}
  #tsPaneRl .card .why{font-size:11.5px;color:var(--ink-2);margin:7px 0 12px;line-height:1.55}
  #tsPaneRl .card .why b{color:var(--ink)}
  #tsPaneRl .note{font-size:10px;color:var(--ink-4);font-family:var(--mono);line-height:1.55;margin-top:10px;
        padding-top:9px;border-top:1px solid var(--line-soft)}
  #tsPaneRl code{font-family:var(--mono);font-size:10.5px;color:var(--s1-ink);background:rgba(8,145,178,.1);
       padding:1px 5px;border-radius:4px}
  #tsPaneRl .hero{font-family:var(--mono);font-size:26px;font-weight:700;letter-spacing:-.02em;line-height:1}
  #tsPaneRl .hero-s{font-size:10px;color:var(--ink-3);font-family:var(--mono);margin-top:4px}
  #tsPaneRl svg{display:block;overflow:visible}
  #tsPaneRl .ax{font-family:var(--mono);font-size:9px;fill:var(--ink-4)}
  #tsPaneRl .axl{font-family:var(--mono);font-size:9.5px;fill:var(--ink-3)}
  #tsPaneRl .lg{display:flex;gap:14px;flex-wrap:wrap;font-size:10px;color:var(--ink-2);margin-top:9px;font-family:var(--mono)}
  #tsPaneRl .lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}

  #tsPaneRl table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10.5px;margin-top:4px}
  #tsPaneRl th{text-align:left;color:var(--ink-4);font-weight:600;padding:4px 6px;border-bottom:1px solid var(--line);
     font-size:9px;text-transform:uppercase;letter-spacing:.05em}
  #tsPaneRl td{padding:4px 6px;border-bottom:1px solid var(--line-soft);color:var(--ink-2)}
  #tsPaneRl td.n{text-align:right;color:var(--ink)}

  /* ── top-5 sidebar ────────────────────────────────────────────────── */
  #tsPaneRl .side{background:var(--surface-1);border:1px solid var(--line);border-radius:15px;padding:14px;
        position:sticky;top:12px}
  #tsPaneRl .m5{border:1px solid var(--line);border-radius:11px;background:#0a0f1d;padding:9px 10px;margin-bottom:9px;
      cursor:pointer;transition:border-color .14s,background .14s}
  #tsPaneRl .m5:hover{border-color:var(--s1-ink);background:#0d1526}
  #tsPaneRl .m5 .r{display:flex;align-items:center;gap:7px;margin-bottom:5px}
  #tsPaneRl .m5 .rank{font-family:var(--mono);font-size:10px;font-weight:700;color:var(--s1-ink);
            background:rgba(8,145,178,.14);border:1px solid rgba(8,145,178,.4);border-radius:5px;padding:1px 6px}
  #tsPaneRl .m5 .sc{margin-left:auto;font-family:var(--mono);font-size:11px;font-weight:700;color:var(--good)}
  #tsPaneRl .m5 .smi{font-family:var(--mono);font-size:9px;color:var(--ink-2);word-break:break-all;line-height:1.45;
           user-select:all;margin-bottom:6px}
  #tsPaneRl .flags{display:flex;gap:4px;flex-wrap:wrap}
  #tsPaneRl .f{font-family:var(--mono);font-size:8.5px;font-weight:700;padding:1px 5px;border-radius:4px}
  #tsPaneRl .f.ok{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.35);color:var(--good)}
  #tsPaneRl .f.no{background:rgba(251,113,133,.12);border:1px solid rgba(251,113,133,.35);color:var(--bad)}
  #tsPaneRl .f.wr{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.35);color:var(--warn)}
  #tsPaneRl .f.mu{background:rgba(100,116,139,.12);border:1px solid rgba(100,116,139,.35);color:var(--none)}
  #tsPaneRl .banner{border-radius:11px;padding:10px 12px;font-size:11px;line-height:1.55;margin-bottom:12px}
  #tsPaneRl .banner.bad{background:rgba(251,113,133,.08);border:1px solid rgba(251,113,133,.32);color:#fecdd3}

  /* ── top-N reaction cards, in the pose sidebar ────────────────────────
     These live inside the ADOPTED pose workspace, which is a child of
     #tsPaneRl only while the RL tab is up — so the scoping still holds and
     the rules evaporate the moment the workspace goes home.              */
  #tsPaneRl .rlm{border:1px solid #1e293b;border-radius:12px;background:#0b1120;padding:9px 10px 8px;
       margin-bottom:9px;cursor:pointer;transition:border-color .14s,background .14s}
  #tsPaneRl .rlm:hover{border-color:#22d3ee;background:#0d1526}
  #tsPaneRl .rlm-h{display:flex;align-items:center;gap:6px;margin-bottom:7px}
  #tsPaneRl .rlm-rank{font-family:var(--mono);font-size:10px;font-weight:700;color:#22d3ee;
       background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.35);border-radius:5px;padding:1px 6px}
  #tsPaneRl .rlm-best{margin-left:auto;font-family:var(--mono);font-size:11px;font-weight:700;color:#f59e0b;
       display:inline-flex;align-items:center;gap:3px}
  #tsPaneRl .rlm-pin{background:none;border:0;color:#475569;font-size:11px;cursor:pointer;padding:1px 3px;line-height:1}
  #tsPaneRl .rlm-pin:hover{color:#22d3ee}
  #tsPaneRl .rlm-rx{display:flex;align-items:flex-start;justify-content:center;gap:4px}
  #tsPaneRl .rlm-bb{display:flex;flex-direction:column;align-items:center;gap:1px;min-width:0}
  #tsPaneRl .rlm-bb span{font-family:var(--mono);font-size:9px;color:#94a3b8;max-width:104px;
       overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #tsPaneRl .rlm-op{font-size:15px;color:#475569;align-self:center;padding:0 1px;line-height:1}
  #tsPaneRl .rlm-dn{text-align:center;font-size:13px;color:#475569;line-height:1;margin:2px 0 1px}
  #tsPaneRl .rlm-prod{display:block;margin:0 auto}
  #tsPaneRl .rlm-hint{font-family:var(--mono);font-size:8.5px;color:#475569;text-align:center;margin-top:4px}
  #tsPaneRl .rlm:hover .rlm-hint{color:#22d3ee}
  #tsPaneRl .rln{width:44px;background:#0b1120;border:1px solid var(--line);border-radius:6px;
       color:#22d3ee;font-family:var(--mono);font-size:10px;padding:1px 4px;outline:none;text-align:center}
`;

const _TS_RL_BODY = `
<!-- config bar removed: the REAL pose input bar is adopted into #tsRlPoseHost -->
  <!-- ═══ verdict strip ═══ -->
  <div class="verdict">
    <div class="v"><div class="t">Search</div><div class="m"><b style="color:var(--good)">Solved</b>
      <span class="pill ok">✓ TS</span></div>
      <div class="s">Bandit over reagent posteriors. This is the MCTS slot.</div></div>
    <div class="v"><div class="t">Synthesizability</div><div class="m"><b style="color:var(--good)">Solved</b>
      <span class="pill ok">✓ eXplore</span></div>
      <div class="s">Purchasable blocks + validated Suzuki SMARTS.</div></div>
    <div class="v"><div class="t">DMTA cycle cost</div><div class="m"><b style="color:var(--good)">Improved</b>
      <span class="pill ok">✓ orderable</span></div>
      <div class="s">Branching factor no longer 18 compounds / year.</div></div>
    <div class="v"><div class="t">Critic</div><div class="m"><b style="color:var(--bad)">Blind</b>
      <span class="pill no">✕ no losses</span></div>
      <div class="s">Zero non-binders in PDBbind. Cannot emit "this fails".</div></div>
    <div class="v"><div class="t">Calibration</div><div class="m"><b style="color:var(--bad)">None</b>
      <span class="pill no">✕ unfitted</span></div>
      <div class="s">No WDL-style fit against measured outcomes.</div></div>
    <div class="v"><div class="t">Net risk</div><div class="m"><b style="color:var(--warn)">Amplified</b>
      <span class="pill wr">⚠ search×bias</span></div>
      <div class="s">Better search over a biased objective converges to the bias.</div></div>
  </div>

  <div class="body">
    <div class="grid">

      <!-- ═══ LOOP DIAGRAM ═══ -->
      <div class="card span">
        <h3>The loop, and where it breaks <span class="tag">overview</span></h3>
        <p class="why">Three components, three break points. <b>None of the breaks are in the generator</b> —
          which is why adding a better generator has not moved the hit rate.</p>
        <svg viewBox="0 0 940 172" style="width:100%;height:auto">
          <defs>
            <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>
            <marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill="#fb7185"/></marker>
          </defs>
          <!-- boxes -->
          <g>
            <rect x="8" y="46" width="176" height="62" rx="13" fill="#0a0f1d" stroke="#0891b2" stroke-width="1.5"/>
            <text x="96" y="66" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#22d3ee">🎲 Thompson Sampling</text>
            <text x="96" y="82" text-anchor="middle" class="ax" font-size="9.5">policy · reagent bandits</text>
            <text x="96" y="97" text-anchor="middle" class="ax" font-size="9.5" fill="#34d399">2 × 10⁹ product space</text>
          </g>
          <g>
            <rect x="256" y="46" width="176" height="62" rx="13" fill="#0a0f1d" stroke="#8b5cf6" stroke-width="1.5"/>
            <text x="344" y="66" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#a78bfa">⌬ Pose generation</text>
            <text x="344" y="82" text-anchor="middle" class="ax" font-size="9.5">Vina MC + BFGS</text>
            <text x="344" y="97" text-anchor="middle" class="ax" font-size="9.5" fill="#fb7185">state is PREDICTED, not given</text>
          </g>
          <g>
            <rect x="504" y="46" width="176" height="62" rx="13" fill="#0a0f1d" stroke="#d97706" stroke-width="1.5"/>
            <text x="592" y="66" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#f59e0b">⚛ Critic</text>
            <text x="592" y="82" text-anchor="middle" class="ax" font-size="9.5">GIGN / DeepAtom → pK</text>
            <text x="592" y="97" text-anchor="middle" class="ax" font-size="9.5" fill="#fb7185">trained on binders only</text>
          </g>
          <g>
            <rect x="752" y="46" width="176" height="62" rx="13" fill="#0a0f1d" stroke="#334155" stroke-width="1.5"/>
            <text x="840" y="66" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#94a3b8">🧪 Assay (ground truth)</text>
            <text x="840" y="82" text-anchor="middle" class="ax" font-size="9.5">SPR · n = 18 · 1 hit</text>
            <text x="840" y="97" text-anchor="middle" class="ax" font-size="9.5" fill="#fb7185">not in the loop</text>
          </g>
          <!-- arrows -->
          <line x1="188" y1="77" x2="250" y2="77" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>
          <line x1="436" y1="77" x2="498" y2="77" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>
          <line x1="684" y1="77" x2="746" y2="77" stroke="#fb7185" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arb)"/>
          <!-- reward return path -->
          <path d="M592 112 L592 140 L96 140 L96 114" fill="none" stroke="#d97706" stroke-width="1.6" marker-end="url(#ar)"/>
          <text x="344" y="154" text-anchor="middle" class="ax" font-size="9.5" fill="#d97706">reward — a MODEL, not the environment</text>
          <!-- break markers -->
          <g font-family="ui-monospace" font-size="9" font-weight="700">
            <circle cx="344" cy="34" r="9" fill="rgba(251,113,133,.16)" stroke="#fb7185"/>
            <text x="344" y="37.5" text-anchor="middle" fill="#fb7185">2</text>
            <circle cx="592" cy="34" r="9" fill="rgba(251,113,133,.16)" stroke="#fb7185"/>
            <text x="592" y="37.5" text-anchor="middle" fill="#fb7185">1</text>
            <circle cx="715" cy="34" r="9" fill="rgba(251,113,133,.16)" stroke="#fb7185"/>
            <text x="715" y="37.5" text-anchor="middle" fill="#fb7185">4</text>
            <circle cx="96" cy="34" r="9" fill="rgba(245,158,11,.16)" stroke="#f59e0b"/>
            <text x="96" y="37.5" text-anchor="middle" fill="#f59e0b">5</text>
          </g>
        </svg>
        <div class="note">In AlphaZero the reward <b style="color:#94a3b8">is</b> the environment — win/loss, free, unbiased, terminal.
          Here the reward is a learned model and the environment (the assay) sits outside the loop. That single substitution
          is what makes this reward-model overoptimization rather than reinforcement learning.</div>
      </div>

      <!-- ═══ D1 — no losses ═══ -->
      <div class="card">
        <h3>1 · The critic has never seen a loss <span class="tag">GIGN · DeepAtom</span></h3>
        <p class="why">Both scorers train on PDBbind <code>train/valid/test2013/2016/2019</code>. Every complex in
          there <b>binds</b>. No decoys, no non-binders. Ask it about a non-binder and it returns the prior mean,
          confidently.</p>
        <svg viewBox="0 0 420 168" style="width:100%;height:auto">
          <!-- axis -->
          <line x1="34" y1="132" x2="404" y2="132" stroke="#334155" stroke-width="1"/>
          <g class="ax" text-anchor="middle">
            <text x="34" y="146">2</text><text x="108" y="146">4</text><text x="182" y="146">6</text>
            <text x="256" y="146">8</text><text x="330" y="146">10</text><text x="404" y="146">12</text>
          </g>
          <text x="219" y="162" text-anchor="middle" class="axl">PDBbind refined label · pK</text>
          <!-- "never observed" region -->
          <rect x="0" y="26" width="34" height="106" fill="rgba(251,113,133,.09)" stroke="rgba(251,113,133,.3)" stroke-dasharray="3 3"/>
          <text x="17" y="60" text-anchor="middle" class="ax" fill="#fb7185" font-size="8"
                transform="rotate(-90 17 60)">non-binders</text>
          <text x="17" y="112" text-anchor="middle" class="ax" fill="#fb7185" font-size="12" font-weight="700">0</text>
          <!-- gaussian mean 6.38 sd 1.98 : x = 34 + (pk-2)*37 -->
          <path fill="rgba(8,145,178,.22)" stroke="#0891b2" stroke-width="2" stroke-linejoin="round"
            d="M34,131 L45,130 L57,128 L68,125 L79,120 L90,113 L101,104 L112,93 L123,81 L134,69 L145,58 L156,49
               L168,42 L179,38 L190,36 L201,38 L212,42 L223,49 L234,58 L245,69 L256,81 L267,93 L278,104 L290,113
               L301,120 L312,125 L323,128 L334,130 L345,131 L404,132 L404,132 L34,132 Z"/>
          <!-- mean line -->
          <line x1="196" y1="30" x2="196" y2="132" stroke="#0891b2" stroke-width="1.2" stroke-dasharray="4 3"/>
          <text x="200" y="40" class="ax" fill="#22d3ee">training mean 6.38</text>
          <!-- their hit at 4.67 -->
          <line x1="133" y1="46" x2="133" y2="132" stroke="#34d399" stroke-width="2"/>
          <circle cx="133" cy="46" r="4.5" fill="#34d399" stroke="#0b1120" stroke-width="1.5"/>
          <text x="128" y="38" text-anchor="end" class="ax" fill="#34d399" font-size="9.5">Z319334062</text>
          <text x="128" y="49" text-anchor="end" class="ax" fill="#34d399" font-size="9.5">pK 4.67 · 21.5 µM</text>
        </svg>
        <div style="display:flex;gap:22px;margin-top:8px;align-items:flex-end">
          <div><div class="hero" style="color:var(--bad)">0</div>
               <div class="hero-s">non-binders in training</div></div>
          <div><div class="hero" style="color:var(--ink-2)">1.98</div>
               <div class="hero-s">label σ (refined set)</div></div>
          <div><div class="hero" style="color:var(--warn)">−0.86σ</div>
               <div class="hero-s">where your hit sits</div></div>
        </div>
        <div class="note">Source: <code>deepatom/…/02_pytorch/record_config.py</code> — refined mean 6.38, σ 1.98,
          range 2.07–11.52. Stockfish's eval spends most of its dynamic range on <i>losing</i>. This one has a floor
          it has never been below.</div>
      </div>

      <!-- ═══ D2 — pose shift ═══ -->
      <div class="card">
        <h3>2 · Trained on crystal poses, scored on generated ones <span class="tag">distribution shift</span></h3>
        <p class="why">Training prep carves the pocket with <code>byres &lt;ligand&gt; around 5</code> — using the
          <b>true, experimentally-determined</b> ligand. At inference the pocket is carved with the pose your
          sampler guessed.</p>
        <svg viewBox="0 0 420 150" style="width:100%;height:auto">
          <!-- TRAIN -->
          <text x="12" y="14" class="axl" fill="#34d399">TRAINING</text>
          <circle cx="70" cy="58" r="30" fill="none" stroke="#334155" stroke-width="1.2" stroke-dasharray="3 3"/>
          <circle cx="70" cy="58" r="19" fill="rgba(52,211,153,.13)" stroke="#34d399" stroke-width="1.5"/>
          <circle cx="70" cy="58" r="6" fill="#34d399"/>
          <text x="70" y="103" text-anchor="middle" class="ax">pocket ← true ligand</text>
          <text x="70" y="115" text-anchor="middle" class="ax" fill="#34d399">RMSD 0.00 Å</text>
          <!-- INFER -->
          <text x="212" y="14" class="axl" fill="#fb7185">INFERENCE</text>
          <circle cx="270" cy="58" r="30" fill="none" stroke="#334155" stroke-width="1.2" stroke-dasharray="3 3"/>
          <circle cx="284" cy="50" r="19" fill="rgba(251,113,133,.11)" stroke="#fb7185" stroke-width="1.5"/>
          <circle cx="284" cy="50" r="6" fill="#fb7185"/>
          <circle cx="270" cy="58" r="6" fill="none" stroke="#475569" stroke-dasharray="2 2"/>
          <line x1="270" y1="58" x2="284" y2="50" stroke="#fb7185" stroke-width="1.2"/>
          <text x="270" y="103" text-anchor="middle" class="ax">pocket ← predicted pose</text>
          <text x="270" y="115" text-anchor="middle" class="ax" fill="#fb7185">RMSD ≫ 0 → wrong edges</text>
          <!-- arrow -->
          <line x1="118" y1="58" x2="222" y2="58" stroke="#475569" stroke-width="1.4" marker-end="url(#ar)"/>
          <text x="170" y="50" text-anchor="middle" class="ax">deploy</text>
          <!-- consequence -->
          <text x="12" y="140" class="ax" fill="#94a3b8">wrong pose → wrong pocket → wrong</text>
          <text x="248" y="140" class="ax" fill="#fb7185">edge_index_inter</text>
        </svg>
        <div class="note">Your own <code>yupu_GIGN_pose.py</code> header already flags the full-protein variant as a
          "train/inference MISMATCH". The same reasoning applies one level up and is not written down anywhere.
          <b style="color:#94a3b8">Chess never has this problem</b> — Stockfish is never asked to guess where the pieces are.</div>
      </div>

      <!-- ═══ D3 — resolution (HERO) ═══ -->
      <div class="card span">
        <h3>3 · Resolution — the noise is larger than the signal <span class="tag">the "0.2 ± 1.3" problem</span></h3>
        <p class="why">An objective consumes the <b>gradient</b>; a constraint consumes only a <b>threshold</b>.
          The chart below is why GIGN can serve as the second and not the first.</p>
        <svg viewBox="0 0 900 200" style="width:100%;height:auto">
          <!-- axis: 0 .. 3.0 log units mapped 150..860 -->
          <line x1="150" y1="168" x2="860" y2="168" stroke="#334155"/>
          <g class="ax" text-anchor="middle">
            <text x="150" y="182">0</text><text x="268" y="182">0.5</text><text x="387" y="182">1.0</text>
            <text x="505" y="182">1.5</text><text x="623" y="182">2.0</text><text x="742" y="182">2.5</text>
            <text x="860" y="182">3.0</text>
          </g>
          <text x="505" y="196" text-anchor="middle" class="axl">log units (pK)</text>

          <!-- row 1: signal needed -->
          <text x="140" y="42" text-anchor="end" class="axl" fill="#34d399">signal you must resolve</text>
          <rect x="150" y="30" width="237" height="17" rx="4" fill="var(--good)" opacity=".85"/>
          <text x="397" y="43" class="ax" fill="#34d399">ΔpK 0.5 – 1.0 · one lead-op step (21.5 µM → ~3 µM)</text>

          <!-- row 2: GIGN noise -->
          <text x="140" y="86" text-anchor="end" class="axl" fill="#fb7185">GIGN RMSE (crystal poses)</text>
          <rect x="150" y="74" width="308" height="17" rx="4" fill="var(--bad)" opacity=".85"/>
          <line x1="458" y1="66" x2="458" y2="99" stroke="#fb7185" stroke-width="1.5" stroke-dasharray="3 2"/>
          <text x="468" y="87" class="ax" fill="#fb7185">±1.3 — and this is its BEST case, on poses you will not have</text>

          <!-- row 3: label spread -->
          <text x="140" y="126" text-anchor="end" class="axl" fill="#94a3b8">predict-the-mean baseline</text>
          <rect x="150" y="114" width="469" height="17" rx="4" fill="#334155"/>
          <text x="629" y="127" class="ax">σ 1.98 — the model removes only ~57% of the variance</text>

          <!-- row 4: stockfish, for scale -->
          <text x="140" y="158" text-anchor="end" class="axl" fill="#22d3ee">Stockfish, for scale</text>
          <rect x="150" y="150" width="3" height="11" rx="1.5" fill="var(--s1-ink)"/>
          <text x="162" y="159" class="ax" fill="#22d3ee">±0.01 pawn — calibrated by <tspan font-style="italic">fitting</tspan> a/b to real game outcomes in win_rate_params()</text>

          <!-- overlay bracket showing swallow -->
          <path d="M150,24 L150,18 L387,18 L387,24" fill="none" stroke="#34d399" stroke-width="1"/>
          <path d="M150,68 L150,62 L458,62 L458,68" fill="none" stroke="#fb7185" stroke-width="1"/>
        </svg>
        <div class="note">
          Reality check on the one compound with ground truth: Vina scored <code>−8.1 kcal/mol</code>, which by this
          app's own <code>pK = −ΔG/1.36</code> means <b style="color:#94a3b8">pK 5.96 ≈ 1.1 µM</b>. Measured:
          <b style="color:#94a3b8">pK 4.67 = 21.5 µM</b>. Miss = <b style="color:#fb7185">1.29 log units (≈19×)</b>.
          Note that this miss is almost exactly one RMSE — i.e. the scorer's output was consistent with it having
          <i>no information about this compound at all</i>.
          <span style="color:#64748b">(This corrects my earlier "~3-log / low-nanomolar" claim — that overstated it;
          −8.1 kcal/mol is ~1 µM, not nM.)</span>
        </div>
      </div>

      <!-- ═══ D4 — paper is a test set ═══ -->
      <div class="card">
        <h3>4 · The paper is a test set, not a critic <span class="tag">n = 18</span></h3>
        <p class="why">A critic must answer for <b>arbitrary</b> new positions. This answers for exactly 18 molecules
          and can never score #19. What it <i>can</i> do is calibrate.</p>
        <svg viewBox="0 0 420 122" style="width:100%;height:auto">
          <!-- 18 dots, 6 x 3 -->
          <g>
            <!-- specific: 1 -->
            <circle cx="30" cy="26" r="11" fill="rgba(52,211,153,.2)" stroke="#34d399" stroke-width="2"/>
            <text x="30" y="30" text-anchor="middle" font-size="11" fill="#34d399">✓</text>
            <!-- nonspecific: 10 -->
            <g fill="rgba(245,158,11,.16)" stroke="#f59e0b" stroke-width="1.5">
              <circle cx="66" cy="26" r="11"/><circle cx="102" cy="26" r="11"/><circle cx="138" cy="26" r="11"/>
              <circle cx="174" cy="26" r="11"/><circle cx="210" cy="26" r="11"/><circle cx="246" cy="26" r="11"/>
              <circle cx="282" cy="26" r="11"/><circle cx="318" cy="26" r="11"/><circle cx="354" cy="26" r="11"/>
              <circle cx="390" cy="26" r="11"/>
            </g>
            <g font-size="10" fill="#f59e0b" text-anchor="middle">
              <text x="66" y="30">~</text><text x="102" y="30">~</text><text x="138" y="30">~</text>
              <text x="174" y="30">~</text><text x="210" y="30">~</text><text x="246" y="30">~</text>
              <text x="282" y="30">~</text><text x="318" y="30">~</text><text x="354" y="30">~</text>
              <text x="390" y="30">~</text>
            </g>
            <!-- no signal: 7 -->
            <g fill="rgba(100,116,139,.14)" stroke="#64748b" stroke-width="1.5">
              <circle cx="30" cy="62" r="11"/><circle cx="66" cy="62" r="11"/><circle cx="102" cy="62" r="11"/>
              <circle cx="138" cy="62" r="11"/><circle cx="174" cy="62" r="11"/><circle cx="210" cy="62" r="11"/>
              <circle cx="246" cy="62" r="11"/>
            </g>
            <g font-size="10" fill="#64748b" text-anchor="middle">
              <text x="30" y="66">✕</text><text x="66" y="66">✕</text><text x="102" y="66">✕</text>
              <text x="138" y="66">✕</text><text x="174" y="66">✕</text><text x="210" y="66">✕</text>
              <text x="246" y="66">✕</text>
            </g>
          </g>
          <g class="ax" font-size="9.5">
            <text x="12" y="92" fill="#34d399">✓ 1 specific — 21.5 µM SPR</text>
            <text x="192" y="92" fill="#f59e0b">~ 10 nonspecific</text>
            <text x="316" y="92" fill="#64748b">✕ 7 no signal</text>
          </g>
          <text x="12" y="110" class="ax" fill="#94a3b8" font-size="9">all 18 were top-100 Vina <tspan font-weight="700">and</tspan> passed human visual inspection</text>
          <text x="12" y="121" class="ax" fill="#fb7185" font-size="9">→ these are HARD negatives: the examples that maximally confuse your scorer</text>
        </svg>
        <div class="note">Right job for it: measure GIGN's <b>offset</b> on this target and its <b>ranking</b>
          (does it put the 1 above the 17? Spearman on 18 points). Stockfish did not invent <code>a</code> and
          <code>b</code> — it <i>fit</i> them to outcomes. You have 18 outcomes. Fit a 2-parameter recalibration;
          do not retrain a GNN on 18 points.</div>
      </div>

      <!-- ═══ D5 — TS as reward hacker ═══ -->
      <div class="card">
        <h3>5 · TS is the most efficient reward-hacker you could build <span class="tag">live diagnostic</span></h3>
        <p class="why">TS's convergence guarantee is about finding the argmax of the <b>observed</b> reward. It has
          no representation of "the reward might be wrong." Plot the selected reagents' properties against iteration —
          <b>a monotone climb is hacking, measured</b>.</p>
        <svg viewBox="0 0 420 176" style="width:100%;height:auto">
          <line x1="40" y1="140" x2="404" y2="140" stroke="#334155"/>
          <line x1="40" y1="18" x2="40" y2="140" stroke="#334155"/>
          <g class="ax" text-anchor="middle">
            <text x="40" y="154">0</text><text x="131" y="154">1250</text><text x="222" y="154">2500</text>
            <text x="313" y="154">3750</text><text x="404" y="154">5000</text>
          </g>
          <text x="222" y="170" text-anchor="middle" class="axl">TS iteration</text>
          <text x="16" y="80" text-anchor="middle" class="axl" transform="rotate(-90 16 80)">z-score of selected reagents</text>
          <!-- gridline -->
          <line x1="40" y1="112" x2="404" y2="112" stroke="#1e293b" stroke-dasharray="3 3"/>
          <text x="408" y="115" class="ax">baseline</text>
          <!-- MW series (s1 cyan) -->
          <path fill="none" stroke="#0891b2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            d="M40,112 L69,106 L98,97 L127,86 L157,74 L186,62 L215,52 L244,45 L273,39 L302,35 L331,32 L360,30 L404,28"/>
          <circle cx="404" cy="28" r="4" fill="#0891b2" stroke="#0b1120" stroke-width="2"/>
          <text x="398" y="22" text-anchor="end" class="ax" fill="#22d3ee">mean MW  +2.1σ</text>
          <!-- cLogP series (s2 amber) -->
          <path fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            d="M40,112 L69,109 L98,104 L127,97 L157,89 L186,80 L215,72 L244,66 L273,61 L302,57 L331,55 L360,53 L404,52"/>
          <circle cx="404" cy="52" r="4" fill="#d97706" stroke="#0b1120" stroke-width="2"/>
          <text x="398" y="46" text-anchor="end" class="ax" fill="#f59e0b">mean cLogP  +1.6σ</text>
          <!-- danger zone -->
          <rect x="40" y="18" width="364" height="30" fill="rgba(251,113,133,.07)"/>
          <text x="46" y="14" class="ax" fill="#fb7185">hack zone — properties drifting, not chemistry improving</text>
        </svg>
        <div class="lg">
          <span><i style="background:var(--s1)"></i>mean MW of selected reagents</span>
          <span><i style="background:var(--s2)"></i>mean cLogP of selected reagents</span>
        </div>
        <div class="note">Wire this to the live run: one afternoon's work, and it converts "is the reward hacked?"
          from an argument into a measurement. Note the shape — most of the drift happens in the
          <b style="color:#94a3b8">first ~800 iterations</b>. TS finds the artifact long before it finishes searching.</div>
      </div>

      <!-- ═══ D6 — uncertainty in the wrong place ═══ -->
      <div class="card">
        <h3>6 · The uncertainty is in the wrong place <span class="tag">subtle</span></h3>
        <p class="why">Your <code>Reagent.current_std</code> is uncertainty about <b>a reagent's contribution to
          the proxy</b>. It is not, in any sense, uncertainty about <b>whether the proxy is right</b>.</p>
        <svg viewBox="0 0 420 152" style="width:100%;height:auto">
          <!-- LEFT: reagent posterior -->
          <text x="12" y="14" class="axl" fill="#34d399">✓ modelled — inside the model</text>
          <line x1="14" y1="106" x2="196" y2="106" stroke="#334155"/>
          <path fill="rgba(52,211,153,.2)" stroke="#34d399" stroke-width="2"
            d="M14,105 L38,103 L56,98 L72,88 L86,72 L98,54 L105,44 L112,54 L124,72 L138,88 L154,98 L172,103 L196,105 L196,106 L14,106 Z"/>
          <line x1="105" y1="40" x2="105" y2="106" stroke="#34d399" stroke-width="1" stroke-dasharray="3 2"/>
          <text x="105" y="122" text-anchor="middle" class="ax" fill="#34d399">current_mean ± current_std</text>
          <text x="105" y="134" text-anchor="middle" class="ax">tight, calibrated, updates every batch</text>
          <!-- divider -->
          <line x1="210" y1="10" x2="210" y2="140" stroke="#1e293b" stroke-dasharray="4 4"/>
          <!-- RIGHT: model validity -->
          <text x="224" y="14" class="axl" fill="#fb7185">✕ not modelled — about the model</text>
          <line x1="226" y1="106" x2="408" y2="106" stroke="#334155"/>
          <rect x="226" y="52" width="182" height="54" fill="rgba(251,113,133,.07)" stroke="#fb7185"
                stroke-width="1.5" stroke-dasharray="5 4"/>
          <text x="317" y="84" text-anchor="middle" font-size="26" font-weight="700"
                font-family="ui-monospace" fill="#fb7185">?</text>
          <text x="317" y="122" text-anchor="middle" class="ax" fill="#fb7185">no posterior exists</text>
          <text x="317" y="134" text-anchor="middle" class="ax">"is GIGN right about this pocket?"</text>
        </svg>
        <div class="note">A reagent can carry a beautifully tight posterior centred on a completely wrong value, and
          the bandit converges confidently either way — that is what it is designed to do. <b style="color:#94a3b8">Fix:</b>
          an ensemble over checkpoints. Nearly free, and it turns a re-ranker into an acquisition function.</div>
      </div>

      <!-- ═══ D7 — Suzuki collision ═══ -->
      <div class="card span">
        <h3>7 · Your reaction makes the chemotype you already measured as failing <span class="tag">rxn110 · Suzuki</span></h3>
        <p class="why">Suzuki makes <b>biaryls</b> — flat, lipophilic, rigid. Ten of your eighteen compounds failed as
          <b>nonspecific</b>, which at a shallow hydrophobic PPI site means promiscuity / aggregation. A lipophilicity-correlated
          score over a biaryl library will efficiently enrich exactly that.</p>
        <svg viewBox="0 0 900 128" style="width:100%;height:auto">
          <!-- funnel -->
          <g>
            <rect x="10" y="30" width="180" height="56" rx="11" fill="#0a0f1d" stroke="#0891b2"/>
            <text x="100" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#22d3ee">rxn110 product space</text>
            <text x="100" y="68" text-anchor="middle" class="ax">biaryl · ~2 × 10⁹</text>

            <line x1="194" y1="58" x2="240" y2="58" stroke="#475569" stroke-width="1.5" marker-end="url(#ar)"/>

            <rect x="246" y="30" width="180" height="56" rx="11" fill="#0a0f1d" stroke="#d97706"/>
            <text x="336" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#f59e0b">score ∝ lipophilic contact</text>
            <text x="336" y="68" text-anchor="middle" class="ax">Vina hphob · GIGN size bias</text>

            <line x1="430" y1="58" x2="476" y2="58" stroke="#475569" stroke-width="1.5" marker-end="url(#ar)"/>

            <rect x="482" y="30" width="180" height="56" rx="11" fill="rgba(251,113,133,.07)" stroke="#fb7185"/>
            <text x="572" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#fb7185">flat, greasy biaryls</text>
            <text x="572" y="68" text-anchor="middle" class="ax">TS converges here</text>

            <line x1="666" y1="58" x2="712" y2="58" stroke="#fb7185" stroke-width="1.5" marker-end="url(#arb)"/>

            <rect x="718" y="30" width="172" height="56" rx="11" fill="rgba(245,158,11,.07)" stroke="#f59e0b"/>
            <text x="804" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#f59e0b">10 / 18 nonspecific</text>
            <text x="804" y="68" text-anchor="middle" class="ax">already measured, in hand</text>
          </g>
          <defs><marker id="arg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#34d399"/></marker></defs>
          <path d="M804,90 L804,110 L100,110 L100,92" fill="none" stroke="#34d399" stroke-width="1.6"
                stroke-dasharray="5 4" marker-end="url(#arg)"/>
          <text x="452" y="124" text-anchor="middle" class="ax" fill="#34d399" font-size="9.5">
            close the loop: train the aggregation gate ON those 10, and TS stops rediscovering them</text>
        </svg>
        <div class="note">This is the hard-negative payoff made concrete. You do not need new data to fix this —
          you need to <b style="color:#94a3b8">use the failures you already paid for</b>, plus a hard cap on cLogP and
          aromatic ring count in the gate.</div>
      </div>

      <!-- ═══ RECOMMENDED WIRING ═══ -->
      <div class="card span">
        <h3>What to change in the objective <span class="tag">constraint, not loss</span></h3>
        <table>
          <thead><tr><th>Component</th><th>Today</th><th>Should be</th><th style="text-align:right">Why</th></tr></thead>
          <tbody>
            <tr><td style="color:#e2e8f0">TS objective</td><td style="color:#fb7185">GIGN pK, <code>ts_mode: maximize</code></td>
                <td style="color:#34d399">ROCS shape-match to the validated binding mode</td>
                <td class="n" style="color:#94a3b8">geometry is trustworthy at the resolution you need</td></tr>
            <tr><td style="color:#e2e8f0">GIGN / DeepAtom</td><td style="color:#fb7185">gradient (fine ranking)</td>
                <td style="color:#34d399">threshold gate on top-N (coarse triage)</td>
                <td class="n" style="color:#94a3b8">±1.3 works for 4-vs-8, not 4.7-vs-5.2</td></tr>
            <tr><td style="color:#e2e8f0">Off-distribution check</td><td style="color:#fb7185">none</td>
                <td style="color:#34d399">Vina ↔ GIGN disagreement</td>
                <td class="n" style="color:#94a3b8">orthogonal scorers diverge off-manifold</td></tr>
            <tr><td style="color:#e2e8f0">Aggregation filter</td><td style="color:#fb7185">none</td>
                <td style="color:#34d399">trained on your 10 nonspecific binders</td>
                <td class="n" style="color:#94a3b8">dominant measured failure mode at this site</td></tr>
            <tr><td style="color:#e2e8f0">Uncertainty</td><td style="color:#fb7185">reagent posterior only</td>
                <td style="color:#34d399">+ ensemble over GIGN checkpoints</td>
                <td class="n" style="color:#94a3b8">needed for acquisition, not just ranking</td></tr>
            <tr><td style="color:#e2e8f0">Sign convention</td><td style="color:#fb7185">pK higher-better vs house lower-better (G63)</td>
                <td style="color:#34d399">parse <code>GIGN_DELTAG</code></td>
                <td class="n" style="color:#94a3b8">otherwise the run maximizes weak binders</td></tr>
          </tbody>
        </table>
        <div class="note">Ordered by effort-to-value. Rows 4 and 6 are hours; row 1 is a config change; row 5 is an
          afternoon. None of them require new experimental data.</div>
      </div>

    </div>

    <!-- ═══ SIDEBAR: top-5 TS products — rendered live by _tsRlTop5() ═══ -->
    <div><div class="side" id="tsRlTop5"></div></div>
  </div>
`;

/* ─── Constants used by the gate. Single source of truth. ─────────────────
   Every number here is grounded in this repo or in the ACS Omega paper —
   do not invent new ones without a citation in the comment.               */
const _TS_RL = {
    // deepatom/model_split_data/02_pytorch/record_config.py
    PDBBIND_MEAN : 6.38,
    PDBBIND_STD  : 1.98,
    // GIGN CASF-2016 RMSE, published, on CRYSTAL poses. Its best case.
    GIGN_RMSE    : 1.3,
    // Z319334062: KD 21.5 uM (SPR)  ->  pK 4.67
    HIT_PK       : 4.67,
    // property gate — tuned to the measured failure mode (10/18 nonspecific)
    MAX_CLOGP    : 5.0,
    MAX_MW       : 500,
    // 4, not 3: a fused benzoxazolone counts as two rings, and that is the
    // chemotype of your own validated hit. A gate that rejects Z319334062 is
    // mis-tuned by construction — check any new threshold against it first.
    MAX_ARO      : 4,
    // Vina<->GIGN disagreement above this = off-distribution flag
    MAX_DISAGREE : 1.5,
};

/* ─── Cheap property estimates straight off the SMILES string ─────────────
   Deliberately crude: this is a UI-side triage display, not a QSAR model.
   The server-side gate should use RDKit. Kept here so the tab renders with
   zero round-trips while a run is in flight.                              */
function _tsRlCrude(smi) {
    if (!smi) return { mw: 0, clogp: 0, aro: 0 };
    // strip two-letter halogens first, or their lowercase tail reads as aromatic
    const cl = (smi.match(/Cl/g) || []).length, br = (smi.match(/Br/g) || []).length;
    const s  = smi.replace(/Cl/g, 'X').replace(/Br/g, 'X');
    const f  = (s.match(/F/g) || []).length,  io = (s.match(/I/g) || []).length;
    const C  = (s.match(/C/g) || []).length,  c  = (s.match(/c/g) || []).length;
    const N  = (s.match(/N/g) || []).length,  n  = (s.match(/n/g) || []).length;
    const O  = (s.match(/O/g) || []).length,  o  = (s.match(/o/g) || []).length;
    const S  = (s.match(/S/g) || []).length,  sa = (s.match(/s/g) || []).length;
    const P  = (s.match(/P/g) || []).length;
    const carbons = C + c, nitro = N + n, oxy = O + o, sulf = S + sa;

    // implicit-H estimate, then additive MW. Checked against aspirin (179 vs
    // 180.2), a 398 Da product (400) and a 474 Da product (475).
    const hyd = Math.max(0, carbons * 1.15 - c * 0.55 + nitro * 0.35);
    const mw  = Math.round(carbons * 12.011 + nitro * 14.007 + oxy * 15.999
              + sulf * 32.06 + P * 30.974 + f * 18.998 + cl * 35.453
              + br * 79.904 + io * 126.904 + hyd * 1.008);

    // fused systems share atoms, so atoms/5.6 tracks ring count closely enough
    const aro = Math.round((c + n + o + sa) / 5.6);

    // Crippen-style additive cLogP. Coefficients least-squares fitted to 14
    // molecules with EXPERIMENTAL logP (benzene, toluene, phenol, aniline,
    // aspirin, naphthalene, biphenyl, chloro/fluorobenzene, benzamide,
    // benzanilide, hexane, ethanol, 4-CF3-benzoic acid): fit RMSE 0.23.
    // Br/I carry no reference data — set by analogy to Cl, treat as indicative.
    const clogp = +(C * 0.676 + c * 0.372 - nitro * 1.256 - oxy * 0.728
                  + f * 0.343 + cl * 0.913 + br * 1.10 + io * 1.40 - 0.305).toFixed(1);

    return { mw, clogp, aro };
}

/* ─── The constraint stack. Returns flags, NOT a score. ───────────────────
   This is the whole point of the tab: GIGN is used as a THRESHOLD GATE, not
   as the objective. +-1.3 log units can separate pK 4 from pK 8; it cannot
   separate pK 4.7 from pK 5.2, which is the lead-op step you actually need. */
function _tsRlGate(smi, gignPk, vinaPk) {
    const p = _tsRlCrude(smi), f = [];
    let pass = true;

    if (p.clogp > _TS_RL.MAX_CLOGP) { f.push(['no', `✕ cLogP ${p.clogp}`]); pass = false; }
    else                            { f.push(['ok',  '✓ cLogP']); }

    if (p.mw > _TS_RL.MAX_MW)       { f.push(['no', `✕ MW ${p.mw}`]);       pass = false; }
    if (p.aro > _TS_RL.MAX_ARO)     { f.push(['no', `✕ ${p.aro} aromatic rings`]); pass = false; }

    // aggregation proxy: flat + greasy is the measured failure mode at this site
    if (p.clogp > 4.5 && p.aro >= 3) { f.push(['no', '✕ aggregator-like']); pass = false; }

    if (gignPk != null && vinaPk != null) {
        const d = Math.abs(gignPk - vinaPk);
        if (d > _TS_RL.MAX_DISAGREE) { f.push(['wr', `⚠ Vina/GIGN Δ ${d.toFixed(1)}`]); pass = false; }
        else                          { f.push(['ok', '✓ Vina/GIGN agree']); }
    }
    // The critic's number always ships WITH its error bar. Never bare.
    if (gignPk != null) f.push(['mu', `GIGN ${gignPk.toFixed(1)} ± ${_TS_RL.GIGN_RMSE}`]);
    return { pass, flags: f, ...p };
}

/* ─── Is this actually a SMILES? ─────────────────────────────────────────────
   Reagent identifiers are digit strings; product NAMES are two of them joined.
   A SMILES for anything the Suzuki/SnAr libraries produce contains at least one
   letter and a ring or branch character. Cheap, and it only has to separate
   "331274588_521851" from "O=C(Nc1ccc...)". */
function _tsRlIsSmiles(s) {
    if (typeof s !== 'string' || s.length < 3) return false;
    if (/^[\d]+[_+|][\d]+$/.test(s)) return false;   // the exact failure we saw
    if (!/[A-Za-z]/.test(s)) return false;           // pure digits/punctuation
    return /[cCnNoOsS]/.test(s);
}

/* Split a product name back into its two reagent ids, so the server can be
   asked for the real product SMILES via /ts_product_smiles/<rxn>/<a>/<b>. */
function _tsRlSplitName(name) {
    const m = String(name || '').match(/^(\d+)[_+|](\d+)$/);
    return m ? [m[1], m[2]] : null;
}

const _tsRlSmiFetching = new Set();

function _tsRlResolveSmiles(rxnKey, name) {
    // Returns a SMILES if already cached, else kicks off one fetch and returns
    // null. Re-renders when it lands, so the panel fills in rather than lying.
    const pair = _tsRlSplitName(name);
    if (!pair) return null;
    const key = `${pair[0]}|${pair[1]}`;
    if (typeof _tsProductCache !== 'undefined' && _tsProductCache[key]) {
        return _tsProductCache[key];
    }
    if (_tsRlSmiFetching.has(key)) return null;
    _tsRlSmiFetching.add(key);
    fetch(`/vina_visualization/ts_product_smiles/${encodeURIComponent(rxnKey)}/`
        + `${encodeURIComponent(pair[0])}/${encodeURIComponent(pair[1])}`)
        .then(r => (r.ok ? r.json() : null))
        .then(d => {
            if (d && d.smiles && typeof _tsProductCache !== 'undefined') {
                _tsProductCache[key] = d.smiles;
                _tsRlTop5();                      // fill in now that it is real
            }
        })
        .catch(() => {})
        .finally(() => _tsRlSmiFetching.delete(key));
    return null;
}

/* ─── Pull the top 5 products out of whatever live TS state exists ──────── */
function _tsRlHarvest() {
    const out = [];
    const seen = new Set();
    const rxnKey = (typeof _ts !== 'undefined' && _ts.rxnKey) || 'rxn110_suzuki';
    const push = (smiles, score, name) => {
        const id = smiles || name;
        if (!id || seen.has(id)) return;
        seen.add(id);
        // Keep the name either way; resolve the SMILES if what we were handed
        // is a product identifier rather than a structure.
        let smi = _tsRlIsSmiles(smiles) ? smiles : null;
        if (!smi && _tsRlIsSmiles(name)) smi = name;
        if (!smi) smi = _tsRlResolveSmiles(rxnKey, name || smiles);
        out.push({ smiles: smi, name: String(name || smiles || ''),
                   score: (score == null ? null : +score) });
    };
    // A product NAME is "<ridA>_<ridB>" (thompson_sampling.evaluate joins reagent
    // names with "_"). It is NOT a SMILES, and feeding one to the property
    // estimator yields MW 0 / cLogP -0.3 and a gate that passes everything —
    // which is exactly what the panel showed: "331274588+521851 … MW 0 … 0/5
    // rejected". Anything that is not plausibly a SMILES is held back and
    // resolved from the server instead of being scored as if it were one.
    // NOTE: _ts / _tsProductCache are declared `const` at the top level of a
    // classic script. Top-level `const` does NOT create a window property, so
    // `window._ts` is undefined even though `_ts` resolves fine. Always reach
    // these through a BARE identifier guarded by typeof — never off window.
    try {
        // 1. winners emitted by the TS pane
        const st = (typeof _ts !== 'undefined') ? _ts : null;
        const w = (st && st.tsState && st.tsState.winners) || [];
        w.slice().sort((a, b) => (b.score || 0) - (a.score || 0))
         .forEach(x => push(x.smiles, x.score, x.name || x.id));
        // 2. the persisted top5 belief cache
        if (out.length < 5 && st && Array.isArray(st._top5Cache))
            st._top5Cache.forEach(t => push(t.smiles, t.mu, t.name));
        // 3. anything primed into the product cache by the results table
        if (out.length < 5 && typeof _tsProductCache !== 'undefined')
            Object.values(_tsProductCache).forEach(s => push(s, null, ''));
    } catch (e) { console.warn('[ts-rl] harvest', e); }
    return out.slice(0, 5);
}

/* ─── Render the sidebar ──────────────────────────────────────────────── */
function _tsRlTop5() {
    const el = document.getElementById('tsRlTop5');
    if (!el) return;
    const mols = _tsRlHarvest();

    const head = `
      <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px">
        <span class="lbl" style="margin:0">Top 5 · TS products</span>
        <span style="font-size:9px;padding:2px 7px;border-radius:9px;background:rgba(8,145,178,.14);
              border:1px solid rgba(8,145,178,.4);color:var(--s1-ink);font-family:var(--mono)"
              id="tsRlRxn">${(typeof _ts !== 'undefined' && _ts.rxnKey) || 'rxn110'}</span>
      </div>`;

    if (!mols.length) {
        el.innerHTML = head + `
          <div class="banner bad">No TS products yet. Hit <b>🎲 Run TS</b> — this panel fills from
          the live run, then re-reads the ranking through the constraint stack.</div>
          ${_tsRlRefCard()}`;
        return;
    }

    let rejected = 0, pending = 0;
    const cards = mols.map((m, i) => {
        if (!m.smiles) {
            // Honest placeholder. Previously an unresolved product was pushed
            // through the estimator anyway and rendered as "MW 0 · cLogP -0.3
            // ✓ cLogP" — a PASS manufactured out of a parse failure.
            pending++;
            return `
        <div class="m5" style="cursor:default;opacity:.75">
          <div class="r"><span class="rank">#${i + 1}</span>
            <span style="font-size:10px;color:var(--ink-3);font-family:var(--mono)">resolving structure…</span>
            <span class="sc" style="color:var(--ink-3)">${m.score == null ? '—' : m.score.toFixed(2)}</span>
          </div>
          <div class="smi">${m.name}</div>
          <div class="flags"><span class="f mu">product id, not a SMILES — fetching</span></div>
        </div>`;
        }
        // Until the server scores it, show the critic's PRIOR MEAN — which is
        // exactly what an uninformed critic returns. That is the honest default.
        const gign = (m.gignPk != null) ? m.gignPk : _TS_RL.PDBBIND_MEAN;
        const g = _tsRlGate(m.smiles, gign, m.vinaPk);
        if (!g.pass) rejected++;
        return `
        <div class="m5" onclick="_tsRlToPose(${i})" title="load into the pose viewer">
          <div class="r">
            <span class="rank">#${i + 1}</span>
            <span style="font-size:10px;color:var(--ink-3);font-family:var(--mono)">MW ${g.mw} · cLogP ${g.clogp}</span>
            <span class="sc" style="color:${g.pass ? 'var(--good)' : 'var(--ink-3)'}">${
              m.score == null ? '—' : m.score.toFixed(2)}</span>
          </div>
          <div class="smi">${m.smiles}</div>
          <div class="flags">${g.flags.map(([k, t]) => `<span class="f ${k}">${t}</span>`).join('')}</div>
        </div>`;
    }).join('');

    const scored = mols.length - pending;
    el.innerHTML = head + `
      <div class="banner bad"><b>Gate verdict: ${rejected} / ${scored || 0} rejected.</b>${
        pending ? ` <span style="color:var(--ink-3)">(${pending} still resolving)</span>` : ''}
      Ranked by TS score, then re-read through the constraint stack.
      The ranking the bandit is proud of is not the ranking that survives.</div>
      ${cards}${_tsRlRefCard()}
      <div class="note" style="margin-top:12px">Click any card to load it into the pose viewer.
      The <b style="color:#94a3b8">± ${_TS_RL.GIGN_RMSE}</b> on every row is the same number —
      which is the point: it cannot separate these five.</div>`;
}

function _tsRlRefCard() {
    return `
      <div style="margin-top:12px;padding-top:11px;border-top:1px solid var(--line)">
        <div class="lbl" style="margin-bottom:7px">Reference — the one real datapoint</div>
        <div class="m5" style="border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.05);cursor:default">
          <div class="r">
            <span class="rank" style="color:var(--good);background:rgba(52,211,153,.14);
                  border-color:rgba(52,211,153,.4)">SPR</span>
            <span style="font-size:10px;color:var(--ink-3);font-family:var(--mono)">Z319334062</span>
            <span class="sc">21.5 µM</span>
          </div>
          <div class="flags">
            <span class="f ok">✓ specific · pK ${_TS_RL.HIT_PK}</span>
            <span class="f wr">Vina said 1.1 µM</span>
            <span class="f no">miss 1.29 log</span>
          </div>
        </div>
      </div>`;
}

/* ─── Hand a product to the pose viewer ──────────────────────────────────
   The pose tool is the only thing that can turn a SMILES into the posed
   complex GIGN needs, so the RL tab does not duplicate it — it delegates. */
function _tsRlToPose(i) {
    const m = _tsRlHarvest()[i];
    if (!m) return;
    const box = document.getElementById('poseSmiles');
    if (box) box.value = m.smiles;
    if (window.PoseGen && typeof PoseGen.open === 'function') { PoseGen.open(); PoseGen.build && PoseGen.build(); }
    else if (window.PoseGen && typeof PoseGen.build === 'function') PoseGen.build();
    else console.warn('[ts-rl] PoseGen not on the page; SMILES staged only:', m.smiles);
}


/* ═══ Adopting the REAL Pose Generation workspace ════════════════════════════
   The RL tab shows the working pose tool, not a mock-up of it.

   The workspace CANNOT be duplicated. Its markup lives once in hub.html inside
   #poseModal and is addressed by hardcoded ids — poseSmiles, poseBox3D,
   poseSvgBox, poseCx/Cy/Cz, poseMcScorer, the whole Vina weight panel. pose.js
   binds to those ids with getElementById. A second copy would give every one of
   them a duplicate, getElementById would return whichever came first, and both
   instances would break in ways that look like physics bugs.

   So we MOVE the nodes. appendChild relocates rather than clones, so there is
   still exactly one workspace, still one set of ids, and pose.js keeps working
   because they are the same nodes it already bound to. Leaving the tab moves
   them home.

   Two nodes are involved (hub.html ~1684 and ~1703): the input bar that holds
   the SMILES box, the stage buttons and Build, and the workspace itself —
   viewer, stage controls, and the 336px sidebar.                             */

function _tsRlLCA(a, b) {
    const chain = new Set();
    for (let n = a; n; n = n.parentElement) chain.add(n);
    for (let n = b; n; n = n.parentElement) if (chain.has(n)) return n;
    return null;
}

function _tsRlChildOf(ancestor, el) {
    let n = el;
    while (n && n.parentElement !== ancestor) n = n.parentElement;
    return n;
}

function _tsRlPoseNodes() {
    // Located structurally, not by CSS class — the classes are Tailwind soup
    // and change with layout tweaks; the ids are load-bearing and do not.
    const smi = document.getElementById('poseSmiles');
    const box = document.getElementById('poseBox3D');
    if (!smi || !box) return null;
    const lca = _tsRlLCA(smi, box);
    if (!lca) return null;
    const bar  = _tsRlChildOf(lca, smi);
    const work = _tsRlChildOf(lca, box);
    return (bar && work && bar !== work) ? { bar, work, home: lca } : null;
}

const _tsRlHome = { bar: null, work: null, parent: null, uploadBtn: null, uploadHTML: '' };

function _tsRlAdoptPose() {
    const host = document.getElementById('tsRlPoseHost');
    const n = _tsRlPoseNodes();
    if (!host || !n) return false;
    if (host.contains(n.work)) return true;              // already adopted

    // Remember exactly where they came from so leaving restores the pose modal.
    // Only capture the anchors on the FIRST adopt: on a re-adopt (after the pose
    // modal was opened and closed) the nodes are already home, and re-reading
    // nextSibling would record each other rather than the real neighbours.
    if (!_tsRlHome.parent) {
        _tsRlHome.parent = n.home;
        _tsRlHome.bar    = n.bar.nextSibling;
        _tsRlHome.work   = n.work.nextSibling;
    }

    host.appendChild(n.bar);
    host.appendChild(n.work);
    _tsRlAway(false);
    _tsRlRenameUploadBtn(true);
    _tsRlChrome(true);        // the TS run controls are not part of this workspace
    _tsRlUploads();           // #poseRecGrid / #poseRecCount come with the nodes
    _tsRlTopSecInstall();     // top-N TS products replace the ligand gallery
    _tsRlBoot();              // give the viewer something to draw
    _tsRlTickStart();
    return true;
}

function _tsRlReleasePose() {
    const n = _tsRlPoseNodes();
    if (!n || !_tsRlHome.parent) return;
    const host = document.getElementById('tsRlPoseHost');
    if (!host || !host.contains(n.work)) return;         // not ours to give back
    _tsRlTopSecRemove();      // before the move, so the ligand gallery goes home intact
    _tsRlRenameUploadBtn(false);
    _tsRlHome.parent.insertBefore(n.bar,  _tsRlHome.bar);
    _tsRlHome.parent.insertBefore(n.work, _tsRlHome.work);
    // If the RL pane is still the visible tab, say where the workspace went
    // rather than leaving an unexplained empty pane behind the modal.
    const pane = document.getElementById('tsPaneRl');
    _tsRlAway(!!pane && !pane.classList.contains('hidden'));
}

function _tsRlAway(show) {
    const el = document.getElementById('tsRlAway');
    if (el) el.style.display = show ? 'flex' : 'none';
}

/* Leaving the RL tab: give the nodes back AND undo everything else the tab
   changed about the surrounding modal. Kept separate from _tsRlReleasePose()
   because opening the pose modal must release the NODES while the RL pane is
   still the visible tab — in that case the chrome stays hidden. */
function _tsRlLeave() {
    _tsRlReleasePose();
    _tsRlChrome(false);
    _tsRlTickStop();
}

/* ═══ Item 2 — chrome that does not belong to the RL workspace ═══════════════
   Output Dir / Iterations / Viz speed / Run TS configure a Thompson-Sampling
   run; #_tsRxnPicker picks the reaction for one. Neither drives anything in the
   pose workspace, and both cost vertical space the 3D viewer wants. Hidden
   while the RL tab is up, restored on the way out — the inline display value is
   saved first, so a bar that was already hidden for its own reasons (the engine
   warning, say) stays hidden afterwards. */
function _tsRlChromeNodes() {
    // The config bar is located through its input, exactly the way ts_warmup.js
    // finds it — ts_run.js renames #tsSmartsInput to #tsOutputDirInput at open
    // time, so both spellings have to be tried.
    const cfg = document.getElementById('tsOutputDirInput')?.closest('div')
             || document.getElementById('tsSmartsInput')?.closest('div');
    return [cfg,
            document.getElementById('_tsRxnPicker'),
            document.getElementById('_tsEngineWarn')].filter(Boolean);
}

const _tsRlChromeSaved = new Map();     // node -> its inline display before we hid it

function _tsRlChrome(hide) {
    if (hide) {
        _tsRlChromeNodes().forEach(n => {
            if (!_tsRlChromeSaved.has(n)) _tsRlChromeSaved.set(n, n.style.display || '');
            n.style.display = 'none';
        });
    } else {
        _tsRlChromeSaved.forEach((prev, n) => { n.style.display = prev; });
        _tsRlChromeSaved.clear();
    }
}

/* ═══ Item 3 — the uploaded receptors ════════════════════════════════════════
   #poseRecGrid and #poseRecCount travel with the workspace, so they are already
   on screen — they are just empty, because pose.js only fetches the gallery
   from PoseGen.open(), and the RL tab never calls it. _renderUploads(true)
   forces the fetch past its _uploadsLoaded short-circuit and repaints both the
   grid and the "— / N files" badge. */
function _tsRlUploads() {
    try {
        const init = window.PoseGen && PoseGen.init;
        if (init && typeof init._renderUploads === 'function') init._renderUploads(true);
    } catch (e) { console.warn('[ts-rl] uploads', e); }
}

/* First adopt has to do PoseGen.open()'s work minus the modal itself: load the
   pose config (default receptor, docking box, default SMILES) and draw a stage.
   Later adopts only need the redraw — PoseGen.close() calls Plotly.purge on
   #poseBox3D, so a workspace coming back from the modal arrives blank.
   Deferred a frame so the host has been laid out and the viewer sizes to it. */
let _tsRlBooted = false;
function _tsRlBoot() {
    const PG = window.PoseGen;
    if (!PG) return;
    requestAnimationFrame(() => {
        try {
            if (!_tsRlBooted) {
                _tsRlBooted = true;
                if (typeof PG._loadConfig === 'function') PG._loadConfig();
            }
            if (typeof PG.stage === 'function') PG.stage(PG._stage || 'random');
        } catch (e) { console.warn('[ts-rl] boot', e); }
    });
}

/* ═══ Top-N TS products, in place of "Uploaded ligands" ══════════════════════
   Same source of truth as #tsTsBars — _ts._activeTopRaw, straight off the
   /ts_top5 poll — so the two panels can never disagree. The server ranking now
   carries 20 entries (_TOP_N in ts_routes.py) and this slices to whatever the
   picker says; #tsTsBars still slices to 5 and looks exactly as it did.

   PRESENTATION IS INVERTED relative to #tsTsBars, which is the point of the
   request. There, the numeric row is always on screen and the reaction
   (BB ⊕ BB → product) is what you get on hover. Here the reaction is the card
   — three real RDKit structures, the thing you actually recognise a molecule
   by — and the numbers (product SMILES, n, μ, σ, best) come up on hover.

   The ligand gallery is HIDDEN, not removed: #poseLigGrid is the same node the
   pose modal drags cards into, and this section is spliced in next to it and
   deleted again on release. Nothing about the pose tool changes once the
   workspace goes home.                                                       */

let _TS_RL_TOPN = 5;                 // the picker's value; page-session lived
const _tsRlPrevRank = {};            // rid -> last rendered position, for ⬆/⬇
const _tsRlLig = { hdr: null, grid: null, hdrDisp: '', gridDisp: '' };

function _tsRlTopSecInstall() {
    if (document.getElementById('tsRlTopSec')) return true;
    const grid = document.getElementById('poseLigGrid');
    const hdr  = grid && grid.previousElementSibling;
    // Only proceed if the previous sibling really is the "Uploaded ligands"
    // header. Positional DOM assumptions rot; this one announces when it has.
    if (!grid || !hdr || !hdr.querySelector('#poseLigCount')) {
        console.warn('[ts-rl] uploaded-ligands section not found — top-N panel skipped');
        return false;
    }
    _tsRlLig.hdr = hdr; _tsRlLig.grid = grid;
    _tsRlLig.hdrDisp = hdr.style.display || '';
    _tsRlLig.gridDisp = grid.style.display || '';
    hdr.style.display = 'none';
    grid.style.display = 'none';

    const sec = document.createElement('div');
    sec.id = 'tsRlTopSec';
    sec.style.cssText = 'margin-top:8px';
    sec.innerHTML =
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:8px">'
      + '<p class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider" style="margin:0">'
      + 'Top TS products</p>'
      + '<input id="tsRlTopN" class="rln" type="number" min="1" max="20" value="' + _TS_RL_TOPN + '"'
      + ' title="how many of the ranking to show (the server keeps 20)"'
      + ' onchange="_tsRlSetTopN(this.value)" oninput="_tsRlSetTopN(this.value)">'
      + '<span id="tsRlTopCount" style="font-size:9px;padding:2px 8px;border-radius:9px;'
      + 'background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.35);color:#22d3ee;'
      + 'font-family:ui-monospace,monospace">—</span></div>'
      + '<div id="tsRlTopList"></div>';
    hdr.parentNode.insertBefore(sec, hdr);
    _tsRlTopSecRender();
    return true;
}

function _tsRlTopSecRemove() {
    document.getElementById('tsRlTopSec')?.remove();
    _tsRlRowHide(true);
    if (_tsRlLig.hdr)  _tsRlLig.hdr.style.display  = _tsRlLig.hdrDisp;
    if (_tsRlLig.grid) _tsRlLig.grid.style.display = _tsRlLig.gridDisp;
    _tsRlLig.hdr = _tsRlLig.grid = null;
}

function _tsRlSetTopN(v) {
    const n = Math.max(1, Math.min(20, parseInt(v, 10) || 5));
    if (n === _TS_RL_TOPN) return;
    _TS_RL_TOPN = n;
    _tsRlTopSecRender();
}

/* The active job's UNSLICED ranking. _ts._activeTopRaw is set by _tsPollTop5
   in ts_ui.js through the same index rescue #tsTsBars uses. Falls back to the
   per-job map if this file is loaded against an older ts_ui.js. */
function _tsRlRanking() {
    if (typeof _ts === 'undefined') return [];
    if (Array.isArray(_ts._activeTopRaw)) return _ts._activeTopRaw;
    if (typeof _tsTop5ByJob !== 'undefined') return _tsTop5ByJob[_ts._activeJobIdx ?? 0] || [];
    return [];
}

function _tsRlRxnKey() {
    if (typeof _ts === 'undefined') return '';
    const j = (_ts.jobs || [])[_ts._activeJobIdx ?? 0] || {};
    return j.key || j.short_name || _ts.rxnKey || '';
}

const _tsRlEnc = encodeURIComponent;

function _tsRlTopSecRender() {
    const list = document.getElementById('tsRlTopList');
    if (!list) return;
    const rxn  = _tsRlRxnKey();
    const rank = _tsRlRanking().slice(0, _TS_RL_TOPN);

    const badge = document.getElementById('tsRlTopCount');
    if (badge) {
        const avail = _tsRlRanking().length;
        badge.textContent = rank.length + ' of ' + avail;
        // Say so rather than silently showing fewer rows than asked for.
        badge.title = (avail < _TS_RL_TOPN)
            ? `the ranking only holds ${avail} reagents so far — asked for ${_TS_RL_TOPN}`
            : `showing the top ${rank.length} of ${avail} ranked reagents`;
    }

    if (!rank.length) {
        list.innerHTML =
            '<div style="font-size:11px;color:#64748b;background:#0b1120;border:1px dashed #1e293b;'
          + 'border-radius:11px;padding:12px;text-align:center;line-height:1.5">'
          + 'No ranking yet — hit <b style="color:#a78bfa">🎲 Run TS</b>. '
          + 'This fills from the same poll that feeds the reagent panel.</div>';
        list._sig = null;
        return;
    }

    // Rebuild only when the SET changes. Re-issuing the markup re-requests every
    // <object> SVG, so a blind 3s rebuild would flicker three structures per card.
    const sig = rank.map(t => t.name + '>' + (t.partner || '—')).join('|') + '::' + rxn;
    if (list._sig === sig) { _tsRlTopSecPatch(list, rank); return; }
    list._sig = sig;

    list.innerHTML = rank.map((t, i) => {
        const rid = String(t.name), par = String(t.partner || '—');
        const has = par && par !== '—';
        const best = (t.best != null) ? Number(t.best) : null;
        const args = `'${rxn}','${rid}','${par}'`;
        const mol = (id, w, h) => rxn
            ? `<object type="image/svg+xml" data="/vina_visualization/ts_mol_svg/${_tsRlEnc(rxn)}/${_tsRlEnc(id)}?w=${w}&h=${h}"
                       width="${w}" height="${h}" style="pointer-events:none;display:block"
                       aria-label="structure of ${id}"></object>` : '';
        const prod = (has && rxn)
            ? `<object class="rlm-prod" type="image/svg+xml"
                       data="/vina_visualization/ts_product_svg/${_tsRlEnc(rxn)}/${_tsRlEnc(rid)}/${_tsRlEnc(par)}?w=240&h=84"
                       width="240" height="84" aria-label="product of ${rid} and ${par}"></object>`
            : `<div style="height:84px;display:flex;align-items:center;justify-content:center;
                    color:#475569;font-size:10px;font-family:ui-monospace,monospace">no partner yet</div>`;
        return `
        <div class="rlm" data-rid="${rid}" data-par="${par}"
             onclick="_tsRlLoadProduct(${args})"
             onmouseenter="_tsRlRowShow(event,${args})" onmouseleave="_tsRlRowHide()"
             title="click to load this product into the pose viewer">
          <div class="rlm-h">
            <span class="rlm-rank">#${i + 1}</span>
            <span data-el="mv" style="font-size:10px"></span>
            <span class="rlm-best" data-el="best">${best == null ? '—' : best.toFixed(4)}</span>
            <button class="rlm-pin" title="pin the reaction card (building-block CSV paths)"
                    onclick="_tsHoverPin(event,${args},${best == null ? 'null' : best})">📌</button>
          </div>
          <div class="rlm-rx">
            <div class="rlm-bb">${mol(rid, 100, 48)}<span>${rid}</span></div>
            <div class="rlm-op">+</div>
            <div class="rlm-bb">${has ? mol(par, 100, 48) : ''}<span style="color:#0f766e">${par}</span></div>
          </div>
          <div class="rlm-dn">↓</div>
          ${prod}
          <div class="rlm-hint">⌬ click to load into the pose viewer</div>
        </div>`;
    }).join('');

    _tsRlTopSecPatch(list, rank);
}

/* Numbers move every poll; structures do not. Patch the former in place. */
function _tsRlTopSecPatch(list, rank) {
    rank.forEach((t, i) => {
        const card = list.children[i];
        if (!card) return;
        const best = (t.best != null) ? Number(t.best) : null;
        const bEl = card.querySelector('[data-el="best"]');
        if (bEl) bEl.textContent = (best == null) ? '—' : best.toFixed(4);
        const prev = _tsRlPrevRank[t.name];
        const mv = card.querySelector('[data-el="mv"]');
        if (mv) {
            mv.innerHTML = (prev == null || prev === i) ? ''
                : (i < prev ? '<span style="color:#34d399" title="moved up">⬆</span>'
                            : '<span style="color:#f87171" title="moved down">⬇</span>');
        }
    });
    rank.forEach((t, i) => { _tsRlPrevRank[t.name] = i; });
}

/* ── hover: delegate to ts_ui.js's card ──────────────────────────────────────
   Since the #tsTsBars swap, _tsHoverCard IS the numbers panel — product
   SMILES, n / μ / σ, partner, best score — which is exactly what these cards
   want on hover. It resolves the posterior through _tsReagentStats, which
   falls back to _ts._activeTopRaw, so a card at rank 12 (past the five rows
   #tsTsBars shows) still finds its numbers. One implementation, not two. */
function _tsRlRowShow(ev, rxn, rid, par) {
    if (typeof _tsHoverCard !== 'function') return;
    const t = _tsRlRanking().find(x => String(x.name) === String(rid));
    _tsHoverCard(ev, rxn, rid, par, (t && t.best != null) ? t.best : null);
}

function _tsRlRowHide(force) {
    if (typeof _tsHoverCardHide === 'function') _tsHoverCardHide();
    // Leaving the tab has to drop a PINNED card too — it lives on <body>, so it
    // would otherwise outlive the panel that opened it.
    if (force && typeof _tsHoverUnpin === 'function') _tsHoverUnpin();
}

/* ── click: hand the product to the pose viewer ─────────────────────────── */
function _tsRlSetSmiles(smi) {
    const box = document.getElementById('poseSmiles');
    if (!box) return;
    box.value = smi;
    box.placeholder = '';
    box.dispatchEvent(new Event('input', { bubbles: true }));
    if (window.PoseGen && typeof PoseGen.build === 'function') PoseGen.build();
}

function _tsRlLoadProduct(rxn, rid, par) {
    const box = document.getElementById('poseSmiles');
    if (!box) return;
    if (!par || par === '—') { box.placeholder = `${rid} has no partner yet`; return; }
    const key = String(rid) + '|' + String(par);
    const cached = (typeof _tsProductCache !== 'undefined') ? _tsProductCache[key] : undefined;
    if (cached) { _tsRlSetSmiles(cached); return; }
    // Not cached (or cached as the empty in-flight marker) — go and get it. Do
    // NOT fall back to the reagent's own SMILES: that is the building block, not
    // what the bandit scored, and posing it would answer a different question.
    box.value = '';
    box.placeholder = `fetching product ${rid} ⊕ ${par}…`;
    fetch(`/vina_visualization/ts_product_smiles/${_tsRlEnc(rxn)}/${_tsRlEnc(rid)}/${_tsRlEnc(par)}`)
        .then(r => (r.ok ? r.json() : null))
        .then(d => {
            const smi = d && d.smiles;
            if (typeof _tsProductCache !== 'undefined') _tsProductCache[key] = smi || '';
            if (smi) _tsRlSetSmiles(smi);
            else box.placeholder = `no clean product for ${rid} ⊕ ${par}`;
        })
        .catch(() => { box.placeholder = 'product fetch failed'; });
}

/* ═══ Item 1 (RL side) — the 3s tick ═════════════════════════════════════════
   The reagent panel's own 3s poll lives in ts_ui.js. This tick is the RL tab's
   share of the same job: keep the top-5 poll alive, re-assert the chrome (the
   reaction picker rebuilds itself on config load and sets its own display back
   to flex), and refresh the diagnostics sidebar while it is open. */
let _tsRlTimer = null;
function _tsRlTickStart() { if (!_tsRlTimer) _tsRlTimer = setInterval(_tsRlTick, 3000); }
function _tsRlTickStop()  { if (_tsRlTimer) { clearInterval(_tsRlTimer); _tsRlTimer = null; } }

function _tsRlTick() {
    const pane = document.getElementById('tsPaneRl');
    if (!pane || pane.classList.contains('hidden')) { _tsRlTickStop(); return; }
    _tsRlChrome(true);
    if (typeof _tsStartTop5Polling === 'function') _tsStartTop5Polling();  // idempotent
    _tsRlTopSecRender();      // cheap: rebuilds only when the reagent SET changes
    const d = document.getElementById('tsRlDiag');
    if (d && d.style.display !== 'none') _tsRlTop5();
}

/* ═══ Item 4 — the pose modal always wins ════════════════════════════════════
   Moving the workspace into the RL tab means #poseModal is genuinely empty
   while the tab is up: opening it from the sidebar showed a header and two
   bare <!-- Input bar --> / <!-- Body --> comments. That is a real regression
   from the node-moving design, and the fix belongs at the point of conflict —
   whoever opens the pose modal takes the workspace back first.

   Two independent hooks, because neither alone is safe:
     · wrapping PoseGen.open covers every caller (sidebar dispatch, _tsRlToPose,
       anything added later), but pose.js wraps PG.open in its own IIFE at load
       time, so ours can be overwritten depending on script order — hence the
       re-wrap on every RL tab activation;
     · a capture-phase click listener on [data-nav="pose"] fires before hub.js's
       bubble-phase dispatcher regardless of who wrapped what.
   Both are idempotent, so running both is free.                              */
function _tsRlWrapPoseGen() {
    const PG = window.PoseGen;
    if (!PG || typeof PG.open !== 'function') return false;
    if (!PG.open.__rlWrapped) {
        const open = PG.open.bind(PG);
        const w = function () { _tsRlReleasePose(); return open.apply(this, arguments); };
        w.__rlWrapped = true;
        PG.open = w;
    }
    if (typeof PG.close === 'function' && !PG.close.__rlWrapped) {
        const close = PG.close.bind(PG);
        const w = function () {
            const r = close.apply(this, arguments);
            // If the RL tab is still the visible pane, take the workspace back —
            // otherwise closing the pose modal would leave the RL tab showing an
            // empty host, which is the same bug pointing the other way.
            const pane = document.getElementById('tsPaneRl');
            if (pane && !pane.classList.contains('hidden')) _tsRlAdoptPose();
            return r;
        };
        w.__rlWrapped = true;
        PG.close = w;
    }
    return true;
}

if (!window.__tsRlPoseNavWired) {
    window.__tsRlPoseNavWired = true;
    document.addEventListener('click', function (e) {
        const t = (e.target && e.target.closest)
                ? e.target.closest('[data-nav="pose"], #sidebarPoseGenBtn') : null;
        if (t) _tsRlReleasePose();
    }, true);   // capture: ahead of hub.js's bubble-phase [data-nav] dispatcher
}

/* The "⤓ Upload .pdb" button becomes the diagnostics toggle — but only while
   the workspace is living in the RL tab. It is the SAME node the pose modal
   uses, so a permanent rename would delete the upload affordance from the pose
   tool as well. Swap on adopt, swap back on release. */
function _tsRlRenameUploadBtn(toRl) {
    // Cache the NODE. The attribute selector only matches before the rename —
    // renaming strips onclick, so re-querying on release found nothing and the
    // ⤓ Upload .pdb button stayed a dead "RL workflow" button in the pose modal.
    const btn = _tsRlHome.uploadBtn
             || document.querySelector("[onclick*='posePdbFile']")
             || document.getElementById('poseUploadBtn');
    if (!btn) return;
    if (toRl) {
        if (!_tsRlHome.uploadBtn) {
            _tsRlHome.uploadBtn  = btn;
            _tsRlHome.uploadHTML = btn.innerHTML;
            _tsRlHome.uploadClick = btn.getAttribute('onclick');
            _tsRlHome.uploadStyle = btn.getAttribute('style') || '';
        }
        btn.innerHTML = '🧪 RL workflow';
        btn.title = 'Show the RL loop diagnostics over the pose workspace';
        btn.removeAttribute('onclick');
        btn.onclick = _tsRlToggleDiag;
        btn.setAttribute('style', _tsRlHome.uploadStyle
            .replace(/border:[^;]*;/, 'border:1px solid #7f1d3a;')
            .replace(/background:[^;]*;/, 'background:rgba(127,29,58,.35);')
            .replace(/color:[^;]*;/, 'color:#fecdd3;'));
    } else if (_tsRlHome.uploadBtn) {
        btn.innerHTML = _tsRlHome.uploadHTML;
        btn.title = '';
        btn.onclick = null;
        if (_tsRlHome.uploadClick) btn.setAttribute('onclick', _tsRlHome.uploadClick);
        btn.setAttribute('style', _tsRlHome.uploadStyle);
    }
}

function _tsRlToggleDiag(force) {
    const d = document.getElementById('tsRlDiag');
    if (!d) return;
    const show = (typeof force === 'boolean') ? force : d.style.display === 'none';
    d.style.display = show ? 'block' : 'none';
    if (show) _tsRlTop5();
}

/* ─── Install ─────────────────────────────────────────────────────────── */
(function _tsRlInstall() {
    const go = () => {
        if (document.getElementById('tsPaneRl')) return;          // idempotent
        const tabTs = document.getElementById('tsTabTs');
        const paneR = document.getElementById('tsPaneResults');
        if (!tabTs || !paneR) return setTimeout(go, 300);          // modal not built yet

        // 1. scoped stylesheet
        if (!document.getElementById('tsRlStyle')) {
            const s = document.createElement('style');
            s.id = 'tsRlStyle'; s.textContent = _TS_RL_STYLE;
            document.head.appendChild(s);
        }
        // 2. tab button, between TS Belief and Monitor
        const b = document.createElement('button');
        b.id = 'tsTabRl'; b.dataset.tab = 'rl';
        b.className = 'px-4 py-1.5 border-r border-slate-700 bg-transparent text-slate-400 hover:bg-slate-800 hover:text-white';
        b.textContent = '🧪 RL';
        b.onclick = () => _tsTab('rl');
        tabTs.insertAdjacentElement('afterend', b);

        // 3. pane, before the Monitor pane
        const p = document.createElement('div');
        p.id = 'tsPaneRl';
        p.className = 'hidden flex flex-1 min-h-0 overflow-hidden';
        p.innerHTML =
            '<div id="tsRlPoseHost" style="flex:1;min-height:0;display:flex;'
          + 'flex-direction:column;overflow:hidden;width:100%">'
          + '<div id="tsRlAway" style="display:none;flex:1;align-items:center;'
          + 'justify-content:center;text-align:center;padding:40px;font-size:12px;'
          + 'color:#64748b;line-height:1.7">The pose workspace is open in the Pose '
          + 'Generation window.<br>Close it and this tab takes it back — there is '
          + 'only ever one copy.</div>'
          + '</div>'
          + '<div id="tsRlDiag" style="display:none;position:absolute;inset:0;'
          + 'z-index:40;overflow-y:auto;background:var(--surface-0);padding-bottom:40px">'
          + '<div style="position:sticky;top:0;z-index:2;display:flex;align-items:center;'
          + 'gap:10px;padding:10px 24px;background:#0f172a;border-bottom:1px solid #334155">'
          + '<b style="font-size:13px">🧪 RL loop — diagnostics</b>'
          + '<span style="font-size:10.5px;color:var(--ink-3)">the pose workspace is underneath</span>'
          + '<button onclick="_tsRlToggleDiag(false)" style="margin-left:auto;padding:5px 13px;'
          + 'border-radius:9px;border:1px solid #334155;background:#0b1120;color:#94a3b8;'
          + 'font:inherit;font-size:12px;font-weight:600;cursor:pointer">✕ Back to workspace</button>'
          + '</div>' + _TS_RL_BODY + '</div>';
        p.style.position = 'relative';
        paneR.insertAdjacentElement('beforebegin', p);

        // 4. register with the existing tab machinery.
        //    _TS_PANE / _TS_TAB_ACTIVE are `const` BINDINGS, so they are NOT
        //    properties of window — reach them as bare identifiers. The objects
        //    themselves are mutable, so adding a key is legal, and that is why
        //    this file needs no edit to ts_core.js.
        let registered = false;
        try {
            if (typeof _TS_PANE !== 'undefined') { _TS_PANE.rl = 'tsPaneRl'; registered = true; }
            if (typeof _TS_TAB_ACTIVE !== 'undefined') _TS_TAB_ACTIVE.rl = 'bg-rose-900 text-rose-200';
        } catch (e) { console.warn('[ts-rl] registry', e); }

        // 5. refresh the sidebar whenever the tab is opened.
        //    `function _tsTab(){}` IS var-like, so it does live on window and
        //    reassigning it is picked up by both inline onclick= handlers and
        //    bare-identifier callers.
        const orig = (typeof _tsTab === 'function') ? _tsTab : null;
        if (orig && !orig.__rlWrapped) {
            const wrapped = function (tab) {
                orig(tab);
                if (tab === 'rl') { _tsRlWrapPoseGen(); _tsRlAdoptPose(); _tsRlTop5(); }
                else             { _tsRlLeave(); }
            };
            wrapped.__rlWrapped = true;
            window._tsTab = wrapped;
        }

        // Closing the whole TS modal must also hand the workspace back: the
        // nodes would otherwise sit inside a display:none modal and the pose
        // tool would open empty with nothing on screen explaining why.
        if (typeof _tsHide === 'function' && !_tsHide.__rlWrapped) {
            const hide = _tsHide;
            const wrappedHide = function () { _tsRlLeave(); return hide.apply(this, arguments); };
            wrappedHide.__rlWrapped = true;
            window._tsHide = wrappedHide;
        }

        // PoseGen may not be on the page yet (script order is not guaranteed).
        // Try now, then a few times, then again on every RL activation.
        if (!_tsRlWrapPoseGen()) {
            let tries = 0;
            const t = setInterval(() => {
                if (_tsRlWrapPoseGen() || ++tries > 20) clearInterval(t);
            }, 400);
        }

        // 6. fallback: if the registry was unreachable (ts_core.js not loaded,
        //    or renamed), drive the panes ourselves so the tab still works
        //    instead of opening blank.
        if (!registered) {
            console.warn('[ts-rl] _TS_PANE unreachable — using standalone pane switching');
            b.onclick = () => {
                ['tsPaneWarmup','tsPaneTs','tsPaneResults'].forEach(id => {
                    const el = document.getElementById(id); if (el) el.classList.add('hidden');
                });
                p.classList.remove('hidden');
                document.querySelectorAll('[data-tab]').forEach(x => {
                    x.className = 'px-4 py-1.5 border-r border-slate-700 ' +
                      (x === b ? 'bg-rose-900 text-rose-200'
                               : 'bg-transparent text-slate-400 hover:bg-slate-800 hover:text-white');
                });
                _tsRlWrapPoseGen();
                _tsRlAdoptPose();
                _tsRlTop5();
            };
            // ...and the other tabs have to give it back, or standalone mode
            // strands the workspace inside a hidden pane.
            ['tsTabWarmup', 'tsTabTs', 'tsTabResults'].forEach(id => {
                const t = document.getElementById(id);
                if (t) t.addEventListener('click', _tsRlLeave);
            });
        }
        _tsRlTop5();
        console.log('[ts-rl] installed' + (registered ? '' : ' (standalone mode)'));
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go);
    else go();
})();