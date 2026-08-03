// =============================================================================
// chembert_chat.js
// All ChemBERT Attention Visualizer JavaScript:
//   - Modal show/hide (_adjShow, _adjHide)
//   - Mode toggle (single / compare)
//   - Model selector and preset loading
//   - Single-compound visualization (_runSingle)
//   - Compare mode (_runCompare)
//   - 3D scatter + bonds (_render3D)
//   - Bar chart (_renderBar, _highlightAtom)
//   - Atom table (_renderTable)
//   - Prepare training data (_runPrepare, _autofillSmilesPath)
//   - Fine-tune panel (_startFinetune, _ftStream, _ftDone, _ftLog)
//   - Smart scroll (_smartScroll)
//   - Mini-chat aliases + _adjShow/_adjHide hooks for shared panel
//     (delegates to elion_mini_chat.js theme + _attnMiniWelcome)
//
// Depends on: elion_mini_chat.js (must load first)
// Loaded by: hub.html dynamic script loader
// =============================================================================

// ══ ChemBERT JS ══════════════════════════════════════════════════════════════
// ═════════════════════════════════════════════════════════════════════════════
// CHEM-BERT 3D Visualizer  —  model switching + single + compare
// ═════════════════════════════════════════════════════════════════════════════

let _currentMode      = 'single';
let _activeModelPath  = '';       // updated whenever a model is selected/loaded
let _activeModelTag   = '';       // 'finetuned' | 'pretrained' | ''

// ── Init ──────────────────────────────────────────────────────────────────────
$(function () {
    $('#adjWeightBtn').on('click', _adjShow);
    $('#smilesA').on('keypress',     function(e){ if(e.which===13) _runSingle(); });
    $('#smilesCompA').on('keypress', function(e){ if(e.which===13) _runCompare(); });
    $('#smilesCompB').on('keypress', function(e){ if(e.which===13) _runCompare(); });
    $('#customModelPath').on('keypress', function(e){ if(e.which===13) _loadCustomPath(); });
    _fetchPresets();
});

// ── Modal show/hide ───────────────────────────────────────────────────────────
function _adjShow() { $('#adjModal').removeClass('hidden').addClass('flex'); }
function _adjHide() { $('#adjModal').addClass('hidden').removeClass('flex'); }

// ── Mode toggle ───────────────────────────────────────────────────────────────
function _setMode(mode) {
    _currentMode = mode;
    if (mode === 'single') {
        $('#modeSingle').addClass('active'); $('#modeCompare').removeClass('active');
        $('#inputSingle').show();           $('#inputCompare').hide();
        $('#panelSingle').show();           $('#panelCompare').hide();
    } else {
        $('#modeCompare').addClass('active'); $('#modeSingle').removeClass('active');
        $('#inputCompare').show();            $('#inputSingle').hide();
        $('#panelCompare').show();            $('#panelSingle').hide();
    }
}

// ── Model selector ────────────────────────────────────────────────────────────
function _fetchPresets() {
    const url = (_activeTool === 'vina')
        ? '/vina_visualization/chembert_models'
        : '/attention_visualization/chembert_models';
    $.getJSON(url, function(d) {
        if (d.status !== 'success') return;
        const container = $('#presetBtns').empty();
        d.presets.forEach(function(p) {
            const btn = $('<button>')
                .addClass('preset-btn')
                .text(p.label)
                .attr('data-path', p.path)
                .attr('data-tag', p.tag)
                .on('click', function() {
                    _selectPreset(p.path, p.tag, $(this));
                });
            container.append(btn);
        });
        // Auto-select finetuned if available
        const first = container.find('.preset-btn').first();
        if (first.length) first.click();
    });
}

function _selectPreset(path, tag, btn) {
    _activeModelPath = path;
    _activeModelTag  = tag;
    $('#presetBtns .preset-btn').removeClass('active pretrained-active');
    btn.addClass(tag === 'pretrained' ? 'pretrained-active' : 'active');
    $('#customModelPath').val('');
    _updateModelBadge(tag);
}

function _loadCustomPath() {
    const path = $('#customModelPath').val().trim();
    if (!path) return;
    _activeModelPath = path;
    _activeModelTag  = '';   // unknown until server responds
    $('#presetBtns .preset-btn').removeClass('active pretrained-active');
    _updateModelBadge('loading');
}

function _updateModelBadge(tag) {
    const badge = $('#modelTagBadge');
    badge.removeClass('tag-finetuned tag-pretrained tag-none');
    if (tag === 'finetuned') {
        badge.addClass('tag-finetuned').text('✓ Finetuned');
    } else if (tag === 'pretrained') {
        badge.addClass('tag-pretrained').text('✓ Pretrained (no score)');
    } else if (tag === 'loading') {
        badge.addClass('tag-none').text('Custom — will detect on run');
    } else {
        badge.addClass('tag-none').text('— not loaded —');
    }
}

// ── Score display (tag-aware) ─────────────────────────────────────────────────
function _renderScore(scoreVal, modelTag, valElId, noteElId) {
    const valEl  = $('#' + valElId);
    const noteEl = $('#' + noteElId);

    if (modelTag === 'pretrained') {
        valEl.text('N/A').attr('class', 'score-val score-na');
        noteEl.text('Pretrained model — no regression head');
        return;
    }
    if (scoreVal === null || scoreVal === undefined) {
        valEl.text('—').attr('class', 'score-val score-neutral');
        noteEl.text('kcal/mol');
        return;
    }
    const cls = scoreVal < -7 ? 'score-better' : scoreVal < -5 ? 'score-neutral' : 'score-worse';
    valEl.text(scoreVal.toFixed(3) + ' kcal/mol').attr('class', 'score-val ' + cls);
    noteEl.text('Binding affinity (more negative = stronger)');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function _setLoading(spinId, plotId, msg) {
    $('#'+spinId).show().find('span:last').text(msg || 'Loading…');
    $('#'+plotId).css('opacity', 0);
}
function _clearLoading(spinId, plotId) {
    $('#'+spinId).hide(); $('#'+plotId).css('opacity', 1);
}

// ── Single mode ───────────────────────────────────────────────────────────────
function _runSingle() {
    const smiles = $('#smilesA').val().trim();
    if (!smiles)            { alert('Enter a SMILES string.'); return; }
    if (!_activeModelPath)  { alert('Select a model first.'); return; }

    $('#runSingle').prop('disabled', true);
    _setLoading('spinnerA', 'plotA', 'Computing…');
    $('#scoreA').text('…').attr('class', 'score-val score-neutral');
    $('#scoreANote').text('kcal/mol');
    $('#statusA').text('Fetching…');

    $.ajax({
        url: '/attention_visualization/adj_3d_viz', type: 'POST', contentType: 'application/json',
        data: JSON.stringify({ smiles: smiles, model_path: _activeModelPath }),
        success: function(d) {
            $('#runSingle').prop('disabled', false);
            if (d.status !== 'success') {
                $('#spinnerAMsg').text('Error: ' + d.message); return;
            }
            // Update badge from server-confirmed tag
            if (d.model_tag && d.model_tag !== _activeModelTag) {
                _activeModelTag = d.model_tag;
                _updateModelBadge(d.model_tag);
            }
            _render3D('plotA', d.atoms, d.bonds, false);
            _renderBar('barA', d.weight_vector, 'plotA', 'tableA');
            _renderTable('tableA', d.atoms, 'barA', 'plotA', d.weight_vector);
            _renderScore(d.predicted_score, d.model_tag, 'scoreA', 'scoreANote');
            $('#statusA').text(d.atoms.length + ' atoms · ' + d.bonds.length + ' bonds');
            _clearLoading('spinnerA', 'plotA');
            // Highlight Compare as next step + notify mini-chat
            setTimeout(() => {
                _clearAttnHighlights();
                _highlightBtnUntilClick('modeCompare');
                const mini = document.getElementById('miniChat');
                if (mini && mini.style.display !== 'none') {
                    const score = d.predicted_score?.toFixed(3) ?? '?';
                    _attnMiniAppend('ai',
                        'Done! Score: **' + score + ' kcal/mol**\n\n' +
                        'Click **Compare** to analyze two compounds side-by-side, or ask me about the attention weights.'
                    );
                }
            }, 600);
        },
        error: function(xhr) {
            $('#runSingle').prop('disabled', false);
            const msg = (xhr.responseJSON||{}).message || 'Server error';
            $('#spinnerAMsg').text('Error: ' + msg);
        }
    });
}

// ── Compare mode ──────────────────────────────────────────────────────────────
function _runCompare() {
    const sa = $('#smilesCompA').val().trim(), sb = $('#smilesCompB').val().trim();
    if (!sa || !sb)         { alert('Enter SMILES for both compounds.'); return; }
    if (!_activeModelPath)  { alert('Select a model first.'); return; }

    $('#runCompare').prop('disabled', true);
    $('#compareErrors').hide();
    _setLoading('spinnerCA', 'plotCA', 'Computing A…');
    _setLoading('spinnerCB', 'plotCB', 'Computing B…');
    $('#scoreCA,#scoreCB').text('…').attr('class','score-val score-neutral');
    $('#scoreDelta').hide();

    $.ajax({
        url: '/attention_visualization/chembert_compare', type: 'POST', contentType: 'application/json',
        data: JSON.stringify({ smiles_a: sa, smiles_b: sb, model_path: _activeModelPath }),
        success: function(d) {
            $('#runCompare').prop('disabled', false);
            if (d.status !== 'success') {
                const e = d.errors || {};
                $('#compareErrors').html(
                    (e.smiles_a ? '<div>A: '+e.smiles_a+'</div>' : '') +
                    (e.smiles_b ? '<div>B: '+e.smiles_b+'</div>' : '') +
                    (d.message  ? '<div>'+d.message+'</div>' : '')
                ).show();
                $('#spinnerCAMsg').text('Error'); $('#spinnerCBMsg').text('Error');
                return;
            }

            const tag = d.a.model_tag || '';
            if (tag && tag !== _activeModelTag) {
                _activeModelTag = tag; _updateModelBadge(tag);
            }

            // Render A
            _render3D('plotCA', d.a.atoms, d.a.bonds, true);
            _renderBar('barCA', d.a.weight_vector, 'plotCA', 'tableCA');
            _renderTable('tableCA', d.a.atoms, 'barCA', 'plotCA', d.a.weight_vector);
            _renderScore(d.a.predicted_score, tag, 'scoreCA', 'scoreCAnote');
            $('#statusCA').text(d.a.atoms.length + ' atoms · ' + d.a.bonds.length + ' bonds');
            _clearLoading('spinnerCA', 'plotCA');

            // Render B
            _render3D('plotCB', d.b.atoms, d.b.bonds, true);
            _renderBar('barCB', d.b.weight_vector, 'plotCB', 'tableCB');
            _renderTable('tableCB', d.b.atoms, 'barCB', 'plotCB', d.b.weight_vector);
            _renderScore(d.b.predicted_score, tag, 'scoreCB', 'scoreCBnote');
            $('#statusCB').text(d.b.atoms.length + ' atoms · ' + d.b.bonds.length + ' bonds');
            _clearLoading('spinnerCB', 'plotCB');

            // Delta (only for finetuned)
            const sa = d.a.predicted_score, sb = d.b.predicted_score;
            if (tag === 'finetuned' && sa !== null && sb !== null) {
                const delta = sb - sa;
                $('#scoreDeltaVal')
                    .text((delta >= 0 ? '+' : '') + delta.toFixed(3))
                    .css('color', delta < 0 ? '#4ade80' : '#f87171');
                $('#scoreDelta').show();

                // Colour the winners
                if (sa < sb) {
                    $('#scoreCA').attr('class','score-val score-better');
                    $('#scoreCB').attr('class','score-val score-worse');
                } else if (sb < sa) {
                    $('#scoreCB').attr('class','score-val score-better');
                    $('#scoreCA').attr('class','score-val score-worse');
                }
            }
        },
        error: function(xhr) {
            $('#runCompare').prop('disabled', false);
            const msg = (xhr.responseJSON||{}).message || 'Server error';
            $('#compareErrors').html('<div>'+msg+'</div>').show();
            $('#spinnerCAMsg,#spinnerCBMsg').text('Error');
        }
    });
}

// ── 3-D scatter + bonds ───────────────────────────────────────────────────────
function _render3D(divId, atoms, bonds, compact) {
    const atomTrace = {
        type:'scatter3d', mode:'markers+text',
        x:atoms.map(a=>a.x), y:atoms.map(a=>a.y), z:atoms.map(a=>a.z),
        text:         atoms.map(a=>a.symbol),
        textfont:     { size: compact?8:10, color:'#ffffff' },
        textposition: 'top center',
        marker: {
            size:  atoms.map(a=>(compact?5:7)+a.weight_norm*12),
            color: atoms.map(a=>a.weight_norm),
            colorscale:[[0,'#3b4cc0'],[.25,'#88bbee'],[.5,'#dddddd'],[.75,'#ee8866'],[1,'#b40426']],
            cmin:0, cmax:1, showscale:!compact,
            colorbar: compact ? undefined : {
                title:    { text:'weight_a (norm)', font:{color:'#64748b',size:9} },
                tickfont: { color:'#64748b', size:8 },
                len:.45, thickness:9, x:1.01,
            },
            line:{width:1,color:'#0f172a'},
        },
        customdata: atoms.map(a=>({idx:a.idx,raw:a.weight_raw.toFixed(4),norm:a.weight_norm.toFixed(4)})),
        hovertemplate:
            '<b>%{text}</b> (idx %{customdata.idx})<br>' +
            'raw: %{customdata.raw}  norm: %{customdata.norm}<br>' +
            'xyz: (%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>',
        name:'Atoms',
    };
    const bx=[],by=[],bz=[];
    bonds.forEach(b=>{
        bx.push(atoms[b.begin].x,atoms[b.end].x,null);
        by.push(atoms[b.begin].y,atoms[b.end].y,null);
        bz.push(atoms[b.begin].z,atoms[b.end].z,null);
    });
    Plotly.react(divId,
        [{type:'scatter3d',mode:'lines',x:bx,y:by,z:bz,
          line:{color:'#334155',width:3},hoverinfo:'skip',name:'Bonds'}, atomTrace],
        { paper_bgcolor:'transparent', plot_bgcolor:'transparent',
          margin:{l:0,r:0,t:0,b:0},
          scene:{ bgcolor:'#05070f',
            xaxis:{showgrid:false,zeroline:false,showticklabels:false},
            yaxis:{showgrid:false,zeroline:false,showticklabels:false},
            zaxis:{showgrid:false,zeroline:false,showticklabels:false},
            aspectmode:'data' },
          showlegend:false, font:{color:'#94a3b8'} },
        { responsive:true, displayModeBar:true,
          modeBarButtonsToRemove:['toImage','resetCameraLastSave3d'],
          displaylogo:false }
    ).then(()=>{ $('#'+divId).css('opacity',1); });
}

// ── Bar chart ─────────────────────────────────────────────────────────────────
function _renderBar(divId, wv, plotDivId, tableDiv) {
    // x-axis is now atom idx (same as 3D hover idx and atom table idx).
    // top_indices are already in atom space from routes.py.
    const top  = [...wv.top_indices].sort((a,b)=>a-b);
    const vals = top.map(i=>wv.values[i]);
    const syms = top.map(i=>(wv.atom_symbols||[])[i]||'?');
    const cols = vals.map(v=>v>=0?'#22d3ee':'#f87171');

    Plotly.react(divId, [{
        type:'bar', x:top, y:vals,
        marker:{color:cols},
        customdata:syms,
        hovertemplate:'<b>%{customdata}%{x}</b><br>weight_a: %{y:.4f}<extra></extra>',
        selected:   {marker:{opacity:1}},
        unselected: {marker:{opacity:0.4}},
    }], {
        paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        margin:{l:30,r:2,t:2,b:22},
        xaxis:{title:{text:'Atom idx',font:{color:'#475569',size:8}},
               tickfont:{color:'#475569',size:7},gridcolor:'#1e293b'},
        yaxis:{tickfont:{color:'#475569',size:7},gridcolor:'#1e293b',zerolinecolor:'#334155'},
        bargap:.15, clickmode:'event',
    }, {responsive:true,displayModeBar:false});

    // Click bar → highlight matching atom row in table and pulse in 3D
    const barEl = document.getElementById(divId);
    barEl.removeAllListeners && barEl.removeAllListeners('plotly_click');
    barEl.on('plotly_click', function(ev) {
        if (!ev.points.length) return;
        const atomIdx = ev.points[0].x;
        _highlightAtom(atomIdx, tableDiv, plotDivId, wv);
    });
}

function _highlightAtom(atomIdx, tableDiv, plotDivId, wv) {
    // 1. Pulse the atom row in the table
    if (tableDiv) {
        const rows = document.querySelectorAll('#'+tableDiv+' [data-atom-idx]');
        rows.forEach(r => {
            const isMatch = parseInt(r.dataset.atomIdx) === atomIdx;
            r.style.background  = isMatch ? 'rgba(34,211,238,0.15)' : '';
            r.style.borderColor = isMatch ? '#22d3ee' : '';
            if (isMatch) r.scrollIntoView({block:'nearest', behavior:'smooth'});
        });
    }
    // 2. Enlarge the atom in the 3D plot (Plotly restyle on trace 1 = atoms)
    if (plotDivId) {
        const plotEl = document.getElementById(plotDivId);
        if (!plotEl || !plotEl.data || plotEl.data.length < 2) return;
        const atoms = plotEl.data[1];  // atomTrace is second trace
        const n = (atoms.x||[]).length;
        const sizes = Array.from({length:n}, (_,i)=>
            i===atomIdx ? 22 : (7 + (atoms.marker.color[i]||0)*12)
        );
        const opacs = Array.from({length:n}, (_,i)=> i===atomIdx ? 1 : 0.55);
        Plotly.restyle(plotDivId, {'marker.size':[sizes], 'marker.opacity':[opacs]}, [1]);
        // Reset after 2s
        setTimeout(()=>{
            const defSizes = Array.from({length:n}, (_,i)=>(7+(atoms.marker.color[i]||0)*12));
            const defOpacs = Array(n).fill(1);
            Plotly.restyle(plotDivId, {'marker.size':[defSizes], 'marker.opacity':[defOpacs]}, [1]);
        }, 2000);
    }
}

// ── Prepare Training Data ─────────────────────────────────────────────────────

function _runPrepare() {
    const subsetPath    = $('#prepSubsetPath').val().trim();
    const referencePath = $('#prepReferencePath').val().trim();
    const outputPath    = $('#prepOutputPath').val().trim();

    if (!subsetPath || !referencePath || !outputPath) {
        $('#prepStatus').text('⚠ Fill in all three paths first.').css('color','#fbbf24');
        return;
    }

    $('#prepRunBtn').prop('disabled', true).text('⏳ Processing…');
    $('#prepStatus').text('Merging…').css('color','#94a3b8');
    $('#prepPreview').addClass('hidden');

    $.ajax({
        url: '/attention_visualization/prepare_smiles', type: 'POST', contentType: 'application/json',
        data: JSON.stringify({
            subset_path:    subsetPath,
            reference_path: referencePath,
            output_path:    outputPath,
        }),
        success: function(d) {
            $('#prepRunBtn').prop('disabled', false).text('⚙️ Prepare & Save');
            if (d.status !== 'success') {
                $('#prepStatus').text('✗ ' + d.message).css('color','#f87171');
                return;
            }
            $('#prepStatus')
                .text('✓ ' + d.total_rows.toLocaleString() + ' rows → ' + d.output_path)
                .css('color','#4ade80');
            const tbody = $('#prepTableBody').empty();
            (d.preview || []).forEach(function(row) {
                const smiles = row.SMILES || '';
                const label  = row.LABELS !== undefined ? Number(row.LABELS).toFixed(4) : '—';
                const short  = smiles.length > 42 ? smiles.slice(0,39) + '…' : smiles;
                tbody.append(
                    '<tr class="hover:bg-slate-800/40">' +
                    '<td class="px-3 py-1 font-mono" title="' + smiles + '">' + short + '</td>' +
                    '<td class="px-3 py-1 text-right font-mono text-cyan-300">' + label + '</td>' +
                    '</tr>'
                );
            });
            $('#prepPreview').removeClass('hidden');
        },
        error: function(xhr) {
            $('#prepRunBtn').prop('disabled', false).text('⚙️ Prepare & Save');
            const msg = (xhr.responseJSON || {}).message || 'Server error';
            $('#prepStatus').text('✗ ' + msg).css('color','#f87171');
        }
    });
}

function _autofillSmilesPath() {
    const out = $('#prepOutputPath').val().trim();
    if (!out) return;
    $('#ftSmilesFile').val(out);
    $('#ftSmilesFile').css('border-color','#22d3ee');
    setTimeout(function(){ $('#ftSmilesFile').css('border-color',''); }, 1800);
    document.getElementById('ftSmilesFile').scrollIntoView({behavior:'smooth', block:'center'});
}

// ── Fine-tune panel ───────────────────────────────────────────────────────────
let _ftEventSource = null;

function _openFinetunePanel() {
    $('#finetuneModal').removeClass('hidden').addClass('flex');
}
function _closeFinetunePanel() {
    $('#finetuneModal').addClass('hidden').removeClass('flex');
}

// File browse helper — resolves the server-side path from the File object's name.
// Since the browser only gives us the filename, we inject the full path the user
// typed in the adjacent text input, or fall back to the filename alone.
function _ftBrowse(inputEl, targetId) {
    const file = inputEl.files[0];
    if (!file) return;
    // Use the value already in the path box as a directory prefix if present,
    // otherwise just set the filename so the user can complete the path.
    const existing = $('#' + targetId).val().trim();
    const dir = existing.includes('/') ? existing.substring(0, existing.lastIndexOf('/') + 1) : '';
    $('#' + targetId).val(dir + file.name);
    // Reset the file input so the same file can be re-selected
    inputEl.value = '';
}

function _startFinetune() {
    const smilesFile      = $('#ftSmilesFile').val().trim();
    const pretrainedModel = $('#ftPretrainedModel').val().trim();
    const task            = $('#ftTask').val();
    const maxEpochs       = parseInt($('#ftMaxEpochs').val()) || 15;
    const maxTime         = parseInt($('#ftMaxTime').val()) || 720;

    if (!smilesFile) {
        alert('Please specify a SMILES file path.');
        return;
    }

    // Close any running stream
    if (_ftEventSource) { _ftEventSource.close(); _ftEventSource = null; }

    // Show log area, hide form's start button during run
    $('#finetuneLog').removeClass('hidden');
    $('#ftLogLines').empty();
    $('#ftLogStatus').text('Starting…').css('color', '#34d399');
    $('#ftSpinner').show();
    $('#ftStartBtn').prop('disabled', true).text('⏳ Running…');

    _ftLog('▶ Submitting fine-tuning job…', '#94a3b8');

    $.ajax({
        url: '/attention_visualization/finetune_chembert', type: 'POST', contentType: 'application/json',
        data: JSON.stringify({
            smiles_file:      smilesFile,
            pretrained_model: pretrainedModel || undefined,
            task:             task,
            max_epochs:       maxEpochs,
            max_time:         maxTime,
        }),
        success: function(d) {
            if (d.status !== 'started') {
                _ftLog('✗ ' + (d.message || 'Server error'), '#f87171');
                _ftDone(false);
                return;
            }
            _ftLog('✓ Job started — ID: ' + d.job_id, '#34d399');
            _ftLog('━'.repeat(55), '#334155');
            _ftStream(d.job_id);
        },
        error: function(xhr) {
            const msg = (xhr.responseJSON || {}).message || 'Server error';
            _ftLog('✗ ' + msg, '#f87171');
            _ftDone(false);
        }
    });
}

function _ftStream(jobId) {
    $('#ftLogStatus').text('Running…');
    _ftEventSource = new EventSource('/attention_visualization/finetune_status/' + jobId);

    _ftEventSource.onmessage = function(e) {
        const line = e.data;
        if (line === '__DONE__') {
            _ftEventSource.close();
            _ftEventSource = null;
            _ftDone(true);
            return;
        }
        // Colour-code common patterns
        let color = '#cbd5e1';
        if (line.startsWith('ERROR'))                   color = '#f87171';
        else if (/epoch \d+/i.test(line))               color = '#fbbf24';
        else if (/validation loss improved/i.test(line))color = '#4ade80';
        else if (/rmse|loss|auc/i.test(line))           color = '#7dd3fc';
        else if (/#{3,}/.test(line))                    color = '#94a3b8';
        _ftLog(line, color);
    };

    _ftEventSource.onerror = function() {
        _ftLog('⚠ Connection lost — job may still be running on server.', '#fbbf24');
        _ftDone(false);
    };
}

function _ftDone(success) {
    $('#ftSpinner').hide();
    if (success) {
        $('#ftLogStatus').text('✓ Finished').css('color', '#4ade80');
        _ftLog('━'.repeat(55), '#334155');
        _ftLog('✓ Fine-tuning complete. New checkpoint saved to chembert/', '#4ade80');
        // Refresh model presets so the new checkpoint appears
        _fetchPresets();
    } else {
        $('#ftLogStatus').text('✗ Error').css('color', '#f87171');
    }
    $('#ftStartBtn').prop('disabled', false).text('🚀 Start Fine-tuning');
}

function _ftLog(line, color) {
    const el = $('<div>').text(line).css('color', color || '#cbd5e1');
    $('#ftLogLines').append(el);
    // Auto-scroll
    const container = document.getElementById('ftLogLines');
    container.scrollTop = container.scrollHeight;
}

function _clearFinetuneLog() {
    $('#ftLogLines').empty();
}

// ── Atom table ────────────────────────────────────────────────────────────────
function _renderTable(divId, atoms, barDivId, plotDivId, wv) {
    const sorted = [...atoms].sort((a,b)=>b.weight_norm-a.weight_norm);
    const hdr  = '<p class="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1 pt-1">Atoms ↓  —  click to locate in chart + 3D</p>';
    const rows = sorted.map(a=>
        '<div data-atom-idx="'+a.idx+'" '+
             'style="cursor:pointer;border-radius:4px;border:1px solid transparent;transition:background .15s" '+
             'class="flex items-center justify-between py-0.5 px-0.5 border-b border-slate-800/40">' +
            '<span class="flex items-center gap-1">' +
                '<span class="w-2 h-2 rounded-full flex-shrink-0" style="background:'+a.color+'"></span>' +
                '<span class="font-mono">'+a.symbol+'<sub class="text-slate-600">'+a.idx+'</sub></span>' +
            '</span>' +
            '<span class="font-mono text-slate-300">'+a.weight_raw.toFixed(4)+'</span>' +
        '</div>'
    ).join('');
    const container = $('#'+divId);
    container.html(hdr+rows);
    // Click row → highlight bar + pulse 3D atom
    container.find('[data-atom-idx]').on('click', function() {
        const atomIdx = parseInt($(this).data('atom-idx'));
        if (barDivId) {
            // Select the bar for this atom
            const barEl = document.getElementById(barDivId);
            if (barEl && barEl.data) {
                const xs = barEl.data[0].x;
                const barPos = xs.indexOf(atomIdx);
                if (barPos >= 0) {
                    Plotly.restyle(barDivId, {
                        selectedpoints: [[barPos]],
                    }, [0]);
                    setTimeout(()=>Plotly.restyle(barDivId,{selectedpoints:[null]},[0]), 2000);
                }
            }
        }
        _highlightAtom(atomIdx, divId, plotDivId, wv);
    });
}

// ── Smart scroll: only follow bottom if user hasn't scrolled up ───────────────
function _smartScroll(elId) {
    const el = document.getElementById(elId);
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distFromBottom < 80) el.scrollTop = el.scrollHeight;
}


// ══ Attention Visualizer Mini-chat ════════════════════════════════════════════
let _attnMiniStream = null;

// ── ChemBERT mini-chat now uses the SHARED miniChat panel ────────────────────
// All _attnMini* calls delegate to their _miniChat* equivalents so ChemBERT
// behaves identically to Vina, DeepAtom, and TS.

function _attnMiniShow()  { _miniChatShow(); }
function _attnMiniClose() { _miniChatClose(); }

function _attnMiniAppend(role, text) {
    return _miniChatAppend(role, text);
}

// _attnMiniSend: reads from miniChatInput (not the removed attnMiniInput)
// and delegates to the shared _miniChatSend() which already routes to the
// correct endpoint via _chatEndpoint() based on _activeTool.
function _attnMiniSend() {
    _miniChatSend();
}

// ── Hook _adjShow to apply ChemBERT theme + welcome ──────────────────────────
const _origAdjShowAttn = _adjShow;
_adjShow = function() {
    _origAdjShowAttn();
    setTimeout(() => {
        // Apply ChemBERT theme to the shared panel (from elion_mini_chat.js)
        if (typeof _miniChatSetContext === 'function' && typeof _MINICHAT_ATTN_THEME !== 'undefined') {
            _miniChatSetContext(_MINICHAT_ATTN_THEME);
        }
        _miniChatShow();
        const out = document.getElementById('miniChatOutput');
        if (out) {
            out.innerHTML = '';
            if (typeof _attnMiniWelcome === 'function') {
                _attnMiniWelcome();
            }
        }
    }, 300);
};

// ── Hook _adjHide to restore Vina theme when ChemBERT modal closes ───────────
const _origAdjHideAttn = (typeof _adjHide === 'function') ? _adjHide : () => {};
_adjHide = function() {
    _origAdjHideAttn();
    _miniChatClose();
    if (typeof _miniChatSetContext === 'function' && typeof _MINICHAT_VINA_THEME !== 'undefined') {
        _miniChatSetContext(_MINICHAT_VINA_THEME);
    }
};