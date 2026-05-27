// attn.js — ChemBERT / Attention Weight Visualizer
// strict mode removed — uses cross-file var assignments


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
window._adjShow = function() { $('#adjModal').removeClass('hidden').addClass('flex'); }
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
    $.getJSON('/attention_visualization/chembert_models', function(d) {
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
                const mini = document.getElementById('attnMiniChat');
                if (mini && mini.style.display === 'flex') {
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

// ══ Qwen Chat (Claude-style) ══════════════════════════════════════════════════
let _chatStream = null;
let _msgCount   = 0;

function _chatAppend(role, text) {
    const out = document.getElementById('chatOutput');
    if (_msgCount === 0) {
        const es = document.getElementById('chatEmptyState');
        if (es) es.style.display = 'none';
        const pills = document.getElementById('chatQuickPills');
        if (pills) pills.style.display = 'none';
    }
    _msgCount++;

    const wrap = document.createElement('div');
    wrap.style.cssText = 'width:100%;max-width:760px;margin:0 auto;padding:0 24px;box-sizing:border-box;';

    const div = document.createElement('div');
    div.className = role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai';

    if (role === 'ai') {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
        const av = document.createElement('div');
        av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
        av.textContent = '🧬';
        row.appendChild(av);
        row.appendChild(div);
        wrap.appendChild(row);
    } else {
        wrap.style.display = 'flex';
        wrap.style.justifyContent = 'flex-end';
        wrap.appendChild(div);
    }

    div.innerHTML = (text || '')
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
        .replace(/`([^`]+)`/g,'<code>$1</code>')
        .replace(/\n/g,'<br>');

    out.appendChild(wrap);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _chatShowTyping() {
    const out  = document.getElementById('chatOutput');
    const wrap = document.createElement('div');
    wrap.style.cssText = 'width:100%;max-width:760px;margin:0 auto;padding:0 24px;box-sizing:border-box;';
    wrap.id = 'chatTypingWrap';
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:flex-start;gap:10px;';
    const av  = document.createElement('div');
    av.style.cssText = 'width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#22d3ee,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px;';
    av.textContent = '🧬';
    const dots = document.createElement('div');
    dots.className = 'chat-bubble-ai';
    dots.id = 'chatTyping';
    dots.style.padding = '10px 0';
    dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    row.appendChild(av); row.appendChild(dots);
    wrap.appendChild(row);
    out.appendChild(wrap);
    out.scrollTop = out.scrollHeight;
}
function _chatHideTyping() {
    document.getElementById('chatTypingWrap')?.remove();
    document.getElementById('chatTyping')?.remove();
}

function _chatGetContext() {
    const ctx = {};
    // SMILES from the single input
    const smilesEl = document.getElementById('smiles1') || document.querySelector('.smiles-mono');
    if (smilesEl && smilesEl.value) ctx.smiles = smilesEl.value.trim();
    // Score
    const scoreEl = document.querySelector('.score-val') || document.getElementById('scoreVal');
    if (scoreEl) { const v = parseFloat(scoreEl.textContent); if (!isNaN(v)) ctx.score = v; }
    // Current model
    const activeModel = document.querySelector('.preset-btn.active, .preset-btn.pretrained-active');
    if (activeModel) ctx.model = activeModel.textContent.trim();
    // Mode
    const modeEl = document.querySelector('.mode-btn.active');
    if (modeEl) ctx.mode = modeEl.textContent.trim();
    return ctx;
}

function _chatSend() {
    const input = document.getElementById('chatInput');
    const text  = input.value.trim();
    if (!text) return;
    if (_chatStream) return;  // already streaming
    input.value = '';
    input.style.height = 'auto';
    _chatAppend('user', text);
    _chatShowTyping();

    const ctx = _chatGetContext();
    let aiDiv = null, fullText = '', firstToken = true;
    _chatStream = 'main';

    fetch('/attention_visualization/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        function pump() {
            reader.read().then(({ done, value }) => {
                if (done) { _chatHideTyping(); _chatStream = null; return; }
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split('\n\n');
                buf = parts.pop();
                for (const part of parts) {
                    const evM   = part.match(/^event:\s*(\w+)/m);
                    const dataM = part.match(/^data:\s*(.+)$/m);
                    if (!evM || !dataM) continue;
                    const ev = evM[1];
                    let payload;
                    try { payload = JSON.parse(dataM[1]); } catch { payload = dataM[1]; }

                    if (ev === 'token') {
                        if (firstToken) { _chatHideTyping(); firstToken = false; aiDiv = _chatAppend('ai', ''); fullText = ''; }
                        fullText += payload;
                        if (aiDiv) {
                            aiDiv.innerHTML = fullText
                                .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                                .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
                                .replace(/`([^`]+)`/g,'<code>$1</code>')
                                .replace(/\n/g,'<br>');
                            document.getElementById('chatOutput').scrollTop = 99999;
                        }
                    } else if (ev === 'ui_action') {
                        if (payload) handleAttnUiAction(payload);
                    } else if (ev === 'done') {
                        _chatHideTyping(); _chatStream = null;
                        $('#streamBubble').removeAttr('id'); $('#streamContent').removeAttr('id');
                    } else if (ev === 'error') {
                        _chatHideTyping(); _chatStream = null;
                        _chatAppend('ai', '⚠️ ' + (payload || 'Server error'));
                    }
                }
                pump();
            }).catch(err => { _chatHideTyping(); _chatStream = null; _chatAppend('ai', '⚠️ Stream error: ' + err.message); });
        }
        pump();
    }).catch(err => { _chatHideTyping(); _chatStream = null; console.error('[Elion] fetch error:', err); _chatAppend('ai', '⚠️ Could not reach Qwen server. Is it running on port 8001?'); });
}

function _qprompt(text) {
    document.getElementById('chatInput').value = text;
    _chatSend();
}

function _chatClear() {
    const out = document.getElementById('chatOutput');
    out.innerHTML = '';
    _msgCount = 0;
    const es = document.getElementById('chatEmptyState');
    if (es) { out.appendChild(es); es.style.display = 'flex'; }
    const pills = document.getElementById('chatQuickPills');
    if (pills) pills.style.display = 'flex';
    fetch('/attention_visualization/chat/clear', { method: 'POST' });
}

// Keyboard shortcut
document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); document.getElementById('chatInput').focus(); }
});

// ── Attn UI action map ────────────────────────────────────────────────────────
const _ATTN_ACTIONS = {
    open_visualizer: { btnId: 'adjWeightBtn' },
    run_finetune:    { btnId: 'finetuneBtn' },
    load_model:      { btnId: 'loadModelBtn' },
    compare_mode:    { btnId: 'modeCompare' },
    show_3d:         { btnId: 'runSingle' },
};

const _ATTN_ALL_BTNS = ['adjWeightBtn','finetuneBtn','loadModelBtn','modeCompare','modeSingle','runSingle','runCompare'];
const _ATTN_ALL_INPUTS = ['smilesA', 'smilesCompA', 'smilesCompB', 'customModelPath'];

function _clearAttnHighlights() {
    _ATTN_ALL_BTNS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('btn-pulse-wait');
    });
    _ATTN_ALL_INPUTS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('input-pulse-wait');
    });
}

function handleAttnUiAction(ui_action) {
    if (!ui_action || ui_action.action === 'none') return;

    // Clear previous highlights
    setTimeout(() => _clearAttnHighlights(), 0);

    if (ui_action.action === 'open_visualizer') {
        setTimeout(() => {
            _chatAppend('ai', 'Now click the flashing **Open Visualizer** button ↑ to launch the visualizer.');
            _highlightBtnUntilClick('adjWeightBtn');
        }, 400);
        return;
    }

    if (ui_action.action === 'compare_mode') {
        setTimeout(() => {
            _highlightBtnUntilClick('modeCompare');
        }, 400);
        return;
    }

    if (ui_action.action === 'show_3d') {
        // Gold-pulse SMILES input + Visualize button
        setTimeout(() => {
            _highlightInputUntilType('smilesA');
            _highlightBtnUntilClick('runSingle');
        }, 400);
        return;
    }

    // All other actions: gold pulse on mapped button
    const entry = _ATTN_ACTIONS[ui_action.action];
    if (!entry) return;
    setTimeout(() => _highlightBtnUntilClick(entry.btnId), 400);
}

// ── Gold highlight (same system as vina) ─────────────────────────────────────
function _highlightBtnUntilClick(btnId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.classList.remove('btn-pulse-wait', 'input-pulse-wait');
    void btn.offsetWidth;
    btn.classList.add('btn-pulse-wait');
    const stop = () => { btn.classList.remove('btn-pulse-wait'); btn.removeEventListener('click', stop); };
    btn.addEventListener('click', stop);
}

function _highlightInputUntilType(inputId) {
    const el = document.getElementById(inputId);
    if (!el) return;
    el.classList.remove('input-pulse-wait');
    void el.offsetWidth;
    el.classList.add('input-pulse-wait');
    const stop = () => { el.classList.remove('input-pulse-wait'); el.removeEventListener('input', stop); };
    el.addEventListener('input', stop);
}


// ══ Attention Visualizer Mini-chat ════════════════════════════════════════════
let _attnMiniStream = null;

function _attnMiniShow() {
    document.getElementById('attnMiniChat').style.display = 'flex';
}
function _attnMiniClose() {
    document.getElementById('attnMiniChat').style.display = 'none';
}

function _attnMiniAppend(role, text) {
    const out = document.getElementById('attnMiniOutput');
    const div = document.createElement('div');
    div.style.cssText = role === 'user'
        ? 'align-self:flex-end;max-width:85%;background:#0e7490;color:#ecfeff;border-radius:10px 10px 2px 10px;padding:6px 10px;font-size:12px;line-height:1.5;'
        : 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.5;';
    div.innerHTML = (text || '')
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
        .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:10px">$1</code>')
        .replace(/\n/g,'<br>');
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _attnMiniSend() {
    const inp  = document.getElementById('attnMiniInput');
    const text = inp.value.trim();
    if (!text || _attnMiniStream) return;
    inp.value = '';
    _attnMiniAppend('user', text);

    const out = document.getElementById('attnMiniOutput');

    // Typing indicator
    const typing = document.createElement('div');
    typing.id = 'attnMiniTyping';
    typing.style.cssText = 'color:#64748b;font-size:12px;padding:4px 0;';
    typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    out.appendChild(typing); out.scrollTop = out.scrollHeight;

    // Build context from current visualizer state
    const ctx = {
        guided_mode: true,
        smiles: document.getElementById('smilesA')?.value.trim() || '',
        model:  document.getElementById('modelTagBadge')?.textContent.trim() || '',
        mode:   'single',
    };

    let aiDiv = null, fullText = '', firstToken = true;
    _attnMiniStream = true;

    fetch('/attention_visualization/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        function pump() {
            reader.read().then(({ done, value }) => {
                if (done) { document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null; return; }
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split('\n\n');
                buf = parts.pop();
                for (const part of parts) {
                    const evM   = part.match(/^event:\s*(\w+)/m);
                    const dataM = part.match(/^data:\s*(.+)$/m);
                    if (!evM || !dataM) continue;
                    const ev = evM[1];
                    let payload;
                    try { payload = JSON.parse(dataM[1]); } catch { payload = dataM[1]; }

                    if (ev === 'token') {
                        if (firstToken) { document.getElementById('attnMiniTyping')?.remove(); firstToken = false; aiDiv = _attnMiniAppend('ai', ''); fullText = ''; }
                        fullText += payload;
                        if (aiDiv) {
                            aiDiv.innerHTML = fullText
                                .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                                .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
                                .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:10px">$1</code>')
                                .replace(/\n/g,'<br>');
                            document.getElementById('attnMiniOutput').scrollTop = 99999;
                        }
                    } else if (ev === 'ui_action') {
                        if (payload) handleAttnUiAction(payload);
                    } else if (ev === 'done') {
                        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                    } else if (ev === 'error') {
                        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                        _attnMiniAppend('ai', '⚠️ ' + (payload || 'Error'));
                    }
                }
                pump();
            }).catch(err => {
                document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                _attnMiniAppend('ai', '⚠️ Stream error: ' + err.message);
            });
        }
        pump();
    }).catch(() => {
        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
        _attnMiniAppend('ai', '⚠️ Could not reach Qwen server.');
    });
}

// ── Hook _adjShow to show mini-chat with welcome message ──────────────────────
let _origAdjShowAttn = _adjShow;
window._adjShow = function() {
    _origAdjShowAttn();
    // Show mini-chat after a short delay
    setTimeout(() => {
        _attnMiniShow();
        if (document.getElementById('attnMiniOutput').children.length === 0) {
            _attnMiniAppend('ai',
                'Hi! I can guide you through this visualizer.\n\n' +
                '**Step 1:** Select a model (Finetuned or Pretrained) from the bar above.\n' +
                '**Step 2:** Enter a SMILES string in the input field.\n' +
                '**Step 3:** Click ⚡ **Visualize** to render the 3D attention weights.'
            );
        }
    }, 300);
};



// ── Attention mini-chat ───────────────────────────────────────────────────────
// ══ Attention Visualizer Mini-chat ════════════════════════════════════════════
_attnMiniStream = null;

function _attnMiniShow() {
    document.getElementById('attnMiniChat').style.display = 'flex';
}
function _attnMiniClose() {
    document.getElementById('attnMiniChat').style.display = 'none';
}

function _attnMiniAppend(role, text) {
    out = document.getElementById('attnMiniOutput');
    div = document.createElement('div');
    div.style.cssText = role === 'user'
        ? 'align-self:flex-end;max-width:85%;background:#0e7490;color:#ecfeff;border-radius:10px 10px 2px 10px;padding:6px 10px;font-size:12px;line-height:1.5;'
        : 'align-self:flex-start;max-width:92%;color:#cbd5e1;font-size:12px;line-height:1.5;';
    div.innerHTML = (text || '')
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
        .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:10px">$1</code>')
        .replace(/\n/g,'<br>');
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
    return div;
}

function _attnMiniSend() {
    inp  = document.getElementById('attnMiniInput');
    text = inp.value.trim();
    if (!text || _attnMiniStream) return;
    inp.value = '';
    _attnMiniAppend('user', text);

    out = document.getElementById('attnMiniOutput');

    // Typing indicator
    typing = document.createElement('div');
    typing.id = 'attnMiniTyping';
    typing.style.cssText = 'color:#64748b;font-size:12px;padding:4px 0;';
    typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    out.appendChild(typing); out.scrollTop = out.scrollHeight;

    // Build context from current visualizer state
    ctx = {
        guided_mode: true,
        smiles: document.getElementById('smilesA')?.value.trim() || '',
        model:  document.getElementById('modelTagBadge')?.textContent.trim() || '',
        mode:   'single',
    };

    aiDiv = null, fullText = '', firstToken = true;
    _attnMiniStream = true;

    fetch(_chatEndpoint(), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, context: ctx })
    }).then(resp => {
        reader  = resp.body.getReader();
        decoder = new TextDecoder();
        buf = '';

        function pump() {
            reader.read().then(({ done, value }) => {
                if (done) { document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null; return; }
                buf += decoder.decode(value, { stream: true });
                parts = buf.split('\n\n');
                buf = parts.pop();
                for (const part of parts) {
                    evM   = part.match(/^event:\s*(\w+)/m);
                    dataM = part.match(/^data:\s*(.+)$/m);
                    if (!evM || !dataM) continue;
                    ev = evM[1];
                    payload;
                    try { payload = JSON.parse(dataM[1]); } catch { payload = dataM[1]; }

                    if (ev === 'token') {
                        if (firstToken) { document.getElementById('attnMiniTyping')?.remove(); firstToken = false; aiDiv = _attnMiniAppend('ai', ''); fullText = ''; }
                        fullText += payload;
                        if (aiDiv) {
                            aiDiv.innerHTML = fullText
                                .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                                .replace(/\*\*(.+?)\*\*/g,'<strong style="color:#7dd3fc">$1</strong>')
                                .replace(/`([^`]+)`/g,'<code style="background:#1e293b;padding:1px 4px;border-radius:3px;font-size:10px">$1</code>')
                                .replace(/\n/g,'<br>');
                            _smartScroll('attnMiniOutput');
                        }
                    } else if (ev === 'ui_action') {
                        if (payload) {
                            if (_activeTool === 'vina') handleVinaUiAction(payload);
                            else handleAttnUiAction(payload);
                        }
                    } else if (ev === 'done') {
                        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                    } else if (ev === 'error') {
                        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                        _attnMiniAppend('ai', '⚠️ ' + (payload || 'Error'));
                    }
                }
                pump();
            }).catch(err => {
                document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
                _attnMiniAppend('ai', '⚠️ Stream error: ' + err.message);
            });
        }
        pump();
    }).catch(() => {
        document.getElementById('attnMiniTyping')?.remove(); _attnMiniStream = null;
        _attnMiniAppend('ai', '⚠️ Could not reach Qwen server.');
    });
}

// ── Hub _adjShow hook (shows mini-chat on open) ───────────────────────────────
// ── Hook _adjShow to show mini-chat with welcome message ──────────────────────
_origAdjShowAttn = _adjShow;
window._adjShow = function() {
    _origAdjShowAttn();
    // Show mini-chat after a short delay
    setTimeout(() => {
        _attnMiniShow();
        if (document.getElementById('attnMiniOutput').children.length === 0) {
            _attnMiniAppend('ai',
                'Hi! I can guide you through this visualizer.\n\n' +
                '**Step 1:** Select a model (Finetuned or Pretrained) from the bar above.\n' +
                '**Step 2:** Enter a SMILES string in the input field.\n' +
                '**Step 3:** Click ⚡ **Visualize** to render the 3D attention weights.'
            );
        }
    }, 300);
};