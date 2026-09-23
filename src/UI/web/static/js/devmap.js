// =============================================================================
// devmap.js — hover a button, see which file and line it runs
// -----------------------------------------------------------------------------
// Backed by /devmap/index (uiapp/routes/devmap_routes.py). The index arrives as
// one JSON document on first hover and is cached for the life of the page, so
// every subsequent tooltip is resolved locally with no network round trip —
// a fetch per mouseover would lag the cursor and put a request on the server
// for every pass across a toolbar.
//
// What a tooltip shows, in the order it is useful:
//
//     Handler   _vinaDock()               web/static/js/vina.js:826
//     Markup    #vinaDockBtn              web/templates/hub.html:1102
//     Calls     POST /vina_visualization/vina_dock
//                                         uiapp/routes/vina_dock_routes.py:499
//
// Resolution order for "what does this element run":
//   1. an inline on* attribute        → first identifier called
//   2. data-nav="<key>"               → hub.js's delegated dispatcher, then the
//                                       function that case actually calls
//   3. a JS-attached listener         → cannot be read from the DOM; the id is
//                                       reported instead, which is enough to
//                                       grep for. This is stated in the tooltip
//                                       rather than left as an empty row, so a
//                                       blank result is never ambiguous between
//                                       "nothing found" and "nothing to find".
//
// OFF by default. Turn it on with the configured hotkey (Alt+Ctrl/⌘+Shift+D
// unless changed) or the badge in the bottom-left corner. Click any file:line
// row to copy it to the clipboard.
//
// Configuration lives in the engine's input_TS.yml under `visualizer.devmap`
// and reaches the browser via GET /devmap/config:
//
//     visualizer:
//       devmap:
//         enabled:  false            # state on a browser that has never toggled
//         hotkey:   "alt+ctrl+shift+d"
//         pin_modifier: "alt"        # hold while hovering to pin; "" disables
//         show_badge:   true
//         hover_delay_ms: 260
//
// The overlay starts disabled and STAYS disabled until that fetch resolves, so
// there is no frame in which tooltips appear on a deployment that configured
// them off. Failure to fetch leaves it off for the same reason.
// =============================================================================

(function () {
    'use strict';

    if (window.__devmapLoaded) return;
    window.__devmapLoaded = true;

    // v2: the default flipped from on to off, and the v1 key was written on
    // every toggle including the ones that merely re-affirmed the old default.
    // Reusing it would leave the overlay on for anyone who had ever pressed the
    // shortcut, which is exactly the population this change is for.
    var STORE_KEY = 'elion.devmap.enabled.v2';

    // Mirrors uiapp/config.DEVMAP_DEFAULTS. Used only until /devmap/config
    // answers, and as the fallback if it never does.
    var CFG = {
        enabled: false,
        show_badge: true,
        hover_delay_ms: 260,        // long enough that sweeping across a toolbar
                                    // does not strobe tooltips, short enough to
                                    // feel like a tooltip rather than a wait
        pin_modifier: 'alt',
        combo: { ctrl: true, alt: true, shift: true, meta: false,
                 key: 'd', code: 'KeyD', label: 'Alt+Ctrl/⌘+Shift+D' }
    };

    var index = null;              // the cached /devmap/index payload
    var indexPromise = null;
    var enabled = false;           // never true before /devmap/config resolves
    var hoverTimer = null;
    var currentTarget = null;
    var tipEl = null;
    var badgeEl = null;
    var pinned = false;            // pin_modifier-hover pins the tooltip so it
                                   // can be moused into and its rows clicked

    function readPref() {
        try {
            var v = window.localStorage.getItem(STORE_KEY);
            return v === null ? null : v === '1';   // null = never toggled here
        } catch (e) { return null; }                // private mode / blocked storage
    }
    function writePref(v) {
        try { window.localStorage.setItem(STORE_KEY, v ? '1' : '0'); } catch (e) {}
    }

    // ── Config ───────────────────────────────────────────────────────────────
    function loadConfig() {
        return fetch('/devmap/config')
            .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
            .then(function (d) {
                if (!d || !d.ok || !d.devmap) throw new Error('no devmap config');
                var c = d.devmap;
                if (typeof c.enabled === 'boolean')      CFG.enabled = c.enabled;
                if (typeof c.show_badge === 'boolean')   CFG.show_badge = c.show_badge;
                if (typeof c.hover_delay_ms === 'number') CFG.hover_delay_ms = c.hover_delay_ms;
                if (typeof c.pin_modifier === 'string')  CFG.pin_modifier = c.pin_modifier;
                if (c.combo && c.combo.key)              CFG.combo = c.combo;
                if (c.hotkey_error) console.warn('[devmap]', c.hotkey_error);
            })
            .catch(function (err) {
                console.warn('[devmap] config unavailable, staying off:', err);
            });
    }

    // Exact modifier match, so a binding of Alt+Ctrl+Shift+D is NOT fired by
    // Ctrl+Shift+D — which matters because Chrome binds that to "bookmark all
    // tabs". `ctrl` accepts ⌘ as well, the Ctrl/⌘ convention used across this
    // UI; `meta` is the strict form.
    //
    // `code` is checked before `key` because `key` depends on the layout and on
    // the modifiers themselves: macOS reports Option+D as "∂", so any binding
    // containing Alt could never match on `key` alone.
    function comboMatches(e, c) {
        if (!c || !c.key) return false;
        var hit = (c.code && e.code === c.code) ||
                  String(e.key || '').toLowerCase() === c.key;
        if (!hit) return false;
        if (c.ctrl ? !(e.ctrlKey || e.metaKey) : !!e.ctrlKey) return false;
        if (!!c.alt !== !!e.altKey) return false;
        if (!!c.shift !== !!e.shiftKey) return false;
        if (c.meta) { if (!e.metaKey) return false; }
        else if (!c.ctrl && e.metaKey) return false;
        return true;
    }

    function pinHeld(e) {
        switch (CFG.pin_modifier) {
            case 'alt':   return !!e.altKey;
            case 'ctrl':  return !!(e.ctrlKey || e.metaKey);
            case 'shift': return !!e.shiftKey;
            case 'meta':  return !!e.metaKey;
            default:      return false;          // "" → pin-on-hover disabled
        }
    }
    function pinModLabel() {
        return { alt: 'Alt', ctrl: 'Ctrl/⌘', shift: 'Shift', meta: '⌘' }[CFG.pin_modifier] || '';
    }
    function pinLabel() {
        var m = pinModLabel();
        return m ? 'hold ' + m + ' to pin' : '';
    }

    // ── Index ────────────────────────────────────────────────────────────────
    function loadIndex() {
        if (index) return Promise.resolve(index);
        if (indexPromise) return indexPromise;
        indexPromise = fetch('/devmap/index')
            .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
            .then(function (data) {
                if (!data || !data.ok) throw new Error('devmap index unavailable');
                index = data;
                console.log('[devmap] index loaded —',
                    data.meta.n_symbols, 'symbols,',
                    data.meta.n_routes, 'routes,',
                    data.meta.files_indexed, 'files in', data.meta.build_ms + 'ms');
                return index;
            })
            .catch(function (err) {
                console.warn('[devmap] index failed to load:', err);
                indexPromise = null;                 // allow a retry on next hover
                throw err;
            });
        return indexPromise;
    }

    // ── Resolution ───────────────────────────────────────────────────────────
    var IDENT_CALL = /\b([A-Za-z_$][\w$]*)\s*\(/;
    var ON_ATTRS = ['onclick', 'onchange', 'oninput', 'onsubmit', 'ondblclick'];

    // Event plumbing, not application logic. A modal's inner container carries
    // onclick="event.stopImmediatePropagation()" purely to stop a backdrop
    // click closing it; reporting that as "the handler" is worse than reporting
    // nothing, because it points at a real function in a real file that has
    // nothing to do with what the user is hovering.
    var PLUMBING = /^(stopImmediatePropagation|stopPropagation|preventDefault|returnValue)$/;

    function handlerFromAttrs(el) {
        for (var i = 0; i < ON_ATTRS.length; i++) {
            var raw = el.getAttribute(ON_ATTRS[i]);
            if (!raw) continue;
            // Take the first call that is not plumbing, so
            // `onclick="event.stopPropagation(); _vinaDock()"` resolves to _vinaDock.
            var re = /\b([A-Za-z_$][\w$]*)\s*\(/g, m;
            while ((m = re.exec(raw)) !== null) {
                if (!PLUMBING.test(m[1])) {
                    return { name: m[1], via: ON_ATTRS[i], source: raw.trim() };
                }
            }
        }
        return null;
    }

    function defsFor(name) {
        return (index && index.symbols && index.symbols[name]) || [];
    }
    function markupFor(id) {
        return (index && index.elements && index.elements[id]) || [];
    }
    function routesFor(url) {
        if (!index || !index.routes) return [];
        var norm = String(url).split('?')[0];
        if (index.routes[norm]) return index.routes[norm];
        // The indexer normalises Flask converters and template literals to
        // `<var>`; try that shape too before giving up.
        var generic = norm.replace(/\/[^/]*\$\{[^}]*\}[^/]*/g, '/<var>');
        return index.routes[generic] || [];
    }

    function resolve(el) {
        var out = {
            id: el.id || '',
            tag: el.tagName.toLowerCase(),
            label: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 48),
            handler: null, defs: [], markup: [], calls: [], nav: null, note: ''
        };

        if (out.id) out.markup = markupFor(out.id);

        var h = handlerFromAttrs(el);
        if (h) {
            out.handler = h;
            out.defs = defsFor(h.name);
        }

        // Delegated data-nav dispatch (hub.js). The card itself has no onclick;
        // one listener on document routes by attribute, so the honest answer is
        // two hops: the dispatcher's case line, then what that case calls.
        var navKey = el.getAttribute && el.getAttribute('data-nav');
        if (!navKey && el.closest) {
            var navEl = el.closest('[data-nav]');
            if (navEl) navKey = navEl.getAttribute('data-nav');
        }
        if (navKey && index && index.nav && index.nav[navKey]) {
            var sites = index.nav[navKey];
            out.nav = { key: navKey, sites: sites };
            if (!out.defs.length) {
                var targets = [];
                sites.forEach(function (s) { (s.targets || []).forEach(function (t) { targets.push(t); }); });
                targets.forEach(function (t) {
                    defsFor(t).forEach(function (d) {
                        out.defs.push(Object.assign({ symbol: t }, d));
                    });
                });
                if (!out.handler && targets.length) {
                    out.handler = { name: targets[targets.length - 1], via: 'data-nav', source: 'data-nav="' + navKey + '"' };
                }
            }
        }

        // Endpoints reached from the handler's body, each mapped back to the
        // Flask view that serves it.
        out.defs.forEach(function (d) {
            (d.calls || []).forEach(function (c) {
                out.calls.push({ method: c.method, url: c.url, routes: routesFor(c.url) });
            });
        });

        if (!out.handler && !out.defs.length) {
            out.note = out.id
                ? 'No inline handler — a listener is attached in JS. Grep for the id.'
                : 'No inline handler and no id — likely a purely presentational element.';
        }
        return out;
    }

    // ── Tooltip ──────────────────────────────────────────────────────────────
    function ensureTip() {
        if (tipEl) return tipEl;
        tipEl = document.createElement('div');
        tipEl.id = 'devmapTip';
        tipEl.style.cssText = [
            'position:fixed;z-index:2147483600;display:none;',
            'max-width:620px;min-width:260px;',
            'background:#020617;border:1px solid #1e3a5f;border-radius:10px;',
            'box-shadow:0 12px 44px rgba(0,0,0,.75);',
            'font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;',
            'font-size:11px;line-height:1.55;color:#94a3b8;',
            'padding:0;overflow:hidden;pointer-events:none;'
        ].join('');
        // Pinning turns pointer events back on so rows become clickable; while
        // unpinned the tooltip must not intercept the mouse or hovering the
        // element under it would flicker.
        tipEl.addEventListener('mouseleave', function () { if (pinned) hideTip(); });
        document.body.appendChild(tipEl);
        return tipEl;
    }

    function row(label, value, path) {
        var pathHtml = '';
        if (path) {
            pathHtml =
                '<span class="devmap-path" data-path="' + esc(path) + '" ' +
                'style="color:#67e8f9;cursor:' + (pinned ? 'pointer' : 'default') + ';' +
                'text-decoration:underline;text-underline-offset:2px;' +
                'text-decoration-color:rgba(103,232,249,.35);">' + esc(path) + '</span>';
        }
        return '<div style="display:flex;gap:10px;padding:2px 0;">' +
               '<span style="color:#475569;flex-shrink:0;width:58px;">' + esc(label) + '</span>' +
               '<span style="flex:1;min-width:0;word-break:break-all;">' + value + '</span>' +
               (pathHtml ? '<span style="flex-shrink:0;margin-left:10px;">' + pathHtml + '</span>' : '') +
               '</div>';
    }

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function render(info) {
        var html = '';

        html += '<div style="padding:7px 11px;background:#060d1a;border-bottom:1px solid #1e3a5f;' +
                'display:flex;align-items:center;gap:8px;">' +
                '<span style="color:#22d3ee;font-weight:700;font-size:10px;letter-spacing:.08em;">SOURCE</span>' +
                '<span style="color:#e2e8f0;font-size:11px;flex:1;overflow:hidden;text-overflow:ellipsis;' +
                'white-space:nowrap;">' + esc(info.label || info.tag) + '</span>' +
                (info.id ? '<span style="color:#475569;font-size:10px;">#' + esc(info.id) + '</span>' : '') +
                '</div>';

        html += '<div style="padding:8px 11px;">';

        if (info.handler) {
            html += row('Handler',
                '<span style="color:#4ade80;">' + esc(info.handler.name) + '()</span>' +
                '<span style="color:#334155;"> via ' + esc(info.handler.via) + '</span>', '');
        }

        if (info.defs.length) {
            info.defs.forEach(function (d) {
                var name = d.symbol ? d.symbol + '()' : (d.kind || '');
                html += row('Defined',
                    '<span style="color:#a5b4fc;">' + esc(d.symbol ? name : d.kind) + '</span>' +
                    '<span style="color:#334155;"> ' + esc(d.lang) + '</span>',
                    d.file + ':' + d.line);
            });
        }

        if (info.nav) {
            info.nav.sites.forEach(function (s) {
                html += row('Dispatch',
                    '<span style="color:#fbbf24;">case \'' + esc(info.nav.key) + '\'</span>',
                    s.file + ':' + s.line);
            });
        }

        if (info.markup.length) {
            info.markup.slice(0, 3).forEach(function (m) {
                html += row('Markup', '<span style="color:#94a3b8;">#' + esc(info.id) + '</span>',
                    m.file + ':' + m.line);
            });
        }

        if (info.calls.length) {
            html += '<div style="height:1px;background:#0f2338;margin:6px 0;"></div>';
            info.calls.slice(0, 8).forEach(function (c) {
                var methodColor = c.method === 'SSE' ? '#e879f9'
                                : c.method === 'POST' ? '#fbbf24' : '#34d399';
                var val = '<span style="color:' + methodColor + ';">' + esc(c.method) + '</span> ' +
                          '<span style="color:#cbd5e1;">' + esc(c.url) + '</span>';
                if (c.routes.length) {
                    c.routes.forEach(function (r) {
                        html += row('Calls', val + '<span style="color:#334155;"> → ' +
                            esc(r.view) + '()</span>', r.file + ':' + r.line);
                        val = '';   // only label the first line of a multi-route group
                    });
                } else {
                    html += row('Calls', val + '<span style="color:#7f1d1d;"> (no matching route)</span>', '');
                }
            });
        }

        if (info.note) {
            html += '<div style="color:#475569;padding-top:4px;font-style:italic;">' +
                    esc(info.note) + '</div>';
        }

        html += '</div>';

        html += '<div style="padding:5px 11px;background:#060d1a;border-top:1px solid #0f2338;' +
                'color:#334155;font-size:9.5px;display:flex;justify-content:space-between;gap:10px;">' +
                '<span>' + (pinned ? 'click a path to copy' : esc(pinLabel())) + '</span>' +
                '<span>' + esc(CFG.combo.label) + ' to toggle</span></div>';

        return html;
    }

    function positionTip(el) {
        var tip = ensureTip();
        var r = el.getBoundingClientRect();
        tip.style.visibility = 'hidden';
        tip.style.display = 'block';
        var tr = tip.getBoundingClientRect();

        // Prefer below-right of the element; flip when that would leave the
        // viewport. A tooltip clipped by the window edge is worse than useless
        // because the file path is what gets cut off.
        var top = r.bottom + 8;
        if (top + tr.height > window.innerHeight - 8) top = Math.max(8, r.top - tr.height - 8);
        var left = r.left;
        if (left + tr.width > window.innerWidth - 8) left = Math.max(8, window.innerWidth - tr.width - 8);

        tip.style.top = top + 'px';
        tip.style.left = left + 'px';
        tip.style.visibility = 'visible';
    }

    function showTip(el, alt) {
        loadIndex().then(function () {
            if (currentTarget !== el) return;         // pointer moved on while we waited
            var info = resolve(el);
            if (!info.handler && !info.defs.length && !info.markup.length && !info.nav) {
                hideTip();
                return;
            }
            pinned = !!alt;
            var tip = ensureTip();
            tip.innerHTML = render(info);
            tip.style.pointerEvents = pinned ? 'auto' : 'none';
            positionTip(el);
        }).catch(function () { /* index unavailable — stay silent */ });
    }

    function hideTip() {
        if (tipEl) { tipEl.style.display = 'none'; tipEl.style.pointerEvents = 'none'; }
        pinned = false;
        currentTarget = null;
    }

    // ── Wiring ───────────────────────────────────────────────────────────────
    var CANDIDATE = 'button,[onclick],[data-nav],a[href],input[type=button],input[type=submit],select,[role=button]';

    document.addEventListener('mouseover', function (e) {
        if (!enabled) return;
        var el = e.target && e.target.closest ? e.target.closest(CANDIDATE) : null;
        if (!el || el === currentTarget) return;
        if (tipEl && tipEl.contains(e.target)) return;
        currentTarget = el;
        clearTimeout(hoverTimer);
        var pin = pinHeld(e);
        hoverTimer = setTimeout(function () { showTip(el, pin); }, CFG.hover_delay_ms);
        // showTip() re-checks: an element with no handler, no id and no
        // data-nav has nothing to report, and an empty tooltip following the
        // cursor around is worse than no tooltip.
    }, true);

    document.addEventListener('mouseout', function (e) {
        if (!enabled) return;
        var el = e.target && e.target.closest ? e.target.closest(CANDIDATE) : null;
        if (!el) return;
        clearTimeout(hoverTimer);
        // A pinned tooltip survives the pointer leaving the element so its rows
        // can be reached; its own mouseleave closes it.
        if (!pinned) hideTip();
    }, true);

    document.addEventListener('click', function (e) {
        var path = e.target && e.target.closest ? e.target.closest('.devmap-path') : null;
        if (!path || !pinned) return;
        e.preventDefault();
        e.stopPropagation();
        var text = path.getAttribute('data-path') || '';
        var done = function () {
            var old = path.textContent;
            path.textContent = 'copied ✓';
            setTimeout(function () { path.textContent = old; }, 900);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(done, function () {});
        } else {
            var ta = document.createElement('textarea');
            ta.value = text; document.body.appendChild(ta); ta.select();
            try { document.execCommand('copy'); done(); } catch (err) {}
            document.body.removeChild(ta);
        }
    }, true);

    document.addEventListener('keydown', function (e) {
        if (comboMatches(e, CFG.combo)) {
            e.preventDefault();
            setEnabled(!enabled);
        }
        if (e.key === 'Escape') hideTip();
    });

    // ── Badge ────────────────────────────────────────────────────────────────
    function ensureBadge() {
        if (badgeEl) return badgeEl;
        badgeEl = document.createElement('button');
        badgeEl.id = 'devmapBadge';
        badgeEl.title = 'Source map — hover any button to see the file and line it runs.\n' +
                        (pinModLabel() ? 'Hold ' + pinModLabel() +
                                      ' while hovering to pin the tooltip and copy a path.\n' : '') +
                        CFG.combo.label + ' toggles.\n' +
                        'Configured in input_TS.yml under visualizer.devmap.';
        badgeEl.style.cssText = [
            'position:fixed;left:14px;bottom:14px;z-index:2147483500;',
            'padding:4px 9px;border-radius:8px;cursor:pointer;',
            'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;',
            'letter-spacing:.04em;transition:opacity .15s,border-color .15s;opacity:.5;'
        ].join('');
        badgeEl.addEventListener('mouseover', function () { badgeEl.style.opacity = '1'; });
        badgeEl.addEventListener('mouseout', function () { badgeEl.style.opacity = '.5'; });
        badgeEl.addEventListener('click', function (ev) {
            ev.stopPropagation();
            setEnabled(!enabled);
        });
        document.body.appendChild(badgeEl);
        return badgeEl;
    }

    function paintBadge() {
        if (!CFG.show_badge) {
            if (badgeEl && badgeEl.parentNode) badgeEl.parentNode.removeChild(badgeEl);
            badgeEl = null;
            return;
        }
        var b = ensureBadge();
        if (enabled) {
            b.style.background = '#052e1c';
            b.style.border = '1px solid #065f46';
            b.style.color = '#34d399';
            b.textContent = '⌖ src map on';
        } else {
            b.style.background = '#111827';
            b.style.border = '1px solid #1e293b';
            b.style.color = '#475569';
            b.textContent = '⌖ src map off';
        }
    }

    function setEnabled(v) {
        enabled = !!v;
        writePref(enabled);            // an explicit toggle outranks the yml
                                       // default from here on, in this browser
        if (!enabled) hideTip();
        else loadIndex().catch(function () {});   // boot skipped the warm-up
        paintBadge();
    }

    // As setEnabled, but leaves localStorage alone — used by devmap.reset(),
    // which is specifically trying to observe the un-toggled default.
    function setEnabledSilently(v) {
        enabled = !!v;
        if (!enabled) hideTip();
        paintBadge();
    }

    function boot() {
        // Config first, and nothing is painted or warmed until it lands. The
        // overlay is off in the meantime (`enabled` starts false), so a slow
        // config fetch shows nothing rather than showing the wrong thing.
        loadConfig().then(function () {
            var saved = readPref();                   // null = never toggled here
            enabled = (saved === null) ? !!CFG.enabled : saved;
            paintBadge();

            // Only warm the index when the overlay is actually on. Prefetching
            // 200–400 KB on every page load for a tool that ships disabled is
            // exactly the cost this default exists to avoid; the first hover
            // after enabling pays for it instead.
            if (!enabled) return;
            var warm = function () { loadIndex().catch(function () {}); };
            if (window.requestIdleCallback) window.requestIdleCallback(warm, { timeout: 4000 });
            else setTimeout(warm, 2500);
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
    else boot();

    // Console helpers — `devmap.where('_vinaDock')` while reading a stack trace.
    window.devmap = {
        enable: function () { setEnabled(true); },
        disable: function () { setEnabled(false); },
        config: function () { return JSON.parse(JSON.stringify(CFG)); },
        // Forget this browser's toggle and fall back to the yml default —
        // the only way to verify what a fresh visitor actually sees.
        reset: function () {
            try { window.localStorage.removeItem(STORE_KEY); } catch (e) {}
            return loadConfig().then(function () { setEnabledSilently(!!CFG.enabled); return CFG; });
        },
        reload: function () { index = null; indexPromise = null; return loadIndex(); },
        index: function () { return index; },
        where: function (name) {
            return loadIndex().then(function () {
                var d = defsFor(name);
                if (!d.length) { console.warn('[devmap] no definition for', name); return d; }
                d.forEach(function (x) { console.log(name, '→', x.file + ':' + x.line, '·', x.snippet); });
                return d;
            });
        },
        route: function (url) {
            return loadIndex().then(function () {
                var r = routesFor(url);
                r.forEach(function (x) {
                    console.log(x.methods.join('/'), x.rule, '→', x.file + ':' + x.line, '·', x.view + '()');
                });
                return r;
            });
        }
    };
})();
