// =============================================================================
// deepatom_chat.js
// DeepAtom mini-chat panel wiring — show/hide/welcome for the shared panel.
//
// Depends on: elion_mini_chat.js (must load first — owns _MINICHAT_DEEPATOM_THEME
//             and _deepatomMiniWelcome)
//
// deepatom.js owns: all CNN saliency visualisation, 3D/bar/atom-table rendering,
//                   Make Atomtypes, SSE log streaming.
// This file owns:   modal open/close + shared panel handoff.
// =============================================================================

// ── Open / Close ──────────────────────────────────────────────────────────────

function _deepatomShow() {
    document.getElementById('deepatomModal').classList.remove('hidden');
    document.getElementById('deepatomModal').classList.add('flex');

    setTimeout(() => {
        // Apply DeepAtom theme to the shared floating panel
        if (typeof _miniChatSetContext === 'function' &&
            typeof _MINICHAT_DEEPATOM_THEME !== 'undefined') {
            _miniChatSetContext(_MINICHAT_DEEPATOM_THEME);
        }
        _miniChatShow();

        // Always show DeepAtom welcome when opening:
        // clear whatever the previous tool left and render fresh DeepAtom content.
        const out = document.getElementById('miniChatOutput');
        if (out) {
            out.innerHTML = '';
            if (typeof _deepatomMiniWelcome === 'function') {
                _deepatomMiniWelcome();
            }
        }

        const inp = document.getElementById('daLigandPath');
        if (inp) inp.focus();
    }, 300);
}

function _deepatomHide() {
    document.getElementById('deepatomModal').classList.add('hidden');
    document.getElementById('deepatomModal').classList.remove('flex');
    _miniChatClose();
    // Restore Vina theme when leaving DeepAtom
    if (typeof _miniChatSetContext === 'function' &&
        typeof _MINICHAT_VINA_THEME !== 'undefined') {
        _miniChatSetContext(_MINICHAT_VINA_THEME);
    }
}

// ── Mini-chat aliases (keep deepatom.js callers working unchanged) ─────────────
// deepatom.js calls _miniChatAppend directly — no separate aliases needed.
// These stubs are here only for forward-compat in case deepatom.js uses them.

function _deepatomMiniShow()  { _miniChatShow(); }
function _deepatomMiniClose() { _miniChatClose(); }