// =============================================================================
// pose_deepatom_dock.js — the DeepAtom affinity predictor's draggable readout
// -----------------------------------------------------------------------------
// The GIGN panel's twin, from the same factory (pose_affinity_dock.js). Drag
// #poseDaCard by its title bar onto the pocket view -> it becomes #poseDaFloat:
// dG by default, a pK toggle, a hover-able sparkline, close returns it.
//
// Child ids derive from `float`: #poseDaFloatBar / Title / Val / Chart / Sub /
// Close / Run / Unit.
//
// Two things differ from GIGN beyond the ids:
//
//   * Cyan, not violet. Two floats can be open over the same pocket at once and
//     they must be distinguishable at a glance, because their numbers are
//     different predictions of the same quantity and confusing them is the
//     whole risk. (The sidebar cards are both violet; that is fine, they are
//     never side by side.)
//
//   * The busy list carries only #poseDaScoreBtn. #poseDaPdbBtn on the card is
//     "generate pose .pdb", not a score action, and rewriting its label during
//     scoring would be wrong.
//
// Load after pose.js and pose_affinity_dock.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityDock) { console.warn('[deepatom-dock] pose_affinity_dock.js not loaded'); return; }

    window._poseDaDockApi = window.PoseAffinityDock({
        key:        'deepatom',
        card:       'poseDaCard',
        float:      'poseDaFloat',
        store:      'elion.pose.daDock',
        color:      '#22d3ee',
        border:     '#0e4f63',
        name:       'DeepAtom',
        titles:     { dg: 'DeepAtom ΔG (lower)', pk: 'DeepAtom pK (higher)' },
        pkId:       'poseDaPk',
        outDirId:   'poseDaOutDir',
        busyIds:    ['poseDaScoreBtn'],
        scoreLabel: '⚛ Score best pose',
        setter:     '_daSet',
        score:      'scoreDeepAtom',
        lastPk:     'daPk',
        scorerVal:  'deepatom',
    });

    window._poseDaUnit   = function (u) { return window._poseDaDockApi.unit(u); };
    window._poseDaDock   = function (x, y) { window._poseDaDockApi.dock(x, y); return 'docked'; };
    window._poseDaUndock = function () { window._poseDaDockApi.undock(); return 'undocked'; };
})();