// =============================================================================
// pose_gator_dock.js — the GatorAffinity predictor's draggable readout
// -----------------------------------------------------------------------------
// The third panel from the same factory (pose_affinity_dock.js) that produced
// pose_gign_dock.js and pose_deepatom_dock.js. Drag #poseGatorCard by its title
// bar onto the pocket view -> it becomes #poseGatorFloat: dG by default, a pK
// toggle, a hover-able sparkline, close returns it to the sidebar.
//
// Child ids derive from `float`: #poseGatorFloatBar / Title / Val / Chart / Sub /
// Close / Run / Unit.
//
// Two GatorAffinity-specific choices, and they are the same two the DeepAtom
// file had to make:
//
//   * Emerald, not violet and not cyan. THREE floats can now be open over the
//     same pocket at once, and they are three different predictions of one
//     quantity — confusing them is the whole risk, so each gets its own colour.
//     (#a78bfa GIGN · #22d3ee DeepAtom · #34d399 GatorAffinity.)
//
//   * busyIds carries BOTH #poseGatorScoreBtn (the one in the ③ Monte-Carlo
//     control strip) and #poseGatorRunBtn (the one on the sidebar card). pose.js
//     only ever rewrites the first, exactly as it only ever rewrote
//     #poseGignScoreBtn — driving both from here fixes the card button's missing
//     busy text without editing pose.js, which keeps this file removable.
//
// Load after pose.js (it wraps PG.mc methods) and after pose_affinity_dock.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityDock) { console.warn('[gator-dock] pose_affinity_dock.js not loaded'); return; }

    window._poseGatorDockApi = window.PoseAffinityDock({
        key:        'gator',
        card:       'poseGatorCard',
        float:      'poseGatorFloat',
        store:      'elion.pose.gatorDock',
        color:      '#34d399',
        border:     '#14532d',
        name:       'GatorAffinity',
        titles:     { dg: 'GatorAffinity ΔG (lower)', pk: 'GatorAffinity pK (higher)' },
        pkId:       'poseGatorPk',
        outDirId:   'poseGatorOutDir',
        busyIds:    ['poseGatorScoreBtn', 'poseGatorRunBtn'],
        scoreLabel: '⚛ Score (GatorAffinity)',
        setter:     '_gatorSet',
        score:      'scoreGator',
        lastPk:     'gatorPk',
        scorerVal:  'gatoraffinity',
    });

    window._poseGatorUnit   = function (u) { return window._poseGatorDockApi.unit(u); };
    window._poseGatorDock   = function (x, y) { window._poseGatorDockApi.dock(x, y); return 'docked'; };
    window._poseGatorUndock = function () { window._poseGatorDockApi.undock(); return 'undocked'; };
})();
