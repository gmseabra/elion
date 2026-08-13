// =============================================================================
// pose_gign_dock.js — the Yupu_GIGN affinity predictor's draggable readout
// -----------------------------------------------------------------------------
// All the behaviour lives in pose_affinity_dock.js, which DeepAtom instantiates the
// same way (pose_deepatom_dock.js). This file is only what is genuinely specific to
// GIGN: which card, which colours, which PG.mc methods to wrap.
//
// Drag #poseGignCard by its title bar onto the pocket view -> it becomes
// #poseGignFloat, a readout shaped like #poseProcPanel with a dG/pK toggle, a
// hover-able sparkline and a close button that sends it back.
//
// Child ids derive from `float`: #poseGignFloatBar / Title / Val / Chart / Sub /
// Close / Run / Unit.
//
// Load after pose.js (it wraps PG.mc methods) and after pose_affinity_dock.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityDock) { console.warn('[gign-dock] pose_affinity_dock.js not loaded'); return; }

    window._poseGignDockApi = window.PoseAffinityDock({
        key:        'gign',
        card:       'poseGignCard',
        float:      'poseGignFloat',
        store:      'elion.pose.gignDock',
        color:      '#a78bfa',
        border:     '#3b2a6b',
        name:       'GIGN',
        titles:     { dg: 'GIGN ΔG (lower)', pk: 'GIGN pK (higher)' },
        pkId:       'poseGignPk',
        outDirId:   'poseGignOutDir',
        // pose.js only ever updated #poseGignScoreBtn, so the sidebar card's own
        // #poseGignRunBtn never showed the busy text. Driving both from here
        // fixes that without editing pose.js, keeping this removable.
        busyIds:    ['poseGignScoreBtn', 'poseGignRunBtn'],
        scoreLabel: '⚛ Score (GIGN)',
        setter:     '_gignSet',
        score:      'scoreGign',
        lastPk:     'gignPk',
        scorerVal:  'yupu_gign',
    });

    window._poseGignUnit   = function (u) { return window._poseGignDockApi.unit(u); };
    window._poseGignDock   = function (x, y) { window._poseGignDockApi.dock(x, y); return 'docked'; };
    window._poseGignUndock = function () { window._poseGignDockApi.undock(); return 'undocked'; };
})();