// =============================================================================
// pose_gign_trajectory.js — register Yupu_GIGN with the trajectory engine
// -----------------------------------------------------------------------------
// All the machinery lives in pose_affinity_trajectory.js; this file is the GIGN
// half of its configuration, in the same relationship pose_gign_dock.js has to
// pose_affinity_dock.js. If you are looking for how records are chosen,
// serialised, queued or gated, read that file — nothing here decides any of it.
//
// This was a 289-line standalone module until DeepAtom needed the same feature.
// Copying it would have produced two independent queues feeding one GPU and two
// capture paths that could disagree about which poses were the records; the two
// floats over the same pocket are only comparable if they were handed the same
// twelve poses, byte for byte.
//
// GIGN-specific facts, and there are only three:
//
//   * /pose/gign_score already stages into <out_dir>/gign_<name>/, so it wants
//     the trajectory root as-is. `perRecordDir` stays false.
//   * PG.mc._gignBusy is the flag the hand-clicked "⚛ Score (GIGN)" button sets;
//     the shared pump waits on it, so a manual score and the queue cannot
//     overlap.
//   * The float's subtitle is #poseGignFloatSub, derived from the dock's
//     `float: 'poseGignFloat'`.
//
// Load after pose.js and pose_affinity_trajectory.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityTrajectory) {
        console.warn('[gign-traj] pose_affinity_trajectory.js not loaded');
        return;
    }

    window.PoseAffinityTrajectory.register({
        key:          'gign',
        label:        'GIGN',
        endpoint:     '/pose/gign_score',
        setter:       '_gignSet',
        busyFlag:     '_gignBusy',
        subId:        'poseGignFloatSub',
        store:        'elion.pose.gignTraj',
        perRecordDir: false,
        dock:         '_poseGignDockApi',   // so a score can be tagged with its pose
    });

    // Back-compat: the previous module exported these names and they are in the
    // muscle memory of anyone who has been debugging this. They now scope to
    // GIGN alone; _trajOn()/_trajOff() with no argument cover every scorer.
    window._gignTrajOn    = function () { return window.PoseAffinityTrajectory.on('gign'); };
    window._gignTrajOff   = function () { return window.PoseAffinityTrajectory.off('gign'); };
    window._gignTrajMinDg = function (v) { return window.PoseAffinityTrajectory.minDg(v); };
    window._gignTrajState = function () { return window.PoseAffinityTrajectory.state(); };
})();