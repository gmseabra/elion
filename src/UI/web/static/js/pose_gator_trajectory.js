// =============================================================================
// pose_gator_trajectory.js — register GatorAffinity with the trajectory engine
// -----------------------------------------------------------------------------
// All the machinery lives in pose_affinity_trajectory.js; this file is the
// GatorAffinity third of its configuration, in the same relationship
// pose_gign_trajectory.js and pose_deepatom_trajectory.js have to it. If you are
// looking for how records are chosen, serialised, queued or gated, read that
// file — nothing here decides any of it.
//
// GatorAffinity-specific facts, and there are only three:
//
//   * /pose/gatoraffinity_score stages into <out_dir>/gator_<name>/, the same
//     shape /pose/gign_score uses, so it wants the trajectory root as-is.
//     `perRecordDir` stays FALSE. This is the GIGN answer, not the DeepAtom one:
//     DeepAtom needs a root per record because its pipeline is a
//     virtual-screening driver that reprocesses every complex under the root it
//     is handed. gatoraffinity_pose.py reads one staged complex named on the
//     command line and nothing else, so a shared root costs nothing and keeps
//     every record's pocket, blocks and log in one place.
//
//   * PG.mc._gatorBusy is the flag the hand-clicked "⚛ Score (GatorAffinity)"
//     button sets; the shared pump waits on it, so a manual score and the queue
//     cannot overlap.
//
//   * The float's subtitle is #poseGatorFloatSub, derived from the dock's
//     `float: 'poseGatorFloat'`.
//
// On cost: this is a graph model like GIGN rather than a voxel pipeline like
// DeepAtom, but it carries the ATOMICA backbone and a fragmentation step, so
// budget several seconds per pose — between the two. The engine's queue is
// serial ACROSS scorers because they all land on one GPU, so arming all three
// on a 12-record run means the three curves finish in sequence, not at once.
// _trajState() shows the backlog; _trajOff('gator') disarms just this one and
// the setting persists.
//
// Load after pose.js and pose_affinity_trajectory.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityTrajectory) {
        console.warn('[gator-traj] pose_affinity_trajectory.js not loaded');
        return;
    }

    window.PoseAffinityTrajectory.register({
        key:          'gator',
        label:        'GatorAffinity',
        endpoint:     '/pose/gatoraffinity_score',
        setter:       '_gatorSet',
        busyFlag:     '_gatorBusy',
        subId:        'poseGatorFloatSub',
        store:        'elion.pose.gatorTraj',
        perRecordDir: false,
        dock:         '_poseGatorDockApi',   // so a score can be tagged with its pose
    });

    window._gatorTrajOn    = function () { return window.PoseAffinityTrajectory.on('gator'); };
    window._gatorTrajOff   = function () { return window.PoseAffinityTrajectory.off('gator'); };
    window._gatorTrajState = function () { return window.PoseAffinityTrajectory.state(); };
})();
