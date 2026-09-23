// =============================================================================
// pose_deepatom_trajectory.js — register the DeepAtom affinity predictor with
//                               the trajectory engine
// -----------------------------------------------------------------------------
// GIGN's twin, from the same engine (pose_affinity_trajectory.js). This is what
// takes #poseDaFloat from "one score so far" — whatever you last clicked — to a
// curve over the same record poses GIGN sees.
//
// ONE setting differs from GIGN, and it is not cosmetic:
//
//   perRecordDir: true
//
// /pose/deepatom_score writes <root>/Dataset_VS/<name>/<name>_complex.pdb and
// then runs the pipeline as `<script> -t vs -d <root>`. The pipeline processes
// **every** complex under <root>/Dataset_VS — it is a virtual-screening driver,
// that is its job. So a shared root means record 12 re-runs arpeggio atom-typing
// and grid generation over records 1..11 as well: 78 complexes for a 12-record
// run, growing as n²/2, each one a result you already have.
//
// Giving every record its own root — <trajectory_dir>/<rec>/ — keeps each call
// at exactly one complex, which is both the correctness fix (the CSV the route
// parses holds one row, not a growing table it has to pick from) and the reason
// this feature is affordable at all. The atomtypes/ and npz/ intermediates land
// beside their own complex, which is also where you want them when a single
// record fails and you need to look at it.
//
// DeepAtom is seconds-to-minutes per pose against GIGN's ~1 s, and the engine's
// queue is deliberately serial across scorers because both land on one GPU.
// Expect the GIGN curve to finish well before this one; _trajState() shows the
// backlog. If that is not a trade you want on a given run, _trajOff('deepatom')
// and the setting persists.
//
// Load after pose.js and pose_affinity_trajectory.js.
// =============================================================================

(function () {
    'use strict';
    if (!window.PoseAffinityTrajectory) {
        console.warn('[deepatom-traj] pose_affinity_trajectory.js not loaded');
        return;
    }

    window.PoseAffinityTrajectory.register({
        key:          'deepatom',
        label:        'DeepAtom',
        endpoint:     '/pose/deepatom_score',
        setter:       '_daSet',
        busyFlag:     '_daBusy',
        subId:        'poseDaFloatSub',
        store:        'elion.pose.daTraj',
        perRecordDir: true,
        dock:         '_poseDaDockApi',     // so a score can be tagged with its pose
    });

    window._daTrajOn    = function () { return window.PoseAffinityTrajectory.on('deepatom'); };
    window._daTrajOff   = function () { return window.PoseAffinityTrajectory.off('deepatom'); };
    window._daTrajState = function () { return window.PoseAffinityTrajectory.state(); };
})();