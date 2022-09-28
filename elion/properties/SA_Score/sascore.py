#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#
# ===============================================
# GMS
# Note: The SA is on a scale from 1-10, where:
#        1 == Very easy to synthezise.
#       10 == Very hard to synthezise.
# Usage:
#       import util.sascorer
#       sascorer.calculatescore(mol) --> float: the SA score.
#
# THIS IS A SLIGHTLY MODIFIED VERSION OF THE ORIGINAL CODE HERE:
# https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle

import math
from collections import defaultdict

import os.path as op

_fscores = None

class SA_Scorer:

    def __init__(self,fscores_file='fpscores'):
        self.fscores = self._readFragmentScores(fscores_file)

    def _readFragmentScores(self, name):
        import gzip
        #global _fscores
        # generate the full path filename:
        if name is None: name == 'fpscores'
        if name == "fpscores":
            name = op.join(op.dirname(__file__), name)
        data = pickle.load(gzip.open('%s.pkl.gz' % name))
        outDict = {}
        for i in data:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        fscores = outDict
        print("Initialized fragment scores.")
        return fscores

    def _numBridgeheadsAndSpiro(self, mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro

    def _calculate_score(self, m):
        """
        Gets an RDKIT Mol object and returns its SASCore.
        """
        #if self.fscores is None:
        #    _readFragmentScores()

        # fragment score
        # Scores are calculated on Morgan circular fnigerprints of radius = 2
        _fscores = self.fscores
        fp = rdMolDescriptors.GetMorganFingerprint(m,2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0

        # --GMS
        # In the generator case, sometimes a weird molecule gets to this point,
        # and gets an all-zero fingerprint. That causes a division-by-zero error
        # right after this for loop.
        #
        # To avoid problems with such weird molecules, let's just skip the whole 
        # calculation in this case.

        if len(fps) > 0:
            for bitId, v in fps.items():
                nf += v
                sfp = bitId
                score1 += _fscores.get(sfp, -4) * v
            score1 /= nf

            # features score
            nAtoms = m.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
            ri = m.GetRingInfo()
            nBridgeheads, nSpiro = self._numBridgeheadsAndSpiro(m, ri)
            nMacrocycles = 0
            for x in ri.AtomRings():
                if len(x) > 8:
                    nMacrocycles += 1

            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = 0.
            # ---------------------------------------
            # This differs from the paper, which defines:
            #  macrocyclePenalty = math.log10(nMacrocycles+1)
            # This form generates better results when 2 or more macrocycles are present
            if nMacrocycles > 0:
                macrocyclePenalty = math.log10(2)

            score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

            # correction for the fingerprint density
            # not in the original publication, added in version 1.1
            # to make highly symmetrical molecules easier to synthetise
            score3 = 0.
            if nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * .5

            sascore = score1 + score2 + score3

            # need to transform "raw" value into scale between 1 and 10
            min = -4.0
            max = 2.5
            sascore = 11. - (sascore - min + 1) / (max - min) * 9.
            # smooth the 10-end
            if sascore > 8.:
                sascore = 8. + math.log(sascore + 1. - 9.)
            if sascore > 10.:
                sascore = 10.0
            elif sascore < 1.:
                sascore = 1.0

        else:
            smi = Chem.MolToSmiles(m,isomericSmiles=False,kekuleSmiles=False,canonical=False)

            print("[SASCORER ERROR] Molecule has an all-zero fingerprint! SASCORE set to 10.")
            print("[SASCORER ERROR] SMILES:", smi)
            sascore = 10

        return sascore

    def predict(self, mols):
        """
        Gets a list of RDKit Mol objects, and prints a
        table with SMILES, SAScore for each molecule.
        """
        scores = []
        for i, m in enumerate(mols):
            if m:
                scores.append(self._calculate_score(m))
            else:
                scores.append(10)
        return scores

if __name__ == '__main__':
    import sys
    import time

    sa_scorer = SA_Scorer()

    suppl = Chem.SmilesMolSupplier(sys.argv[1], nameColumn=-1)
    #sa_scorer.processMols(suppl)
    
    # This is broken now!!
    for mol in suppl:
        score = sa_scorer.predict(mol)
        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp('_Name')
        print(f"{smiles:<70}  {name:>5}  {score:6.3f}")

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
