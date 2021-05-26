The repository contains code to generate novel fragments for FBDD using the "syntax corrected" dual encoder model reported in Bisland et al (2021).
See https://pubs.acs.org/doi/10.1021/acs.jcim.0c01226


Requirements:
- Ubuntu 64 bit
- Anaconda3 
- CUDA >= 10.0.130

Notes:
- Note that the PSO optimisation requires a SMARTS file containg structural alerts
- A default file is provided from sureChembl with some additions to rein in the model's imagination
- See https://www.surechembl.org/knowledgebase/169485-non-medchem-friendly-smarts
- Original source: Sushko et al (2012), Journal of Chemical Information and Modeling 2012 52 (8), 2310-2316
- The user is advised to supply their own more comprehensive file for optimal performance.
- This should be named "SMARTS_unwanted_functionalities.txt" and placed in the data folder, replacing the existing file.
- 5000 seed molecules and fingerprint features used in the paper are provided in the data folder
- These have the substitutions Cl->X and Br->Y - output generated molecules will have these substituted back
- To use your own seed molecules you will need to make these substitutions - manually, or use smiles_substitute() in model_utils.py
- The model vocabulary is: [,n,3,),=,-,],!,O,N,1,s,C,+,E,2,X,Y,c,(,H,4,o,K,S,F,#, where "!" and "E" are start and stop characters
- To generate required fingerprint features for your own molecules use the functions in fingerprint_utils.py - mol list should have header smiles
- Note that the above substitutions are needed to one-hot encode for the LSTM - the fingerprint functions need UNSUBSTITUTED smiles
- The rdkit BaseFeatures.fdef file by Greg Landstrum is included in the data folder for pharmacophore fingerprints
- The BSD 3-clause license covering rdkit code redistribution is included at the bottom of this file (just in case - I'm not a lawyer, barely even a coder)
- Finally, to process the deafult 5K seeds we use 5 threads in multiprocessing - if you have fewer seed molecules, you might want to change this

Usage:
- install the conda environment in environment.yml and activate with $conda activate dualEncoder
- to run with the default 5K seeds provided just do $python genFrags_v1.py in the top level of the cloned repo, then let it grind away
- the output file is fragmentGenerator_output.csv which contains the seed and generated smiles, sas, privileged fragment score, and overall PSO
- overall PSO < -4.5 roughly is probably pretty good. Between -3.5 and -4 may also be of interest
- positive overall scores will have structural alerts. Value 3 means no valid smiles was found.
- fragment prediction scores greater than about 0.4 are the positive class in ROC analysis

- to generate required fingerprints for your own molecules as seeds:

from synCor import fingerprint_utils as fu
mols = fu.filterFails("pathToYourMols.csv")
fp, ph = fu.smiles_to_fingerprints("filtered_mols.csv")
fullFP = fu.varianceFilter(fp, ph)


Reproduced rdkit BSD 3-clause license:
Copyright (c) 2006-2015, Rational Discovery LLC, Greg Landrum, and Julie Penzotti and others
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

