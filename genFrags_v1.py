######################## Use syntax corrected dual encoder to generate fragments ######################################################
#multithreading
import multiprocessing
from multiprocessing import Pool
files = ["5000_random_library_smiles_fold1", "5000_random_library_smiles_fold2","5000_random_library_smiles_fold3", "5000_random_library_smiles_fold4", "5000_random_library_smiles_fold5"]

#check generation time - can remove later
import time

#setup tensorflow session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# system utilities
import os
import sys

#numpy and pandas
import numpy as np
import pandas as pd

# pyswarms for latent space sampling
import pyswarms as ps

# model utilities
from synCor import model_utils as mu

# pyswarms for PSO
import pyswarms as ps

# rdkit - note that synthetic accessibility score is a Contrib function- see https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
# suppress rdkit warnings to console
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Load the pre-built models
encoder, decoder, states_and_pred = mu.load_dual_models()

# Load structural alerts file - note that for optimal output the user should supply their own detailed alerts file in SMARTS format
alerts = pd.read_csv(mu.find("SMARTS_unwanted_functionalities.txt", os.getcwd()), sep='\t', names=['smarts', 'alert'])
smarts = alerts.smarts.values

# load smiles and fingerprints data - get the files from command line input or load default files
#mols = pd.read_csv('5000_random_library_smiles_duplicatesOut_testFold.csv', header = 0, names=['smiles'])
#fps = np.load('5000_random_library_smiles_fingerprints_testFold.npy')

# load model parameters
params = mu.load_params()

# generate decode dictionary matrix with ascii representation of RNN vocabulary characters
dictMat = mu.generate_decode_matrix(params["reverse_dictionary"])

#sampling depth around seed molecules - see inner loop of main function
sampleDepth = 10

#initial target cost to move PSO swarm for each molecule in user list - NOTE lower score is better
initialTargetCost = -2

################################################### OBJECTIVE FUNCTION #################################################################

# The function to optimise in PSO - see the docs at https://pyswarms.readthedocs.io/en/latest/intro.html
# The first argument needs to an m x n array where m = number of particles and n = search dimension; here n is latent space dimension = 64
# Must return a vector of length m with obective function scores

def genSmiles(latVecArray, emb, dic, stateMod, decMod, structFlags):

  #return array of shape (nParticles, )
  out = np.zeros((len(latVecArray), ))

  #predict from the vector - states[0] and [1] are reset states for lstm, states[2] is classifier prediction
  for vec in range(len(latVecArray)):
  
    #get fragment prediction and states
    states = stateMod.predict(latVecArray[vec:vec+1])
    
    #fragment prediction - add to output - weight this
    out[vec] += 2 * states[2]

    asciiMat = mu.decodeMatrix(decodeModel=decMod, stateMatrix=states, embed=emb, fwdDict=dic)
    smiles = mu.smiles_from_mat(mat=asciiMat, dictMat=dictMat)

    # Replace the X -> Cl and Y -> Br dictionary encodings to give (hopefully) valid smiles
    sm1 = smiles.replace("X", "Cl")
    sm2 = sm1.replace("Y", "Br")

    #validity check - add to score if molecule valid then check QED and add to score
    if Chem.MolFromSmiles(sm2) != None:
      try:
        smi = Chem.CanonSmiles(sm2)
        mol = Chem.MolFromSmiles(smi)
        
        #add 3 points for a valid molecule
        out[vec] += 3

        #check for unwanted functionalities - remove 4 points if find any
        match = [mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in structFlags]
        if np.any(match): out[vec] += -4

        #synthetic accessibility score - this has value between 1 (best) and 10 (worst)
        #we remove SAS/10 points from each molecule
        out[vec] += -sascorer.calculateScore(mol)/10

        #small penalty for 7 and 8 member rings
        if(mol.HasSubstructMatch(Chem.MolFromSmarts('[r7]')) or mol.HasSubstructMatch(Chem.MolFromSmarts('[r8]'))):
          out[vec] += -1

        #small penalty for low or high fsp3
        if(rdMolDescriptors.CalcFractionCSP3(mol) < 0.2 or rdMolDescriptors.CalcFractionCSP3(mol) > 0.8):
          out[vec] += -1

        #penalty for low or high heavy atom count
        if mol.GetNumHeavyAtoms() < 8 or mol.GetNumHeavyAtoms() > 20:
          out[vec] += -2

      except:
        #if any part of above throws exception, set cost to -3
        out[vec] = -3

    elif Chem.MolFromSmiles(sm2) == None:
      out[vec] = -3
  
  #make output negative to minimise
  out = -out
  return out

##################################################################################################################

def PSO(file):
  #load the smiles and fingerprints files - the files list above will be passed to the Pool object in __main__
  csvFile = mu.find(file + ".csv", os.getcwd())
  npyFile = mu.find(file + "_fingerprint.npy", os.getcwd())
  mols = pd.read_csv(csvFile, header=0)
  fps = np.load(npyFile)
  
  #perform pso to generate 10 molecules from each seed
  #empty list to hold output
  resultList = []

  for seed in range(len(mols)):
    #vectorise expects a LIST - me must give it that or it will default to a tensor of the length of the smiles: weird redults will follow
    smi = [mols['smiles'].iloc[seed]]

    # one-hot encode the SEED smiles - note that we don't need y here - y is only used in training but we use the same vectorisation function
    x, y = mu.vectorise(smi, params["embed_length"], params["dictionary_length"], params["forward_dictionary"])

    #get the molecule fingerprint and reshape for encoder
    fp = fps[seed]
    fp = np.reshape(fp, (1,87))

    # convert to latent-space vectors
    Lat = encoder.predict([x[0:1], fp[0:1]])
    Lat = np.reshape(Lat, (64,))    
    #find the zeros of the latent vector
    zeroComps = []
    for i in range(len(Lat)):
      if Lat[i] == 0:
        zeroComps.append(i)

    # Initialize swarm
    options = params["swarm_config"]
    
    #set a target cost to move the swarm
    currBestCost = initialTargetCost
    
    #set the current smiles - substitute X->Cl, Y->Br
    currSmi1 = mu.smiles_substitute(smi[0], "forward")

    #generate 10 molecules from each seed - now have a moving boundary if promising molecules found
    for vec in range(sampleDepth):
      Lat = np.reshape(Lat, (64,))
    
      #track movement of swarm
      tracker = 0

      #set the maximum component size and zeros of the latent vector - the latent vector will be reset below if new molecules found
      x_max = Lat + 1
      x_max[zeroComps] = 0.00001
      
      #the latent space is output of a relu - minimum value is 0
      x_min = Lat - 1
      x_min[zeroComps] = 0

      #check for negativity
      negComps = []
      for i in range(len(x_min)):
        if x_min[i] < 0:
          x_min[i] = 0

      bounds = (x_min, x_max)

      #do PSO - need to call the optimiser each time or will get stuck
      optimizer = ps.single.GlobalBestPSO(n_particles=params["n_particles"], dimensions=64, options=options, bounds=bounds, center=Lat, bh_strategy="reflective")

      cost, pos = optimizer.optimize(genSmiles, params["n-iterations"], emb=params["embed_length"], dic=params["forward_dictionary"], stateMod=states_and_pred, decMod=decoder, structFlags=smarts)
      final = np.reshape(pos, (1,64))

      # Check if the cost is better than the target and move the swarm if it is
      if cost < currBestCost:
        currBestCost = cost
        Lat = final
        tracker = 1

      #get the SMILES for the final best position - states[0] and [1] are reset states for lstm, states[2] is classifier prediction
      states = states_and_pred.predict(final[0:1])
      asciiMat = mu.decodeMatrix(decodeModel=decoder, stateMatrix=states, embed=params["embed_length"], fwdDict=params["forward_dictionary"])
      smiles = mu.smiles_from_mat(mat=asciiMat, dictMat=dictMat)
      sm2 = mu.smiles_substitute(smiles, "forward")

      #write out smiles and fragment prediction
      # note that states[2] is an array: [[prediction]] - call states[2].sum() to get the value

      #only write out canonical smiles
      try:
        smi = Chem.CanonSmiles(sm2)
        s = sascorer.calculateScore(Chem.MolFromSmiles(smi))
        resultList.append([currSmi1, smi, cost, states[2].sum(), s])
        #check if the swarm has moved and reset the current generator if so
        if tracker == 1:
          currSmi1 = smi

      except:
        continue

  return resultList

#########################################################################################################################


# Do the PSO

if __name__ == "__main__":
  #perform pso with multithreading to generate 10 molecules from each seed

  #check the running time
  tic = time.perf_counter()
  
  #this is necessary on linux systems where the default method is "fork" which causes problems with CUDA
  multiprocessing.set_start_method('spawn', force=True)
  
  #setup the pool  
  with Pool(5) as p:
    finalOutput = p.map(PSO, files)
    p.close()
    p.join()

  flat_list = [item for sublist in finalOutput for item in sublist]
  flat = pd.DataFrame(flat_list, columns=["seed.smiles", "genMol.smiles", "objectiveScore", "fragmentScore", "SAS"])
  print(flat)
  print(len(flat))

  flat.to_csv("fragmentGenerator_output.csv", index=False)
  
  #check the running time
  toc = time.perf_counter()
  
  #save the running time
  total = toc - tic
  np.save("totalTime.npy", total)