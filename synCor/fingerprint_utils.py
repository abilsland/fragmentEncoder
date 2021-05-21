#general imports
import numpy as np
import pandas as pd
import os
import pickle

#tensorflow
import tensorflow
from tensorflow.keras.models import load_model, Model

#rdkit and deepchem imports
import deepchem as dc
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate


################################## file finder #################################################
def find(name, path):
  #find files for loading in directory

  #search directiories
  for root, dirs, files in os.walk(path):
    if name in files:

      #return the path to the file
      return os.path.join(root, name)
###################################### END #####################################################


###################### generate ecfp4 and pharmacophore fingerprints #########################
#Convert smiles to ecfp4 and pharmacophore fingerprints on deepchem/rdkit
#Input file should be of smiles type with header "smiles"

def smiles_to_fingerprints(csvfile, writeFiles = True):

  #load molecules from csv
  mols = pd.read_csv(csvfile, header=0)
  smi = mols.smiles.values

  #create rdkit mol list
  rdkitmol = [Chem.MolFromSmiles(m) for m in smi]

  #check the molecule list and save any entries that failed for the user to inspect
  #it is better to know that some failed - the filterFails() utility below will get rid of them, however
  #so, in the case of fails just run again with the output file from that function
  try:
    if None in rdkitmol:
      raise ValueError('unparsable molecules')
  except (ValueError, IndexError):
    idx = [i for i, m in enumerate(rdkitmol) if m == None]
    errs = mols.iloc[idx]
    errs.to_csv("mols.err", index=False)
    exit("Some molecules could not not be parsed - saving to mols.err")

  #If all molecules pass, get the ecfp4 fingerprints with deepchem
  #Note that life would be much easier in some ways using rdkit Morgan fingerprints but the model was trained on deepchem ecfp4
  feat = dc.feat.CircularFingerprint(size=1024)
  fingerprint = feat.featurize(rdkitmol)
  fingerprint = np.array(fingerprint, dtype = "int8")

  #Now get the rdkit 2d pharmacophore fingerprints - we just use the public base features file of rdkit for feature definitions
  #Greg Landrum - https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef
  #Point to the base features  
  fdefName = find("BaseFeatures.fdef", os.getcwd())

  #These are used to build a feature factory as described in rdkit
  featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)

  #The mol list exists, so we just build a signature factory from the feature factory
  sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
  sigFactory.SetBins([(0,2),(2,5)])
  sigFactory.Init()
  sigFactory.GetSigSize()

  #Generate the fingerprint
  pharmacophores = [Generate.Gen2DFingerprint(m,sigFactory) for m in rdkitmol]
  pharmacophores = np.array(pharmacophores, dtype="int8")

  #Default is to write the files to the current directory unless writeFiles=False in the call
  #Turn off saving by setting writeFiles = False in the call
  if writeFiles:
    np.save("ecfp4.npy", fingerprint)
    np.save("pharmacophores.npy", pharmacophores)

  return fingerprint, pharmacophores
############################### END ########################################################



############ auto-filtering of the molecules if some failed and we don't care why ##########
def filterFails(csvfile):
  #load molecules from csv
  mols = pd.read_csv(csvfile, header=0)
  smi = mols.smiles.values

  #create rdkit mol list
  rdkitmol = [Chem.MolFromSmiles(m) for m in smi]

  #check the mol list and return indices of any entries that failed
  idx = [i for i, m in enumerate(rdkitmol) if m == None]

  #write the good molecules back to a file that can be called in smiles|_to_fingerprints()
  mols = mols.loc[~mols.index.isin(idx)]
  mols.to_csv("filtered_mols.csv", index=False)
  return mols
############################# END ###########################################################


################ exclude the low variance features from the paper ###########################
def varianceFilter(fingerprints, pharmacophores, writeFile = True):
  
  #Perform variance filtering of ecfp4 and pharmacophore fingerprints using previously selected features
  #The model was trained on a specific combined fingerprints feature set
  #These were features passing a low variance threshold in the privleged fragment classifier training set
  #Here we retain these features from molecules processed above

  #Load the ecfp preselected features data
  ecfpFeats = np.load(find('fragmentHits_ecfpVarianceFilter_retainedFeatures.npy', os.getcwd()))
  print('loaded ecfp features')

  #Slice out the features on axis 1
  ecfpSelected = fingerprints[:, ecfpFeats]

  #Load the pharmacophores preselected features data
  pharmFeats = np.load(find('fragmentHits_pharmacophoresVarianceFilter_retainedFeatures.npy', os.getcwd()))
  print('loaded pharmacophore features')

  #Slice out the features on axis 1
  pharmSelected = pharmacophores[:, pharmFeats]

  #Generate the combined fingerprint
  fp = np.hstack([ecfpSelected, pharmSelected])
  
  #just to make sure....
  fp = fp.astype("int8")
  print("final fingerprint shape - we should have dim 87 in axis 1:")
  print(fp.shape)

  if writeFile:
    #default behviour is save file for later use
    np.save('Full_fingerprint_varianceFiltered.npy', fp)

  return fp
############################# END ############################################################