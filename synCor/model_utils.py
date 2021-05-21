#general imports
import numpy as np
import pandas as pd
import os
import pickle

#tensorflow
import tensorflow
from tensorflow.keras.models import load_model, Model


############################### one-hot encode the smiles ###################################
def vectorise(smilesList, maxSmilesLength, dictLength, fwdDict):

  #create a one-hot encoded 3d array representing each smiles in the list
  #We thank Esben Jannik Bjerrum for source code available at:
  #https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras

  #generate a numpy zeros 3d array of the appropriate size for the input
  one_hot = np.zeros((len(smilesList), maxSmilesLength, dictLength), dtype="int8")

  #loop through each smiles in the input
  for i, smile in enumerate(smilesList):
  #place a 1 in position [i, 0, fwdDict["!"]] for the ith smiles
    one_hot[i, 0, fwdDict["!"]] = 1

    #For the rest of the string, place a 1 in position [j+1, fwdDict[char]] for each encoding
    #note that we need to offset j because it starts at zero and we want avoid overwriting the start character encoding just made
    for j, char in enumerate(smile):
      one_hot[i, j+1, fwdDict[char]]=1

    #now, encode the end character in all positions up to the embedding length
    one_hot[i, len(smile)+1:, fwdDict["E"]]=1

  #Now, return the training and target arrays
  #The target array is offset by 1 - start character encoding is therefore omitted
  return one_hot[:,0:-1,:], one_hot[:,1:,:]
####################################### END #####################################################



################################## n-sphere sampling ############################################
def sampleNsphere():

  #This is NOT a very good way to sample latent space!!!
  #However, we find it can be usefully employed to generate invalid and undesirable smiles
  #These can subsequently be used for syntax correction in training as outlined in the paper
  print("")
###################################### END #####################################################



################################## file finder #################################################
def find(name, path):
  #find files for loading in directory

  #search directiories
  for root, dirs, files in os.walk(path):
    if name in files:

      #return the path to the file
      return os.path.join(root, name)
###################################### END #####################################################



################################## load parameters #############################################
def load_params():
  # load configuration parameters - these include model dictionaries, SMILES embedding dimension, dictionary length, and PSO configuration parameters 

  #find the file and load
  with open(find("model_parameters.p", os.getcwd()), 'rb') as f:
    params = pickle.load(f)

  #return the parameters
  return params
###################################### END #####################################################




################################## load models #############################################
def load_dual_models():
  # load prebuilt models - states_and_pred returns both the classifier fragment prediction score and the LSTM cell and hidden states
  #find the files and load
  encoder = load_model(find("synCor_encoder_final.h5", os.getcwd()))
  decoder = load_model(find("synCor_smiles_decoder_final.h5", os.getcwd()))
  l2s = load_model(find("synCor_L2S_final.h5", os.getcwd()))
  classifier = load_model(find("synCor_classifier_final.h5", os.getcwd()))

  # bolt the classifier and latent to states models together
  latVec_in = tensorflow.keras.Input(shape = (64,))
  states = l2s(latVec_in)
  pred = classifier(latVec_in)
  states_and_pred = Model(latVec_in, [states, pred])

  #return the parameters
  return encoder, decoder, states_and_pred
###################################### END #####################################################




################################## generate dictionary matrix ##################################
def generate_decode_matrix(dict):
  # generates a diagonal matrix with the ascii encoding of the ith character in the decode (ie, REVERSE) dictionary in the [i,i] position
  # We perform matrix multiplication with this on the RNN output to decode
  # This gives slight speedup relative to growing string by repeated dictionary lookups
  dictMat = np.zeros((len(dict),len(dict)), dtype="int")
  for i in range(len(dict)):
    dictMat[i,i] = ord(dict[i])

  return dictMat
###################################### END #####################################################


######################### generate prediction matrix from LSTM outputs ########################
def decodeMatrix(decodeModel, stateMatrix, embed, fwdDict):
  #reset the LSTM cells with decoded c and h states - the output of dense layers downstream of latent embedding
  #decodeModel is the SMILES decoder model 
  #embed is the SMILES embedding dimension
  decodeModel.layers[1].reset_states(states=[stateMatrix[0],stateMatrix[1]])
  
  #the start character is !
  #give this to the decoder first
  currentVec = np.zeros((1,1,len(fwdDict)))
  predictMat = np.zeros((1,1,len(fwdDict)))
  currentVec[0,0,fwdDict["!"]] = 1

  #Predict next character - loop up to maximum iteration of embedding size
  for i in range(embed):
    predictMat = np.vstack([predictMat, currentVec])
    nextIdx = np.argmax(decodeModel.predict(currentVec))

    #reset the current vector
    currentVec = np.zeros((1,1,len(fwdDict)))
    currentVec[0,0,nextIdx] = 1

    #the padding character in training is "E" - terminate the loop if this is output
    if nextIdx == fwdDict["E"]:
      break

  #return the prediction matrix, excluding the initial vector of zeros
  return predictMat[1:,:,:]
###################################### END #####################################################



########### obtain SMILES from ascii matric returned by decodeMatrix() ########################
def smiles_from_mat(mat, dictMat):
  outSmi = np.squeeze(mat)
  outSmi = np.matmul(outSmi, dictMat)
  #check if only a single character has been generated - if so, axis=1 does not exist
  if outSmi.ndim < 2:
    outSmi = np.max(outSmi)
  else:
    outSmi = np.max(outSmi, axis=1)
  outSmi = outSmi.astype(int)
  outSmi = outSmi.tolist()
  outSmi = [chr(i) for i in outSmi]
  smi = ""
  for c in outSmi:
    smi += c
  smi = smi.strip("!")
  return smi
###################################### END #####################################################



######################## substitute SMILES characters Cl/X, Br/Y ###############################
def smiles_substitute(smiles, direction):
  #forward substitution is used to convert from the model vocabulary to SMILES
  if direction == "forward":
    sm1 = smiles.replace("X", "Cl")
    sm2 = sm1.replace("Y", "Br")

  #backsubstitution can be used if the user supplies a list of SMILES that are not already converted to model vocab
  elif direction == "backward":
    sm1 = smiles.replace("Cl", "X")
    sm2 = sm1.replace("Br", "Y")

  else:
    sm2 = smiles
    print("substitution direction was not supplied or was not valid")

  return sm2
###################################### END #####################################################