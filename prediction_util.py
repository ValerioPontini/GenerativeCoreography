from absl import app
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
import pickle
from smplx import SMPL
from DataAPI.aist_plusplus.features.manual import extract_all_frame_manual_features
import torch


def main(_):
	#Leggi la predizione dal csv
	df = pd.read_csv("predictedValues.csv", header=None) 

	#Converti in una matrice 
	data = df.to_numpy() #(N, 225)

	#Elimina il padding di 6 
	data = data[:, 6:]

	#salvo la matrice trans
	trans = data[:, :3]
	#tolgo trans dai motion inputs
	data = data[:, 3:]

	#elimino la rotazione applicata all'inizio
	data = R.from_matrix(data.reshape(-1, 3, 3)).as_rotvec().reshape(data.shape[0], -1)
	print(data.shape)

	
	

	smpl = SMPL(model_path="/Volumes/GENCOREO/mint/smpl", gender='MALE', batch_size=1)


	keypoints3d = smpl.forward(
      global_orient=torch.from_numpy(data[:, :3]).float(),
      body_pose=torch.from_numpy(data[:, 3:]).float(),
      transl=torch.from_numpy(trans).float(),
      )

	joints = keypoints3d.joints.detach().numpy()
	
	joints = joints[:20, :]

	vettore= joints[0][0]
	counter = 0
	rot = []
	for frame in joints:
		for riga in frame:
			angolo = R.from_rotvec(riga, degrees=True).magnitude()
			rot.append(angolo)
	print(rot)
	joints = np.insert(joints, 3, 0, axis=2)
	for frame in joints:
		for riga in frame:
			riga[3]=rot.pop(0)
		np.savetxt(f"./predictions/firstAttempt{counter}.csv", frame, delimiter=',')
		counter += 1
	
if __name__ == '__main__':
  app.run(main)
