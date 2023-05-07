import numpy as np
import pandas as pd

filename=input("Enter filename (with .csv):")
ens = pd.read_csv(filename,header=None)

acc = 0

for i in range(ens.shape[0]):
	sump = 0
	sumn = 0

	if ens.iat[i,1]:
		sump += ens.iat[i,2]
	else:
		sumn += ens.iat[i,4]
	
	if ens.iat[i,6]:
		sump += ens.iat[i,7]
	else:
		sumn += ens.iat[i,9]

	if ens.iat[i,11]:
		sump += ens.iat[i,12]
	else:
		sumn += ens.iat[i,14]

	if ens.iat[i,16]:
		sump += ens.iat[i,17]
	else:
		sumn += ens.iat[i,19]

	if ens.iat[i,21]:
		sump += ens.iat[i,22]
	else:
		sumn += ens.iat[i,24]

	if sump > sumn:
		y_pred = 1
	else:
		y_pred = 0

	if ens.iat[i,0] == y_pred:
		acc += 1

print(f"Accuracy: {acc/ens.shape[0]}")

feats = ens[[0,1,6,11,16,21]]

ofilename=input("Enter output filename (with .csv):")
feats.to_csv(ofilename, header=False, index=False)
