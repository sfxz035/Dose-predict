import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
# path = './data/data.0.1.mat'
# a = sio.loadmat(path)
# RTDose = a.get('RTDose')
# RTSt = a.get('RTSt')
# plt.imshow(RTDose)
# plt.show()
# plt.imshow(RTSt)
# plt.show()
path = './data/data.0.13.npz'
b = np.load(path)
RTDose = b['RTDose']
RTSt = b['RTSt']
plt.imshow(RTDose)
plt.show()
plt.imshow(RTSt[:,:,0])
plt.show()
c = 1