import os
import json
import numpy as np
import matplotlib.pyplot as plt

fold = os.getcwd()
site = 'UM'

with open(fold + '/{}_save.json'.format(site), 'r') as file:
    # latent = np.array(json.load(file))
    tmp = json.load(file)
    target = np.array(tmp['target'])
    output = np.array(tmp['output'])

plt.figure()
for i in range(target.shape[0]):
    plt.plot(target[i, 2, :])
plt.show()

for i in range(20):
    plt.figure()
    plt.plot(target[i, 2, :], 'r.-', linewidth=1.5, label='target')
    plt.plot(output[i, 2, :], 'b.-', linewidth=1.5, label='output')
    plt.show()

# index = [0, 1, 2, 3, 4, 91, 92, 93, 94]
# # plt.figure()
# # for i in index:
# #     plt.plot(latent[i, 0, 0, :])
# # plt.show()
# plt.figure()
# for i in range(1):
#     plt.plot(latent[:, 0, 0, i])
# plt.show()
#
# for i in range(8):
#     plt.figure()
#     plt.hist(np.vstack((latent[:10,:],latent[-6:,:])).reshape(-1,8)[:,i],bins =50)
#     plt.figure()
#     plt.hist(latent[10:89, :].reshape(-1, 8)[:, i], bins=50)
#     plt.show()

