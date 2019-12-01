import os
import json
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

site = 'UM'
num_asd = 43
num_region = 200
with open(os.getcwd() + '/{}_data.json'.format(site), 'r') as file:
    data = json.load(file)
    order = np.array(data['Order'])

with open(os.getcwd() + '/{}_latent.json'.format(site), 'r') as file:
    latent = np.array(json.load(file))      # [95, 200, 8]

asd = []
hc = []
for i in range(95):
    if int(order[i][-1][9:12]) <= 326:
        asd.append(latent[i])   # [43, 200, 8]
    else:
        hc.append(latent[i])
asd = np.swapaxes(np.array(asd), 0, 1)      # [200, 43, 8]
hc = np.swapaxes(np.array(hc), 0, 1)

print(order.shape)
print(latent.shape)

save_dir = os.getcwd() + '/results'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def save_vis(save_dir, r):
    print('Processing Region', r)
    X1 = asd[r, :]
    X2 = hc[r, :]
    X = np.concatenate((X1, X2))
    tsne = TSNE(n_components=2,perplexity= 100,init='pca')
    X_2d = tsne.fit_transform(X)
    X1_2d = X_2d[:num_asd, :]
    X2_2d = X_2d[num_asd:, :]
    plt.figure()
    plt.scatter(X1_2d[:, 0], X1_2d[:, 1], marker='x', c='r')
    plt.scatter(X2_2d[:, 0], X2_2d[:, 1], marker='x', c='g')
    plt.savefig(os.path.join(save_dir, 'region_' + str(r) + '_tsne.jpg'))
    print('Saving Region', r)
    plt.close()


save_vis(save_dir, 5)

for i in range(num_region):
    save_vis(save_dir, i)