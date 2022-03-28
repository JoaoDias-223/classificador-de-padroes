import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


noPCA_config = {
    "layers": (40, 40, 40),
    "iterations": 50000
}

PCA_config = {
    "layers": (30, 30, 30),
    "iterations": 50000,
    "components": 2
}

no_pca_layers = ''.join(map(str, map(lambda x: f'{x}', noPCA_config['layers'])))
pca_layers = ''.join(map(str, map(lambda x: f'{x}', PCA_config['layers'])))

data = load_iris()

features =data.data
target = data.target


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis')

Classificador = MLPClassifier(
    hidden_layer_sizes=noPCA_config["layers"],
    alpha=1,
    max_iter=noPCA_config["iterations"]
)
Classificador.fit(features,target)
predicao = Classificador.predict(features)

plt.subplot(2,2,3)
plt.scatter(features[:,0], features[:,1], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis',s=15)

pca = PCA(n_components=PCA_config['components'], whiten=True, svd_solver='randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_)*100))

plt.subplot(2,2,2)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis')

predicao = Classificador.predict(features)

plt.subplot(2,2,4)
plt.scatter(features[:,2], features[:,3], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(features[:,2], features[:,3], c=target,marker='o',cmap='viridis',s=15)
plt.savefig(f"no_pca/graphs/{no_pca_layers}l_{noPCA_config['iterations']}i")


ClassificadorPCA = MLPClassifier(
    hidden_layer_sizes=PCA_config["layers"],
    alpha=1,
    max_iter=PCA_config["iterations"]
)

ClassificadorPCA.fit(pca_features,target)
predicao = ClassificadorPCA.predict(pca_features)


plt.subplot(2,2,4)
plt.cla()
plt.scatter(pca_features[:,0], pca_features[:,1], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis',s=15)
plt.savefig(f"pca/graphs/{pca_layers}l_{PCA_config['iterations']}i_{PCA_config['components']}c")


plot_confusion_matrix(Classificador, features, target,include_values=True,display_labels=data.target_names)
plt.savefig(f"no_pca/confusion_matrices/{no_pca_layers}l_{noPCA_config['iterations']}i")

plot_confusion_matrix(ClassificadorPCA, pca_features, target,include_values=True,display_labels=data.target_names)
plt.savefig(f"pca/confusion_matrices/{pca_layers}l_{PCA_config['iterations']}i_{PCA_config['components']}c")
