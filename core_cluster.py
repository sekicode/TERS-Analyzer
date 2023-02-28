# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MyCluster:
    def __init__(self) -> None:
        pass

    def do_pca(self, data, n_components=3):
        data = np.array(data)
        df = pd.DataFrame(data[:, :, 1])

        X = df.values
        X = StandardScaler().fit_transform(X)

        if n_components == 0:
            return copy.deepcopy(X)

        n = min([int(n_components), len(data)])

        pca = PCA(n_components=n)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
            ['PC{}'.format(i) for i in range(1, n+1)]])

        print(pca.explained_variance_ratio_)
        self.pca_variance_ratio_ = pca.explained_variance_ratio_
        self.pca = pca
        self.principalDf = principalDf

        return principalDf

    def kmeans(self, data, n_clusters=3):
        # if n_components:
        #     finalDf = self.do_pca(n_components)
        # else:
        #     data = np.array(self.data)
        #     finalDf = pd.DataFrame(data[:, :, 1])

        # finalDf = self._handle_non_numerical_data(finalDf)

        finalDf = self._handle_non_numerical_data(data)
        xx = np.array(finalDf.astype(float))
        xx = preprocessing.scale(xx)

        clf = KMeans(n_clusters=n_clusters)
        clf.fit(xx)

        prediction = np.zeros(shape=len(xx))
        for i in range(len(xx)):
            predict_me = np.array(xx[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction[i] = clf.predict(predict_me)

        self.prediction = prediction

        return self.prediction

    def dtw_clustering(self, data):
        data = np.array(data)
        xx = np.array(data[:, :, 1].astype(float))
        series = preprocessing.scale(xx)

        from dtaidistance import dtw
        from dtaidistance import clustering
        # ds = dtw.distance_matrix_fast(series)

        # Custom Hierarchical clustering
        model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
        cluster_idx1 = model1.fit(series)
        # Keep track of full tree by using the HierarchicalTree wrapper class
        model2 = clustering.HierarchicalTree(model1)
        cluster_idx2 = model2.fit(series)
        # You can also pass keyword arguments identical to instantiate a Hierarchical object
        model2 = clustering.HierarchicalTree(
            dists_fun=dtw.distance_matrix_fast, dists_options={})
        cluster_idx3 = model2.fit(series)
        # SciPy linkage clustering
        model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        cluster_idx4 = model3.fit(series)
        model2.plot("2.png")

    def _handle_non_numerical_data(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        columns = df.columns.values
        for column in columns:
            text_digit_vals = {}

            def convert_to_int(val):
                return text_digit_vals[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x += 1

                df[column] = list(map(convert_to_int, df[column]))

        return df

    def test(self, df_PCA):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1', fontsize=15)
        ax.set_ylabel('PC2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        colors = ['r', 'g', 'b']

        for color in zip(colors):
            ax.scatter(df_PCA.loc[:, 'PC1'],
                       df_PCA.loc[:, 'PC2'],
                       c=color,
                       s=50)

        ax.grid()

    def test2(self, data, pca):
        print(pca.components_)

        fig, ax = plt.subplots()
        for i, v in enumerate(pca.components_):
            X = data[0, :, 0]
            l = plt.plot(X, v)
            c = l[0].get_color()
            plt.text(X[int(len(X)*0.9)],
                     v[int(len(X)*0.9)],
                     "PC%i" % (i + 1),
                     color=c)
