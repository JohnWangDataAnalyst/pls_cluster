# -*- coding: utf-8 -*-

"""
Authors: Zheng Wang

input: small dataset with multiple features (coloumn)
outputs: num cluster labels with p value
"""


import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
class PLS_cluster:

    def __init__(self, data):
        self.data = data
    

    def get_cluster_num(self, threshold=1.5):
        u,s,v = np.linalg.svd(self.data, full_matrices=False)
        s.sort()
        s=s[::-1]
        s_n = 100*(s-s.min())/(s.max()-s.min())
        #print(s_n)
        rank = np.log(-s_n[1:]+s_n[:-1] )
        #print(rank)
        print(rank >= threshold)

        self.cluster_n = (rank >= threshold).sum()+1
        self.X = self.data.dot(v.T[:,:self.cluster_n])
        

    def get_labels(self):
        kmeans = KMeans(n_clusters=self.cluster_n, random_state=0).fit(self.X)
        self.labels = kmeans.labels_
        data_new = []
        cluster_info = []
        for i in range(self.cluster_n):
            data_new.append(self.data[np.where(kmeans.labels_ == i)])
            cluster_info.append(self.data[np.where(kmeans.labels_ == i)].shape[0])
        self.data_new = np.concatenate(data_new, axis = 0)
        self.cluster_info = np.array(cluster_info)

    def validation(self, perm_n =500):
        cluster_mean = []
        for i in range(self.cluster_n):
            if i == 0:
                cluster_mean.append(self.data_new[:self.cluster_info[0]].mean(0))
            else:
                cluster_mean.append(self.data_new[self.cluster_info[:i].sum():self.cluster_info[:i].sum()+self.cluster_info[i]].mean(0))
        #print(cluster_mean)
        R = np.array(cluster_mean).dot(np.array(cluster_mean).T)
     
        d,u =np.linalg.eigh(R)
        d.sort()
        d=d[::-1]
        eg= -d[self.cluster_n-1]+ d[self.cluster_n-2]
        print(eg)
        egs_perm = []
        for i in range(perm_n):
            ind = np.arange(self.data.shape[0])
            np.random.shuffle(ind)
            data_new = self.data[ind]
            #print(data_new.shape)
            cluster_mean = []
            for i in range(self.cluster_n):
                if i == 0:
                    cluster_mean.append(data_new[:self.cluster_info[0]].mean(0))
                else:
                    cluster_mean.append(data_new[self.cluster_info[:i].sum():self.cluster_info[:i].sum()+self.cluster_info[i]].mean(0))
            #print(cluster_mean)
            R = np.array(cluster_mean).dot(np.array(cluster_mean).T)

            d,u =np.linalg.eigh(R)
            d.sort()
            d=d[::-1]
        
            egs_perm.append(-d[self.cluster_n-1]+d[self.cluster_n-2])
        z = (eg - np.array(egs_perm).mean())/np.array(egs_perm).std()
        print(z)
        self.egs_perm = np.array(egs_perm)
        self.p_value = norm.sf(z)
        self.egs = np.array(egs_perm)