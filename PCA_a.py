import numpy as nm
import pandas as pd


'''PCA 操作流程：      去平均值，即每一位特征减去各自的平均值
    计算协方差矩阵
    计算协方差矩阵的特征值与特征向量
    对特征值从大到小排序
    保留最大的个特征向量
    将数据转换到个特征向量构建的新空间中
    
    选择主成分的个数
    '''


def zero_mean(datamat):                    # 计算均值并求协方差矩阵
    mean_val = nm.mean(datamat, axis=0)    # 按列求均值，求每一维的特征平均值   axis=0 对各列求均值， axis=1 对各行求均值
    new_data = datamat - mean_val
    cov_data = nm.cov(new_data, rowvar=0)  # rowvar= 0 传入数据一行代表一个样本  rowvar = 1 传入数据 一列是一个样本
    # print(cov_data)
    return cov_data, new_data, mean_val


def pca(cov_data_mat, new_data, mean_vals, n):
    eig_val, eig_vec = nm.linalg.eig(cov_data_mat)  # eig_val 特征值 行向量  eig_vect 特征向量 列向量
    eig_val_sort = nm.argsort(eig_val)      #argsort  从小到大的索引值
    n_eig_val = eig_val_sort[-1:-(n+1):-1]
    n_eig_vec = eig_vec[:, n_eig_val]
    low_datamat = nm.dot(new_data, n_eig_vec)
    recon_data = (low_datamat * nm.transpose(n_eig_vec))+mean_val

    return low_datamat, recon_data


if __name__ == '__main__':

   # row_mat = pd.read_csv('E:/L python/li-hang book/data/train_binary.csv', header= 0)
   # data_mat = row_mat.values

    # data_mats = data_mat[0:10, 490:500]
   data_mats =nm.array([[2.5,2.4],  [0.5,0.7],  [2.2,2.9],  [1.9,2.2],  [3.1,3.0],  [2.3,2.7],  [2.0,1.6],  [1.0,1.1],  [1.5,1.6],
               [1.1,0.9]])
   cov_data, new_data, mean_val = zero_mean(data_mats)
   low_data ,recon_data = pca(cov_data, new_data, mean_val, 1)
   print(low_data)
   print(recon_data)





