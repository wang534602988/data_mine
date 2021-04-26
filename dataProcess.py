import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

continues = [0, 2, 4, 10, 11, 12]  # 记录数值型数据的维度
categories = [1, 3, 5, 6, 7, 8, 9]  # 记录类别型数的维度


# 类别数据转数值型
def cate_encode(arrays):
    enc = preprocessing.OrdinalEncoder()
    if len(arrays) == 1:
        result = enc.fit_transform(arrays.T)
    else:
        result = enc.fit_transform(arrays)
    return result


# 缺失值处理
def imputation(arrays, missing, strategy):
    arrays[arrays == missing] = np.nan  # 将？转化为nan
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    if len(arrays.shape) == 1:
        result = imp.fit_transform(arrays.reshape(1, -1))
    else:
        result = imp.fit_transform(arrays)
    return result


# 正则化
def normal(arrays):
    X_normalized = preprocessing.StandardScaler()
    return X_normalized.fit_transform(arrays)


# 数值型数据批量处理
def continues_process(arrays):
    arrays = imputation(arrays, "?", 'mean')
    result = normal(arrays)
    return result


# 类别型数据批量处理：
def category_process(arrays):
    arrays = imputation(arrays, '?', "most_frequent")
    result = cate_encode(arrays)
    return result


# 加载数据
def load_data(filename):
    data = pd.read_csv(filename).values
    x = data[:, 0:-2]
    y = data[:, -1]
    return x, y


# 读取文件输出array数组
def out_data(filename):
    x, y = load_data(filename)
    cont = continues_process(x[:, continues])
    cate = category_process(x[:, categories])
    label = category_process(y)
    new_x = np.hstack((cate, cont))
    return new_x, label


# 写入数据
def write_data(filename, matrix):
    result = ''
    try:
        np.savetxt(filename, matrix, fmt='%f', delimiter=',')
        result = filename
    except:
        print('文件写入出错')
    return result

# 整体写入    
def write(filename, x_outname, y_outname):
    x, y = load_data(filename)
    cont = continues_process(x[:, continues])  # 取出数值型数据所在列
    cate = category_process(x[:, categories])  # 取出类别型数据所在列
    label = category_process(y)  # 去除标签列
    new_x = np.hstack((cate, cont))  # 处理后的数值型数据与类别型数据进行水平合并
    write_data(x_outname, new_x)  # 将新数据写入文件
    write_data(y_outname, label)
    return new_x, label
