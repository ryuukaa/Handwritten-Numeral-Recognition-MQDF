# -*- coding: utf-8 -*-
import struct
import numpy as np
import matplotlib.pyplot as plt

# 读取训练集图片数据集
with open('train-images-idx3-ubyte', 'rb') as trainimg:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', trainimg.read(16))
    images = np.fromfile(trainimg, dtype=np.uint8).reshape(images_num, rows * cols)

# 读取训练集标签数据集
with open('train-labels-idx1-ubyte', 'rb') as trainlab:
    labels_magic, labels_num = struct.unpack('>II', trainlab.read(8))
    labels = np.fromfile(trainlab, dtype=np.uint8)

# 读取测试集图片数据集
with open('t10k-images-idx3-ubyte', 'rb') as testimg:
    timg_magic, timg_num, t_rows, t_cols = struct.unpack('>IIII', testimg.read(16))
    t_images = np.fromfile(testimg, dtype=np.uint8).reshape(timg_num, rows * cols)
# 读取测试集标签数据集
with open('t10k-labels-idx1-ubyte', 'rb') as testlab:
    tlab_magic, tlab_num = struct.unpack('>II', testlab.read(8))
    t_labels = np.fromfile(testlab, dtype=np.uint8)

# 测试取出一张图片和对应标签
choose_num = 50 # 随机指定一个编号
label = labels[choose_num]
image = images[choose_num].reshape(28,28)
plt.imshow(image)
plt.title('the label is : {}'.format(label))
plt.show()

# 打印数据信息
print('labels_magic is {}'.format(labels_magic,),
      'labels_num is {} '.format(labels_num),
      'labels is {} '.format(labels))
print('images_magic is {} '.format(images_magic),
      'images_num is {} '.format(images_num),
      'rows is {} '.format(rows),
      'cols is {} '.format(cols))
print('(images is {} )'.format(images))

def mqdf(x_j, e_value_i, e_vector_i, u_i, k):
    delta = np.sum(e_value_i[k : 784]) / (784 - k)#计算δ，δ可取k或k+1
    epsilon = (np.sum((x_j - u_i) ** 2) - np.sum(np.dot((x_j - u_i), np.array(e_vector_i[:, 0:k])) ** 2))#计算ε
    a = np.sum((np.dot((x_j - u_i), np.array(e_vector_i[:, 0:k])) ** 2) / e_value_i[0:k])
    b = epsilon / delta
    c = np.sum(np.log(e_value_i[0:k].real))
    d = (784 - k) * np.log(delta)
    g2 = - a - b - c - d
    return g2

if __name__ == '__main__':
    train_label = np.ones([len(labels),1])
    for i in range(len(labels)):
        train_label[i] = labels[i]
    train_image = np.ones([len(images), 784])
    for i in range(len(images)):
        train_image[i] = images[i]
    test_lab = np.ones([len(t_labels), 1])
    for i in range(len(t_labels)):
        test_lab[i] = t_labels[i]
    test_lab = test_lab[0:10000, 0].reshape(10000, 1)
    train_img = train_image[0:55000, :] #55000*784
    trimg_valid = train_image[55000:60000, :] #5000*784
    train_lab = train_label[0:55000, :].reshape(55000, 1)
    trlab_valid = train_label[55000:60000, :].reshape(5000, 1)
    tra_imglab = np.hstack((train_img, train_lab)) #55000*785
    # MNIST数据集的标签是介于0到9的数字
    #计算协方差、特征值、特征向量
    x = [[], [], [], [], [], [], [], [], [], []]
    cov = [[], [], [], [], [], [], [], [], [], []]
    e_value = [[], [], [], [], [], [], [], [], [], []]
    e_vector = [[], [], [], [], [], [], [], [], [], []]
    u = [[], [], [], [], [], [], [], [], [], []]#每一样本类的样本均值
    for k in range(55000):
        for i in range(10):
            if tra_imglab[k, 784] == i:#如果标签匹配
                x[i].append(tra_imglab[k, 0:784])#将55000个数据中0-9的数字分别存入x的列表
    for i in range(10):
        cov[i] = np.cov(np.mat(x[i]).T) / 784 #mat将列表转为矩阵计算0-9每个样本协方差，默认行为变量计算方式
        e_value[i], e_vector[i] = np.linalg.eig(cov[i])#计算矩阵特征向量
        u[i] = np.mean(x[i], axis=0)#求取均值
                                        # axis不设置值，对 m*n 个数求均值;
                                        # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
                                        # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    print('this is validation:')
    k_list = []
    k = 15#取超参数为25，当k=15-30时，找到最优k
    for n in range(15):
        k += 1
        k_list.append(k)
    overall_accuracy = []
    for k in (k_list):#在取得k值范围内
        prediction = []#存储预测值
        for j in range(len(trimg_valid)):
            x_j = trimg_valid[j]
            P = []
            #分别计算每个样本的g
            for i in range(10):
                e_value_i = e_value[i].real
                e_vector_i = e_vector[i].real
                u_i = u[i]
                g = mqdf(x_j, e_value_i, e_vector_i, u_i, k)
                P.append(g)
            likely_class = P.index(max(P))
            prediction.append(likely_class)#最大值下标
        count = 0
        prediction = np.array(prediction)
        prediction = prediction.reshape(5000, 1)
        #进行验证
        for i in range(0, 5000):
            if prediction[i, 0] == trlab_valid[i, 0]:
                count += 1
        accuracy = 100 * float(count / 5000.0)
        overall_accuracy.append(accuracy)
        print("k = %.2f, validation accuracy rate = %.2f %%" % (k, accuracy))
    best_k = k_list[overall_accuracy.index(max(overall_accuracy))]
    print('******  best k = %.2f  ******' %(best_k))
    plt.plot(k_list, overall_accuracy, color="r", marker = "o", markersize=8)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.show()
    #测试集进行测试
    print('this is test:')
    prediction = []
    k=best_k
    for j in range(len(t_images)):
        x_j = t_images[j]
        P = []
        for i in range(10):
            e_value_i = e_value[i].real
            e_vector_i = e_vector[i].real
            u_i = u[i]
            g = mqdf(x_j, e_value_i, e_vector_i, u_i, k)
            P.append(g)
        likely_class = P.index(max(P))
        prediction.append(likely_class)
    count = 0
    prediction = np.array(prediction)
    prediction = prediction.reshape(10000, 1)
    for i in range(0, 10000):
        if prediction[i, 0] == test_lab[i, 0]:#预测值如果等于测试集数字则继续
            count += 1
    accuracy = 100 * (count / 10000.0)
    print("k = %.2f, accuracy rate = %.2f %%" % (k, accuracy))


