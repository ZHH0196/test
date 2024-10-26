import scipy.io 
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

# 将数据导入
def ImportData():
    X99_normal = scipy.io.loadmat('99.mat')['X099_DE_time']              
    X108_InnerRace_007 = scipy.io.loadmat('108.mat')['X108_DE_time']
    X121_Ball_007 = scipy.io.loadmat('121.mat')['X121_DE_time']
    X133_Outer_007 = scipy.io.loadmat('133.mat')['X133_DE_time']
    X172_InnerRace_014 = scipy.io.loadmat('172.mat')['X172_DE_time']
    X188_Ball_014 = scipy.io.loadmat('188.mat')['X188_DE_time']
    X200_Outer_014 = scipy.io.loadmat('200.mat')['X200_DE_time']
    X212_InnerRace_021 = scipy.io.loadmat('212.mat')['X212_DE_time']
    X225_Ball_021 = scipy.io.loadmat('225.mat')['X225_DE_time']
    X237_Outer_021 = scipy.io.loadmat('237.mat')['X237_DE_time']
    return [X99_normal,X108_InnerRace_007,X121_Ball_007,X133_Outer_007, X172_InnerRace_014,X188_Ball_014,X200_Outer_014,X212_InnerRace_021,X225_Ball_021,X237_Outer_021]

# 采样
def Sampling(Data, interval_length, samples_per_block):
    #根据区间长度计算采样块数
    No_of_blocks = (round(len(Data)/interval_length) - round(samples_per_block/interval_length)-1)
    SplitData = np.zeros([No_of_blocks, samples_per_block])
    for i in range(No_of_blocks):
        SplitData[i,:] = (Data[i*interval_length:(i*interval_length)+samples_per_block]).T
    return SplitData


def DataPreparation(Data, interval_length, samples_per_block):
    for count,i in enumerate(Data):
        SplitData = Sampling(i, interval_length, samples_per_block)
        y = np.zeros([len(SplitData),10])
        y[:,count] = 1
        y1 = np.zeros([len(SplitData),1])
        y1[:,0] = count
        # 堆叠并标记数据
        if count==0:
            X = SplitData
            LabelPositional = y
            Label = y1
        else:
            X = np.append(X, SplitData, axis=0)
            LabelPositional = np.append(LabelPositional,y,axis=0)
            Label = np.append(Label,y1,axis=0)
    return X, LabelPositional, Label
Data = ImportData()
interval_length = 200  #信号间隔长度
samples_per_block = 1681  #每块样本点数

#数据前处理
X, Y_CNN, Y = DataPreparation(Data, interval_length, samples_per_block) 

print('Shape of Input Data =', X.shape)
print('Shape of Label Y_CNN =', Y_CNN.shape)
print('Shape of Label Y =', Y.shape)

XX = {'X':X}
scipy.io.savemat('Data.mat', XX)

kSplits = 5
kfold = KFold(n_splits=kSplits, random_state=32, shuffle=True)


# 一维卷积神经网络1D-CNN分类
# reshape数据
Input_1D = X.reshape([-1,1681,1])

# 数据集划分
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, train_size=0.75,test_size=0.25, random_state=101)


# 定义1D-CNN模型
class CNN_1D():
    def __init__(self):
        self.model = self.CreateModel()

    def CreateModel(self):
        model = models.Sequential([
            layers.Conv1D(filters=16, kernel_size=3, strides=2, activation='relu', input_shape=(1681, 1)),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(10),
            layers.Softmax()
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model
accuracy_1D = []

# 训练模型
for train, test in kfold.split(X_1D_train,y_1D_train):
  Classification_1D = CNN_1D()
  history = Classification_1D.model.fit(X_1D_train[train], y_1D_train[train], verbose=1, epochs=12)


# 评估模型在训练集上的准确性
kf_loss, kf_accuracy = Classification_1D.model.evaluate(X_1D_train[test], y_1D_train[test]) 
accuracy_1D.append(kf_accuracy)
CNN_1D_train_accuracy = np.average(accuracy_1D)*100
print('CNN 1D train accuracy =', CNN_1D_train_accuracy)

# 评估模型在测试集上的准确性
CNN_1D_test_loss, CNN_1D_test_accuracy = Classification_1D.model.evaluate(X_1D_test, y_1D_test)
CNN_1D_test_accuracy*=100
print('CNN 1D test accuracy =', CNN_1D_test_accuracy)

# 定义混淆矩阵
def ConfusionMatrix(Model, X, y):
  y_pred = np.argmax(Model.model.predict(X), axis=1)
  ConfusionMat = confusion_matrix(np.argmax(y, axis=1), y_pred)
  return ConfusionMat

# 绘制1D-CNN的结果
plt.figure(1)
plt.title('Confusion Matrix - CNN 1D Train') 
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_train, y_1D_train) , annot=True, fmt='d',annot_kws={"fontsize":8},cmap="YlGnBu")
plt.show()

plt.figure(2)
plt.title('Confusion Matrix - CNN 1D Test') 
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_test, y_1D_test) , annot=True, fmt='d',annot_kws={"fontsize":8},cmap="YlGnBu")
plt.show()

plt.figure(3)
plt.title('Train - Accuracy - CNN 1D')
plt.bar(np.arange(1,kSplits+1),[i*100 for i in accuracy_1D])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.ylim([70,100])
plt.show()

plt.figure(4)
plt.title('Train vs Test Accuracy - CNN 1D')
plt.bar([1,2],[CNN_1D_train_accuracy,CNN_1D_test_accuracy])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.xticks([1,2],['Train', 'Test'])
plt.ylim([70,100])
plt.show()

# 保存模型
Classification_1D.model.save('cnn_1d_model.h5')
print("模型已保存为 cnn_1d_model.h5")