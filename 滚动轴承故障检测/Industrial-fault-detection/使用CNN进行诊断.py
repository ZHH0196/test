import scipy.io 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from keras import models
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 加载模型
loaded_model = models.load_model('cnn_1d_model.h5')
print("模型已加载")

# 导入数据
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
    return [X99_normal, X108_InnerRace_007, X121_Ball_007, X133_Outer_007, X172_InnerRace_014, X188_Ball_014, X200_Outer_014, X212_InnerRace_021, X225_Ball_021, X237_Outer_021]

# 获取数据
data = ImportData()

# 故障类型映射字典
fault_type_mapping = {
    0: '正常状态',
    1: '内圈故障 (0.007英寸)',
    2: '滚动体故障 (0.007英寸)',
    3: '外圈故障 (0.007英寸)',
    4: '内圈故障 (0.014英寸)',
    5: '滚动体故障 (0.014英寸)',
    6: '外圈故障 (0.014英寸)',
    7: '内圈故障 (0.021英寸)',
    8: '滚动体故障 (0.021英寸)',
    9: '外圈故障 (0.021英寸)'
}
# 绘制故障检测并标记故障位置的时序图
def PlotFaultWithDetection(data, y_pred, start=0, end=500):
    plt.style.use('dark_background')      
    for i, signal in enumerate(data):
        plt.figure(figsize=(12, 6))
        
        # 如果 end 为 None，则绘制到信号的最后一个数据点
        if end is None:
            end = len(signal)
        
        # 绘制给定范围内的数据点
        plt.plot(signal[start:end], label=f'Class {i}')
        
        # 打印并标记故障类别
        fault_class = fault_type_mapping[y_pred[i]]
        print(f"Class {i}: 预测故障类别为 {fault_class}")
 
        # 在图上标记故障
        plt.title(f'Class {i} - 二维时间序列图 - 预测故障类别: {fault_class}', fontsize=14, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('振动信号', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 保存图形
        # plt.savefig(f'TimeSeries_Class_{i}_Fault.png', dpi=300)
        plt.show()

# 假设 y_pred 是模型预测的结果，这里我们使用模型进行预测
# 数据前处理
def DataPreparation(data, interval_length, samples_per_block):
    for count, i in enumerate(data):
        SplitData = Sampling(i, interval_length, samples_per_block)
        y = np.zeros([len(SplitData), 10])
        y[:, count] = 1
        y1 = np.zeros([len(SplitData), 1])
        y1[:, 0] = count
        # 堆叠并标记数据
        if count == 0:
            X = SplitData
            LabelPositional = y
            Label = y1
        else:
            X = np.append(X, SplitData, axis=0)
            LabelPositional = np.append(LabelPositional, y, axis=0)
            Label = np.append(Label, y1, axis=0)
    return X, LabelPositional, Label

# 采样
def Sampling(data, interval_length, samples_per_block):
    No_of_blocks = (round(len(data) / interval_length) - round(samples_per_block / interval_length) - 1)
    SplitData = np.zeros([No_of_blocks, samples_per_block])
    for i in range(No_of_blocks):
        SplitData[i, :] = (data[i * interval_length:(i * interval_length) + samples_per_block]).T
    return SplitData

interval_length = 200  # 信号间隔长度
samples_per_block = 1681  # 每块样本点数

# 数据前处理
X, Y_CNN, Y = DataPreparation(data, interval_length, samples_per_block) 

# 重塑数据
Input_1D = X.reshape([-1, 1681, 1])

# 数据集划分
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, train_size=0.01, test_size=0.99, random_state=101)

# 使用加载的模型进行评估
loaded_model_test_loss, loaded_model_test_accuracy = loaded_model.evaluate(X_1D_test, y_1D_test)
loaded_model_test_accuracy *= 100
print('加载模型的测试准确率 =', loaded_model_test_accuracy)

# 使用加载的模型进行预测
y_pred = np.argmax(loaded_model.predict(X_1D_test), axis=1)

# 调用时序图绘制，并标记故障信息
PlotFaultWithDetection(data, y_pred, start=0)  # 例如，绘制前1000个数据点并标记故障

# 定义混淆矩阵
def ConfusionMatrix(Model, X, y):
    y_pred = np.argmax(Model.predict(X), axis=1)
    ConfusionMat = confusion_matrix(np.argmax(y, axis=1), y_pred)
    return ConfusionMat

# 绘制故障分类结果图
def PlotFaultClassification(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred, palette="coolwarm",legend=False)
    plt.title('故障分类结果', fontsize=14, fontweight='bold')
    plt.xlabel('预测故障类别', fontsize=12)
    plt.ylabel('组', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.style.use('dark_background')
    # plt.savefig('Fault_Classification_Results.png', dpi=300)
    plt.show()

# 绘制故障检测效果图
def PlotDetectionAccuracy(train_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.bar(['训练精度', '测试精度'], [train_acc, test_acc], color=['#1f77b4', '#ff7f0e'])
    plt.title('检测精度 - 训练 vs 测试', fontsize=14, fontweight='bold')
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.style.use('dark_background')
    # plt.savefig('Detection_Accuracy.png', dpi=300)
    plt.show()

# 绘制故障定位图
def PlotFaultLocation(y_pred, X_test):
    plt.figure(figsize=(10, 6))
    for i in range(10):  # 假设有10种故障类别
        fault_indices = np.where(y_pred == i)
        if len(fault_indices[0]) > 0:
            plt.plot(X_test[fault_indices[0][0], :, 0], label=f'Fault Class {i}')
    plt.title('故障定位图', fontsize=14, fontweight='bold')
    plt.xlabel('时间序列', fontsize=12)
    plt.ylabel('振动信号', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.style.use('dark_background')
    # plt.savefig('Fault_Location.png', dpi=300)
    plt.show()

# 混淆矩阵可视化
plt.figure(1)
plt.title('混淆矩阵 - CNN 1D 测试') 
sns.heatmap(ConfusionMatrix(loaded_model, X_1D_test, y_1D_test), annot=True, fmt='d', annot_kws={"fontsize": 8}, cmap="YlGnBu")
plt.style.use('dark_background')
# plt.savefig('CNN_1D_Confusion_Matrix.png', dpi=300)
plt.show()

# 调用故障分类结果图
PlotFaultClassification(y_1D_test, y_pred)

# 调用故障检测效果图 
train_acc = 99.23
test_acc = loaded_model_test_accuracy
PlotDetectionAccuracy(train_acc, test_acc)

# 调用故障定位图
PlotFaultLocation(y_pred, X_1D_test)


# 绘制时域图、频域图、功率谱图和指标参数表
def PlotSignalAnalysis(data, start=0, end=1000, sampling_rate=12000):
    for i, signal in enumerate(data):
        plt.figure(figsize=(16, 12))
        
        # 时域图
        plt.subplot(3, 2, 1)
        plt.plot(signal[start:end], label=f'Class {i}')
        plt.title(f'Class {i} - 时域图', fontsize=14, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('振动信号', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 频域图（FFT频谱）
        N = end - start  # 信号长度
        yf = fft(signal[start:end])
        xf = fftfreq(N, 1 / sampling_rate)[:N//2]
        
        plt.subplot(3, 2, 3)
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title(f'Class {i} - 频域图 (FFT)', fontsize=14, fontweight='bold')
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 滚动体故障的功率谱图
        # 修正信号为一维数组，确保 welch 函数的输入是正确的
        signal_segment = signal[start:end].flatten()  # 确保信号为一维
        freqs, psd = welch(signal_segment, fs=sampling_rate, nperseg=1000)
        
        plt.subplot(3, 2, 5)
        plt.semilogy(freqs, psd)  # 确保 freq 和 psd 的维度一致
        plt.title(f'Class {i} - 功率谱图 (PSD)', fontsize=14, fontweight='bold')
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('功率谱密度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 保存和显示图像
        plt.tight_layout()
        # plt.savefig(f'Signal_Analysis_Class_{i}.png', dpi=300)
        plt.show()

# 调用信号分析绘制函数
PlotSignalAnalysis(data, start=0, end=1000)