import scipy.io
import pandas as pd

# 读取.mat文件
mat = scipy.io.loadmat('Data.mat')

# 打印.mat文件中的所有变量名称
print(mat.keys())

# 遍历.mat文件中的所有变量
for var_name in mat:
    # 跳过特殊变量
    if var_name.startswith('__'):
        continue
    
    # 获取变量数据
    data = mat[var_name]
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 将DataFrame写入.csv文件，文件名为变量名
    df.to_csv(f'{var_name}.csv', index=False)