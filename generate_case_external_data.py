# generate extenal_input.npy of [num, features]
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from config import time_delta, time_period
original_data_path = './External Features/数据/'
case_data_path = './case_external_data/'

def generate_case_external_data():
    files = os.listdir(original_data_path)
    for file in tqdm(files):
        print('Processing file: ', file)
              
        file_path = original_data_path + file
        df = pd.read_excel(file_path)

        # # 清除第二到七列的数据
        # df = df.drop(df.columns[[1,2,3,4,5,6]], axis=1)

        # 对异常值999.99, 99999.9, 9999999, 99.99做Nan值填充处理
        df = df.replace(999.99, np.nan)
        df = df.replace(99999.9, np.nan)
        df = df.replace(9999999, np.nan)
        df = df.replace(99.99, np.nan)

        # 对缺失值进行处理（前向填充，后向填充）
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        # # 将df保存为csv文件
        # df.to_csv('./external_input.csv', index=False)

        # 将df数据转换为numpy数组
        external_lines = df.to_numpy()

        # 读取case1的时间
        case_num = file.split('.')[1]
        case_name = 'case' + case_num
        starttime = time_period[case_name]['starttime']
        endtime = time_period[case_name]['endtime']
        current_time = starttime

        # 生成external_input，时间间隔为time_delta（30s）
        external_input = []
        for line in external_lines:
            line_time = datetime.strptime(line[0], '%Y/%m/%d %H:%M')
            line_feature = line[1:]
            line_feature = np.array([float(x) for x in line_feature], dtype=np.float64)

            while current_time < line_time: # 后向填充
                external_input.append(line_feature)
                current_time += time_delta
            external_input.append(line_feature)
            current_time += time_delta

        if current_time <= endtime: # 前向填充
            while current_time <= endtime:
                external_input.append(line_feature)
                current_time += time_delta

        external_input = np.array(external_input)

        # feature scaling
        for i in range(external_input.shape[1]):
            external_input[:, i] = (external_input[:, i] - np.mean(external_input[:, i])) / np.std(external_input[:, i])
        
        # 保存external_input
        case_data_name = case_data_path + case_name + '.npy'
        np.save(case_data_name, external_input)
        

if __name__ == '__main__':
    generate_case_external_data()