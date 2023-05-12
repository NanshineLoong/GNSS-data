import os
import numpy as np
import datetime
from config import time_delta, time_period
from tqdm import tqdm

def find_cases():
    cases = os.listdir('./pos_features/')
    cases = [case for case in cases if 'case' in case]
    return cases

# read files from pos_features or roti_features to generate one case data
def generate_one_case_data(folder, case, start_time, end_time):
    case_path = folder + '/' + case
    files = os.listdir(case_path)
    case_data = []
    
    for file in files:
        features = []

        current_time = start_time
        file_path = case_path + '/' + file
        lacks = 0

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                np_line = np.fromstring(line, dtype=float, sep=' ')
                np_time, np_feature = np_line[:6].astype(int), np_line[6:]
                np_time = datetime.datetime(np_time[0], np_time[1], np_time[2], np_time[3], np_time[4], np_time[5])

                if folder == 'pos_features':
                    target = np.sum(np_feature[-3:]**2)
                    label = 1 if target > 1 else 0
                    np_feature = np.concatenate([np_feature[:-3], np.array([target, label])]) # remove the delta of E, N, U, add the sum of their squares and the label
                    # np_feature = np.append(np_feature[:-3], target, label) # remove the delta of E, N, U, add the sum of their squares and the label

                # if the data is lacked at current_time, use the data of current line to replace
                while current_time < np_time:
                    features.append(np_feature) 
                    current_time += time_delta

                    lacks += 1
                
                features.append(np_feature) 

                current_time += time_delta
            
            # add the last time steps to the tail
            while current_time <= end_time:
                features.append(np_feature) # add the data of last line
                current_time += time_delta

                lacks += 1
                
        features = np.array(features)

        # feature scaling for each sensor(the last column is label)
        for i in range(features.shape[1] - 1):
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

        print(f'{file_path} lack {lacks} time' )
        case_data.append(features)
    
    # stack data into a numpy array
    case_data = np.stack(case_data, axis=1)  # [timestep, sensor, feature]
    return case_data


# generate case.npy of [timestep, sensor, feature]
def generate_case_data():
    cases = find_cases()

    # process each case 
    for case in cases:
        # case = 'case14'

        print(f'processing {case} ...')

        # read data from pos_features
        pos_data = generate_one_case_data(folder='pos_features', case=case, 
                                                 start_time=time_period[case]['starttime'],
                                                 end_time=time_period[case]['endtime'],)

        # read data from roti_features
        roti_data = generate_one_case_data(folder='roti_features', case=case,
                                                 start_time=time_period[case]['starttime'],
                                                 end_time=time_period[case]['endtime'],)

        # combine pos_data and roti_data
        if pos_data.shape[:2] == roti_data.shape[:2]:
            data = np.concatenate((roti_data, pos_data), axis=2)

            # save data into a npy file
            np.save(f'./case_data/{case}.npy', data)
        
if __name__ == '__main__':
    # Each case represent a time period
    # There are total 17 cases
    generate_case_data()