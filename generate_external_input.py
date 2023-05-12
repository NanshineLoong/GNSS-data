# generate 
# external_inputs [num, features] 

import numpy as np
import os
time_steps = 12 # 输入序列的长度
forcast_step = 6 # 预测未来第几个时间步

input_data_path = './input_data/'

def generate_external_input():
    case_name= 'case14'
    file = f'./case_external_data/{case_name}.npy'

    # case_data [steps, features]
    case_data = np.load(file, allow_pickle=True) # [steps, features]
    external_input = case_data[time_steps-1:-forcast_step, :] # [num, features]
    print(external_input.shape)

    save_path = input_data_path + case_name
    # if no save_path, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save
    np.save(save_path + '/external_inputs.npy', external_input)

if __name__ == '__main__':
    generate_external_input()
