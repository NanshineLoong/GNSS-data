# generate 
# local_inputs [num, time_steps, features]
# global_inputs [num, time_steps, sensors] 所有传感器目标属性数据
# global_attn_state [num, sensors, features, time_steps] 所有传感器数据
# decoder_gts [num, 1]

import numpy as np
import os
time_steps = 12 # 输入序列的长度
forcast_step = 6 # 预测未来第几个时间步
target_sensor = 0 # 目标传感器

input_data_path = './input_data/'

def generate_input():
    case_name= 'case14'
    file = f'./case_data/{case_name}.npy'

    case_data = np.load(file) # [steps, sensors, features+label]
    case_data_no_label = case_data[:, :, :-1] # [steps, sensors, features]
    case_data_label = case_data[:, :, -1] # [steps, sensors]

    # generate global_attn_state [num, sensors, features, time_steps], num = steps - time_steps + 1
    global_attn_state = np.transpose(case_data_no_label, [1, 2, 0]) # [sensors, features, steps]
    global_attn_state = np.expand_dims(global_attn_state, axis=0) # [1, sensors, features, steps]
    global_attn_state = np.concatenate([global_attn_state[:, :, :, i:i+time_steps] 
                                        for i in range(global_attn_state.shape[-1] - time_steps - forcast_step + 1)], axis=0) # [num, sensors, features, time_steps]
    # print(global_attn_state.shape)

    # generate global_inputs [num, time_steps, sensors]
    global_inputs = global_attn_state[:, :, -1, :] # [num, sensors, time_steps] 只取最后一个特征(目标属性 ΔE^2 + ΔN^2 +ΔU^2)
    global_inputs = np.transpose(global_inputs, [0, 2, 1]) # [num, time_steps, sensors]
    print(global_inputs.shape)

    # generate local_inputs [num, time_steps, features]
    local_inputs = global_attn_state[:, target_sensor, :, :] # [num, features, time_steps] 只取目标传感器
    local_inputs = np.transpose(local_inputs, [0, 2, 1]) # [num, time_steps, features]

    # generate decoder_gts [num, 1]
    decoder_gts = case_data_label[time_steps+forcast_step-1:, target_sensor] # [num, 1]
    decoder_gts = np.expand_dims(decoder_gts, axis=1) # [num, 1]
    print(decoder_gts.shape)

    save_path = input_data_path + case_name
    # if no save_path, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save
    np.save(save_path + '/local_inputs.npy', local_inputs)
    np.save(save_path + '/global_inputs.npy', global_inputs)
    np.save(save_path + '/global_attn_state.npy', global_attn_state)
    np.save(save_path + '/decoder_gts.npy', decoder_gts)

if __name__ == '__main__':
    generate_input()
