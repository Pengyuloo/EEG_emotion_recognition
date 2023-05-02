import numpy as np

first9_num = 9

# return time_step, fea_dim, hidden_dim, n_labels
def get_dim(db_name, hid_dim):
    if db_name=='SEED':
        #return 62, 1000, 256, 3  ## for channels
        return 200, 310, 512, 3  ## for time_step

# return sub_num, clip_num, time_step, fea_dim
def get_num(db_name):
    if db_name=='SEED':
        #return 15, 15, 62, 1000  ## seed
        return 15, 15, 200, 310  ## for time_step


def data_preprocess(data, db_name):
    _, _, time_step, fea_dim = get_num(db_name)
    num = data.shape[0]
    for i in range(num):
        this = data[i]
        if this.shape[0]> 200 :
            this = this[:200]
        if this.shape[0]<200 :
            add_len = 200-this.shape[0]
            padding = np.zeros([add_len,62,5])
            this = np.concatenate((this,padding))
        # print('this.shape :', this.shape)
        data[i]=this
    data = np.vstack(data)
    # print("data.shape:",data.shape)
    reshape = np.reshape(data, [num, -1])
    # print("reshape.shape :",reshape.shape)
    mean = np.mean(reshape, axis=1)
    mean = np.reshape(mean, [-1,1])
    std = np.std(reshape, axis=1)
    std = np.reshape(std, [-1,1])
    norm = (reshape - mean)  / std
    # print('test***unnorm shape', test_d/ata.shape, 'norm shape', test_norm.shape)
    data  = np.reshape(reshape, [num, time_step, fea_dim])
    print("data.shape:", data.shape)

    return data

# id:loso
# leave one subject out for 1 session
def leave_one_sub_out(feature, label, sub_id, db_name):
    sub_num, clip_num, _, _ = get_num(db_name)

    train_data = feature[:sub_id*clip_num]
    test_data = feature[sub_id*clip_num:(sub_id+1)*clip_num]
    train_data = np.concatenate((train_data, feature[(sub_id+1)*clip_num:]))
    
    train_label = label[:sub_id*clip_num]
    test_label = label[sub_id*clip_num:(sub_id+1)*clip_num]
    train_label = np.concatenate((train_label, label[(sub_id+1)*clip_num:]))
    
    train_data = data_preprocess(train_data, db_name)
    test_data = data_preprocess(test_data, db_name)
    return train_data, test_data, train_label, test_label