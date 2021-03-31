import pickle

def load_pickle(index):
    index_string = str(index)
    with open('PPG_FieldStudy/S{}/S{}.pkl'.format(index_string, index_string), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def unpack_data(data):
    ppg_signal = data['signal']['wrist']['BVP'][:, 0]
    acc_signal = data['signal']['wrist']['ACC'][:, 0]
    heart_rate = data['label'] / 60
    activity = data['activity'].reshape(-1)

    return ppg_signal, acc_signal, heart_rate, activity