import pickle

def load_pickle(index):
    index_string = str(index)
    with open('PPG_FieldStudy/S{}/S{}.pkl'.format(index_string, index_string), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def unpack_data(data):
    ppg_signal = data['signal']['wrist']['BVP']
    acc_signal = data['signal']['wrist']['ACC']
    heart_rate = data['label']

    return ppg_signal, acc_signal, heart_rate