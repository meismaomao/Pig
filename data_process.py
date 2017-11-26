import numpy as np

def input_data():
    train_data = np.loadtxt('./cifar100python/train_data.txt')
    train_label = np.loadtxt('./cifar100python/train_label.txt')
    t_test = np.loadtxt('./cifar100python/t_test.txt')
    t_label = np.loadtxt('./cifar100python/t_label.txt')
    raw_data = []
    raw_data.extend(train_data)
    raw_data.extend(t_test)
    raw_label = np.loadtxt('./cifar100python/raw_label.txt')
    np.random.seed(0)
    np.random.shuffle(raw_data)
    np.random.seed(0)
    np.random.shuffle(raw_label)
    print(np.shape(t_test))
    print(np.shape(raw_data))
    print(np.shape(raw_label))
    return np.array(train_data), np.array(train_label), np.array(t_test), \
           np.array(t_label), np.array(raw_data), np.array(raw_label)

def input_test():
    t_test = np.loadtxt('./cifar100python/t_test.txt')
    t_label = np.loadtxt('./cifar100python/t_label.txt')
    return t_test, t_label

if __name__ == '__main__':
    step = 0
    BATCH_SIZE = 128
    TRAIN_NUM = 50000
    RAW_NUM = 60000
    train_data, train_label, t_test, t_label, raw_data, raw_label = input_data()
    train_offset = (step * BATCH_SIZE) % (TRAIN_NUM - BATCH_SIZE)
    batch_data = train_data[train_offset:(train_offset + BATCH_SIZE), :]
    batch_label = train_label[train_offset:(train_offset + BATCH_SIZE), :]

    raw_offset = (step * BATCH_SIZE) % (RAW_NUM - BATCH_SIZE)
    raw_batch_data = raw_data[raw_offset:(raw_offset + BATCH_SIZE), :]
    raw_batch_label = raw_label[raw_offset:(raw_offset + BATCH_SIZE), :]
