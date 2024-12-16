import copy
import random
import numpy as np
import threading
import time
import gzip
import networkx as nx
from keras import models, layers
from matplotlib import pyplot as plt

'''一些全局参数'''
CLIENT_NUM = 10
ROUNDS = 15
TIPS = 2
CONFIRMED_COUNT = 3
learning_rate = 0.001
batch_size = 50
n_features = 784
n_classes = 10
DATA_DIVIDER = 1  # 自定义数据集总量（根据实验情况自行决定）
SUB_DATA_DIVIDER = 6  # 自定义数据集每轮训练量（根据实验情况自行决定）


'''加载和划分数据'''
# region
def load_images(path):
    """加载图像数据，返回numpy数组"""
    with gzip.open(path, 'rb') as f:
        # 根据MNIST格式，前4个字节是幻数，接下来的4字节是图像数，再接下来的两个4字节分别是行数和列数
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # 接下来的数据是图像像素值，每个图像的像素值为28*28
        # num*rows*cols/5表示为原图像数量的1/5
        images = np.frombuffer(f.read(int(num * rows * cols / DATA_DIVIDER)), dtype=np.uint8).reshape(
            int(num / DATA_DIVIDER), 28, 28, 1)  # 这里用784表示28x28
        return images / 255.0  # 归一化为0-1范围


def load_labels(path):
    """加载标签数据，返回numpy数组"""
    with gzip.open(path, 'rb') as f:
        # 根据MNIST格式，前两个字节是幻数，接下来的两个字节是标签数
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # 接下来的每个字节都是标签值，且每个标签对应一个数组（一行）
        labels = np.frombuffer(f.read(int(num / DATA_DIVIDER)), dtype=np.uint8).reshape(int(num / DATA_DIVIDER))
        # one-hot编码：
        # 1.创建指定大小的零矩阵
        one_hot_labels = np.zeros((len(labels), n_classes), dtype=int)
        # 2.每个标签对应的行列位置置为1，形成每行只有一个1的一维数组
        one_hot_labels[np.arange(len(labels)), labels] = 1
        return one_hot_labels


def load_data():
    """加载数据"""
    train_images_path = './data/mnist/MNIST/raw/train-images-idx3-ubyte.gz'
    train_labels_path = './data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz'
    test_images_path = './data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz'
    # 加载图像和标签
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)
    # 将训练数据划分为多个子集，每个子集对应一个客户端
    train_images = np.array_split(train_images, CLIENT_NUM)
    train_labels = np.array_split(train_labels, CLIENT_NUM)
    size_list = [train_images[i].shape[0] for i in range(CLIENT_NUM)]

    return train_images, test_images, train_labels, test_labels, size_list


# endregion

'''初始化模型'''
def initialize_model():
    """使用Keras初始化模型并包含元数据"""
    delta_model = models.Sequential([
        # 第一个卷积层, 输入指定MNIST图像的shape
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # 池化层
        layers.MaxPooling2D((2, 2)),
        # 展平层，为全连接层做准备
        layers.Flatten(),
        # 全连接层，减少神经元数量以减轻过拟合
        layers.Dense(32, activation='relu'),
        # 输出层，10个神经元对应10个类别
        layers.Dense(10, activation='softmax')
    ])
    delta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 包装模型和元数据
    init_model = {
        'delta_model': delta_model,
        'meta': {
            'n_samples': 0,  # 训练样本的数量
            'avg_cost': 0    # 平均损失值，这将在训练过程中更新
        }
    }
    return init_model

'''本地训练'''
def local_training(node_id, model, X_train, y_train, tx_id):
    """使用Keras进行本地训练"""
    print("client{} begin training..".format(node_id))
    delta_model = model['delta_model']
    # 计算每轮使用的数据子集大小
    subset_size = len(X_train) // SUB_DATA_DIVIDER  # 或者选择固定数量，例如100
    # 随机选择数据子集的起始点
    start_index = random.randint(0, len(X_train) - subset_size)
    end_index = start_index + subset_size
    # 选取训练数据的子集
    X_subset = X_train[start_index:end_index]
    y_subset = y_train[start_index:end_index]

    # 进行训练
    delta_model.fit(X_subset, y_subset, epochs=1, batch_size=batch_size, verbose=0)
    loss, acc = delta_model.evaluate(X_test, y_test, verbose=0)
    # 准确率存入字典
    if node_id not in accuracy_dict:
        accuracy_dict[node_id] = []
    accuracy_dict[node_id].append(round(acc, 4))
    print("{} accuracy: {:.4f}".format(tx_id, acc))
    update_model = {'delta_model': delta_model, 'meta': model['meta']}
    update_model['meta']['n_samples'] = subset_size
    return update_model

'''模型聚合'''
def aggregate_models(selected_tips, local_model):
    """聚合模型"""
    weights1 = G.nodes[selected_tips[0]]['model']['delta_model'].get_weights()
    weights2 = G.nodes[selected_tips[1]]['model']['delta_model'].get_weights()

    n1 = G.nodes[selected_tips[0]]['model']['meta']['n_samples']
    n2 = G.nodes[selected_tips[1]]['model']['meta']['n_samples']

    new_weights = []
    for w1, w2 in zip(weights1, weights2):
        new_weights.append((w1 * n1 + w2 * n2) / (n1 + n2))

    new_model = initialize_model()
    new_model['delta_model'].set_weights(new_weights)
    new_model['meta']= local_model['meta']
    return new_model

'''依据模型相似性选择tips'''
def cosine_similarity(model1, model2):
    """计算两个Keras模型参数之间的余弦相似度"""
    weights1 = model1['delta_model'].get_weights()
    weights2 = model2['delta_model'].get_weights()

    # 展开模型权重
    vec1 = np.concatenate([w.flatten() for w in weights1])
    vec2 = np.concatenate([w.flatten() for w in weights2])

    # 计算余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

def select_tips_based_on_similarity(local_model, tips_set, id):
    """选择与本地Keras模型最相似的前k个tips"""
    similarities = []
    # 计算本地模型与每个tip之间的相似度
    for tip in tips_set:
        tip_model = G.nodes[tip]['model']
        similarity = cosine_similarity(local_model, tip_model)
        similarities.append((tip, similarity))
    # 根据相似度降序排序，选择前k个最相似的tips
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 初始化两个存储容器：一个用于相同ID，一个用于不同ID
    same_id_tips = []
    different_id_tips = []
    for tip, similarity in similarities:
        order = int(tip.split('_')[0])
        if order == id:
            same_id_tips.append((tip, similarity))
        else:
            different_id_tips.append((tip, similarity))
    selected_tips = []
    if len(same_id_tips) >= 1 and len(different_id_tips) >= 1:
        selected_tips.append(same_id_tips[0][0])  # 最相似的同ID tip
        selected_tips.append(different_id_tips[0][0])  # 最相似的不同ID tip
    else:
        selected_tips = [tip for tip, _ in similarities[:TIPS]]
    return selected_tips

'''模拟联邦学习'''
def add_transaction(id, local_model, X_train, y_train, round, start):
    for r in range(start, round):
        # 节点名称
        tx_id = f"{id}_r{r + 1}"
        # 第0轮都使用创世纪块训练
        if r == 0:
            # 本地训练
            update_model = local_training(id, G.nodes['genesis']['model'], X_train, y_train, tx_id)
            G.add_node(tx_id, model=update_model, round=r)  # 记录交易轮次
            G.add_edge(tx_id, 'genesis')
        else:
            while True:
                with tip_lock:
                    if len(tip_set) >= 2:
                        selected_tips = select_tips_based_on_similarity(local_model, tip_set, id)
                        break
                    else:
                        print('轮次{}--客户端{}--tips not enough.'.format(r, id + 1))
                # 短暂休眠以避免独占锁
                time.sleep(0.1)
            # 加权聚合已选交易中的模型--当且仅当tip_set中tips数量足够时
            aggregate_model = aggregate_models(selected_tips, local_model)
            print("{} choose {}&{} aggregate".format(tx_id, selected_tips[0], selected_tips[1]))
            # 本地训练
            update_model = local_training(id, aggregate_model, X_train, y_train, tx_id)
            G.add_node(tx_id, model=update_model, round=r)  # 记录交易轮次
            # 将新交易加入DAG，并更新tips集合
            for tip in selected_tips:
                G.add_edge(tx_id, tip)
                # 更新批准次数
                if tip not in confirmation_count:
                    confirmation_count[tip] = 0
                confirmation_count[tip] += 1
                # 如果批准次数达到2次，移出tip_set，加入confirmed_set
                with tip_lock:
                    if confirmation_count[tip] >= CONFIRMED_COUNT:
                        tip_set.discard(tip)
                        with confirmed_lock:
                            confirmed_set.add(tip)
                            print('confirmed_set_size:{} '.format(len(confirmed_set)))
                            print('confirmed_set:{} '.format(confirmed_set))
            # 当前轮次存入字典
            rounds_dict[f"c_{id}"] = r
        # 新产生的交易加入tips集合
        with tip_lock:
            tip_set.add(tx_id)
            print('tip_set_size:{} '.format(len(tip_set)))
            print('tip_set:{} '.format(tip_set))
            print('-----------------\n')

        local_models[id].update(update_model)


def add_transaction_with_thread(id, local_model, X_train, y_train, round):
    """添加交易到DAG"""
    if id == CLIENT_NUM - 1:
        # 当所有客户端都到达第5轮
        while True:
            rounds_dict_list = list(rounds_dict.values())
            print('rounds_dict_list:----{}'.format(rounds_dict_list))
            if all(r == 5 for r in rounds_dict_list):
                while True:
                    with tip_lock:
                        if len(tip_set) >= 2:
                            selected_tips = select_tips_based_on_similarity(local_model, tip_set, id)
                            break
                        else:
                            print('客户端{}--tips not enough.'.format(id))
                    # 短暂休眠以避免独占锁
                    time.sleep(1)
                print('new client selected_tips:---{}'.format(selected_tips))
                aggregate_model = aggregate_models(selected_tips, local_model)
                local_models[id].update(aggregate_model)
                add_transaction(id, local_models[id], X_train, y_train, round, 6)
                break
    else:
        add_transaction(id, local_models[id], X_train, y_train, round, 0)


def simulate_federated_learning_with_thread(rounds, clients, X_train, y_train):
    # 客户端依次加入线程池
    for i in range(clients):
        t = threading.Thread(target=add_transaction_with_thread,
                             args=(i, local_models[i], X_train[i], y_train[i], rounds))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

'''绘制DAG'''
def draw_dag(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_weight='bold', edge_color='gray')
    plt.title("Tangle DAG with Federated Learning Simulation")
    plt.show()


if __name__ == '__main__':
    '''初始化DAG和创世纪块'''
    G = nx.DiGraph()
    # tips集合和用于保护tip_set的锁
    tip_set = set()
    tip_lock = threading.Lock()
    # 已批准交易集合和用于保护confirmed_set的锁
    confirmed_set = set()
    confirmed_lock = threading.Lock()
    # 线程池
    threads = []
    # 交易的确认次数
    confirmation_count = {}
    # 客户端的当前训练轮次
    rounds_dict = {}
    # 客户端的训练准确率
    accuracy_dict = {}
    # 创世纪块不加入已批准交易集合
    model = initialize_model()
    G.add_node('genesis', model=model, round=-1)
    # 每个客户端都维护一个本地模型，初始时是创世纪块中的模型
    local_models = [copy.deepcopy(initialize_model()) for _ in range(CLIENT_NUM)]
    # 创世纪块中数据量设为2000
    model['meta']['n_samples'] = 2000

    '''划分客户端数据集'''
    X_train, X_test, y_train, y_test, sizes = load_data()
    # 将客户端拥有的数据量存入各自模型meta中
    for i in range(CLIENT_NUM):
        local_models[i]['meta']['n_samples'] = sizes[i]

    '''模拟DAG的联邦学习过程'''
    simulate_federated_learning_with_thread(ROUNDS, CLIENT_NUM, X_train, y_train)
    print('accuracy_dict: {}'.format(accuracy_dict))
    print('rounds_dict: {}'.format(rounds_dict))

    '''绘制DAG'''
    draw_dag(G)
