import copy
import threading
import time
import gzip
import networkx as nx
import tensorflow.compat.v1 as tf
import numpy as np
from matplotlib import pyplot as plt

tf.disable_eager_execution()
'''一些全局参数'''
CLIENT_NUM = 8
ROUNDS = 30
TIPS = 2
CONFIRMED_COUNT = 3
learning_rate = 0.001
batch_size = 50
n_features = 784
n_classes = 10
DATA_DIVIDER = 2  # 自定义数据集总量（根据实验情况自行决定）

'''加载划分数据'''


# region

def load_images(path):
    """加载图像数据，返回numpy数组"""
    with gzip.open(path, 'rb') as f:
        # 根据MNIST格式，前4个字节是幻数，接下来的4字节是图像数，再接下来的两个4字节分别是行数和列数
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # 接下来的数据是图像像素值，每个图像的像素值为28*28
        # num*rows*cols/5表示为原图像数量的1/5
        images = np.frombuffer(f.read(int(num * rows * cols / DATA_DIVIDER)), dtype=np.uint8).reshape(
            int(num / DATA_DIVIDER), 784)  # 这里用784表示28x28
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


def load_data2():
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


# region


def initialize_model():
    """初始化模型"""
    delta_model = {
        'W_hidden': np.random.randn(n_features, 128).astype(np.float32),
        'b_hidden': np.zeros([128], dtype=np.float32),
        'W_output': np.random.randn(128, n_classes).astype(np.float32),
        'b_output': np.zeros([n_classes], dtype=np.float32)
    }
    meta = {'n_samples': 0, 'avg_cost': 0}
    init_model = {'delta_model': delta_model, 'meta': meta}
    return init_model


def local_training(node_id, model, X_train, y_train, tx_id):
    """本地训练"""
    print("client{} begin training..".format(node_id))
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    n_samples = model['meta']['n_samples']
    delta_model = model['delta_model']
    W_hidden, b_hidden, W_output, b_output = (delta_model['W_hidden'], delta_model['b_hidden'],
                                              delta_model['W_output'], delta_model['b_output'])
    hidden_W = tf.Variable(W_hidden, dtype=tf.float32)
    hidden_b = tf.Variable(b_hidden, dtype=tf.float32)
    output_W = tf.Variable(W_output, dtype=tf.float32)
    output_b = tf.Variable(b_output, dtype=tf.float32)

    hidden_layer = tf.nn.relu(tf.matmul(x, hidden_W) + hidden_b)
    prediction = tf.matmul(hidden_layer, output_W) + output_b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            dict_x = X_train[i * batch_size:(i + 1) * batch_size]
            dict_y = y_train[i * batch_size:(i + 1) * batch_size]
            _, c = sess.run([optimizer, loss], feed_dict={x: dict_x, y: dict_y})

            # 获取训练后的权重和偏置
            hidden_W_val, hidden_b_val, output_W_val, output_b_val = sess.run([hidden_W, hidden_b, output_W, output_b])

        acc = accuracy.eval({x: X_test, y: y_test})
        # 准确率存入字典
        if node_id not in accuracy_dict:
            accuracy_dict[node_id] = []
        accuracy_dict[node_id].append(round(acc, 4))
        print("{} accuracy: {:.4f}\n".format(tx_id, acc))

    update_delta_model = {
        'W_hidden': hidden_W_val.tolist(),
        'b_hidden': hidden_b_val.tolist(),
        'W_output': output_W_val.tolist(),
        'b_output': output_b_val.tolist()
    }
    update_model = {'delta_model': update_delta_model, 'meta': model['meta']}
    return update_model


def aggregate_models(selected_tips, local_model):
    """聚合tips"""

    # 获取父节点的模型 delta_model
    parent_model1 = G.nodes[selected_tips[0]]['model']['delta_model']
    parent_model2 = G.nodes[selected_tips[1]]['model']['delta_model']

    # 提取父节点的权重和偏置
    W1_hidden = np.array(parent_model1['W_hidden'])
    b1_hidden = np.array(parent_model1['b_hidden'])
    W1_output = np.array(parent_model1['W_output'])
    b1_output = np.array(parent_model1['b_output'])

    W2_hidden = np.array(parent_model2['W_hidden'])
    b2_hidden = np.array(parent_model2['b_hidden'])
    W2_output = np.array(parent_model2['W_output'])
    b2_output = np.array(parent_model2['b_output'])

    # 获取元数据 meta
    n1 = G.nodes[selected_tips[0]]['model']['meta']['n_samples']
    n2 = G.nodes[selected_tips[1]]['model']['meta']['n_samples']

    # 按数据量加权聚合隐藏层
    aggregated_W_hidden = (W1_hidden * n1 + W2_hidden * n2) / (n1 + n2)
    aggregated_b_hidden = (b1_hidden * n1 + b2_hidden * n2) / (n1 + n2)

    # 按数据量加权聚合输出层
    aggregated_W_output = (W1_output * n1 + W2_output * n2) / (n1 + n2)
    aggregated_b_output = (b1_output * n1 + b2_output * n2) / (n1 + n2)

    # 构造聚合后的 delta_model
    delta_model = {
        'W_hidden': aggregated_W_hidden.tolist(),
        'b_hidden': aggregated_b_hidden.tolist(),
        'W_output': aggregated_W_output.tolist(),
        'b_output': aggregated_b_output.tolist()
    }

    # 构造聚合后的模型
    aggregated_model = {'delta_model': delta_model, 'meta': local_model['meta']}

    return aggregated_model


def cosine_similarity(model1, model2):
    """计算两个模型参数之间的余弦相似度"""
    # 提取模型参数
    W1_hidden = np.array(model1['W_hidden'])
    b1_hidden = np.array(model1['b_hidden'])
    W1_output = np.array(model1['W_output'])
    b1_output = np.array(model1['b_output'])

    W2_hidden = np.array(model2['W_hidden'])
    b2_hidden = np.array(model2['b_hidden'])
    W2_output = np.array(model2['W_output'])
    b2_output = np.array(model2['b_output'])

    # 展开模型参数
    vec1 = np.concatenate([W1_hidden.flatten(), b1_hidden.flatten(), W1_output.flatten(), b1_output.flatten()])
    vec2 = np.concatenate([W2_hidden.flatten(), b2_hidden.flatten(), W2_output.flatten(), b2_output.flatten()])

    # 计算余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def select_tips_based_on_similarity(local_model, tips_set, id):
    """选择与本地模型最相似的前k个tips"""
    similarities = []
    # 计算本地模型与每个tip之间的相似度
    for tip in tips_set:
        tip_model = G.nodes[tip]['model']
        similarity = cosine_similarity(local_model['delta_model'], tip_model['delta_model'])
        similarities.append((tip, similarity))
    # 根据相似度升序排序，选择前k个最相似的tips
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


# endregion

def draw_dag(G):
    """绘制DAG"""
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G, k=0.15, iterations=20)
    # pos = nx.multipartite_layout(G, subset_key="round")  # 根据交易的轮次进行分层布局
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=9, font_weight='bold',
            arrowsize=15, edge_color='gray')
    plt.title("Tangle DAG with Federated Learning Simulation")
    plt.show()

# commit testing
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
    print(model)
    G.add_node('genesis', model=model, round=-1)
    # 每个客户端都维护一个本地模型，初始时是创世纪块中的模型
    local_models = [copy.deepcopy(initialize_model()) for _ in range(CLIENT_NUM)]
    # 创世纪块中数据量设为2000
    model['meta']['n_samples'] = 2000

    '''划分客户端数据集'''
    X_train, X_test, y_train, y_test, sizes = load_data2()
    # 将客户端拥有的数据量存入各自模型meta中
    for i in range(CLIENT_NUM):
        local_models[i]['meta']['n_samples'] = sizes[i]

    '''模拟DAG的联邦学习过程'''
    simulate_federated_learning_with_thread(ROUNDS, CLIENT_NUM, X_train, y_train)
    print('accuracy_dict: {}'.format(accuracy_dict))
    print('rounds_dict: {}'.format(rounds_dict))

    '''绘制DAG'''
    draw_dag(G)

'''设计方法'''
# 训练五轮后，获取全局模型赋值给新客户端，让他继续训练
#
# accuracy_dict: {4: [0.1696, 0.2862, 0.4026, 0.4976, 0.5618, 0.5854, 0.621, 0.646, 0.6646, 0.6728, 0.685, 0.6936, 0.7082, 0.7154, 0.7194, 0.7234, 0.7302, 0.7364, 0.74, 0.7424, 0.7478, 0.748, 0.7514, 0.7522, 0.7584, 0.7622, 0.7672, 0.7688, 0.7724, 0.7744],
#                 1: [0.1668, 0.2858, 0.3496, 0.4846, 0.5314, 0.5798, 0.608, 0.642, 0.6772, 0.6948, 0.711, 0.7252, 0.7332, 0.7422, 0.7502, 0.757, 0.7628, 0.7664, 0.7694, 0.7724, 0.7766, 0.7792, 0.7818, 0.7844, 0.7864, 0.7882, 0.792, 0.7932, 0.7962, 0.7966],
#                 5: [0.1738, 0.291, 0.4126, 0.4972, 0.5202, 0.5872, 0.647, 0.6624, 0.6852, 0.7064, 0.7126, 0.7276, 0.7316, 0.7422, 0.7478, 0.7578, 0.7596, 0.7628, 0.7658, 0.7698, 0.7744, 0.7774, 0.7808, 0.783, 0.7854, 0.7876, 0.7888, 0.7898, 0.7924, 0.7928],
#                 3: [0.166, 0.2818, 0.3992, 0.4918, 0.5502, 0.5874, 0.5998, 0.6156, 0.6498, 0.6692, 0.6834, 0.6958, 0.7118, 0.7208, 0.7258, 0.7308, 0.7394, 0.7456, 0.7524, 0.7552, 0.7638, 0.768, 0.7702, 0.7742, 0.7742, 0.7754, 0.778, 0.7802, 0.7822, 0.7832],
#                 0: [0.1602, 0.2744, 0.3944, 0.49, 0.5772, 0.6202, 0.6624, 0.682, 0.6982, 0.7152, 0.7238, 0.733, 0.7352, 0.7476, 0.7516, 0.7566, 0.7598, 0.7664, 0.7676, 0.7686, 0.771, 0.7744, 0.776, 0.7784, 0.7794, 0.7822, 0.7832, 0.785, 0.7862, 0.7884],
#                 6: [0.1656, 0.2762, 0.395, 0.4958, 0.5256, 0.571, 0.6132, 0.6376, 0.6546, 0.6604, 0.6682, 0.6852, 0.6978, 0.707, 0.712, 0.7262, 0.7344, 0.7382, 0.742, 0.7482, 0.754, 0.7532, 0.7616, 0.766, 0.768, 0.7714, 0.7736, 0.7756, 0.777, 0.7784],
#                 2: [0.1688, 0.282, 0.401, 0.492, 0.5512, 0.5924, 0.6216, 0.6464, 0.6554, 0.6702, 0.6872, 0.6944, 0.7002, 0.709, 0.7192, 0.723, 0.7294, 0.741, 0.7498, 0.7552, 0.7538, 0.758, 0.7588, 0.7672, 0.768, 0.7718, 0.7748, 0.7786, 0.7816, 0.784],
#                 7: [0.3934, 0.5344, 0.6258, 0.6614, 0.6844, 0.7082, 0.7302, 0.7428, 0.7488, 0.7542, 0.7578, 0.7634, 0.7648, 0.7716, 0.774, 0.7754, 0.7784, 0.7798, 0.7816, 0.7846, 0.785, 0.7882, 0.7898, 0.7916]}

# VS
# 训练三轮后，随机矩阵赋值给新客户端 ，让他继续训练
# accuracy_dict: {6: [0.1846, 0.2778, 0.383, 0.447, 0.4972, 0.5442, 0.5624, 0.5884, 0.6196, 0.641, 0.6614, 0.6844, 0.694, 0.703, 0.7118, 0.7156, 0.7212, 0.729, 0.734, 0.7364, 0.7418, 0.7476, 0.7498, 0.7532, 0.755, 0.7592, 0.7622, 0.7658, 0.7676, 0.7686],
#                 0: [0.1864, 0.2892, 0.385, 0.4478, 0.5346, 0.5824, 0.6118, 0.6394, 0.6564, 0.668, 0.6772, 0.688, 0.6976, 0.7044, 0.7098, 0.7146, 0.7206, 0.7252, 0.7312, 0.737, 0.7414, 0.7472, 0.756, 0.7572, 0.7606, 0.763, 0.766, 0.7668, 0.7696, 0.7726],
#                 4: [0.1868, 0.2944, 0.3456, 0.4152, 0.4796, 0.535, 0.5888, 0.619, 0.6314, 0.646, 0.6524, 0.662, 0.6804, 0.686, 0.6948, 0.7088, 0.7172, 0.7294, 0.734, 0.738, 0.7442, 0.749, 0.7508, 0.7516, 0.7516, 0.7578, 0.7616, 0.765, 0.7688, 0.7694],
#                 3: [0.1882, 0.287, 0.3826, 0.4658, 0.4926, 0.528, 0.543, 0.5822, 0.6142, 0.628, 0.646, 0.6526, 0.6672, 0.6744, 0.6826, 0.6904, 0.699, 0.7064, 0.709, 0.713, 0.715, 0.7184, 0.72, 0.7222, 0.7226, 0.7254, 0.7278, 0.73, 0.7318, 0.734],
#                 2: [0.1906, 0.2922, 0.3872, 0.4552, 0.5086, 0.575, 0.6, 0.6276, 0.6356, 0.6516, 0.6688, 0.679, 0.691, 0.6988, 0.704, 0.71, 0.7154, 0.7272, 0.7374, 0.7462, 0.7504, 0.7546, 0.7582, 0.7618, 0.7646, 0.7662, 0.7668, 0.7672, 0.7732, 0.7774],
#                 1: [0.1934, 0.2918, 0.3908, 0.4696, 0.4952, 0.5606, 0.579, 0.6224, 0.6516, 0.6612, 0.6738, 0.6914, 0.7082, 0.722, 0.7338, 0.737, 0.7428, 0.7484, 0.7534, 0.754, 0.7572, 0.7592, 0.7602, 0.7638, 0.7648, 0.7676, 0.7678, 0.7714, 0.7714, 0.7738],
#                 5: [0.1964, 0.302, 0.4026, 0.4824, 0.548, 0.5786, 0.589, 0.6278, 0.6424, 0.6636, 0.681, 0.696, 0.7112, 0.7172, 0.7248, 0.7274, 0.7346, 0.74, 0.745, 0.751, 0.7562, 0.7606, 0.764, 0.7658, 0.7688, 0.7736, 0.7778, 0.7784, 0.782, 0.783],
#                 7: [0.3798, 0.5128, 0.5974, 0.634, 0.649, 0.6706, 0.691, 0.7032, 0.7166, 0.7254, 0.7304, 0.735, 0.7396, 0.7418, 0.7438, 0.7494, 0.7564, 0.7582, 0.7604, 0.766, 0.7706, 0.7734, 0.773, 0.7758]}
