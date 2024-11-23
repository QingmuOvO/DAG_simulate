import copy
import networkx as nx
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

tf.disable_eager_execution()
'''一些全局参数'''
CLIENT_NUM = 5
ROUNDS = 5
TIPS = 2
CONFIRMED_COUNT = 3
n_features = 5
n_class = 2
learning_rate = 0.001
batch_size = 100


# region
def split_data(path, clients_num):
    """切分数据集"""
    # 读取数据
    data = pd.read_csv(path)
    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,
        data["Occupancy"].values.reshape(-1, 1),
        random_state=42)

    # one-hot 编码
    y_train = np.concatenate([1 - y_train, y_train], 1)
    y_test = np.concatenate([1 - y_test, y_test], 1)

    # 计算随机分区大小（分区大小在平均分配数量的1/2~3/2之间）
    minZones = len(X_train) // clients_num // 2
    maxZones = len(X_train) // clients_num * 3 // 2
    sizes = np.random.randint(minZones, maxZones, size=clients_num - 1)

    total_allocated = np.sum(sizes)
    total_sum = len(X_train)
    # 如果已分配的数据量已经超额，则重新配额
    if total_sum - total_allocated < minZones:
        deficit = abs(total_sum - total_allocated) + minZones // 2  # 计算逆差
        average_reduction = deficit // (clients_num - 1)  # 计算平均补差
        # 从每个客户端减去平均减少量
        for i in range(clients_num - 1):
            sizes[i] -= average_reduction  # 重新确定每个客户端的数据量
        sizes = np.append(sizes, total_sum - np.sum(sizes))  # 将剩余的数据集全部分配给最后一个客户端
    # 若没有超额，则全部给最后一个客户端
    else:
        sizes = np.append(sizes, total_sum - total_allocated)
    print('allocated size : {}'.format(sizes))

    # 根据生成的sizes划分训练集给多个client
    X_train_splits = np.split(X_train, np.cumsum(sizes)[:-1])
    y_train_splits = np.split(y_train, np.cumsum(sizes)[:-1])

    # 获取每个客户端的数据量
    sizes = sizes.tolist()

    return X_train_splits, X_test, y_train_splits, y_test, sizes


def initialize_model():
    """初始化模型"""
    W = np.random.randn(n_features, n_class)
    b = np.random.randn(n_class)
    delta_model = {'ser_W': W, 'ser_b': b}
    meta = {'n_samples': 0, 'avg_cost': 0}
    init_model = {'delta_model': delta_model, 'meta': meta}
    return init_model


def local_training(node_id, model, X_train, y_train):
    """本地训练"""
    print("client{} begin training..".format(node_id))
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])

    n_samples = model['meta']['n_samples']
    delta_model = model['delta_model']
    ser_W, ser_b = delta_model['ser_W'], delta_model['ser_b']
    W = tf.Variable(ser_W, dtype=tf.float32)
    b = tf.Variable(ser_b, dtype=tf.float32)

    pred = tf.matmul(x, W) + b
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            dict_x = X_train[i * batch_size:(i + 1) * batch_size]
            dict_y = y_train[i * batch_size:(i + 1) * batch_size]
            _, c = sess.run([train_op, cost], feed_dict={x: dict_x, y: dict_y})

        # 获得训练后的模型参数
        val_W, val_b = sess.run([W, b])
        # 使用测试集进行准确率测试
        acc = accuracy.eval({x: X_test, y: y_test})
        print(f"Test Accuracy: {acc:.4f}")

    update_delta_model = {'ser_W': val_W.tolist(), 'ser_b': val_b.tolist()}
    update_model = {'delta_model': update_delta_model, 'meta': local_models[node_id]['meta']}
    return update_model


def aggregate_models(selected_tips, local_model):
    """聚合tips"""
    # 获取到model中的模型部分delta_model
    parent_model1 = G.nodes[selected_tips[0]]['model']['delta_model']
    parent_model2 = G.nodes[selected_tips[1]]['model']['delta_model']
    W1, b1 = np.array(parent_model1['ser_W']), np.array(parent_model1['ser_b'])
    W2, b2 = np.array(parent_model2['ser_W']), np.array(parent_model2['ser_b'])
    W_local, b_local = np.array(local_model['delta_model']['ser_W']), np.array(local_model['delta_model']['ser_b'])

    # 获取到model中的元数据meta
    n1 = G.nodes[selected_tips[0]]['model']['meta']['n_samples']
    n2 = G.nodes[selected_tips[1]]['model']['meta']['n_samples']
    n_local = local_model['meta']['n_samples']
    n_total = n1 + n2 + n_local

    # 按数据量加权聚合
    aggregated_W = (W1 * n1 + W2 * n2 + W_local * n_local) / n_total
    aggregated_b = (b1 * n1 + b2 * n2 + b_local * n_local) / n_total

    # 更新本地模型
    delta_model = {'ser_W': aggregated_W.tolist(), 'ser_b': aggregated_b.tolist()}
    aggregated_model = {'delta_model': delta_model, 'meta': local_model['meta']}

    return aggregated_model


def cosine_similarity(model1, model2):
    """计算两个模型参数之间的余弦相似度"""
    W1, b1 = np.array(model1['ser_W']), np.array(model1['ser_b'])
    W2, b2 = np.array(model2['ser_W']), np.array(model2['ser_b'])

    # 展开模型参数
    vec1 = np.concatenate([W1.flatten(), b1.flatten()])
    vec2 = np.concatenate([W2.flatten(), b2.flatten()])

    # 计算余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def select_tips_based_on_similarity(local_model, tips_set):
    """选择与本地模型最相似的前k个tips"""
    similarities = []
    # 计算本地模型与每个tip之间的相似度
    for tip in tips_set:
        tip_model = G.nodes[tip]['model']
        similarity = cosine_similarity(local_model['delta_model'], tip_model['delta_model'])
        similarities.append((tip, similarity))
    # 根据相似度升序排序，选择前k个最相似的tips
    similarities.sort(key=lambda x: x[1], reverse=True)
    print('acc_list:{}'.format(similarities))
    selected_tips = [tip for tip, _ in similarities[:TIPS]]
    return selected_tips


def add_transaction(id, local_model, X_train, y_train, round):
    """添加交易到DAG"""
    # 节点名称
    tx_id = f"c{id + 1}_r{round + 1}"
    # 第0轮都使用创世纪块训练
    if round == 0:
        # 本地训练
        update_model = local_training(id, G.nodes['genesis']['model'], X_train, y_train)
        G.add_node(tx_id, model=update_model, round=round)  # 记录交易轮次
        G.add_edge(tx_id, 'genesis')
    else:
        selected_tips = select_tips_based_on_similarity(local_model, tip_set)
        # 加权聚合已选交易中的模型
        aggregate_model = aggregate_models(selected_tips, local_model)
        print("{} choose {}&{} aggregate".format(tx_id, selected_tips[0], selected_tips[1]))
        # 本地训练
        update_model = local_training(id, aggregate_model, X_train, y_train)
        G.add_node(tx_id, model=update_model, round=round)  # 记录交易轮次
        # 将新交易加入DAG，并更新tips集合
        for tip in selected_tips:
            G.add_edge(tx_id, tip)
            # 更新批准次数
            if tip not in confirmation_count:
                confirmation_count[tip] = 0
            confirmation_count[tip] += 1
            # 如果批准次数达到2次，移出tip_set，加入confirmed_set
            if confirmation_count[tip] >= CONFIRMED_COUNT:
                tip_set.discard(tip)
                confirmed_set.add(tip)
    # 新产生的交易加入tips集合
    tip_set.add(tx_id)

    print('tip_set_size:{}'.format(len(tip_set)))
    print('tip_set:{}'.format(tip_set))
    print('confirmed_set_size:{}'.format(len(confirmed_set)))
    print('confirmed_set:{}\n---------'.format(confirmed_set))

    return update_model


def simulate_federated_learning(rounds, clients, X_train, y_train):
    for r in range(rounds):
        print(f"Round {r + 1}/{rounds}")
        for i in range(clients):
            # 每次交易后，更新客户端的本地模型
            local_models[i] = add_transaction(i, local_models[i], X_train[i], y_train[i], r)


# endregion

def draw_dag(G):
    """绘制DAG"""
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.multipartite_layout(G, subset_key="round")  # 根据交易的轮次进行分层布局
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=9, font_weight='bold',
            arrowsize=15, edge_color='gray')
    plt.title("Tangle DAG with Federated Learning Simulation")
    plt.show()


if __name__ == '__main__':
    '''初始化DAG和创世纪块'''
    G = nx.DiGraph()
    # tips集合
    tip_set = set()
    # 已批准交易集合
    confirmed_set = set()
    # 每个交易的确认次数
    confirmation_count = {}
    # 创世纪块不加入已批准交易集合
    model = initialize_model()
    G.add_node('genesis', model=model, round=-1)
    # 每个客户端都维护一个本地模型，初始时是创世纪块中的模型
    local_models = [copy.deepcopy(initialize_model()) for _ in range(CLIENT_NUM)]
    # 创世纪块中数据量设为2000
    model['meta']['n_samples'] = 2000

    '''划分客户端数据集'''
    X_train, X_test, y_train, y_test, sizes = split_data("./data/datatraining.txt", CLIENT_NUM)
    # 将客户端拥有的数据量存入各自模型meta中
    for i in range(CLIENT_NUM):
        local_models[i]['meta']['n_samples'] = sizes[i]

    '''模拟DAG的联邦学习过程'''
    simulate_federated_learning(ROUNDS, CLIENT_NUM, X_train, y_train)

    '''绘制DAG'''
    draw_dag(G)
