import numpy as np


def _hamming_loss(pred, test):
    """ 单样本评估标记结果的Hamming-error指标

    pred, test 为多标记向量，对应位置取值1表示存在该标记，否则不存在
    hamming-loss = (pred .* test) / len(y1)
    """
    assert len(pred) == len(test)
    n_pred = np.array(pred)
    n_test = np.array(test)
    return np.dot(n_pred, n_test) / len(n_pred)


def _one_error(pred, test):
    """ 单样本评估标记结果的One-error指标

    pred 为预测标签的概率向量， test为测试标签1/0向量
    pred 中概率最高的标签在 test 中对应位置为1则返回1，否则返回0
    one-error = I(argmax(pred) in test)
    """
    assert len(pred) == len(test)
    n_pred = np.array(pred)
    n_test = np.array(test)

    argmax = n_pred.argmax()
    return int(n_test[argmax])


def _coverage(pred, test):
    """ 单样本评估标记结果的Coverage指标

    pred 为预测标签的概率向量, test为测试标签1/0向量
    pred 中概率排序的序列值在全部包含test中标签的最大值
    """
    assert len(pred) == len(test)
    n_pred = np.array(pred)
    n_test = np.array(test)

    n_sort = np.argsort(-n_pred)
    n_rank = n_sort[n_test == 1]
    rank_dep = n_rank.max() if len(n_rank) > 0 else 0

    return rank_dep


def _ranking_loss(pred, test):
    """ 单样本评估标记结果的Ranking loss指标

    pred 为预测标签的概率向量, test为测试标签1/0向量
    pred 中概率排序后, 无关标记位于相关标记之前的个数
    """
    assert len(pred) == len(test)
    n_pred = np.array(pred)
    n_test = np.array(test)

    n_sort = np.argsort(-n_pred)
    n_rank = n_sort[n_test == 1]
    rank_dep = n_rank.max() + 1 if len(n_rank) > 0 else 0

    delivery = len(n_rank) * (len(test) - len(n_rank))

    if delivery == 0 or rank_dep - len(n_rank) == 0:
        return 0.0
    else:
        return (rank_dep - len(n_rank)) / delivery


def _average_precision(pred, test):
    """ 单样本评估标记结果的Average precision指标

    pred 为预测标签的概率向量, test为测试标签1/0向量
    pred 中相关标记之前仍为相关标记的数目
    """

    assert len(pred) == len(test)
    n_pred = np.array(pred)
    n_test = np.array(test)

    n_sort = np.argsort(-n_pred) + 1
    n_rank = n_sort[n_test == 1]

    n_rank.sort()
    precision_value = 0.0
    for idx, rank in enumerate(n_rank):
        if rank > 0:
            precision_value += (idx + 1) / rank

    return precision_value / len(n_rank) if len(n_rank) > 0 else 0
