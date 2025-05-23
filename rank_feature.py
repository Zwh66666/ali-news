import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from utils import Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

seed = 42

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')
#额外添加的swing召回的特征梳理
def func_swing_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']
    interacted_items = user_item_dict[user_id]#获取用户的交互记录
    interacted_items = interacted_items[::-1]#反转
    sim_sum = 0
    for loc, i in enumerate(interacted_items):#计算历史文章i和候选文章的相似度 乘以衰减因子 累加
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:
            pass
    return sim_sum

def func_swing_last(x):#计算最后一个交互的文章与候选文章的相似度
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim
#截止到这里

def func_if_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]#获取用户的交互记录 反转 最近的在前
    interacted_items = interacted_items[::-1]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):#计算历史文章i和候选文章的相似度 乘以衰减因子 累加
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:#如果不存在则跳过
            pass
    return sim_sum


def func_if_last(x):#计算最后一个交互的文章与候选文章的相似度
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def func_binetwork_sim_last(x):#计算最后一个文章与binetwork的候选文章的相似度
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = binetwork_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:#检查是否为numpy数组
        return -1
    #计算余弦相似度
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                        article_vec_map[i])#计算与候选文章的余弦相似度并累加
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x):#计算最后一个文章与候选文章的余弦相似度
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(article_vec_map[article_id],
                               article_vec_map[last_item])
    except Exception as e:
        pass
    return sim


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')
    #合并文章的特征到df_feature中
    df_article = pd.read_csv('../data/articles.csv')
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')#将时间戳转换为日期添加到df_feature中

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)#排序先对user_id进行排序，然后对click_timestamp进行排序
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)#将click_article_id重命名为article_id
    df_click = df_click.merge(df_article, how='left')#合并
    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby([
        'user_id'
    ])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']

    # 点击文章的创建时间差的统计值
    df_temp = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].agg(
        user_click_timestamp_created_at_ts_diff_mean='mean',
        user_click_timestamp_created_at_ts_diff_std='std'
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg(
        user_click_datetime_hour_std='std'
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby(['user_id'])['words_count'].agg(
        user_clicked_article_words_count_mean='mean',
        user_click_last_article_words_count=lambda x: x.iloc[-1]
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    df_temp = df_click.groupby('user_id')['created_at_ts'].agg(
        user_click_last_article_created_time=lambda x: x.iloc[-1],
        user_clicked_article_created_time_max='max',
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id')['click_timestamp'].agg(
        user_click_last_article_click_time=lambda x: x.iloc[-1],
        user_clicked_article_click_time_mean='mean',
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('../user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_sum, axis=1)
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')
##swing相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('../user_data/sim/offline/swing_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/swing_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_swing_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_sum, axis=1)
    df_feature['user_last_click_article_swing_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open('../user_data/sim/offline/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    # 将 parallel_apply 替换为普通的 apply
    df_feature['user_last_click_article_binetwork_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_binetwork_sim_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('../user_data/data/offline/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/data/online/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_w2v_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_article_w2w_sim_sum_2'] = df_feature[[
        'user_id', 'article_id'
    ]].apply(lambda x: func_w2w_sum(x, 2), axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/feature.pkl')

    else:
        df_feature.to_pickle('../user_data/data/online/feature.pkl')
