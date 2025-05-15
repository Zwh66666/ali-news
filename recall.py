import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(42)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):#输入recall_result
    # 创建字典存储每个用户的最大和最小相似度分数
    user_score_max = {}
    user_score_min = {}

    # 获取用户的相似度的最大值和最小值
    # 按user_id分组，每组已经按sim_score降序排列sim_score是向量相似度分数
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]  # 最大值(第一个元素)
        user_score_min[user_id] = scores[-1] # 最小值(最后一个元素)

    ans = []
    # 遍历DataFrame中的每一行(user_id, sim_score)
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        # 对每个分数进行Min-Max归一化
        # 公式: (x - min) / (max - min) + 一个很小的值(10^-3)
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()#按照user_id分组，对article_id进行去重
    user_item_dict1 = dict(zip(user_item_['user_id'],user_item_['article_id']))#将user_id和article_id转换为字典

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'], user_item_['article_id']))

    # 初始化计数器
    cnt = 0       # 总item数
    hit_cnt = 0   # 重合item数

    # 遍历df1中的所有用户
    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]  # 获取该用户在df1中的item集合

        cnt += len(item_set1)  # 累加item数量

        # 如果该用户也存在于df2中
        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]  # 获取该用户在df2中的item集合

            inters = item_set1 & item_set2  # 计算两个集合的交集
            hit_cnt += len(inters)  # 累加重合item数量

    # 返回重合率 = 重合item数 / 总item数
    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':#载入数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = [ 'itemcf','faiss', 'binetwork','swing']

    weights = {'itemcf':1,'swing':0.2,'binetwork': 1,'faiss':0.2}#分配权重
    recall_list = []
    recall_dict = {}
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(f'{recall_path}/recall_{recall_method}.pkl')#读取召回结果
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)#归一化相关系数
        recall_result['sim_score'] = recall_result['sim_score'] * weight#分配权重

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result#将召回结果按照方法存储到字典中

    # 求相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')#计算召回相似度

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)#将所有召回结果合并
    recall_score = recall_final[['user_id', 'article_id','sim_score']].groupby(['user_id', 'article_id'])['sim_score'].sum().reset_index()
    #将召回结果按照user_id和article_id分组，分组后对同一用户对同一个物品的推荐记录归为一组，然后对sim_score进行求和
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])#去除重复的user_id和article_id，以防label不一样
    recall_final = recall_final.merge(recall_score, how='left')#将召回结果和相似度结果合并
    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])#按照user_id和sim_score进行排序

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')


