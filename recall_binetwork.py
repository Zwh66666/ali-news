import argparse
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='binetwork 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'binetwork 召回，mode: {mode}')


def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        list).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        list).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):
        sim_dict.setdefault(item, {})

        for user in users:
            tmp_len = len(user_item_dict[user])
            for relate_item in user_item_dict[user]:
                sim_dict[item].setdefault(relate_item, 0)
                sim_dict[item][relate_item] += 1 / \
                    (math.log(len(users)+1) * math.log(tmp_len+1))

    return sim_dict, user_item_dict


def recall_single_thread(df_query, item_sim, user_item_dict):
    """单线程版本的召回函数
    Args:
        df_query: 查询DataFrame
        item_sim: 物品相似度矩阵
        user_item_dict: 用户历史交互字典
    Returns:
        df_data: 召回结果DataFrame
    """
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:1]

        for _, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:100]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)
    return df_data


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/binetwork_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/binetwork_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(item_sim, f)

    # 单线程召回处理
    log.info('开始单线程召回处理')
    df_data = recall_single_thread(df_query, item_sim, user_item_dict)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'binetwork: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_binetwork.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_binetwork.pkl')