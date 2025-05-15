# 导入必要的库
import argparse  # 用于解析命令行参数
import math
import os
import pickle  # 用于序列化/反序列化Python对象
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm  # 进度条显示

from utils import Logger, evaluate  # 自定义工具函数

random.seed(2020)  # 设置随机种子保证可复现性

# 解析命令行参数
parser = argparse.ArgumentParser(description='usercf 召回')
parser.add_argument('--mode', default='valid')  # 运行模式: valid/online
parser.add_argument('--logfile', default='usercf.log')  # 日志文件名

args = parser.parse_args()
mode = args.mode
logfile = args.logfile

# 初始化日志系统
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'usercf 召回，mode: {mode}')


def cal_user_sim(df):
    """计算用户相似度矩阵
    Args:
        df: 包含用户点击行为的DataFrame
    Returns:
        user_sim_dict: 用户相似度字典
        item_user_dict: 物品-用户交互字典
    """
    # 构建物品-用户交互字典
    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        lambda x: list(x)).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    user_cnt = defaultdict(int)  # 用户交互次数统计
    user_sim_dict = defaultdict(lambda: defaultdict(float))  # 使用更高效的数据结构

    # 分批处理物品-用户字典
    batch_size = 1000  # 根据内存情况调整
    items = list(item_user_dict.items())

    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i:i+batch_size]

        for _, users in batch:
            for u1 in users:
                user_cnt[u1] += 1

                for u2 in users:
                    if u1 == u2:
                        continue
                    # 相似度计算(共同交互物品次数)
                    user_sim_dict[u1][u2] += 1 / math.log(1 + len(users))

    # 归一化相似度矩阵
    for u1, related_users in tqdm(user_sim_dict.items()):
        for u2 in related_users:
            user_sim_dict[u1][u2] /= math.sqrt(user_cnt[u1] * user_cnt[u2])

    # 构建用户-物品交互字典
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    return dict(user_sim_dict), user_item_dict


def recall_single_thread(df_query, user_sim, user_item_dict):
    """基于用户相似度的召回函数（单线程版本）
    Args:
        df_query: 查询DataFrame
        user_sim: 用户相似度矩阵
        user_item_dict: 用户历史交互字典
    Returns:
        df_data: 召回结果DataFrame
    """
    data_list = []  # 存储召回结果

    for user_id, item_id in tqdm(df_query.values):
        rank = {}  # 物品排名字典

        if user_id not in user_sim:
            continue

        # 获取与当前用户最相似的K个用户
        similar_users = sorted(user_sim[user_id].items(), 
                              key=lambda d: d[1], 
                              reverse=True)[:50]

        # 基于相似用户的历史交互生成推荐
        for sim_user, wuv in similar_users:
            if sim_user not in user_item_dict:
                continue
                
            for item in user_item_dict[sim_user]:
                # 过滤掉用户已交互的物品
                if user_id in user_item_dict and item in user_item_dict[user_id]:
                    continue
                    
                rank.setdefault(item, 0)
                rank[item] += wuv  # 加权计算得分

        # 取相似度最高的100个物品
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 构建结果DataFrame
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 设置标签(用于评估)
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    # 合并所有结果
    if data_list:
        df_data = pd.concat(data_list, sort=False)
        return df_data
    else:
        return pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])


if __name__ == '__main__':
    # 根据模式加载数据
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/usercf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/usercf_sim.pkl'

    # 计算用户相似度并保存
    user_sim, user_item_dict = cal_user_sim(df_click)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(user_sim, f)

    # 单线程处理召回
    log.info('开始单线程召回处理')
    df_data = recall_single_thread(df_query, user_sim, user_item_dict)

    # 按用户ID和相似度排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)

    # 评估召回效果(仅在验证模式下)
    if mode == 'valid':
        log.info('计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)
        log.debug(
            f'usercf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}')

    # 保存最终召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_usercf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_usercf.pkl')