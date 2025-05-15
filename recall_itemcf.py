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
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')  # 运行模式: valid/online
parser.add_argument('--logfile', default='test.log')  # 日志文件名

args = parser.parse_args()
mode = args.mode
logfile = args.logfile

# 初始化日志系统
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df):
    """计算物品相似度矩阵
    Args:
        df: 包含用户点击行为的DataFrame
    Returns:
        sim_dict: 物品相似度字典
        user_item_dict: 用户-物品交互字典
    """
    # 构建用户-物品交互字典
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_cnt = defaultdict(int)  # 物品出现次数统计
    sim_dict = {}  # 物品相似度字典

    # 计算物品共现矩阵
    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重计算
                # 正向点击权重1.0，反向点击权重0.7
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 距离衰减因子
                loc_weight = loc_alpha * (0.7 ** (np.abs(loc2 - loc1) - 1))

                # 相似度计算(共现次数/用户交互序列长度的对数)
                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    # 归一化相似度矩阵
    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


def recall_single_thread(df_query, item_sim, user_item_dict):
    """基于物品相似度的召回函数（单线程版本）
    Args:
        df_query: 查询DataFrame
        item_sim: 物品相似度矩阵
        user_item_dict: 用户历史交互字典
    Returns:
        df_data: 召回结果DataFrame
    """
    data_list = []  # 存储召回结果

    for user_id, item_id in tqdm(df_query.values):
        rank = {}  # 物品排名字典

        if user_id not in user_item_dict:
            continue

        # 获取用户最近交互的2个物品(按时间倒序)
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        # 基于相似物品生成推荐
        for loc, item in enumerate(interacted_items):
            # 取每个物品最相似的200个物品
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # 加权计算相似度得分(考虑位置衰减)
                    rank[relate_item] += wij * (0.5 ** loc)

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
    df_data = pd.concat(data_list, sort=False)
    return df_data


if __name__ == '__main__':
    # 根据模式加载数据
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    # 计算物品相似度并保存
    item_sim, user_item_dict = cal_sim(df_click)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(item_sim, f)

    # 单线程处理召回
    log.info('开始单线程召回处理')
    df_data = recall_single_thread(df_query, item_sim, user_item_dict)

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
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}')

    # 保存最终召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')