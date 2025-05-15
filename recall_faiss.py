import argparse
import math
import os
import pickle
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='faiss 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='faiss_recall.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'faiss 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         vector_size=256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         epochs=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map


def build_faiss_index(article_vec_map,metric='cos'):
    """构建FAISS索引
    Args:
        article_vec_map: 文章向量映射
    Returns:
        index: FAISS索引
        article_ids: 文章ID列表
    """
    # 提取所有文章ID和向量
    article_ids = []
    article_vecs = []
    
    for article_id, vec in article_vec_map.items():
        article_ids.append(article_id)
        article_vecs.append(vec)
    
    # 转换为numpy数组
    article_ids = np.array(article_ids)
    article_vecs = np.array(article_vecs).astype('float32')
    
    # 构建FAISS索引
    dimension = article_vecs.shape[1]

    if metric == 'ip':
        index = faiss.IndexFlatIP(dimension)  # 内积相似度
    elif metric == 'cos':
        # 余弦相似度需要先对向量做归一化
        faiss.normalize_L2(article_vecs)
        index = faiss.IndexFlatIP(dimension)

    index.add(article_vecs)
    
    return index, article_ids


def recall_single_thread(df_query, article_vec_map, faiss_index, article_ids, user_item_dict):
    """单线程版本的FAISS召回函数
    Args:
        df_query: 查询DataFrame
        article_vec_map: 文章向量映射
        faiss_index: FAISS索引
        article_ids: 文章ID列表
        user_item_dict: 用户-物品交互字典
    Returns:
        df_data: 召回结果DataFrame
    """
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]  # 只取最后一个交互项

        for item in interacted_items:
            if item not in article_vec_map:
                continue
                
            article_vec = article_vec_map[item]
            article_vec = np.array([article_vec]).astype('float32')

            # 使用FAISS搜索相似向量
            sim_scores, indices = faiss_index.search(article_vec, 100)

            # 获取相似文章ID和分数
            sim_scores = sim_scores[0]
            indices = indices[0]

            for idx, score in zip(indices, sim_scores):
                relate_item = article_ids[idx]
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += score

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

    df_data = pd.concat(data_list, sort=False) if data_list else pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])
    return df_data


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 获取文章向量
    if os.path.exists(w2v_file):
        with open(w2v_file, 'rb') as f:
            article_vec_map = pickle.load(f)
        log.info(f'从文件加载文章向量: {w2v_file}')
    else:
        article_vec_map = word2vec(df_click, 'user_id', 'click_article_id', model_path)
        with open(w2v_file, 'wb') as f:
            pickle.dump(article_vec_map, f)
        log.info(f'生成并保存文章向量到: {w2v_file}')

    # 构建FAISS索引
    log.info('构建FAISS索引')
    faiss_index, article_ids = build_faiss_index(article_vec_map)

    # 构建用户-物品交互字典
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 单线程召回处理
    log.info('开始单线程FAISS召回处理')
    df_data = recall_single_thread(df_query, article_vec_map, faiss_index, article_ids, user_item_dict)

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
            f'faiss: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_faiss.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_faiss.pkl')
        df_data.to_pickle('../user_data/data/online/recall_faiss.pkl')