import argparse
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Multiply, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='DIN 排序模型')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
batch_size = args.batch_size
epochs = args.epochs
embedding_dim = args.embedding_dim
learning_rate = args.learning_rate

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'DIN 排序模型，mode: {mode}')

# 注意力层
def attention_layer(query, key, value):
    # 计算注意力分数
    score = tf.reduce_sum(tf.multiply(query, key), axis=-1, keepdims=True)
    # 注意力权重
    attention_weights = tf.nn.softmax(score, axis=1)
    # 加权求和
    context = tf.multiply(value, attention_weights)
    return context

# 构建DIN模型
def build_din_model(user_features, item_features, behavior_features, 
                   user_feature_dims, item_feature_dims, behavior_feature_dims,
                   embedding_dim=16):
    # 用户特征输入
    user_inputs = []
    user_embeddings = []
    
    for feature, dim in user_feature_dims.items():
        inp = Input(shape=(1,), name=f'user_{feature}_input')
        user_inputs.append(inp)
        
        if dim > 0:  # 类别特征
            emb = Embedding(dim, embedding_dim, name=f'user_{feature}_emb')(inp)
            emb = Flatten()(emb)
            user_embeddings.append(emb)
        else:  # 数值特征
            user_embeddings.append(inp)
    
    # 物品特征输入
    item_inputs = []
    item_embeddings = []
    
    for feature, dim in item_feature_dims.items():
        inp = Input(shape=(1,), name=f'item_{feature}_input')
        item_inputs.append(inp)
        
        if dim > 0:  # 类别特征
            emb = Embedding(dim, embedding_dim, name=f'item_{feature}_emb')(inp)
            emb = Flatten()(emb)
            item_embeddings.append(emb)
        else:  # 数值特征
            item_embeddings.append(inp)
    
    # 行为特征输入（用户历史交互）
    behavior_inputs = []
    behavior_embeddings = []
    
    for feature, dim in behavior_feature_dims.items():
        inp = Input(shape=(None,), name=f'behavior_{feature}_input')
        behavior_inputs.append(inp)
        
        if dim > 0:  # 类别特征
            emb = Embedding(dim, embedding_dim, name=f'behavior_{feature}_emb')(inp)
            behavior_embeddings.append(emb)
        else:  # 数值特征，扩展维度以匹配embedding输出
            expanded = tf.expand_dims(inp, axis=-1)
            behavior_embeddings.append(expanded)
    
    # 合并用户特征
    user_concat = Concatenate()(user_embeddings) if len(user_embeddings) > 1 else user_embeddings[0]
    user_dense = Dense(64, activation='relu')(user_concat)
    
    # 合并物品特征
    item_concat = Concatenate()(item_embeddings) if len(item_embeddings) > 1 else item_embeddings[0]
    item_dense = Dense(64, activation='relu')(item_concat)
    
    # 合并行为特征
    if behavior_embeddings:
        behavior_concat = Concatenate(axis=-1)(behavior_embeddings) if len(behavior_embeddings) > 1 else behavior_embeddings[0]
        
        # 注意力机制
        query = Dense(64, activation='relu')(item_dense)
        query = tf.expand_dims(query, axis=1)  # 扩展维度以便广播
        
        key = Dense(64, activation='relu')(behavior_concat)
        value = behavior_concat
        
        attention_output = attention_layer(query, key, value)
        attention_output = tf.reduce_sum(attention_output, axis=1)
        
        # 合并所有特征
        concat_features = Concatenate()([user_dense, item_dense, attention_output])
    else:
        # 如果没有行为特征，只合并用户和物品特征
        concat_features = Concatenate()([user_dense, item_dense])
    
    # 全连接层
    dense1 = Dense(128, activation='relu')(concat_features)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    output = Dense(1, activation='sigmoid')(dense3)
    
    model = Model(inputs=user_inputs + item_inputs + behavior_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['AUC'])
    
    return model

def preprocess_data(df_feature):
    """预处理数据，将特征分为用户特征、物品特征和行为特征"""
    # 示例特征分类，根据实际数据调整
    user_features = ['user_id']  # 用户相关特征
    item_features = ['article_id']  # 物品相关特征
    behavior_features = []  # 行为相关特征，如用户历史交互
    
    # 其他特征根据前缀或含义分类
    for col in df_feature.columns:
        if col in ['label', 'created_at_datetime', 'click_datetime']:
            continue
        if col.startswith('user_'):
            user_features.append(col)
        elif col.startswith('article_'):
            item_features.append(col)
        elif col.startswith('hist_') or 'behavior' in col:
            behavior_features.append(col)
        else:
            # 默认归为物品特征，可根据实际情况调整
            item_features.append(col)
    
    # 对类别特征进行编码
    feature_dims = {}
    for f in df_feature.select_dtypes(['object', 'category']).columns:
        if f in df_feature.columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
            feature_dims[f] = len(lbl.classes_)
    
    # 对数值特征进行标准化
    scaler = StandardScaler()
    for f in df_feature.select_dtypes(['int64', 'float64']).columns:
        if f not in ['user_id', 'article_id', 'label']:
            df_feature[f] = scaler.fit_transform(df_feature[[f]])
            feature_dims[f] = -1  # 标记为数值特征
    
    # 分类特征维度
    user_feature_dims = {f: feature_dims.get(f, 0) for f in user_features}
    item_feature_dims = {f: feature_dims.get(f, 0) for f in item_features}
    behavior_feature_dims = {f: feature_dims.get(f, 0) for f in behavior_features}
    
    return df_feature, user_features, item_features, behavior_features, user_feature_dims, item_feature_dims, behavior_feature_dims

def train_model(df_feature, df_query):
    # 预处理数据
    df_feature, user_features, item_features, behavior_features, user_feature_dims, item_feature_dims, behavior_feature_dims = preprocess_data(df_feature)
    
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]
    
    del df_feature
    gc.collect()
    
    ycol = 'label'
    
    # 准备输入数据
    def prepare_inputs(df, user_features, item_features, behavior_features):
        inputs = {}
        for f in user_features:
            inputs[f'user_{f}_input'] = df[f].values
        for f in item_features:
            inputs[f'item_{f}_input'] = df[f].values
        for f in behavior_features:
            # 这里需要根据实际情况处理序列数据
            # 简化处理：假设行为特征已经是序列形式
            inputs[f'behavior_{f}_input'] = df[f].values
        return inputs
    
    oof = []
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0
    
    # 训练模型
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[user_features + item_features + behavior_features], 
                       df_train[ycol], df_train['user_id'])):
        log.debug(f'\nFold_{fold_id + 1} Training ================================\n')
        
        # 构建模型
        model = build_din_model(
            user_features, item_features, behavior_features,
            user_feature_dims, item_feature_dims, behavior_feature_dims,
            embedding_dim=embedding_dim
        )
        
        # 准备训练数据
        X_train = df_train.iloc[trn_idx]
        Y_train = X_train[ycol].values
        train_inputs = prepare_inputs(X_train, user_features, item_features, behavior_features)
        
        X_val = df_train.iloc[val_idx]
        Y_val = X_val[ycol].values
        val_inputs = prepare_inputs(X_val, user_features, item_features, behavior_features)
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=3, mode='max'),
            ModelCheckpoint(
                filepath=f'../user_data/model/din_model_{fold_id}.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True
            )
        ]
        
        # 训练模型
        history = model.fit(
            train_inputs, Y_train,
            validation_data=(val_inputs, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 加载最佳模型
        model.load_weights(f'../user_data/model/din_model_{fold_id}.h5')
        
        # 验证集预测
        val_pred = model.predict(val_inputs)
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = val_pred.flatten()
        oof.append(df_oof)
        
        # 测试集预测
        test_inputs = prepare_inputs(df_test, user_features, item_features, behavior_features)
        test_pred = model.predict(test_inputs)
        prediction['pred'] += test_pred.flatten() / 5
        
        # 保存模型
        model.save(f'../user_data/model/din_model_{fold_id}.h5')
    
    # 生成线下验证结果
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')
    
    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)

def online_predict(df_test):
    # 预处理数据
    df_test, user_features, item_features, behavior_features, user_feature_dims, item_feature_dims, behavior_feature_dims = preprocess_data(df_test)
    
    # 准备输入数据
    def prepare_inputs(df, user_features, item_features, behavior_features):
        inputs = {}
        for f in user_features:
            inputs[f'user_{f}_input'] = df[f].values
        for f in item_features:
            inputs[f'item_{f}_input'] = df[f].values
        for f in behavior_features:
            inputs[f'behavior_{f}_input'] = df[f].values
        return inputs
    
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0
    
    # 加载模型并预测
    for fold_id in tqdm(range(5)):
        model = tf.keras.models.load_model(f'../user_data/model/din_model_{fold_id}.h5')
        test_inputs = prepare_inputs(df_test, user_features, item_features, behavior_features)
        pred_test = model.predict(test_inputs)
        prediction['pred'] += pred_test.flatten() / 5
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)

if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        online_predict(df_feature)