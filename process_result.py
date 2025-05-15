import pandas as pd

# 读取数据
df = pd.read_csv('d:/second/prediction_result/result.csv')  # 假设数据在CSV文件中

# 分割article_id列并只保留前5个
df['article_id'] = df['article_id'].str.split(' ')
df = pd.concat([
    df.drop(['article_id'], axis=1),
    df['article_id'].apply(lambda x: pd.Series(x[:5]))
], axis=1)

# 重命名列
df.columns = ['user_id'] + [f'article_{i+1}' for i in range(5)]

# 保存结果
df.to_csv('d:/second/processed_data.csv', index=False)