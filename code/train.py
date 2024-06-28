import joblib
import numpy as np
import pandas as pd
import catboost as cb

features = ['hour', 'month', 'stationID', 'mean_hour']
shift_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
                  'SO2', 'O3', 'Benzene', 'Xylene', 'target']
start = 359856  # train.csv和test.csv合并后test的起始位置
shift_time = 24  # 时间后移24小时作为特征

train_data = pd.read_csv('../xfdata/空气质量指数预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('../xfdata/空气质量指数预测挑战赛公开数据/test.csv')
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data[shift_features] = np.log(data[shift_features]+1)
data['new_date'] = pd.to_datetime(data['date'], format='%Y%m%d%H')
data['hour'] = data['new_date'].dt.hour
data['month'] = data['new_date'].dt.month
data['ID_hour'] = 100 * data['stationID'] + data['hour']

for c in shift_features:
    features.append('shift_{}'.format(c))
    features.append('shift_2_{}'.format(c))
    features.append('long_mean_{}'.format(c))
    features.append('short_mean_{}'.format(c))
    features.append('mean_1_{}'.format(c))
    features.append('mean_2_{}'.format(c))
    features.append('max_{}'.format(c))


# 对17个组分别构建特征
def group_feature(df):
    df = df.reset_index(drop=False)
    for col in shift_features:
        df['shift_{}'.format(col)] = df[col].shift(shift_time)
        df['shift_2_{}'.format(col)] = df[col].shift(2 * shift_time)
        df['long_mean_{}'.format(col)] = df['shift_{}'.format(col)].rolling(window=24).mean()
        df['short_mean_{}'.format(col)] = df['shift_{}'.format(col)].rolling(window=6).mean()
        df['max_{}'.format(col)] = df['shift_{}'.format(col)].rolling(window=6).max()

    df = df.set_index('index', drop=True)

    return df


# 分组构建特征后按顺序重新合并数据
def concat_data(df):
    d = group_feature(df[df['stationID'] == 0])

    for n in range(1, 17):
        d = pd.concat([d, group_feature(df[df['stationID'] == n])], axis=0)

    d = d.sort_index()

    return d


# 训练模型并保存
def get_model(train_df, valid_df, train_again_df, features_name, label):
    model = cb.CatBoostRegressor(reg_lambda=0.25, loss_function='MAE', task_type='GPU',
                                 max_depth=7, learning_rate=0.01, min_child_samples=10, random_state=2023,
                                 n_estimators=2000)
    model.fit(train_df[features_name], train_df[label],
              eval_set=[(valid_df[features_name], valid_df[label])],
              early_stopping_rounds=20, verbose=20)
    m = cb.CatBoostRegressor(reg_lambda=0.25, loss_function='MAE', task_type='GPU',
                             max_depth=7, learning_rate=0.01, min_child_samples=10, random_state=2023,
                             n_estimators=5 * len(model.evals_result_['learn']['MAE']))
    m.fit(train_again_df[features_name], train_again_df[label], verbose=20)
    joblib.dump(m, '../user_data/{}.pkl'.format(label))


data = concat_data(data)

# 对同时间的各组数据取平均数
for col in shift_features:
    data['mean_1_{}'.format(col)] = data['shift_{}'.format(col)].rolling(window=7).mean()
    data['mean_2_{}'.format(col)] = data['shift_{}'.format(col)].rolling(window=3).mean()

train = data.loc[: start - 1 - 12240]
valid = data.loc[start - 12240: start - 1]
train_again = data.loc[: start - 1]

train['mean_hour'] = train['ID_hour'].map(train.groupby(['ID_hour'])['target'].mean())
valid['mean_hour'] = valid['ID_hour'].map(train.groupby(['ID_hour'])['target'].mean())
train_again['mean_hour'] = train_again['ID_hour'].map(train_again.groupby(['ID_hour'])['target'].mean())

train = train.loc[248064: start - 1 - 12240]
train_again = train_again.loc[248064: start - 1]

for col in shift_features:
    get_model(train, valid, train_again, features, col)
