import joblib
import numpy as np
import pandas as pd

features = ['hour', 'month', 'stationID', 'mean_hour']
shift_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
                  'SO2', 'O3', 'Benzene', 'Xylene', 'target']
start = 359856
shift_time = 24
shift_num = 30

train_data = pd.read_csv('../xfdata/空气质量指数预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('../xfdata/空气质量指数预测挑战赛公开数据/test.csv')
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data[shift_features] = np.log(data[shift_features]+1)
data['new_date'] = pd.to_datetime(data['date'], format='%Y%m%d%H')
data['hour'] = data['new_date'].dt.hour
data['month'] = data['new_date'].dt.month
data['ID_hour'] = 100 * data['stationID'] + data['hour']
data['mean_hour'] = data['ID_hour'].map(data.loc[: start - 1 - 12240].groupby(['ID_hour'])['target'].mean())

for c in shift_features:
    features.append('shift_{}'.format(c))
    features.append('shift_2_{}'.format(c))
    features.append('long_mean_{}'.format(c))
    features.append('short_mean_{}'.format(c))
    features.append('mean_1_{}'.format(c))
    features.append('mean_2_{}'.format(c))
    features.append('max_{}'.format(c))


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


def concat_data(df):
    d = group_feature(df[df['stationID'] == 0])

    for n in range(1, 17):
        d = pd.concat([d, group_feature(df[df['stationID'] == n])], axis=0)

    d = d.sort_index()

    return d


def pre(num):
    test = data.loc[start + 17 * shift_time * num: start - 1 + 17 * shift_time * (num + 1)]
    for col in shift_features:
        model = joblib.load('../user_data/{}.pkl'.format(col))
        data.loc[start + 17 * shift_time * num: start - 1 + 17 * shift_time * (num + 1),
                 col] = model.predict(test[features])


for i in range(shift_num):
    data = concat_data(data)
    for col in shift_features:
        data['mean_1_{}'.format(col)] = data['shift_{}'.format(col)].rolling(window=7).mean()
        data['mean_2_{}'.format(col)] = data['shift_{}'.format(col)].rolling(window=3).mean()
    pre(i)
data.loc[start:, 'target'] = np.e ** data.loc[start:, 'target'] - 1
data.loc[start:, ['date', 'stationID', 'target']].to_csv('../prediction_result/sample_submit.csv', index=False)
