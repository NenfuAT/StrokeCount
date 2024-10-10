import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def main():
    df = pd.read_csv("./csv/distance1.csv")
    # CSVファイルの読み込み

    df['time'] = df['time'].astype(int)
    # 現在の位置ベクトルの計算
    df['position'] = df[['dx', 'dy', 'dz']].apply(lambda x: np.array([x['dx'], x['dy'], x['dz']]), axis=1)

    # 前回の位置ベクトルの取得
    df['prev_position'] = df['position'].shift(1)

    # ベクトルの変化を計算
    df['delta_position'] = df['position'] - df['prev_position']

    # 基準ベクトル (y軸方向を前方として定義)
    forward_vector = np.array([0, 1, 0])  # y軸方向

    # 手前への移動をカウントするためのしきい値を設定
    threshold = 0.01 # 適切なしきい値を設定(1.1cm)

    # ベクトルのy成分を使って、極大値と極小値を検出
    y_components = df['delta_position'].apply(lambda x: x[1] if isinstance(x, np.ndarray) else 0).values
    
    # 極大値と極小値のインデックスを検出
    maxima_indices = argrelextrema(y_components, np.greater)[0]
    minima_indices = argrelextrema(y_components, np.less)[0]

    count_pull = 0
    pull_times = []
    # 極大値と極小値を順に比較して手前に引いたタイミングをカウント
    for max_idx in maxima_indices:
        # 極大値より後にある極小値を探す
        min_idx = minima_indices[minima_indices > max_idx]
        if len(min_idx) > 0:
            min_idx = min_idx[0]  # 最初の極小値を取得

            # 極大値から極小値への移動量を確認
            delta = df['position'].iloc[min_idx] - df['position'].iloc[max_idx]
            dot_product = np.dot(delta, forward_vector)
            
            # 内積がしきい値を超えた場合に手前に移動したとみなす
            if dot_product < -threshold:  # 手前への移動なので負の内積
                count_pull += 1
                # 極大値と極小値間の時間を計算（ms単位）
                start_time = df['time'].iloc[max_idx]
                end_time = df['time'].iloc[min_idx]
                time_taken = end_time - start_time
                pull_times.append(time_taken)

    # 結果の表示
    print(f'手前に引いた回数: {count_pull}')
    for i, time in enumerate(pull_times):
        print(f'手前に引いた動作 {i+1} の時間: {time} ms')
if __name__ == "__main__":
    main()
