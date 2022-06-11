import streamlit as st
import pandas as pd
import numpy as np
import glob

def load_data(select):
    #print(select)
    data = pd.read_csv(select)
    data["date"] = pd.to_datetime(data["UnixTimeMillis"], unit='ms')
    return data

def main():
    st.title('タイトル')
    st.header('ヘッダを表示')

    file = glob.glob('./data/results/*.csv')
    #print(len(file))

    # サイドバー
    st.sidebar.subheader("input")
    selected = st.sidebar.selectbox(
        '表示するルートを選択：',
        file
    )

    st.header(selected)

    #　座標データの取得
    data = load_data(selected)

    map_data = data[["LatitudeDegrees","LongitudeDegrees"]].to_numpy()
    map = pd.DataFrame(
    map_data,
    columns=['lat', 'lon'])

    st.map(map)

if __name__ == "__main__":
    main()