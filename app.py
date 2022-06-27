import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px
import glob
from dataclasses import dataclass
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

# https://www.kaggle.com/code/robikscube/smartphone-competition-2022-twitch-stream

KEY = st.secrets["google_map_api_key"]

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3
HAVERSINE_RADIUS = 6_371_000

@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)


@dataclass
class BLH:
    lat: np.array
    lng: np.array
    hgt: np.array

# 複雑な計算部分
def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (e2_ * b) * np.sin(t) ** 3, r - (e2 * a) * np.cos(t) ** 3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)

# プロット用の関数
def visualize_traffic(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    color_col="phone",
    label_col="tripId",
    zoom=9,
    opacity=1,
):
    if center is None:
        center = {
            "lat": df[lat_col].mean(),
            "lon": df[lon_col].mean(),
        }
    fig = px.scatter_mapbox(
        df,
        # Here, plotly gets, (x,y) coordinates
        lat=lat_col,
        lon=lon_col,
        # Here, plotly detects color of series
        color=color_col,
        labels=label_col,
        zoom=zoom,
        center=center,
        height=600,
        width=800,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    st.plotly_chart(fig)

def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    # xyz値のカラム
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    # カラム設定
    columns = ["utcTimeMillis"] + ecef_columns

    # gnssデータの定義
    ecef_df = (
        gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
        .dropna()
        .reset_index(drop=True)
    )

    # numpy 変換
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    # 複雑な計算
    blh = ECEF_to_BLH(ecef)

    # 時間をnumpy変換
    TIME = ecef_df["utcTimeMillis"].to_numpy()

    # v1 次元スプライン補間曲線を得られる関数
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame(
        {
            "tripId": tripID,
            "UnixTimeMillis": UnixTimeMillis,
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )

# ベースラインを表示
def plot_gt_vs_baseline(tripId):
    gt = pd.read_csv(f"./data/results/{tripId}_gt.csv")
    gnss = pd.read_csv(f"./data/results/{tripId}_gnss.csv")

    # グランドトゥルースとベースライン予測の組み合わせ
    baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
    
    baseline["isGT"] = False
    gt["isGT"] = True
    gt["tripId"] = tripId

    #データのコンバイン
    combined = (
        pd.concat([baseline, gt[baseline.columns]], axis=0)
        .reset_index(drop=True)
        .copy()
    )

    # Plotting the route
    visualize_traffic(
        combined,
        lat_col="LatitudeDegrees",
        lon_col="LongitudeDegrees",
        color_col="isGT",
        zoom=10,
    )


def load_data(select):
    #print(select)
    data = pd.read_csv(select)
    data["date"] = pd.to_datetime(data["UnixTimeMillis"], unit='ms')
    return data

def main():
    st.title('Smartphone Competition 2022  show map')

    files = glob.glob('./data/results/*_gt.csv')
    name = []
    for file in files:
        name.append(file[15:-7])

    # サイドバー
    st.sidebar.subheader("input")

    # ボタンを押すと切り替え
    methods = st.sidebar.radio("Choose a search method",('text', 'list'))

    if methods == "text":
        selected = st.sidebar.text_input('input tripID', '2020-05-15-US-MTV-1-GooglePixel4XL')
    else:
        selected = st.sidebar.selectbox(
            'chose root ：',
            name
        )

    st.header(selected)

    plot_gt_vs_baseline(selected) #ベースラインの表示

    link = '[Smartphone Competition 2022 [Twitch Stream]](https://www.kaggle.com/code/robikscube/smartphone-competition-2022-twitch-stream)'
    st.text('Code used for baseline ')
    st.markdown(link, unsafe_allow_html=True)
    test = "35.473183,138.57476399999996"

    components.html(
    """<iframe src="https://www.google.com/maps/embed/v1/streetview?key="""+ KEY + 
        """&location=""" + test + """
        &heading=210
        &pitch=10
        &fov=35" 
        width="600" height="450" style="border:0;" allowfullscreen></iframe>"""
    ,height=500
    )

if __name__ == "__main__":
    main()