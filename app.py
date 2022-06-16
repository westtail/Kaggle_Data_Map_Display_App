import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px
import glob
from dataclasses import dataclass
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

# https://www.kaggle.com/code/robikscube/smartphone-competition-2022-twitch-stream

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




def load_data(select):
    #print(select)
    data = pd.read_csv(select)
    data["date"] = pd.to_datetime(data["UnixTimeMillis"], unit='ms')
    return data

def main():
    st.title('Smartphone Competition 2022  show map')

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


def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist


def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df1["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df2["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns
    ecef_df = (
        gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
        .dropna()
        .reset_index(drop=True)
    )
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy()
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


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score

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
    fig.show()


def plot_gt_vs_baseline(tripId):
    gt = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/ground_truth.csv")
    #gnss = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_gnss.csv")
    #imu = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_imu.csv")

    baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)

    # グランドトゥルースとベースライン予測の組み合わせ
    baseline["isGT"] = False
    gt["isGT"] = True
    gt["tripId"] = tripId

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