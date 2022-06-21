# コンペ用のマップ表示型アプリケーション
GUIやデプロイには[streamlit](https://streamlit.io/)を用いています.   
コンペのページhttps://www.kaggle.com/competitions/smartphone-decimeter-2022.    
アプリを動かす際のデータセットページhttps://www.kaggle.com/datasets/westtail/ground-truth-and-gnss-data.   

# TODO
* データの読み取り機能
* googlemapの表示機能
* 検索機能
* ファイル型提出機能


# アプリの機能
トリップIDを選択することでIDに紐づいた経度緯度によりgooglemapからマップ表示される

# ディレクトリ構成
```
.
├── app.py 
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── data
└── README.md

```

# 開発者で動かす場合
環境構築の方法は大きく４段階に分かれています  

## 1. リポジトリをcloneします
```
git clone https://github.com/kuto5046/handm_data_visualize_app.git
```

## 2. 環境を構築します

基本的にはdockerを用いて環境構築しますが、ローカルで作成することもできます

### dockerを用いる場合

dockerコマンドを使って環境を構築します
```shell
cd kaggle_app
docker-compose build
```

### ローカルでインストールする場合

次のコマンドで必要なライブラリをインストールしてください
```shell
pip install -r requirements.txt
```

## 3. 実行に必要なデータをダウンロードする

dataフォルダーにデータを用意する必要がある

手動でダウンロードするか、kaggle apiでダウンロードすることもできる

https://www.kaggle.com/datasets/westtail/ground-truth-and-gnss-data
```
cd data
kaggle datasets download -d westtail/ground-truth-and-gnss-data
unzip archive.zip
```

# 4. 実行する
dockerの場合
```shell
docker-compose up -d
docker exec -it python3 streamlit run app.py   
```

dockerでない場合は以下のコマンドを実行する
```shell
streamlit run app.py
```

アウトプットを確認したい場合はアウトプットされた　URL か　`localhost:8501`にアクセス
