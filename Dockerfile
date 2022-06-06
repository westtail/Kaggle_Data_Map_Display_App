FROM python:3 
# 取ってくるイメージの指定
USER root
# コンテナの最高権限ユーザー

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
RUN apt-get install -y vim less
# 最低限必要なインストール処理

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
# 環境変数の設定

COPY requirements.txt /root
WORKDIR /root
# 作業環境を作成

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
# 必要なライブラリのインストール