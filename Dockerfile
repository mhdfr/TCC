FROM jupyter/datascience-notebook:7a0c7325e470

MAINTAINER Tarsísio Xavier, Matheus H. F. Rafael

RUN pip install xgboost

EXPOSE 8888 8888
