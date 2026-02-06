FROM python:3.11

RUN pip install numpy==1.25.0
RUN pip install pandas==2.2.2
RUN pip install py7zr==0.21.1
RUN pip install pyarrow==16.1.0
RUN pip install fastparquet==2024.5.0
RUN pip install openpyxl==3.1.5
RUN pip install xlrd==2.0.1
RUN pip install recbole==1.2.0
RUN pip install kmeans-pytorch==0.3
RUN pip install wandb==0.17.4
RUN pip install faiss-cpu==1.8.0.post1
RUN pip install ray==2.6.3
RUN pip install recpack==0.3.6
RUN pip install scikit-learn==1.5.1
RUN pip install binpickle==0.3.4
RUN pip install lenskit==2025.1.1
RUN pip install torch==2.4.0
RUN pip install scipy==1.14.0
RUN pip install seaborn==0.13.2
RUN pip install matplotlib==3.9.1
RUN pip install xgboost==2.0.3