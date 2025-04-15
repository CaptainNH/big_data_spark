# Лабораторная работа 2 Spark

## Датасет 
[Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset).

## Запуск
```
# 1. Поднять контейнеры
docker-compose -f docker-compose.yml up -d  
docker-compose -f docker-compose-3d.yml up -d 

# 2. Загрузить данные в hdfs
docker cp student_depression_dataset.csv namenode:/
docker exec -it namenode /bin/bash
hdfs dfs -mkdir /data
hdfs dfs -D dfs.block.size=32M -put student_depression_dataset.csv  /data/
hdfs dfsadmin -setSpaceQuota 1g /data

# 3. Запуск
docker cp app.py spark-master:/
docker exec -it spark-master /bin/bash
apk add --update make automake gcc g++ python-dev linux-headers
pip install numpy psutil
/spark/bin/spark-submit app.py  # обычная версия
/spark/bin/spark-submit app.py -o # оптимизированная версия

# 4. Остановка
docker-compose -f docker-compose.yml down
docker-compose -f docker-compose-3d.yml down
```
## Результаты эксперментов:

![Figure_2](https://github.com/user-attachments/assets/ec22b75f-6cd0-42a2-a91c-4e1179c039a7)

![Figure_1(1)](https://github.com/user-attachments/assets/514f266c-4e24-4333-b4aa-8538809f8817)


