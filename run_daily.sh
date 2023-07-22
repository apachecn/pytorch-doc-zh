#!/bin/bash

sum=0
while :
do
    echo "开始拉取最新代码"
    # git remote add origin_online https://github.com/apachecn/pytorch-doc-zh.git
    git pull origin_online master
    git push origin main
    sum=`expr $sum + 1`
    # 1个小时
    echo $(date "+%Y-%m-%d %H:%M:%S") " --- 更新次数: $sum" >> logs/info.log
    sleep 10
done

echo "$(date) The sum is: $sum"
