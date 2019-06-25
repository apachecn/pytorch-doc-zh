#!/bin/bash
loginfo() { echo "[INFO] $@"; }
logerror() { echo "[ERROR] $@" 1>&2; }

gitbook install
gitbook build ./ _book
# python3 src/script.py "home" "index"

versions="0.2 0.3 0.4 1.0"
for version in $versions;do
    loginfo "==========================================================="
    loginfo "开始", ${version}, "版本编译"

    echo "cp book.json docs/${version}"
    cp book.json docs/${version}

    # 替换 book.json 的编辑地址
    echo "python3 src/script.py ${version} book"
    python3 src/script.py ${version} "book"

    echo "gitbook install docs/${version}"
    gitbook install docs/${version}

    echo "gitbook build docs/${version} _book/docs/${version}"
    gitbook build docs/${version} _book/docs/${version}

    # 注释多余的内容
    # echo "python3 src/script.py ${version} index"
    # python3 src/script.py ${version} "index"
done

# rm -rf /opt/apache-tomcat-9.0.17/webapps/test_book
# cp -r _book /opt/apache-tomcat-9.0.17/webapps/test_book
