#!/bin/bash
loginfo() { echo "[INFO] $@"; }
logerror() { echo "[ERROR] $@" 1>&2; }

gitbook init
gitbook install
python3 src/rename_powered_by_ApacheCN.py "home"
gitbook build ./ _book
python3 src/script.py "home"

versions="0.2 0.3 0.4 1.0"
for version in $versions;do
    loginfo "==========================================================="
    loginfo "开始", ${version}, "版本编译"

    echo "gitbook init docs/${version}"
    gitbook init docs/${version}

    echo "python3 src/rename_powered_by_ApacheCN.py ${version}"
    python3 src/rename_powered_by_ApacheCN.py ${version}

    echo "cp book.json docs/${version}"
    cp book.json docs/${version}

    echo "gitbook install docs/${version}"
    gitbook install docs/${version}

    echo "gitbook build docs/${version} _book/docs/${version}"
    gitbook build docs/${version} _book/docs/${version}
    
    echo "python3 src/script.py ${version}"
    python3 src/script.py ${version}
done
