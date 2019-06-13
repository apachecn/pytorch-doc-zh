# coding: utf-8
import os
import sys


def format_index(filename):
    """
    将基础数据合并为图数据
    :return:
    """

    str1 = """
    </head>
    <body>
    """

    str2 = """
    <script type="text/javascript">
        function hidden_left(){
            document.getElementsByClassName("btn pull-left js-toolbar-action")[0].click()
        }
        // window.onload = hidden_left();
    </script>
    </head>
    <body onload="hidden_left()">
    """

    str3 = "powered by Gitbook"
    str4 = "powered by ApacheCN"
    with open(filename, 'r') as f:
        var_object = f.read()
        var_object = var_object.replace(str1, str2)
        var_object = var_object.replace(str3, str4)
        # print(var_object)

    f = open(filename, "w")
    f.write(var_object)


if __name__ == "__main__":
    version = sys.argv[1]
    if version == "home":
        filename = "_book/index.html"
    else:
        filename = "_book/docs/%s/index.html" % version
    format_index(filename)
