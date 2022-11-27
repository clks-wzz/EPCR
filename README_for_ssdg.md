1. 数据准备
    a. 具体可见以下脚本分别准备4个数据集:
        scripts/get_list_test_OCIM_casia.py
        scripts/get_list_test_OCIM_msu.py
        scripts/get_list_test_OCIM_oulu.py
        scripts/get_list_test_OCIM_replayattack.py

        生成的数据list每一行是:  "img_path bbox_path label\n"
        bbox_path里存放的是bbox:  "
                                x
                                y
                                w
                                h
                                "
    b. 然后将生成的8个txt list(每个数据集一个train list，一个test list) 放入下面的脚本中运行:
        scripts/get_list_all_OCIM_SSL_extra.py
    注意事项： 须将输入目录和输出目录替换为当前服务器拥有的目录

2. 训练+测试
    a. 运行 configs 下的脚本即可
    b. 训练+测试同步进行，评估结果文件放在 ${MODEL_DIR}/score_file.txt 下， 每个 
    c. 刚发现有三个脚本忘记添加--amp 选项了，在脚本里添加 "--amp \" 即可
    注意事项： 同(1)