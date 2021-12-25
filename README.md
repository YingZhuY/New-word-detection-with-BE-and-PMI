# New word detection with BE and PMI

## 环境

* CentOS Linux release 7.7.1908 (Core)
* Python 3.7.10
* numpy 1.20.1
* pygtrie 2.4.2


## 使用方式

* 将 [NWdetection](https://pan.baidu.com/s/16kzk5iWk5UwHywxFgFsAXQ)，提取码：mat9（from 百度云盘）下的文件，下载到代码同目录下，处理后的目录树内容如下

```shell
| 
|--- commonly_used_words.txt
|--- genCandidate.py
|--- random_select_aver.txt
|--- stopZi.txt
|--- WiKi_index.txt
|--- 白皮书 （可换为你的任意语料）
```

* `python pipeline [--name VALUE]` 即可自动运行全部代码，得到新词
  - 例如：`python genCandidate.py --txt-directory ./白皮书 --topK 0.4 --min-pmi 6 --min-entropy 2 --BE-stop --wiki --Re-pattern --delCommon --restore-score`

* 参数说明如下

```shell
$ python genCandidate.py -h
usage: genCandidate.py [-h] [--txt-file TXT_FILE]
                       [--txt-directory TXT_DIRECTORY] [--BE-stop] [--wiki]  
                       [--Re-pattern] [--selectN SELECTN] [--delCommon]      
                       [--min-n MIN_N] [--max-n MAX_N] [--min-freq MIN_FREQ] 
                       [--min-pmi MIN_PMI] [--min-entropy MIN_ENTROPY] --topK
                       TOP_K [--restore-score] [--log-dir LOG_DIR]
                       [--task-name TASK_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --txt-file TXT_FILE   Path to your txt corpus.                                # 指定语料文件路径
  --txt-directory TXT_DIRECTORY                                                 # 指定语料目录路径
                        If you need to process two or more txt files, you can
                        put them in the same directory. Give the directory   
                        here.
  --BE-stop             Filter with BE-stop mode.                               # Way 1 首尾字处理模式
  --wiki                Filter with wiki mode.                                  # Way 2 WiKi index 对应词提权
  --Re-pattern          Filter with Re-pattern mode.                            # Way 3 正则模式过滤
  --selectN SELECTN     Top N to select in the Re-pattern mode (without PUNC)   # 指定正则模式数
  --delCommon           Filter with Re-pattern mode.                            # 去掉通用词
  --min-n MIN_N         The min n of n-gram to extract. Default 2.              # 最小 n-gram 长度
  --max-n MAX_N         The max n of n-gram to extract. Default 6.              # 最大 n-gram 长度
  --min-freq MIN_FREQ   The frequency threshold. Default 5.                     # 词频阈值
  --min-pmi MIN_PMI     The PMI threshold. Default 4. You can define your own   # PMI 阈值
                        min-pmi.
  --min-entropy MIN_ENTROPY                                                     # BE 阈值
                        The Entropy threshold. Default 1. You can define your
                        own min-entropy.
  --topK TOP_K          Output the top k new words. 1. topK>1: the top k        # 控制返回词数
                        words; 2. topK<1: the top k% words; 3. topK=1: all
                        words
  --restore-score       Restore score to a json file.                           # 记录候选词的中间计算结果
  --log-dir LOG_DIR     Directory where to write logs / saved results           # 结果存放目录
  --task-name TASK_NAME                                                         # 任务名
                        Name for this task, you can change a comprehensive
                        one. The result file will be stored in this directory.
```


## 代码说明

算法实现了简单的基于互信息和信息熵的新词发现，同时加上了四个后处理方法

### 【BE-stop】 Way 1: 对首尾字的处理

* 对在 candidate n-gram 中, 首字或者尾字出现次数特别多的进行筛选, 如 "XX的,美丽的,漂亮的" 剔出字典

* 测试中发现这样也会去掉很多合理词，如 “法*”

  -> solve: 处理为对带筛选首尾字进行限制，要求其在停用词表内

> p.s. 停用词来自 [中文停用词](https://github.com/goto456/stopwords)。本程序选择 `cn_stopwords.txt`，取其单字项构成文件 `stopZi.txt`，共 237 项。

### 【wiki】 Way 2: WiKi index 对应词提权

* `WiKi_index.txt` 是从 `zhwiki-20210301-pages-articles-multistream-index.txt.bz2` 中提取、简化、预处理得到的，其每一项都为一个中文维基百科词条。获取方式参见 [GloVe_Chinese_word_embedding](https://github.com/YingZhuY/GloVe_Chinese_word_embedding)

* 维基百科词条是确定可以成词的，我们提高它对应的权重（这里设置为一个经验值 1.2），有助于区别脏词。

### 【Re-pattern】 Way 3: 正则模式过滤

* 使用 `random_select_aver.txt` 文件中的 pattern（取不含 PUNC 的 selectN）。

* selectN 不推荐设置很大，太消耗内存。

* 对匹配到的模式产生的词，如从 `从大分式开始dfdsf` 匹配到 `大分式`
  1. 若 `大分式` 不在 result 中，用子词 `大分式` 替换其父词 `从大分式开始dfdsf`
  2. 若 `大分式` 在 result 中，删除其父词

> p.s. 对于匹配到的子词，注意限制 >= min_n （上限不用判断，因为不可能越剪越长）

### 【delCommon】 Way 4: 去掉通用词

* `commonly_used_words.txt` 来源于 [现代汉语常用词表](https://gist.github.com/indiejoseph/eae09c673460aa0b56db)，从结果中去掉这些通用词，共 56064 项。


## 白皮书语料结果示例 (前 60)
```
最高人民法院
柯尔克孜
稀土
低碳
西藏自治区
公安机关
中国政府
融资租赁
拉美裔
广州市
第二
行政机关
有限公司
检察机关
广东省
伊拉克
深圳
司法解释
钓鱼岛
哈萨克
刑事案件
自贸区
语言文字
京津冀
充分发挥
交通运输
宗教信仰自由
有关部门
损害赔偿
中央政府
贫困人口
消费者
第三
上海市
注册商标
矿产资源
基础设施
环境污染
劳动教养
世界卫生组织
宽带
环境保护
各类
做好
劳动合同
股份有限公司
可持续发展
裁判文书
南沙群岛
网络安全
食品安全
典型案例
厦门海事法院
新中国成立
严厉打击
商业秘密
亚太
青藏高原
对赌协议
微博
```

## References

- [x] [D-TopWords](https://github.com/chenaoxd/dtopwords)
- [x] [现代汉语常用词表](https://gist.github.com/indiejoseph/eae09c673460aa0b56db)
