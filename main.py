import os
import re
import shutil
import math  
import datetime
import pickle
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import nltk
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

import gensim
from gensim.models import word2vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision import transforms, utils
#from pattern.es import lemma
t = datetime.datetime

#===== 資料年份參數 ==============================================================
begin_year          = 2006
end_year            = 2016
end_year           += 1      # range(begin, end)
#===== 資料庫相關參數 ============================================================
HOST                = 'ip'
PORT                =  27017
USERNAME            = "*"
PASSWORD            = "*"
DATABASE_NAME       = 'USPTO'
COLLECTION_NAME     = 'grant'
#===== TF-IDF 相關參數 ===========================================================
TFIDF_FEATURES      = 625    # 特徵（＝詞）數
TFIDF_MIN_DF        = 1      # 出現次數小於這個數字的詞會被捨棄
TFIDF_NGRAM         = (1,2)  # n-gram 產生連字詞的字數範圍 (1,3) * 已棄用
TFIDF_PER_KEYWORDS  = 5      # 每篇文章要留存的關鍵字數
#===== Word2Vec 相關參數 =========================================================
W2V_SG              = 1      # 模型選擇： 0 = CBOW ； 1 = skip-gram
W2V_FEATURES        = 625    # 特徵向量的維度，大約 n0~n00
W2V_WORKERS         = 11     # 訓練並行數，按 CPU 數量決定 
W2V_MIN_COUNT       = 5      # 詞頻小於這個數字的詞會被捨棄
W2V_WINDOW          = 5      # 當前詞和預測詞的最大距離
W2V_NEGATIVE        = 5      # 使用負採樣
W2V_BATCH_WORDS     = 10000  # 每條執行緒收到的訓練詞數量
#===== 卷積相關參數 ================================================================
batch_size          = 305    # 分批時的 單批圖像數量
learning_rate       = 1e-3   # 學習率
num_epoches         = 1      # 迭代次數
train_rate          = 0.9    # 訓練集的比例
#===== 資料儲存路徑 ===============================================================
PATENTS_PATH        = 'patents/'                    # 專利文件
PATENTS_ORI_PATH    = 'cita/patent_ori.csv'         # 專利創意度
TOKEN_PATH          = 'tokens.txt'                  # 已斷詞的詞集
STOPWORDS_PATH      = 'stopwords.txt'               # 停用詞
#===== 測試資料路徑 ==============================================================
EVAL_PATENT_PATH    = PATENTS_PATH + str(begin_year) + '/1.txt'
#================================================================================

def main():
    '''
     工作日誌：
      [ ] 5. 混淆矩陣
    '''
    
    # 確認原始專利資料
    is_Patent_Ready = os.path.isfile(EVAL_PATENT_PATH)
    if is_Patent_Ready:
        print('[%s] 已確認 專利資料集 存在' %t.now())
    else:
        print('[%s] 未偵測到專利資料，開始連線下載' %t.now())
        for year in range(begin_year, end_year):
            fetch_patent(year)
        print('[%s] 資料已下載完畢' %t.now())
        
    # 先進行前處理
    # 如果不先前處理而讓 TF-IDF 和 W2V 分別讀取的話，會因為 Bi-gram 導致字典表對不上
    patents = load_patents()
    token = []
    is_token_Ready = os.path.isfile(TOKEN_PATH)
    if is_token_Ready:
        print('[%s] 已確認 斷詞資料 存在' %t.now())
        token = load_token()
    else:
        print('[%s] 未偵測到斷詞資料，嘗試呼叫前處理方法' %t.now())
        token = Preprocessing(patents)
        
    # 計算創意度
    # 由於計算過程比較長，先外包到 cita 資料夾
    
    # 專利資料分類
    print('[%s] 已確認創意度紀錄存在，開始進行專利分類' %t.now())
    labels = read_labels()
        
    # 去除無效分類以及保留極端值
    data, labels = keep_extreme(token, labels)
    
    # 先轉類型之後比較方便
    data = np.array(data)
    labels = np.array(labels)
        
    # 十分法
    skf = StratifiedKFold(n_splits = 10 , shuffle = True)
    
    round_num = 1
    for train_index, test_index in skf.split(data, labels):
        print('============================ 第 %d 輪 ============================' %round_num)
        train_data  = data[train_index]
        train_label = labels[train_index]
        test_data   = data[test_index]
        test_label  = labels[test_index]
        
        print('[%s] 測試集： %5d　　　訓練集： %5d' %(t.now(), len(train_label), len(test_label)))
        print('[%s] ========== 開始處理訓練集 ==========' %(t.now()))
        
        # TF-IDF Model
        # gensim 的 TF-IDF 無法處理未訓練過的新字，和檢測創新度的精神衝突，故在此處選用 sklearn
        tfidf_vec, tfidf_matrix = train_tfidf(train_data)
        
        # 訓練 Word2Vec
        # 大多有 直接使用現有語料庫訓練 跟 使用更大的（如維基）語料庫訓練的辦法
        # 由於之後的處理大多都在這些專利文檔本身 故現在先使用文檔本身進行訓練
        W2V_model = training_Word2vec(train_data)
        
        # 用 TF-IDF 將每篇文檔的關鍵字先存放起來
        train_keywords = count_keywords(tfidf_vec, tfidf_matrix, TFIDF_PER_KEYWORDS)
        
        # 將每篇文章轉換成向量表
        train_key_vec = keywords_to_vec(train_keywords, W2V_model)
        train_avg_vec = patents_to_avg_vec(train_data, W2V_model)
        train_wei_vec = patents_to_weighted_vec(train_data, W2V_model, tfidf_vec, tfidf_matrix)

        print('[%s] ========== 開始處理測試集 ==========' %(t.now()))
        
        # 訓練 TF-IDF 測試集的矩陣
        test_tfidf_matrix = tfidf_vec.transform(test_data)
        
        # 取出測試集的關鍵字
        test_keywords  = count_keywords(tfidf_vec, test_tfidf_matrix, TFIDF_PER_KEYWORDS)
        
        # 轉換成向量
        test_key_vec = keywords_to_vec(test_keywords, W2V_model)
        test_avg_vec = patents_to_avg_vec(test_data, W2V_model)
        test_wei_vec = patents_to_weighted_vec(test_data, W2V_model, tfidf_vec, tfidf_matrix)
        
        print('[%s] ========== 開始準備訓練 ============' %(t.now()))
        
        train_cnn(train_wei_vec, train_label, test_wei_vec, test_label)
        train_xgb(train_wei_vec, train_label, test_wei_vec, test_label)
        
        round_num += 1
        
    print('[%s] 所有處理已執行完畢' %t.now())

#================================================================================
#        取得專利資料
#================================================================================

def fetch_patent(year):
    '''
    從實驗室的 MongoDB 取得指定年份的資料
     @param year: 欲搜尋資料的年份
     @save:   專利文檔，按年份分別儲存
     @global: 資料庫相關參數
    '''
    collection_year_name = COLLECTION_NAME + str(year)

    print("[%s] 已準備連線" %t.now())
    print("　目標主機："   + HOST + ":" + str(PORT) +\
          "\n　登入身分：" + USERNAME +\
          "\n　資料庫　：" + DATABASE_NAME +\
          "\n　資料表　：" + collection_year_name)
    
    # 開始連線
    client = pymongo.MongoClient(host = HOST, port = PORT)
    client.admin.authenticate(USERNAME, PASSWORD)
    database = client[DATABASE_NAME]
    collection = database[collection_year_name]
    print("[%s] 連線成功" %t.now())
    
    # 查詢
    print('[%s] 開始查詢資料' %t.now())
    # 7 軟體
    # {'main_classification':{'$regex':r"^7"}}
    # 706 AI
    # {'$or':[{'USPCs.main_classification':{'$regex':r"^706"}},{'main_classification':{'$regex':r"^706"}}]}
    results = collection.find({'$or':[{'USPCs.main_classification':{'$regex':r"^706"}},{'main_classification':{'$regex':r"^706"}}]}) # 700xxx 軟體類
    data_count = results.collection.count_documents({'$or':[{'USPCs.main_classification':{'$regex':"^706"}},{'main_classification':{'$regex':"^706"}}]})
    print('[%s] 回傳筆數：%s'%(t.now(), str(data_count)))
    
    if os.path.isfile(PATENTS_CITA_PATH):
        os.remove(PATENTS_CITA_PATH) # 清空專利列表

    i = 1
    for result in results:
        result = dict(result)
        #desc = result["description"]
        desc = result['claims'][0]['text']
        desc = desc.replace('\n',' ')
        
        print("[%s] 資料處理中： %6d / %6d " %(t.now(), i, data_count), end='\r')

        # 儲存資料
        with open(PATENTS_PATH + str(year) + '/' + str(i) + '.txt','w', encoding='utf-8') as f:
            f.write(desc)
            
        patent_num = result["patent_num"]
        patent_cite = result["citation"]
        
        # 儲存專利編號，之後要寫引用次數統計時使用，要注意這邊的操作是基於以下兩點：
        # 一、不直接把專利編號存成檔名是因為到時候向量和關鍵字也是直接另存成同一個檔案
        # 　　不如直接編號表編下去，還可以把其他資料分開寫比較有彈性
        # 二、邊連線邊算創新度之類的太耗時了，很容易連線逾時斷線
        with open(PATENTS_NUM_PATH ,'a', encoding='utf-8') as f:
            f.write(str(year) + ',' + str(i) + ',' + str(patent_num)  + '\n')
            
        i = i + 1

    print('\n[%s] %s 已讀取完畢' %(t.now(), collection_year_name))

#================================================================================

def load_patents():
    '''
    直接讀取原始的專利資料，將預處理的部分交給之後的 TF-IDF 模型時使用
     @return: ['專利1','專利2']
     @filesource: PATENTS_PATH
    '''
    corpus = []
    for year in range(begin_year, end_year):
        for patent in os.listdir(PATENTS_PATH + str(year) + '/'):
            if not patent.startswith('.') and not patent.endswith('.index'):
                with open(PATENTS_PATH + str(year) + '/' + patent, 'r') as f:
                    for line in f:
                        corpus.append(line)
    
    print('[%s] 原始專利資料已載入： %s' %(t.now(), PATENTS_PATH))
    return corpus

#================================================================================
#        資料前處理
#================================================================================

def Preprocessing(docs):
    '''
    將一串文字進行前處理，包括去數字、去符號、停用詞、斷詞、短語識別及詞性還原。
     @param docs: ['文本1','文本2'...]
     @return: 經過斷詞和詞性還原的詞串列，如 [['詞1-1','詞1-2'],['詞2-1','詞2-2']]
     @save: 儲存斷詞結果
     @method: count_tokenize, count_phrases, count_lemmatize
    '''
    
    tokens_list = count_tokenize(docs) # 斷詞
    tokens_list = remove_stopwords(tokens_list, STOPWORDS_PATH) # 去停用詞
    tokens_list = count_lemmatize(tokens_list) # 詞性標註
    #tokens_list = count_phrases(tokens_list) # 連接短語
    
    # 把斷詞結果儲存起來
    with open(TOKEN_PATH, 'a') as f:
        i = 0
        for tokens in tokens_list:
            f.write(','.join(tokens))
            f.write('\n')
            i += 1
            print('[%s] 正在儲存： %6d / %6d' %(t.now(), i, len(tokens_list)), end = '\r')
    print('\n[%s] 資料已完成前處理' %t.now())
            
    return tokens_list

#================================================================================

def count_tokenize(docs):
    '''
    斷詞器，包括去數字、去符號、斷詞的部分。之後如果要加入其他的斷詞方法（如斯坦福）只需要修改這一部分即可。
     @param docs: ['文本1','文本2'...]
     @return: 斷詞後的詞串列，如 [['詞1-1','詞1-2'],['詞2-1','詞2-2']]
    '''
    
    i = 0
    tokens_list = []
    for desc in docs:
        # 重複的部分就加減做當保險，反正不差幾秒
        desc = desc.lower() # 轉成小寫
        desc = ''.join([i for i in desc if not i.isdigit()])  # 去除數字
        desc = re.sub(r'\W', ' ', desc) # 去除符號
        desc = re.sub(r'[^\x00-\x7F]+',' ', desc) # 去除非 ASCII 字元
        desc = re.sub('_','', desc) # 去除邪惡底線
        tokens = nltk.word_tokenize(desc) # 斷詞
        tokens_list.append(tokens)
        i += 1
        print('[%s] 正在進行斷詞： %6d / %6d' %(t.now(), i, len(docs)), end = '\r')
    print('\n[%s] 斷詞已完成' %t.now())
    
    return tokens_list

#================================================================================

def remove_stopwords(tokens_list, stopwords_path):
    '''
    讀取停用詞列表，並去除停用詞
     @filesource: stopwords_path，停用詞字典檔
     @return: 去除停用詞後的詞串列
    '''
    
    stopwords = nltk.corpus.stopwords.words('english') 
    with open(stopwords_path,'r') as f:
        for line in f:
            for word in line.split(','):
                stopwords.append(word)
    
    removed_token_list = []
    for tokens in tokens_list:
        tokens = [w for w in tokens if w not in stopwords] # 去除停用詞
        removed_token_list.append(tokens)
        print('[%s] 正在去除停用詞： %6d / %6d' %(t.now(), i, len(tokens_list)), end = '\r')
    print('[%s] 停用詞已去除完畢' %t.now())
        
    return removed_token_list
        
#================================================================================

def count_phrases(tokens_list):
    '''
    用 Bi-gram 來判斷連續詞（＝短語），例如 new-york 或是 data-set
    儘管 sklearn 的 CountVectorizer 寫法比較直覺，效果也比較好，但會直接轉成向量
    因此為了保留原本的詞庫，方便之後可以分別跑 TF-IDF 和 Word2Vec 等不同路線時詞庫一致，因此使用 Phrases
     @param tokens_list: 斷詞後的詞串列，如  [['詞1-1','詞1-2'],['詞2-1','詞2-2']]
     @return: 銜接短語後的詞串列，如 [['詞1-1','詞1-2'],['詞2-1_詞2-2']]
     @note: 口試後已停用
    '''
    i = 0
    bigram_tokens = []
    bigram = Phrases(tokens_list, min_count=1, threshold=2)
    bigram_phraser = Phraser(bigram)
    for tokens in tokens_list:
        bigram_tokens.append(bigram_phraser[tokens])
        i += 1
        print('[%s] 正在篩選短語： %6d / %6d' %(t.now(), i, len(tokens_list)), end = '\r')
    print('[%s] 短語已辨識完畢' %t.now())
    
    return bigram_tokens

#================================================================================

def count_lemmatize(tokens_list):
    '''
    使用 NLTK 的 lemmatizer 來實作詞性還原。原本使用較為簡單的 Lemma 但發生無法挽回的錯誤，故已廢除。
     @param tokens_list: 斷詞後的詞串列，如  [['詞1-1','詞1-2'],['詞2-1','詞2-2']]
     @return: 篩選過詞性的詞串列，如 [['詞1-1','詞1-2'],['詞2-1']]
     @method: get_wordnet_pos，單純用於縮短詞性的格式轉換部分
     @example: http://www.zmonster.me/2016/01/21/lemmatization-survey.html
    
     @note: 無法解決 Lemma 造成的 StopIteration 問題，可能是 Lemma 的函數有更改或尚不支援 3.7
            包括 Gensim 的詞性轉換也無法使用（因為他們也是基於 Lemma 來做的）
            相關問題請參見：https://github.com/RaRe-Technologies/gensim/issues/2438
    
     @note: 6.19 更新警告：絕對不要嘗試用 conda 安裝 Lemma ，會讓 python 從 3.7 降到 2.x 並引起*災難*！！
            這邊直接使用 NLTK 的詞性分析作取代
    '''
    i = 0
    lemmated_token_list = []
    lemmatizer = WordNetLemmatizer() # 詞性還原器
    for tokens in tokens_list:
        tokens = nltk.pos_tag(tokens) # 詞性標註
        pos_tokens = []
        for word, pos in tokens:
            if len(word) > 2: # 去掉一堆短垃圾，像是 x, y, w, Zi
                wordnet_pos = get_wordnet_pos(pos)
                if wordnet_pos is not None:
                    lemma_word = lemmatizer.lemmatize(word, pos=wordnet_pos) # 詞性還原
                    pos_tokens.append(lemma_word)
        lemmated_token_list.append(pos_tokens)
        i += 1
        print('[%s] 正在進行詞性還原： %6d / %6d' %(t.now(), i, len(tokens_list)), end = '\r')
    print('\n[%s] 詞性還原已完成' %t.now())
    
    return lemmated_token_list

#================================================================================

def get_wordnet_pos(treebank_tag):
    '''
    幫忙將得到的詞性標註轉換成 WordNet 的格式，方便之後作詞型還原
     @example: http://www.zmonster.me/2016/01/21/lemmatization-survey.html
    '''
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    #elif treebank_tag.startswith('J'):
    #    return wordnet.ADJ
    #elif treebank_tag.startswith('R'):
    #    return wordnet.ADV
    else:
        return None

#================================================================================

def load_token():
    '''
    讀取斷好詞的資料，該資料每當有斷詞的時候就會附加在 TOKEN 存放的文件裡
     @return: [['專利1的詞1','專利1的詞2']['專利2的詞1','專利2的詞2']]
     @filesource: TOKEN_PATH
    '''
    token = []
    with open(TOKEN_PATH, 'r') as f:
        for line in f:
            token.append(line.split(','))
    
    print('[%s] 斷詞資料已載入：%s' %(t.now(), TOKEN_PATH))
    return token

#================================================================================
#        TF - IDF
#================================================================================

def train_tfidf(corpus):
    '''
    利用 sklearn 的方法來做 TF-IDF。
    它實際會幫忙我們做一次前處理（如轉小寫和斷詞、停用詞等），再轉成 TF-IDF 的模型。
    但我們的資料是已經斷好詞的資料，因此這邊使用假斷詞器直接繞過去
     @param corpus: 文檔字串組成的串列，如 [['Double','A'],['my','god!']]
     @return: TF-IDF Model
     @save:   TF-IDF Model
     @global: TF-IDF 相關參數
     @method: load_stopwords, my_Tokenizer
     @example: https://www.zybuluo.com/lianjizhe/note/1212780
     @example: https://codeday.me/bug/20180902/241475.html
    '''
    
    print('[%s] 開始訓練 TF-IDF' %(t.now()))
    
    # TF-IDF 模型產生器
    # 警告：添加斷詞器之後，停用詞和正規式等部分似乎已經失效
    tfidf_vec = TfidfVectorizer(max_features = TFIDF_FEATURES,
                                min_df = TFIDF_MIN_DF,
                                analyzer = 'word',
                                preprocessor = fake_tokenizer,
                                tokenizer = fake_tokenizer,
                                lowercase = False)           
    
    tfidf_matrix = tfidf_vec.fit_transform(corpus) # 訓練 TF-IDF
    print('[%s] TF_IDF 模型已訓練完畢' %t.now())
    
    # 檢查用法
    # print(tfidf_vec.get_feature_names()) # 得到語料庫所有不重複的詞
    # print(tfidf_vec.vocabulary_) # 得到每個單詞對應的 ID  
    # print(tfidf_matrix.toarray()) # 得到每個詞的向量，向量的順序對應詞的 ID
    
    # 排序處理
    # feature_array = np.array(tfidf_vec.get_feature_names()) # 將語料庫所有詞（＝特徵）做成矩陣
    # tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1] # 將重要度向量進行排序（易當）
    # n = 5
    # top_n = feature_array[tfidf_sorting][:n] # 取出前 n 個詞
    
    return tfidf_vec, tfidf_matrix

#================================================================================

def fake_tokenizer(corpus):
    '''
    假的斷詞器，用來跳過 TF-IDF 的斷詞步驟
    '''
    return corpus
#================================================================================

def count_keywords(tfidf_vec, tfidf_matrix, keyword_count):
    '''
    將每一篇文章的前Ｎ個關鍵詞存放起來
     @param tfidf_vec: 訓練好的 TF-IDF 向量
     @param tfidf_matrix: 訓練好的 TF-IDF 的權重矩陣
     @param keyword_count: 每一篇要儲存的關鍵字數量
     @return: 每篇文章的關鍵字，如 [[文章1關鍵字1, 文章1關鍵字2],[文章2關鍵字1,文章2關鍵字2]]
     @save: 每篇文章中，TF-IDF 分數前Ｎ高的詞
    '''
    
    word = tfidf_vec.get_feature_names() # 用 TF-IDF 取出每篇文章的關鍵字
    weight = tfidf_matrix.toarray() # 將向量轉成陣列方便使用（易當）
    
    all_keywords = []
    for w in weight:
        loc = np.argsort(-w) # 排序
        keywords = np.array(word)[loc[:keyword_count]] # 轉列表，拿前五高向量回去對字典表拿詞
        all_keywords.append(keywords)
            
    print('[%s] TF-IDF 關鍵字已計算完畢' %(t.now()))
    
    return all_keywords

#================================================================================
#        Word2Vec
#================================================================================

def training_Word2vec(sentences):
    '''
    訓練 Word2Vec 模型
     @param sentences: 斷好詞的文檔串列所組成的全文檔串列，如 [['Double','A'],['my','god!']]
     @save: Word2Vec Model
     @global: W2V 相關參數
     @example: https://www.jianshu.com/p/0425bfe619c3
     @example: https://www.jianshu.com/p/52ee8c5739b6
     @example: https://blog.csdn.net/qq_19707521/article/details/79169826
     @see 負採樣: https://python5566.wordpress.com/2018/03/17/nlp-筆記-negative-sampling/
    '''
    
    print('[%s] 開始訓練 Word2Vec' %(t.now()))
    model = word2vec.Word2Vec(sentences,
                              sg = W2V_SG,
                              workers = W2V_WORKERS,
                              size = W2V_FEATURES,
                              min_count = W2V_MIN_COUNT,
                              window = W2V_WINDOW,
                              batch_words = W2V_BATCH_WORDS,
                              negative = W2V_NEGATIVE,
                              sorted_vocab = 1)
    
    model.init_sims(replace=True)
    print('[%s] Word2Vec 模型已訓練完畢' %t.now())
    
    return model

#================================================================================
#        向量化
#================================================================================

def keywords_to_vec(keywords, W2V_model):
    '''
    將關鍵字列表轉換成 Word2Vec 向量
     @param keywords: 關鍵字列表，如 [[文章1關鍵字1, 文章1關鍵字2],[文章2關鍵字1,文章2關鍵字2]]
     @param W2V_model: 訓練好的 Word2Vec 模型
     @return: 每篇文章關鍵字的向量，如 [[[向量1-1], [向量1-2]],[[向量2-1],[向量2-2]]]
    '''
    #print('[%s] 開始計算關鍵字向量' %t.now())
    
    all_doc_keywords_vec = []
    doc_num = 0
    doc_len = len(keywords)
    for doc in keywords:
        print("[%s] 關鍵字向量轉換中： %6d / %6d " %(t.now(), doc_num+1, doc_len), end='\r')
        word_vec_list = []
        for word in doc:
            word_vec_list.append(W2V_model.wv[word])
        
        vec = np.mean(word_vec_list, axis=0)
        all_doc_keywords_vec.append(vec)
        doc_num += 1
    
    print('\n[%s] 關鍵字已成功轉換為向量' %(t.now()))
    return all_doc_keywords_vec

#================================================================================

def patents_to_avg_vec(tokens_list, W2V_model):
    '''
    將文件列表轉換成 Word2Vec 平均向量
     @param tokens_list: 斷好詞的文件列表，如 [[文章1字1, 文章1字2],[文章2字1,文章2字2]]
     @param W2V_model: 訓練好的 Word2Vec 模型
     @param path: 向量副本存放的路徑位置
     @return: 每篇文章的平均向量，如 [[向量1],[向量2]]
    '''
    #print('[%s] 開始計算平均向量' %t.now())
    
    all_avg_vec = []
    doc_num = 0
    doc_len = len(tokens_list)
    for doc in tokens_list:
        print("[%s] 平均向量轉換中： %6d / %6d " %(t.now(), doc_num+1, doc_len), end='\r')
        doc = [word for word in doc if word in W2V_model.wv.vocab] # 檢查一下有沒有在向量裡，有可能丟了
        vec = np.mean(W2V_model.wv[doc], axis=0)
        all_avg_vec.append(vec)
        doc_num += 1
    
    print('\n[%s] 文件已成功轉換為平均向量' %(t.now()))
    return all_avg_vec

#================================================================================

def patents_to_weighted_vec(tokens_list, W2V_model, tfidf_vec, tfidf_matrix):
    '''
    將文件列表轉換成 Word2Vec 和 TF-IDF 的 加權平均向量
     @param tokens_list: 斷好詞的文件列表，如 [[文章1字1, 文章1字2],[文章2字1,文章2字2]]
     @param W2V_model: 訓練好的 Word2Vec 模型
     @param tfidf_vec: 訓練好的 TF-IDF 向量
     @param tfidf_matrix: 訓練好的 TF-IDF 的權重矩陣
     @return: 每篇文章的平均加權向量，如 [[向量1],[向量2]]
    '''
    #print('[%s] 開始計算加權向量' %t.now())
    
    all_wei_vec = []
    tfidf_words = tfidf_vec.get_feature_names() # TF-IDF 所有字的列表
    tfidf_dict  = tfidf_vec.vocabulary_ # TF-IDF 每個字對應的 ID
    tfidf_weight = tfidf_matrix.toarray() # 將向量轉成陣列方便使用（容易當機）
    
    doc_num = 0
    doc_len = len(tokens_list)
    for doc in tokens_list:
        print("[%s] 平均加權向量轉換中： %6d / %6d " %(t.now(), doc_num+1, doc_len), end='\r')
        # 確認兩邊都存在的詞
        doc = [word for word in doc if word in W2V_model.wv.vocab and word in tfidf_words]
        # 將 TF-IDF 乘上 Word2Vec
        doc = [tfidf_weight[tfidf_dict[word]].T * W2V_model.wv[word] for word in doc]
        # 平均所有字
        vec = np.mean(np.array(doc), axis=0).T
        vec = np.asarray(vec) 
        all_wei_vec.append(vec)
        doc_num += 1
    
    print('\n[%s] 文件已成功轉換為平均加權向量' %(t.now()))
    return all_wei_vec

#================================================================================
#        創意度
#================================================================================

def read_labels(path='cita/patent_ori_old.csv'):
    '''
    讀取專利標籤列表
     @param path: 紀錄原創性評分的檔案
     @return: 標籤列表
    '''
    names = ['index', 'number', 'year', 'ori']
    data = pd.read_csv(path, names = names)
    data_count = len(data)
    
    labels = []
    for i in range(data_count):
        txt = str(i) + '.txt'
        ori = data.iloc[i]['ori']
        
        level = -1
        if ori >= 0.7561:
            level = 1
        elif ori <= 0.4444:
            level = 0
        
        labels.append(level)
        print("[%s] 專利歸類中： %6d / %6d" %(t.now(), (i + 1), data_count), end = '\r')
    print('\n[%s] 專利歸類已完成' %t.now())
    
    return labels

#================================================================================

def keep_extreme(data, labels):
    '''
    將極端值留下，去除未被分類的資料
     @param data: 資料串列
     @param labels: 標籤串列
     @return: 篩選後的資料與標籤
    '''
    data_count = len(labels)
    i = 0
    
    new_data  = []
    new_labels = []
    for datum, label in zip(data, labels):
        if label is not -1:
            new_data.append(datum)
            new_labels.append(label)
            
        i += 1
        print("[%s] 去除無效分類的專利： %6d / %6d" %(t.now(), (i + 1), data_count), end = '\r')
    print('\n[%s] 去除已完成，目前資料總數： %d' %(t.now(), len(new_labels)))
    
    return new_data, new_labels

#================================================================================
#        CNN
#================================================================================
# @example: https://ithelp.ithome.com.tw/articles/10187424
# @example: https://www.pytorchtutorial.com/10-minute-pytorch-4/

# 開始搭建模型之前，需要先了解每一種層的定義
# Conv2d = 卷積層
# 參數：https://blog.csdn.net/qq_36810544/article/details/78786243
# 此處使用的參數依序是：輸入的資料厚度、輸出的資料厚度、卷積核的大小、卷積核的步長、周圍空值填充的方式

# MaxPool2d = 最大池化層
# 參數：https://pytorch.org/docs/stable/nn.html#maxpool2d
# 此處使用的參數是：卷積核的大小，卷積核的步長

# ReLU = 棒棒激活函數，大家的好朋友

# 先定義好 CNN 的內容架構，記得要繼承 NN 類
# 內容必須包含 __init__（進來此類時就執行，也就是向前傳遞）以及 forward（向後傳遞，調權重）兩個方法

class Cnn(nn.Module):
    
    # 向前傳播部分，到時候會傳入兩個值：dim = 指定的維度，class = 分成幾類
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        
        # Sequential 會直接將每一層按照順序加到模型中
        # 請參見：https://www.cnblogs.com/denny402/p/7593301.html
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_dim,
                      out_channels = 25,
                      kernel_size = 3,
                      stride = 1, 
                      padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(25, 25, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        
        # 全連接層的輸入是 圖片寬 * 圖片高 * 塊數量 * 上層輸出
        self.fc = nn.Sequential(
            nn.Linear(900, 4),
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        out = out.view(out.size(0), -1) #item?
        out = self.fc(out)
        return out

#================================================================================

def train_cnn(train_vec, train_label, test_vec, test_label, try_gpu = False):
    '''
    訓練卷積神經網路，網路構造於 class Cnn 中
     @param train_vec, train_label: 訓練集的資料和標籤
     @param test_vec, test_label: 訓練集的資料和標籤
     @return: CNN 分類後的準確度
     @example: https://ithelp.ithome.com.tw/articles/10187424
     @example: https://www.pytorchtutorial.com/10-minute-pytorch-4/
    '''
        
    # 轉換成張量、轉換成資料集
    train_tensor_data = torch.stack([torch.Tensor(i) for i in train_vec])
    train_tensor_data = train_tensor_data.reshape(len(train_vec), 1, 25, 25)
    train_tensor_label = torch.LongTensor(train_label)
    train_dataset = TensorDataset(train_tensor_data, train_tensor_label)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 
    print('[%s] 訓練集已轉換為張量' %t.now())

    test_tensor_data = torch.stack([torch.Tensor(i) for i in test_vec])
    test_tensor_data = test_tensor_data.reshape(len(test_vec), 1, 25, 25)
    test_tensor_label = torch.LongTensor(test_label)
    test_dataset = TensorDataset(test_tensor_data, test_tensor_label)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True) 
    print('[%s] 測試集已轉換為張量' %t.now())
    
    print('[%s] 開始訓練 CNN' %t.now())
    model = Cnn(1, 4)

    # 如果能用 GPU 跑的話就移到 GPU 上面跑
    use_gpu = torch.cuda.is_available()
    if try_gpu and use_gpu:
        model = model.cuda()

    # 定義優化器和損失函數
    # @see: https://reurl.cc/8EWab
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    final_accuracy = 0
    for epoch in range(num_epoches):
        #print('迭代次數： {}'.format(epoch + 1))
        running_loss = 0.0 # 測試集的遺失率
        running_acc = 0.0  # 測試集的準確率

        # 開始訓練，把訓練集各塊輪流丟進去，另外記得從 1 開始
        for i, (img, label) in enumerate(train_loader, 1):
            # 如果能使用 GPU 就把圖和標籤丟過去
            if try_gpu and use_gpu:
                img = img.cuda()
                label = label.cuda()

            # 將圖和標籤轉換成變量，讓他們能按照整個流程即時變更
            # @example: https://morvanzhou.github.io/tutorials/machine-learning/torch/2-02-variable/
            # @example: https://zhuanlan.zhihu.com/p/34298983
            img = Variable(img)
            label = Variable(label)
            
            # 向前傳播
            out = model(img)                             # 執行模型
            loss = criterion(out, label)                 # 計算本輪數值
            running_loss += loss.item() * label.size(0)  # 累加遺失率
            _, pred = torch.max(out, 1)                  # 按梯度返回最大值，也就是分類時機率最高的
            num_correct = (pred == label).sum()          # 用各標籤來計算總準確度
            accuracy = (pred == label).float().mean()    # 平均準確度
            running_acc += num_correct.item()            # 累加準確度

            # 向後傳播
            optimizer.zero_grad()                        # 梯度歸零
            loss.backward()                              # 執行向後傳播，計算新梯度
            optimizer.step()                             # 更新參數

            # 列印本輪結果
            #print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            #    epoch + 1, num_epoches, running_loss / (batch_size * i),
            #    running_acc / (batch_size * i)))

            #with open('Result/avg_train.csv','a') as f:
            #    f.write(str(running_loss / (batch_size * i)) + ',' + 
            #            str(running_acc / (batch_size * i)) + '\n')

        # 列印訓練結果
        #print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        #    epoch + 1, running_loss / (len(test_dataset)), running_acc / (len(
        #        test_dataset))))
        # 完成訓練

        model.eval()  # 把參數固定下來，也就是使用訓練好的模型

        eval_loss = 0 # 訓練集的遺失率
        eval_acc = 0  # 訓練集的準確度

        # 開始測試
        for data in test_loader:
            img, label = data
            
            # 選擇使用 GPU 還是 CPU，並設為變量
            if try_gpu and use_gpu:
                img = Variable(img).cuda() # volatile=True
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            
            out = model(img)                          # 執行模型
            loss = criterion(out, label)              # 計算數值
            eval_loss += loss.item() * label.size(0)  # 累加遺失率
            _, pred = torch.max(out, 1)               # 按梯度返回最大值，也就是分類時機率最高的
            num_correct = (pred == label).sum()       # 總和各類的準確率
            eval_acc += num_correct.item()            # 累加準確度
            
            #print('[%s] CNN 訓練遺失率：%.6f　準確度：%.6f' %(t.now(),
            #                                  eval_loss / (len(test_dataset)),
            #                                  eval_acc / (len(test_dataset))))
            final_accuracy = eval_acc / (len(test_dataset))
    
    print('[%s] CNN 遺失率：%.6f　準確度：%.6f' %(t.now(), final_accuracy))
    return final_accuracy
            
#================================================================================
#        XGBoost
#================================================================================

def train_xgb(train_vec, train_label, test_vec, test_label):
    '''
    訓練 XGBoost 模型
     @param train_vec, train_label: 訓練集的資料和標籤
     @param test_vec, test_label: 訓練集的資料和標籤
     @return: XGBoost 分類後的準確度
     @example: https://zhuanlan.zhihu.com/p/31182879
    '''
    # 轉換成矩陣
    dtrain = xgb.DMatrix(train_vec, label = train_label)
    dtest = xgb.DMatrix(test_vec, label = test_label)
    evallist  = [(dtrain,'train'), (dtest,'test')]
    
    # 參數設置
    param = {'max_depth': 6,              # 樹的深度
             'gamma': 0.2,                # 剪枝
             'eta': 0.3,                  # 學習率
             'nthread': 11,               # CPU 進程數
             'eval_metric': 'merror',
             'silent': 1,                 # 關閉執行時輸出資訊
             'num_class': 2               # 類別數
            }
    num_round = 50  # 循環次數
    
    # 開始訓練
    bst = xgb.train(param, dtrain, num_round, evallist) 
    preds = bst.predict(dtest)

    # 訓練出來的類別
    predictions = [round(value) for value in preds] 
    y_test = dtest.get_label()
    
    # 混淆矩陣
    test_accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    print('[%s] XGBoost 準確度：%.6f' %(t.now(), test_accuracy))

    return predictions

#================================================================================

if __name__ == '__main__':
    main()
