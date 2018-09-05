from gensim import corpora;
from gensim import models;
#语料
raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
#对语料进行预处理
stopList = set("for a of the and to in".split());
#print(stopList);
texts = [[word for word in docment.lower().split() if word not  in stopList] for docment in raw_corpus];
from collections import defaultdict;
#默认的字典 自动加1
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1;
#print(frequency)
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts];
#表示每一句话大于2出现的频率统计
#print(processed_corpus);
#每一个单词关联一个唯一的ID 用genism.corpra.Dictionary 来实现 12个不同的单词每个单词关联唯一的一个ID
dic = corpora.Dictionary(processed_corpus);
#print(dic);
#获得词袋向量表示
new_doc = "human computer interaction";
new_vec = dic.doc2bow(new_doc.lower().split());
#第一个为ID 第二个表示出现的次数
#print(new_vec);
bow_corpus = [dic.doc2bow(text) for  text in processed_corpus];
#print(bow_corpus);
#向量化语料后 转化为另外一个模型 用tf-idf进行转换为另外一个
tfIdf = models.TfidfModel(bow_corpus);
string = "system minors";
string_bow = dic.doc2bow(string.lower().split());
string_tfIdf = tfIdf[bow_corpus];
#print(string_bow);
print(string_tfIdf);