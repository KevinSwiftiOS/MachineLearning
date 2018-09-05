#genism中使用word2Vec模型
from gensim.models import  word2vec;
import gensim;
import logging;
#输出的等级 比print更易使用
logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s',level=logging.INFO)
#第一次使用 加载文档集
sentences = word2vec.Text8Corpus('/Users/hcnucai/Downloads/text8')
model = word2vec.Word2Vec(sentences,size=200)
#保存模型 以便下次使用
model.save('/Users/hcnucai/Downloads/text8.model');
#model = word2vec.Word2Vec('/Users/hcnucai/Downloads/text8.model')
#计算词的相似性
print(model.most_similar(positive=['woman','man','kiss','love'],negative=['girl'],topn=5));
#换种方式
more_examples = ['he his she','big bigger bad','going went being']
for example in more_examples:
    a,b,x = example.split();
    predicted = model.most_similar([x,b],[a])[0][0];
    print("'%s',is to '%s' as to '%s' is to '%s'" %(a,b,x,predicted));
    #找出不同的单词
    print(model.doesnt_match(['fuck', 'head', 'foot', 'hand'])) # fuck
    print(model.doesnt_match("breakfast cereal dinner lunch".split()))  # cereal