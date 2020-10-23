# SVM练习

## 笔记
构建 Pipeline（管道）
为了使得 向量化（vectorizer） => 转换器（transformer） => 分类器（classifier） 过程更加简单,scikit-learn 提供了一个 Pipeline 类，操作起来像一个复合分类器:  
```python
from sklearn.pipeline import Pipeline  
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])
```