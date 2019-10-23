# 機械学習


## 最初にインポートする系
```python
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
```
<br>

```
X,y = mglearn.datasets.make_forge()
```
これで色々データとってこれる  
make_~　以降で取ってくるデータを指定できる

<br>

```
discrete_scatter(X[:,0],X[:,1],y)
```
dicrete_scatter  
第１引数： 散布図に描写する各データのX値  
第２引数： 散布図に描写する各データのY値  
第３引数： 散布図に描写する各データのLABEL  

<br>

```
plt.legend(["Class 0","Class 1"],loc=4)
```
凡例の指定とlocでどこの事象に凡例を示すか指定できる
<br>

# 決定木のアンサンブル  
## ランダムフォレスト
<br>
利点  

決定木における訓練データの過剰適合を決定木複数組み合わせることによって回避できる。  
複数の決定木を作りその結果の平均値を取ることによって過剰適合の度合いを減らせる。  
パラメータのチューニングをしなくても良い


パラメータ  
max_features ・・・特徴量サブセットの大きさ（サブセットは部分集合）
<br>

##　勾配ブースティング回帰木
<br>
概要  
一つ前の決定木の誤りを次の決定木が修正するようにして決定木を順番に作っていく。
<br>
利点  
特徴量のスケールを変換する必要が無いこと
<br>
パラメータ  
learning_rate・・学習率