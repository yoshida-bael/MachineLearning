# 機械学習


## 最初にインポートする系
```python
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import scipy as sp
import sklearn
%matplotlib inline
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
グラフ書く時に使うやつ  
第１引数： 散布図に描写する各データのX値  
第２引数： 散布図に描写する各データのY値  
第３引数： 散布図に描写する各データのラベル
<br>

```
plt.legend(["Class 0","Class 1"],loc=4)
```
凡例の指定とlocでどこの事象に凡例を示すか指定できる
<br>  


## 決定木のアンサンブル  

###  アンサンブル法
複数の機械学習モデルを組み合わせることによってより強力なモデルを構築する方法

### ランダムフォレスト
#### 概要
・決定木における訓練データの過剰適合を解決する手法の一つ。   
・複数の異なった方向に過剰適合した決定木を作り、その結果の平均値を取ることによって過剰適合の度合いを減らす。  
・データを復元抽出して複数のサンプルデータを作成(boostrap sample)
・特徴量のサブセットを用意して個々の決定木で学習
  
#### やり方
ランダムフォレストをインポート
```
from sklearn.ensemble import RandomForestClassifier
```
今回使用するデータセットをインポートして学習させる  
n_samplesでデータを100個復元抽出して作る  
n_estimatorsで決定木の個数を指定
```
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)
forest = RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(X_train,y_train)
```
グラフに描画
```
fig,axes = plt.subplots(2,3,figsize=(20,10))
for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)
mglearn.plots.plot_2d_separator(forest,X_train,fill=True,ax=axes[-1,-1],alpha=.4)
axes[-1,-1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
```
![“Ensemble”](ensemble.png)


  





