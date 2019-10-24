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

## 決定木のアンサンブル  

###  アンサンブル法
複数の機械学習モデルを組み合わせることによってより強力なモデルを構築する方法

### ランダムフォレスト
#### 概要
決定木における訓練データの過剰適合を解決する手法の一つ。
決定木複数組み合わせることによって回避できる。  
複数の異なった方向に過剰適合した決定木を作り、その結果の平均値を取ることによって過剰適合の度合いを減らせる。  
たくさんの決定木を作る必要があり、ある程度のターゲットを予測できてお互いに違っている必要がある。
  

#### 構築方法  
学習するデータの構築  
ブーストラップサンプリング(boostrap sample)  
n_samples個のデータから、データポイントをランダムにn_samples回復元抽出する手法  
  
特徴量サブセットの大きさを選ぶ  
max_featuresで制御することで、決定木の個々のノードが異なるトク著料のサブセットを使って決定を行うようになる。


#### ランダムフォレストの解析
  





