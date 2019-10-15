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