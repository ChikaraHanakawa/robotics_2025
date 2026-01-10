# robotics_2025
こちらは，千葉工業大学大学院 先進工学研究科 未来ロボティクス専攻1年次 後期の講義で取り扱われているリポジトリになります．
本リポジトリには課題として作成した`k-means.py`が含まれています．

## 概要
このプログラムは，教師なし学習の代表的な手法である**K-means法**を実装しました．
NumPyを用いたK-meansアルゴリズムの実装と、その収束過程をアニメーションで可視化するプロジェクトです．

## 環境構築
### 1. 依存ライブラリのインストール
```bash
pip install numpy matplotlib
```
### 2. 実行
```bash
python3 k-means.py
```

### 主な特徴
* **依存ライブラリ最小限**: `scikit-learn`を使わず，NumPyによるベクトル演算でアルゴリズムを実装．
* **アニメーション表示**: 各イテレーションでの重心（Centroid）の移動を`matplotlib.animation`で動的に表現．
* **柔軟なパラメータ**: データ数、次元数、クラスタ数（$k$），シード値を自由に変更可能．


## アルゴリズムの仕組み

### K-meansとは
K-means法は教師なし学習のデータ分類．
教師なし学習とは，教師となる正解データを使わずにデータの関連性を求める手法です．
正解データが不要のため，機械学習を行うのに準備が少ないのがメリットです．デメリットとしては，正解がない事による評価が難しいことです．

### K-meansアルゴリズムの数式表現

1. **データ**
    - このようなデータ $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$ があったとする．

    <img src="fig/explanation1.png" width=300>

2. **初期化**
	- データからランダムに $k$ 個の重心（セントロイド） $\{\mathbf{\mu}_1, \ldots, \mathbf{\mu}_k\}$ を選ぶ．

    <img src="fig/explanation2.png" width=300>

3. **距離計算**
	- 各データ点 $\mathbf{x}_i$ について、すべての重心 $\mathbf{\mu}_j$ とのユークリッド距離 $d_{ij}$ を計算し、最も距離が小さい重心に割り当てます．

        $d_{ij} = \| \mathbf{x}_i - \mathbf{\mu}_j \|_2$
    
    <img src="fig/explanation3.png" width=300>

4. **割り当て**
	- そして、クラスタラベル $c_i$ を次式で決定します．

	  $c_i = \underset{j \in \{1, \ldots, k\}}{\arg\min} \, d_{ij}$

    <img src="fig/explanation4.png" width=300>

5. **重心更新**
	- 各クラスタ $j$ について、割り当てられた点の平均を新たな重心とする．

        $
        \mathbf{\mu}_j = \frac{1}{N_j} \sum_{i: c_i = j} \mathbf{x}_i
        $

	- ここで $N_j$ はクラスタ $j$ に属する点の数．

    <img src="fig/explanation5.png" width=300>

6. **収束判定**
	- 重心 $\mathbf{\mu}_j$ の位置が変化しなくなる
    （$\mathbf{\mu}_j^{\text{old}} \approx \mathbf{\mu}_j^{\text{new}}$）
    まで3から5を繰り返す．

    <img src="fig/explanation6.png" width=300>

---

### 実行結果
<img src="fig/Kmeans_result.gif" width=700>