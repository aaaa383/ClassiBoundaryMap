import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# データの生成
def generate_data(n_samples, n_features, centers, cluster_std, random_state):
    data = make_blobs(n_samples=n_samples, centers=centers,
                      n_features=n_features, cluster_std=cluster_std, random_state=random_state)
    return data


# 機械学習モデルの選択
def choose_model(model_name, **kwargs):
    if model_name == "ロジスティック回帰":
        model = LogisticRegression(C=kwargs.get('C'))
    elif model_name == "k近傍法":
        model = KNeighborsClassifier(n_neighbors=kwargs.get('n_neighbors'))
    elif model_name == "SVM":
        model = SVC(kernel=kwargs.get('kernel'),
                    C=kwargs.get('C'), probability=True)
    elif model_name == "決定木":
        model = DecisionTreeClassifier(max_depth=kwargs.get('max_depth'))
    elif model_name == "ランダムフォレスト":
        model = RandomForestClassifier(n_estimators=kwargs.get(
            'n_estimators'), max_depth=kwargs.get('max_depth'))
    elif model_name == "ニューラルネットワーク":
        model = MLPClassifier(hidden_layer_sizes=kwargs.get(
            'hidden_layer_sizes'), max_iter=kwargs.get('max_iter'))
    elif model_name == "勾配ブースティング決定木(XGboost)":
        model = XGBClassifier(n_estimators=kwargs.get(
            'n_estimators'), max_depth=kwargs.get('max_depth'))
    return model


# グラフ描画
def plot_decision_boundary(model, X, y):
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()

    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    n_classes = len(np.unique(y))

    # カラーマップの設定
    colors_light = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF', '#FFAFAF']
    colors_bold = ['#FF0000', '#00FF00', '#0000FF', '#0F0F0F', '#FF0F0F']

    cmap_light = ListedColormap(colors_light[:n_classes])
    cmap_bold = ListedColormap(colors_bold[:n_classes])

    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                marker='o', cmap=cmap_bold)
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    return fig


# 混合行列を表示
def plot_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(y))

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    disp.plot(ax=ax)

    return fig


# Streamlitアプリ
st.title("機械学習の分類境界を表示")
n_samples = st.sidebar.slider("データ数", 100, 1000, 300)
centers = st.sidebar.slider("クラスター数", 2, 4, 3)
cluster_std = st.sidebar.slider("クラスターの標準偏差", 0, 10, 1)
random_state = st.sidebar.slider("乱数のシード値", 0, 100, 42)

data = generate_data(n_samples, 2, centers, cluster_std, random_state)
X, y = data
X = StandardScaler().fit_transform(X)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state)

model_name = st.sidebar.selectbox("機械学習モデルを選択", [
                                  "ロジスティック回帰", "k近傍法", "SVM", "決定木", "ランダムフォレスト", "ニューラルネットワーク", "勾配ブースティング決定木(XGboost)"])

# それぞれのモデルのハイパーパラメータを選択できる
if model_name == "ロジスティック回帰":
    C = st.sidebar.slider("正則化パラメータ(C)", 0.001, 10.0, 1.0)
    model = choose_model(model_name, C=C)
elif model_name == "k近傍法":
    n_neighbors = st.sidebar.slider("近傍数(k)", 1, 15, 5)
    model = choose_model(model_name, n_neighbors=n_neighbors)
elif model_name == "SVM":
    kernel = st.sidebar.selectbox("カーネル", ['linear', 'rbf', 'poly', 'sigmoid'])
    C = st.sidebar.slider("正則化パラメータ(C)", 0.001, 10.0, 1.0)
    model = choose_model(model_name, kernel=kernel, C=C)
elif model_name == "決定木":
    max_depth = st.sidebar.slider("最大深さ", 1, 15, 5)
    model = choose_model(model_name, max_depth=max_depth)
elif model_name == "ランダムフォレスト":
    n_estimators = st.sidebar.slider("決定木の数", 10, 200, 100)
    max_depth = st.sidebar.slider("最大深さ", 1, 15, 5)
    model = choose_model(
        model_name, n_estimators=n_estimators, max_depth=max_depth)
elif model_name == "ニューラルネットワーク":
    hidden_layer_sizes = st.sidebar.slider("隠れ層のユニット数", 10, 100, 50)
    max_iter = st.sidebar.slider("最大反復回数", 100, 1000, 500)
    activation = st.sidebar.selectbox("活性化関数", [
        "identity", "logistic", "tanh", "relu"])
    solver = st.sidebar.selectbox("最適化手法", [
        "lbfgs", "sgd", "adam"])
    model = choose_model(
        model_name, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, solver=solver)
elif model_name == "勾配ブースティング決定木(XGboost)":
    n_estimators = st.sidebar.slider("決定木の数", 10, 200, 100)
    max_depth = st.sidebar.slider("最大深さ", 1, 15, 5)
    model = choose_model(
        model_name, n_estimators=n_estimators, max_depth=max_depth)

# モデルの学習
model.fit(X_train, y_train)

# 学習データ
pred_train = model.predict(X_train)
st.subheader(f"学習データ: {model_name}の分類境界")
fig, ax = plt.subplots()
plot_decision_boundary(model, X_train, y_train)
st.pyplot(fig)
st.subheader("学習データ: 混合行列")
fig = plot_confusion_matrix(model, X_train, y_train)
st.pyplot(fig)
st.subheader("学習データ: モデルの評価")
st.write(f"##### モデル: {model_name}")
st.write(
    f"##### 学習データの正解率(accuracy): {accuracy_score(y_train, pred_train):.2f}")

st.write("----------------------------------------------------------------------------------")

# 検証データ
pred_test = model.predict(X_test)
st.subheader(f"検証データ: {model_name}の分類境界")
fig, ax = plt.subplots()
plot_decision_boundary(model, X_test, y_test)
st.pyplot(fig)
st.subheader("検証データ: 混合行列")
fig = plot_confusion_matrix(model, X_test, y_test)
st.pyplot(fig)
st.subheader("検証データ: モデルの評価")
st.write(f"##### モデル: {model_name}")
st.write(f"##### 検証データの正解率(accuracy): {accuracy_score(y_test, pred_test):.2f}")
