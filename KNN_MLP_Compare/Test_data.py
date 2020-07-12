from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    """
    import the iris dataset
    feature_names:sepal length,sepal width, petal length, petal width(cm)
    target_name:0:setosa;1:versicolor;2:virginica
    """
    iris = datasets.load_iris()
    X = iris.data # shape=(150,4)
    Y = iris.target # shape=(150,)

    # randomly choose 30 points
    indices = np.random.choice(len(X),30)
    X = X[indices]
    Y = Y[indices]
    return X,Y
def _plot_view(X,Y):
    """
    :param X: feature shape=(batch,4)
    :param Y: target  shape=(batch,)
    :return: plot them in 3D as a scatterplot
    """

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1, figsize=(20, 15))
    ax = Axes3D(fig, elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y,
    cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)
    for name, label in [('Virginica', 0), ('Setosa', 1),('Versicolour', 2)]:
        ax.text3D(X[Y == label, 0].mean(),
                  X[Y == label, 1].mean(),
                  X[Y == label, 2].mean(),
            name,horizontalalignment='center',
            bbox=dict(alpha=.5, edgecolor='w',facecolor='w'),size=25)
    ax.set_title("3D visualization", fontsize=40)
    ax.set_xlabel("Sepal Length [cm]", fontsize=25)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("Sepal Width [cm]", fontsize=25)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("Petal Length [cm]", fontsize=25)
    ax.w_zaxis.set_ticklabels([])
    plt.show()
if __name__ == "__main__":
    X,Y = get_data()
    _plot_view(X,Y)