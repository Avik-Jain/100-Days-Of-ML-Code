import numpy as np
import pandas as pd

dataset = pd.read_csv('..\datasets\Social_Network_Ads.csv')

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

def sigmod(theta):
    import math
    y = 1/(1 + math.e**(-theta))
    return y


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


# out = sigmod(yy_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# plot data set and boundary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

theta = classifier.coef_
b = classifier.intercept_
# line equation: age * theta_0 + salary * theta1 + b = 0
age_plot = [i/10 for i in range(-40, 40)]
salary_plot = -1 * (theta[0, 0] * np.array(age_plot) + b)/theta[0, 1]


def plot_result(x, y, type='train'):
    xlim = [-3, 3]
    ylim = [-2.5, 3.5]
    x_positive = x[np.where(y == 1)]
    x_negative = x[np.where(y == 0)]
    fig_train = plt.figure()
    ax = fig_train.add_subplot(111)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.title('Logistic Regresstion (%s set)' % type)

    X_set, y_set = x_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.55, cmap=ListedColormap(('red', 'green')))

    # ax.plot(age_plot, salary_plot, c='r')
    # plt.fill_between(age_plot, salary_plot, ylim[-1], color='lawngreen')
    # plt.fill_between(age_plot, ylim[0], salary_plot, color='hotpink')
    #
    ax.scatter(x_negative[:, 0], x_negative[:, 1], c='r', label='0')
    ax.scatter(x_positive[:, 0], x_positive[:, 1], c='g', label='1')
    ax.set_xlim((X1.min(), X1.max()))
    ax.set_ylim((X2.min(), X2.max()))
    plt.legend()
    plt.show()


plot_result(x_train, y_train, type="Train")
plot_result(x_test, y_test, type="Test")



