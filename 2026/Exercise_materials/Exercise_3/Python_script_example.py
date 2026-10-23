import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

iris = sns.load_dataset("iris")

print(iris.head())

X = iris[["sepal_length"]]
y = iris["sepal_width"]
model = LinearRegression().fit(X, y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

sns.lmplot(x="sepal_length", y="sepal_width", data=iris)
plt.savefig("iris_sepal_width_length_2025-10-16.png")

setosa = iris.query("species == 'setosa'")
setosa.to_csv("setosa_irises_2025-10-16.csv", index=False)
