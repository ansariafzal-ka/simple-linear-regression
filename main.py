from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
print(data)
X = data["Hours_Studied"]
y = data["Marks_Obtained"]

X = np.array(X).reshape(-1, 1)
y = np.array(y)

model =  LinearRegression()
model.fit(X, y)
new_values = np.array([[2], [5], [3], [1.5], [1], [0], [3.3], [7], [4], [2.5]])
predictions = model.predict(new_values)

plt.scatter(new_values, predictions, color='red', label='Predictions')
plt.plot(X, model.predict(X), color='green', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Hours Studied vs Marks Obtained')
plt.legend()
plt.show()