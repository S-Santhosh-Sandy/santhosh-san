import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


data = pd.read_csv('date.csv')


data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)


X = data[['Date']]
y = data['Rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results)

future_dates = pd.date_range(start='2023-01-11', periods=5)
future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
future_predictions = model.predict(future_dates_ordinal)


future_results = pd.DataFrame({'Date': future_dates, 'Predicted Rainfall': future_predictions})
print(future_results)