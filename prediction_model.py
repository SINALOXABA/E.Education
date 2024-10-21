# prediction_model.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import data_preprocessing

# Load preprocessed data
df = data_preprocessing.generate_data()

# Define features and labels
X = df[['age', 'previous_score', 'attendance_rate', 'homework_completion']]
y = df['quiz_score']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict quiz scores
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Display predictions and actual values
predicted_vs_actual = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(predicted_vs_actual.head())
