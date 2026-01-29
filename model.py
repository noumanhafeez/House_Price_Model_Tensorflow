import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Load dataset
df = pd.read_csv("new_dataset.csv")

# Example: last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)  # Regression output
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=120,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_nn = model.predict(X_test_scaled).flatten()

print("Neural Network MSE:", mean_squared_error(y_test, y_pred_nn))
print("Neural Network R2:", r2_score(y_test, y_pred_nn))


