import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

from logistic_regression_prediction import LogisticRegressionWeatherForecast as ldp
from rule_based_prediction import RuleBasedWeatherForecast as rbp
from monitor_resources import MonitorResource as mr

np.random.seed(42)
data_size = 30000
temperature = np.random.randint(10, 40, data_size)
humidity = np.random.randint(10, 100, data_size)

conditions = []
for t, h in zip(temperature, humidity):
    if t > 30 and h < 50:
        conditions.append("Sunny")
    elif t <= 30 and h >= 70:
        conditions.append("Rainy")
    elif 20 <= t <= 30 and 50 <= h < 70:
        conditions.append("Cloudy")
    else:
        conditions.append("Uncertain")

df = pd.DataFrame({
    "Temperature": temperature,
    "Humidity": humidity,
    "Condition": conditions
})

df["Condition_Code"] = df["Condition"].map({"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Uncertain": 3})

X = df[["Temperature", "Humidity"]]
y = df["Condition_Code"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rule_based_model = rbp()
logistic_model = ldp()

logistic_model.train(X_train, y_train)

#monitor_resources = mr()

# Benchmark Rule-Based Model
start_time = time.time()
#cpu_before, ram_before, gpu_before = monitor_resources.monitor()
rule_based_predictions = [rule_based_model.predict(row["Temperature"],
                          row["Humidity"]) for _,
                          row in X_test.iterrows()]
rule_based_time = time.time() - start_time
#cpu_after, ram_after, gpu_after = monitor_resources.monitor()
rule_based_accuracy = accuracy_score(y_test,
                        [df[df["Condition"] == pred]["Condition_Code"].values[0] for pred in rule_based_predictions])

#time.sleep(3)

# Benchmark Logistic Regression Model
start_time = time.time()
#cpu_before_lr, ram_before_lr, gpu_before_lr = monitor_resources.monitor()
logistic_predictions = logistic_model.predict(X_test)
#cpu_after_lr, ram_after_lr, gpu_after_lr = monitor_resources.monitor()
logistic_time = time.time() - start_time
logistic_accuracy = accuracy_score(y_test,
                         [df[df["Condition"] == pred]["Condition_Code"].values[0] for pred in logistic_predictions])

# Display Results
print("Rule-Based Model:")
print(f"Accuracy: {rule_based_accuracy * 100:.2f}%")
print(f"Execution Time: {rule_based_time:.4f} seconds")
#print(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
#print(f"RAM Usage: {ram_after - ram_before:.2f} MB")
#print(f"GPU Usage: {gpu_after - gpu_before:.2f}%\n")

print("Logistic Regression Model:")
print(f"Accuracy: {logistic_accuracy * 100:.2f}%")
print(f"Execution Time: {logistic_time:.4f} seconds")
#print(f"CPU Usage: {cpu_after_lr - cpu_before_lr:.2f}%")
#print(f"RAM Usage: {ram_after_lr - ram_before_lr:.2f} MB")
#print(f"GPU Usage: {gpu_after_lr - gpu_before_lr:.2f}%")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test,
                [df[df["Condition"] == pred]["Condition_Code"].values[0] for pred in logistic_predictions],
                target_names=["Sunny", "Rainy", "Cloudy", "Uncertain"],
                zero_division=0 ))
