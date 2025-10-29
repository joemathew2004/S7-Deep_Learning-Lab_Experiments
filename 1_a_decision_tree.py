import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Play Tennis Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 
                'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 
                    'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 
                 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 
             'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
                   'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Features and target
X = df.drop('PlayTennis', axis=1)           # X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']] 
y = df['PlayTennis']                        # y = df['PlayTennis'] 

# Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy
model.fit(X, y)

# Display the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=encoders['PlayTennis'].classes_, filled=True)

plt.title("Decision Tree for Play Tennis Dataset")
plt.show()

print("\nEnter new sample values:")
outlook = input("Outlook (Sunny/Overcast/Rain): ")
temperature = input("Temperature (Hot/Mild/Cool): ")
humidity = input("Humidity (High/Normal): ")
wind = input("Wind (Weak/Strong): ")

sample = [[
    encoders['Outlook'].transform([outlook])[0],
    encoders['Temperature'].transform([temperature])[0],
    encoders['Humidity'].transform([humidity])[0],
    encoders['Wind'].transform([wind])[0]
]]

prediction = model.predict(sample)
predicted_label = encoders['PlayTennis'].inverse_transform(prediction)

print(f"\nPrediction for new sample ({outlook}, {temperature}, {humidity}, {wind}):", predicted_label[0])