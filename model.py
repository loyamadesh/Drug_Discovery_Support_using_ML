import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import itertools
import joblib

np.random.seed(42)

# Load your 1200 row dataset
df = pd.read_csv("data.csv")

# Extract unique drugs
drugs = df["Drug_Name"].unique().tolist()

# Create category mapping
categories = {drug:i for i,drug in enumerate(drugs)}

# Create standard weight dictionary from dataset
standard_weights = {}

for drug in drugs:
    avg_wt = df[df["Drug_Name"] == drug]["Molecular_Weight"].mean()
    standard_weights[drug] = float(avg_wt)

# Create interaction matrix between drugs
interaction_matrix = {}
for pair in itertools.combinations(range(len(drugs)),2):
    interaction_matrix[pair] = np.random.uniform(-0.5,1.0)

data = []

# Generate training combinations from dataset drugs
for _ in range(6000):

    num = np.random.choice([2,3])
    selected = np.random.choice(drugs, size=num, replace=False)

    weights = []
    deviations = []
    synergy_score = 0

    for drug in selected:

        std = standard_weights[drug]

        sample = np.random.normal(std, std*0.1)

        deviation = ((sample-std)/std)*100

        weights.append(sample)
        deviations.append(deviation)

    # Calculate synergy
    for pair in itertools.combinations(selected,2):

        cat_pair = tuple(sorted((categories[pair[0]],categories[pair[1]])))

        synergy_score += interaction_matrix.get(cat_pair,0)

    total_weight = sum(weights)

    avg_deviation = np.mean(deviations)

    diversity = len(set([categories[d] for d in selected]))

    # Effectiveness formula
    effectiveness = (
        70
        + synergy_score*15
        - abs(avg_deviation)*3
        - (total_weight/3000)*5
        + diversity*5
        + np.random.normal(0,5)
    )

    data.append([
        total_weight,
        avg_deviation,
        synergy_score,
        diversity,
        effectiveness
    ])

df_train = pd.DataFrame(data,columns=[
    "Total_Wt",
    "Avg_Deviation",
    "Synergy",
    "Diversity",
    "Effectiveness"
])

X = df_train[["Total_Wt","Avg_Deviation","Synergy","Diversity"]]
y = df_train["Effectiveness"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train,y_train)

# Save everything for app.py
joblib.dump({
    "model":model,
    "interaction_matrix":interaction_matrix,
    "categories":categories,
    "standard_weights":standard_weights
},"combination_model.pkl")

print("Model trained with dataset and saved successfully!")