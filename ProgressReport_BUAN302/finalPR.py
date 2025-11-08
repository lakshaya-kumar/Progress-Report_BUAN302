import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# -----------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Clean column names if needed
df.columns = [col.strip() for col in df.columns]

# Encode RiskLevel
df['RiskLevel'] = df['RiskLevel'].map({'low risk': 0, 'mid risk': 1, 'high risk': 2})

# Create readable label
risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
df['RiskLabel'] = df['RiskLevel'].map(risk_map)


# TITLE

st.title(" Maternal Health Risk - Machine Learning Progress Report")
st.markdown("### For BUAN302, submitted by **Lakshaya Kumar**")

st.write("""
This dataset contains clinical information of 1,014 pregnant women, including indicators such as age, blood pressure, blood sugar, body temperature, and heart rate. 
Each record is labelled as **Low**, **Medium**, or **High Risk** based on maternal health indicators. 
For this project, the categorical variable *RiskLevel* was converted into numeric form — 0 for Low, 1 for Medium, and 2 for High to enable training and evaluation of various machine learning classification models.
""")

# -----------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------
st.header("Exploratory Data Analysis")

# Histogram
st.subheader("Histogram - Feature Distributions by Risk Level")
for col in ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]:
    fig = px.histogram(
        df,
        x=col,
        color="RiskLabel",
        nbins=20,
        title=f"{col} Distribution by Risk Level",
        color_discrete_map={
            "Low Risk": "#1f77b4",
            "Medium Risk": "#ff7f0e",
            "High Risk": "#d62728"
        },
        barmode="overlay",
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

# Box Plots
st.subheader("Box Plots - Feature Variation by Risk Level")
for col in ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]:
    fig = px.box(
        df,
        x="RiskLabel",
        y=col,
        color="RiskLabel",
        title=f"{col} Distribution Across Risk Levels",
        color_discrete_map={
            "Low Risk": "#1f77b4",
            "Medium Risk": "#ff7f0e",
            "High Risk": "#d62728"
        },
        points="all"
    )
    fig.update_layout(xaxis_title="Risk Level", yaxis_title=col)
    st.plotly_chart(fig, use_container_width=True)

# Scatter Plot
st.subheader("Scatter Plot - Blood Pressure Relationship by Risk Level")
fig = px.scatter(
    df,
    x="SystolicBP",
    y="DiastolicBP",
    color="RiskLabel",
    size="HeartRate",
    hover_data=["Age", "BS", "BodyTemp"],
    title="Systolic vs Diastolic Blood Pressure (Bubble size = Heart Rate)",
    color_discrete_map={
        "Low Risk": "#1f77b4",
        "Medium Risk": "#ff7f0e",
        "High Risk": "#d62728"
    },
)
fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="black")))
fig.update_layout(
    xaxis_title="Systolic Blood Pressure (mmHg)",
    yaxis_title="Diastolic Blood Pressure (mmHg)",
    legend_title="Risk Category"
)
st.plotly_chart(fig, use_container_width=True)

st.write ("""
The scatter plot shows the relationship between systolic and diastolic blood pressure across the three maternal health risk levels. 
The high-risk group (red) generally clusters toward higher systolic and diastolic blood pressure values, indicating hypertension as a strong determinant of maternal risk. The low-risk group (blue) mostly lies in the lower blood pressure range, reflecting stable cardiovascular conditions. The medium-risk group (orange) is spread across the midrange, overlapping slightly with both ends. The bubble size represents heart rate, showing that higher heart rates often accompany elevated blood pressure in high-risk cases. 
Overall, the plot highlights how increasing blood pressure is strongly associated with higher maternal risk levels.
""")

# Density Plot
st.subheader("Density Plot - Age Distribution by Risk Level")
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"Low Risk": "#1f77b4", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"}
for label, color in colors.items():
    subset = df[df["RiskLabel"] == label]
    sns.kdeplot(subset["Age"], fill=True, label=label, color=color, alpha=0.5, ax=ax)
ax.set_title("Age Distribution Across Risk Levels", fontsize=13)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.legend(title="Risk Category")
st.pyplot(fig)

st.write ("""
The age density curves reveal how maternal age varies across the three risk categories. Low-risk women are concentrated in the younger age bracket, with the density peak around the early 20s. Medium-risk individuals tend to fall between the late 20s and mid-30s, overlapping moderately with both low and high-risk groups. The high-risk curve (red) shifts noticeably toward older ages, peaking around 35–40 years, showing that maternal risk rises with age.
""")

# Correlation Heatmap (numeric columns only)
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['float64', 'int64'])
fig = px.imshow(
    numeric_df.corr(),
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="Correlation Matrix of Numeric Features"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# DATA PREP
# -----------------------------------------------------------
X = df.drop(['RiskLevel', 'RiskLabel'], axis=1)
y = df['RiskLevel']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------
st.header("Model Training & Evaluation")

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Display model comparison
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)

st.subheader("Model Comparison")
st.dataframe(results_df)
fig = px.bar(results_df, x='Model', y='Accuracy', color='Model', title='Model Accuracy Comparison')
st.plotly_chart(fig)

st.write("""
The Decision Tree and Random Forest models achieved the highest accuracy of 81.8%, indicating strong predictive ability and suggesting that the relationships between features and risk levels are non-linear and hierarchical. Bagging also performed well with 80.7% accuracy. 
Gaussian Naive Bayes had the lowest accuracy (~57.6%), likely due to its assumption of feature independence, which doesn’t hold true for clinical variables.
""")

# -----------------------------------------------------------
# DECISION TREE VISUALISATION
# -----------------------------------------------------------
st.header(" Decision Tree Visualisation")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(models['Decision Tree'], filled=True, feature_names=X.columns, class_names=['Low', 'Medium', 'High'])
st.pyplot(fig)

# -----------------------------------------------------------
# CONFUSION MATRIX & CLASSIFICATION REPORT
# -----------------------------------------------------------
st.header(" Confusion Matrix & Classification Report")
selected_model = st.selectbox("Select a model to view details", list(models.keys()))
model = models[selected_model]
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("""
Thank you
""")
