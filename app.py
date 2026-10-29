import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Illustrative sample data for demonstration only.
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 5, 6, 7, 8, 3, 6, 4, 7, 8, 9],
    "Attendance": [50, 60, 70, 80, 85, 90, 95, 98, 55, 75, 80, 85, 90, 95, 65, 88, 70, 92, 96, 99],
    "Previous_Marks": [35, 40, 50, 60, 70, 75, 85, 90, 38, 52, 65, 72, 78, 88, 48, 74, 58, 80, 91, 95],
    "Final_Marks": [38, 45, 55, 65, 72, 80, 88, 95, 40, 56, 67, 75, 82, 92, 50, 78, 60, 85, 93, 97]
}
df = pd.DataFrame(data)
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# Fit on all demo records for a simple live prediction demo.
model = LinearRegression()
model.fit(X, y)

st.title("🎓 Student Performance Predictor")
st.write(
    "Estimate a student's exam marks using study hours, attendance, and previous marks. "
    "This is an educational demonstration, not an official assessment tool."
)

st.info(
    "Demo limitation: this app uses a small, illustrative sample dataset. "
    "Predictions and metrics are not validated for real students. Replace the sample data "
    "with a larger, representative dataset before drawing conclusions."
)

st.header("Enter student details")
study_hours = st.slider("Study hours per day", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85, step=1)
previous_marks = st.slider("Previous exam marks (%)", min_value=0, max_value=100, value=70, step=1)

if st.button("Predict performance", type="primary", use_container_width=True):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })
    prediction = float(model.predict(new_student)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.subheader("Prediction result")
    st.metric("Estimated final marks", f"{prediction:.1f} / 100")

    if prediction < 40:
        st.warning("Suggested focus: review fundamentals and make a regular study plan.")
    elif prediction < 70:
        st.info("Suggested focus: revise regularly and practice questions from weaker topics.")
    else:
        st.success("Suggested focus: maintain consistent study habits and practice exam-style questions.")

st.divider()
st.header("Explore the sample dataset")
st.caption("The points below are illustrative training examples, not real student records.")
st.dataframe(df, use_container_width=True, hide_index=True)

fig, ax = plt.subplots()
ax.scatter(df["Study_Hours"], df["Final_Marks"])
ax.set_xlabel("Study hours per day")
ax.set_ylabel("Final marks")
ax.set_title("Sample study hours vs final marks")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.header("Model evaluation (illustrative only)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
evaluation_model = LinearRegression().fit(X_train, y_train)
test_predictions = evaluation_model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:.2f} marks")
col2.metric("R² score", f"{r2:.2f}")
st.caption(
    "The dataset is very small and constructed for demonstration. These scores should not "
    "be interpreted as real-world model accuracy."
)
