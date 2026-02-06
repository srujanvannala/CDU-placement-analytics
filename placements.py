import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Resume Keywords
# --------------------------------------------------
SKILLS = [
    "python", "java", "c", "c++", "sql", "machine learning",
    "data science", "deep learning", "html", "css", "javascript"
]

PROJECT_WORDS = ["project", "developed", "designed", "implemented"]
INTERNSHIP_WORDS = ["internship", "intern", "training"]
CERTIFICATION_WORDS = ["certification", "certified", "certificate"]
ACHIEVEMENT_WORDS = ["award", "winner", "achievement", "hackathon"]

# --------------------------------------------------
# Resume Processing Functions (FIXED)
# --------------------------------------------------
def extract_resume_text(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")
    return text.lower()

def analyze_resume(text):

    def count(words):
        total = 0
        for word in words:
            pattern = r"\b" + re.escape(word) + r"\b"
            total += len(re.findall(pattern, text))
        return total

    skills = min(count(SKILLS) * 3, 30)
    projects = min(count(PROJECT_WORDS) * 5, 20)
    internships = min(count(INTERNSHIP_WORDS) * 10, 20)
    certifications = min(count(CERTIFICATION_WORDS) * 5, 15)
    achievements = min(count(ACHIEVEMENT_WORDS) * 5, 15)

    total = skills + projects + internships + certifications + achievements

    return {
        "Skills": skills,
        "Projects": projects,
        "Internships": internships,
        "Certifications": certifications,
        "Achievements": achievements,
        "Total": total
    }

# --------------------------------------------------
# Synthetic Dataset for ML
# --------------------------------------------------
np.random.seed(42)

data = pd.DataFrame({
    "CGPA": np.round(np.random.uniform(5.0, 9.8, 500), 2),
    "Internships": np.random.randint(0, 4, 500),
    "Certifications": np.random.randint(0, 6, 500),
    "Coding": np.random.randint(1, 6, 500),
    "Communication": np.random.randint(1, 6, 500),
    "Projects": np.random.randint(1, 6, 500)
})

data["Placed"] = (
    (data["CGPA"] >= 6.5) &
    (data["Coding"] >= 3) &
    (data["Communication"] >= 3)
).astype(int)

X = data.drop("Placed", axis=1)
y = data["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Smart Placement Analytics", layout="wide")
st.title("🎓 CHAITANYA DEEMED TO BE UNIVERSITY PLACEMENT ANALYTICS SYSTEM")

# Sidebar Inputs
st.sidebar.header("📌 Student Profile")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)
internships = st.sidebar.selectbox("Internships", [0, 1, 2, 3])
certifications = st.sidebar.selectbox("Certifications", [0, 1, 2, 3, 4, 5])
coding = st.sidebar.slider("Coding Skills (1–5)", 1, 5, 3)
communication = st.sidebar.slider("Communication Skills (1–5)", 1, 5, 3)
projects = st.sidebar.slider("Projects (1–5)", 1, 5, 3)

input_data = np.array([[cgpa, internships, certifications,
                        coding, communication, projects]])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1] * 100

# --------------------------------------------------
# Placement Prediction
# --------------------------------------------------
st.subheader("📊 Placement Prediction")

c1, c2, c3 = st.columns(3)
c1.metric("Placement Probability", f"{probability:.2f}%")
c2.metric("Model Accuracy", f"{accuracy*100:.2f}%")
c3.metric("Prediction", "WILL BE PLACED ✅" if prediction else "WILL NOT PLACE ❌")

# --------------------------------------------------
# Skill Gap Analysis
# --------------------------------------------------
st.subheader("🧩 Skill Gap Analysis")

required = {"Coding": 4, "Communication": 4, "Projects": 4}
student = {"Coding": coding, "Communication": communication, "Projects": projects}

gap_df = pd.DataFrame({
    "Skill": required.keys(),
    "Required Level": required.values(),
    "Your Level": student.values()
})

st.dataframe(gap_df)

fig, ax = plt.subplots()
ax.bar(gap_df["Skill"], gap_df["Required Level"])
ax.bar(gap_df["Skill"], gap_df["Your Level"])
ax.set_ylabel("Skill Level")
ax.set_title("Skill Gap Comparison")
st.pyplot(fig)

# --------------------------------------------------
# Resume Upload & Analysis
# --------------------------------------------------
st.subheader("📄 Resume Upload & Analysis")

uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    resume_text = extract_resume_text(uploaded_file)
    result = analyze_resume(resume_text)

    st.metric("📊 Resume Score", f"{result['Total']} / 100")

    score_df = pd.DataFrame({
        "Section": list(result.keys())[:-1],
        "Score": list(result.values())[:-1]
    })

    st.bar_chart(score_df.set_index("Section"))

    st.subheader("💡 Resume Improvement Suggestions")

    if result["Skills"] < 20:
        st.write("📌 Add more relevant technical skills.")
    if result["Projects"] < 10:
        st.write("📌 Include detailed projects.")
    if result["Internships"] < 10:
        st.write("📌 Add internships or industrial training.")
    if result["Certifications"] < 8:
        st.write("📌 Include professional certifications.")
    if result["Achievements"] < 8:
        st.write("📌 Highlight awards or hackathons.")

    if result["Total"] >= 80:
        st.success("✅ Strong resume – placement ready!")

# --------------------------------------------------
# Final Recommendation
# --------------------------------------------------
st.subheader("🚀 Final Recommendation")

if probability >= 70:
    st.success("🎯 You are well prepared for placements!")
else:
    st.warning("📌 Improve skills and resume to increase placement chances.")
