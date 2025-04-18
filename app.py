import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_keywords(text, num_keywords=10):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:num_keywords]]

def get_similarity(resume_text, job_text):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0] * 100, 2)

# Streamlit UI
st.set_page_config(page_title="Resume Analyzer + Job Matcher")
st.title("📄 Resume Analyzer + Job Matcher")
st.markdown("Upload your resume and a job description to check your match score and get suggestions.")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")
job_description = st.text_area("Paste Job Description Here", height=200)

if st.button("Analyze"):
    if uploaded_file and job_description.strip() != "":
        with st.spinner("Processing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            keywords = get_keywords(resume_text, 10)
            match_score = get_similarity(resume_text, job_description)

        st.subheader("📊 Match Score")
        st.success(f"✅ Your resume matches {match_score}% with the job description.")

        st.subheader("🔑 Top Keywords in Your Resume")
        st.write(", ".join(keywords))

        if match_score < 60:
            st.warning("You may want to add more relevant skills or experience from the job description.")
    else:
        st.error("Please upload a resume and paste a job description.")
