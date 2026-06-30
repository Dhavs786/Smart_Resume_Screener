# TalentAI: AI-Powered Resume Screener & Candidate Ranking System

TalentAI is an advanced candidate ranking and screening system designed for modern recruiters. It parses multiple PDF resumes, performs NLTK-based cleaning, computes deep semantic embedding matching, and evaluates candidate qualifications with a custom ATS scoring algorithm.

## Features
1. **Resume PDF Text Extraction**: Uses `pdfplumber` to extract clean text from resumes.
2. **NLP Preprocessing**: Lowercasing, tokenization, stopword removal, and lemmatization.
3. **Deep Semantic Embeddings**: Utilizes Sentence Transformers (`all-MiniLM-L6-v2`) to perform contextual alignment.
4. **Weighted ATS Score**:
   - **50%** Semantic similarity.
   - **30%** Direct skill matching.
   - **10%** Experience/Education layout.
   - **10%** Formatting and contact details check.
5. **Demonstration Sandbox**: Click **"Generate Demo Resumes"** inside the app sidebar to generate 4 test resumes automatically.
6. **Detailed Candidate Profiling**:
   - Side-by-side view of matched, missing, and extra skills.
   - Tailored interview questions.
   - AI Candidate Summary (with optional Gemini, OpenAI, or Groq API integrations).
7. **Interactive Resume Chatbot**: Chat directly with any candidate's resume content.
8. **Multi-Job Placement Matrix**: Compare candidate fitness across 3 different roles side-by-side.

## Installation & Setup

1. Make sure Python 3.8+ is installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Run the NLTK downloader to ensure the tokenizers and corpora are cached:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
   ```

## Running the Application

Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to access the dashboard.
