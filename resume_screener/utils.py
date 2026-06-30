import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import pypdf
import json

# Predefined common skills list for keyword mapping
SKILL_DB = [
    # Programming Languages
    "python", "sql", "java", "c\\+\\+", "c#", "javascript", "typescript", "r", "go", "rust", "ruby", "php", "html", "css", "bash", "shell", "c",
    # AI/ML/DL
    "machine learning", "deep learning", "natural language processing", "nlp", "computer vision", "generative ai", "genai", "llm", "rag", 
    "reinforcement learning", "linear regression", "decision trees", "random forest", "xgboost", "neural networks", "transformers",
    # Frameworks & Libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy", "fastapi", "flask", "django", "react", "angular", "vue", 
    "node\\.js", "express", "spring boot", "hugging face", "langchain", "llamaindex", "streamlit", "jquery", "bootstrap",
    # Databases
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra", "dynamodb", "oracle", "sql server", "firebase", "neo4j",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "ci/cd", "terraform", "ansible", "linux", "unix", "maven",
    # Data & BI
    "power bi", "tableau", "excel", "spark", "hadoop", "kafka", "snowflake", "databricks", "data engineering", "data science", 
    "business intelligence", "data analytics", "qlikview"
]

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file object (or file path).
    """
    text = ""
    # Try pdfplumber first
    try:
        if isinstance(pdf_file, str):
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            # File-like object (BytesIO)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}. Trying pypdf...")
        # Fallback to pypdf
        try:
            pdf_file.seek(0) if hasattr(pdf_file, 'seek') else None
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e2:
            print(f"pypdf extraction failed: {e2}")
            
    return text.strip()

def preprocess_text(text):
    """
    Preprocesses text: lowercasing, tokenization, stopword removal, lemmatization.
    Returns both a clean string and the list of processed tokens.
    """
    if not text:
        return "", []
    
    # Lowercase conversion
    text_lower = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text_lower)
    
    # Stopword removal & non-alphabetic filtering
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in cleaned_tokens]
    
    return " ".join(lemmatized_tokens), lemmatized_tokens

def extract_contact_info(text):
    """
    Extracts candidate name, email, and phone number from resume text.
    """
    email = "Not Found"
    phone = "Not Found"
    name = "Candidate"
    
    if not text:
        return name, email, phone

    # Email pattern
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = email_match.group(0)
        
    # Phone pattern (supports various formats: +1-234-567-8901, (123) 456-7890, etc.)
    phone_match = re.search(r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,12})', text)
    if phone_match:
        phone = phone_match.group(0)

    # Name heuristic: Look at the first few lines of the resume text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in lines[:5]:
            # Filter out common headings or email/phone lines to guess the name
            if "@" in line or any(char.isdigit() for char in line):
                continue
            if any(word in line.lower() for word in ["resume", "cv", "curriculum", "vitae", "contact", "summary", "profile", "education", "experience"]):
                continue
            # Assume first line that doesn't match above is the name
            # Cap at 4 words
            words = line.split()
            if 1 <= len(words) <= 4:
                name = line
                break
                
    return name, email, phone

def extract_skills(text):
    """
    Detects and extracts skills from text using pre-defined skill list.
    """
    extracted = set()
    text_lower = text.lower()
    
    for skill in SKILL_DB:
        # Create regex for word boundary, making sure special chars like C++ are handled
        pattern = r'\b' + skill + r'\b'
        if skill == "c\\+\\+":
            pattern = r'c\+\+'
        elif skill == "c#":
            pattern = r'c#'
        elif skill == "node\\.js":
            pattern = r'\bnode\.js\b|\bnodejs\b'
            
        if re.search(pattern, text_lower):
            # Format nicely for display
            display_name = skill.replace('\\', '').title()
            if display_name.lower() == "sql":
                display_name = "SQL"
            elif display_name.lower() == "nlp":
                display_name = "NLP"
            elif display_name.lower() == "aws":
                display_name = "AWS"
            elif display_name.lower() == "gcp":
                display_name = "GCP"
            elif display_name.lower() == "bi":
                display_name = "BI"
            elif display_name.lower() == "api":
                display_name = "API"
            elif display_name.lower() == "llm":
                display_name = "LLM"
            elif display_name.lower() == "rag":
                display_name = "RAG"
            elif display_name.lower() == "git":
                display_name = "Git"
            elif display_name.lower() == "ci/cd":
                display_name = "CI/CD"
            elif display_name.lower() == "db":
                display_name = "DB"
            elif display_name.lower() == "ml":
                display_name = "ML"
            extracted.add(display_name)
            
    return sorted(list(extracted))

def classify_resume(skills):
    """
    Classifies a candidate's profile into standard tech categories based on extracted skills.
    """
    skills_lower = [s.lower() for s in skills]
    
    ml_ai_keywords = ["pytorch", "tensorflow", "deep learning", "nlp", "natural language processing", "generative ai", "genai", "llm", "rag", "computer vision", "transformers"]
    ds_da_keywords = ["scikit-learn", "r", "pandas", "numpy", "tableau", "power bi", "data science", "data analytics", "excel", "business intelligence", "sql"]
    frontend_keywords = ["react", "angular", "vue", "javascript", "typescript", "html", "css", "jquery", "bootstrap"]
    backend_keywords = ["node.js", "nodejs", "django", "flask", "fastapi", "spring boot", "postgresql", "mongodb", "redis", "mysql", "java", "c++", "c#", "go", "rust"]
    devops_keywords = ["docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "terraform", "ansible", "git", "ci/cd", "linux"]
    
    ml_matches = len([s for s in skills_lower if s in ml_ai_keywords or any(kw in s for kw in ["machine learning", "deep learning", "nlp", "vision", "ai"])])
    ds_matches = len([s for s in skills_lower if s in ds_da_keywords or "data" in s or "analytic" in s])
    fe_matches = len([s for s in skills_lower if s in frontend_keywords])
    be_matches = len([s for s in skills_lower if s in backend_keywords])
    do_matches = len([s for s in skills_lower if s in devops_keywords or "cloud" in s])
    
    scores = {
        "Machine Learning / AI Engineer": ml_matches,
        "Data Scientist / Analyst": ds_matches,
        "Frontend Developer": fe_matches,
        "Backend Developer": be_matches,
        "Cloud / DevOps Engineer": do_matches
    }
    
    max_cat = max(scores, key=scores.get)
    if scores[max_cat] == 0:
        return "General Technical Professional"
        
    return max_cat

def calculate_ats_score(resume_text, jd_text, resume_skills, jd_skills, semantic_similarity):
    """
    Simulates ATS scoring:
    - Semantic Similarity (Cosine): 50%
    - Skill Match Percentage: 30%
    - Experience/Education Presence check: 10%
    - Formatting & Contact check: 10%
    """
    # 1. Semantic Score (0 - 100)
    semantic_score = float(semantic_similarity * 100)
    
    # 2. Skill Match Score (0 - 100)
    if not jd_skills:
        skill_score = 100.0
    else:
        matched_skills = set(resume_skills).intersection(set(jd_skills))
        skill_score = (len(matched_skills) / len(jd_skills)) * 100.0
        
    # 3. Experience & Education Check (0 - 100)
    exp_edu_score = 0.0
    resume_lower = resume_text.lower()
    if any(word in resume_lower for word in ["experience", "work history", "employment", "professional background", "positions held"]):
        exp_edu_score += 50
    if any(word in resume_lower for word in ["education", "degree", "university", "college", "academic", "bachelor", "master", "phd"]):
        exp_edu_score += 50
        
    # 4. Formatting & Contact Check (0 - 100)
    formatting_score = 0.0
    name, email, phone = extract_contact_info(resume_text)
    if name != "Candidate":
        formatting_score += 30
    if email != "Not Found":
        formatting_score += 35
    if phone != "Not Found":
        formatting_score += 35
        
    # Combine scores with weights
    ats_score = (semantic_score * 0.50) + (skill_score * 0.30) + (exp_edu_score * 0.10) + (formatting_score * 0.10)
    
    return round(ats_score, 1), {
        "semantic_similarity_score": round(semantic_score, 1),
        "skills_match_score": round(skill_score, 1),
        "experience_education_score": round(exp_edu_score, 1),
        "formatting_contact_score": round(formatting_score, 1)
    }

def get_sentence_transformer_model():
    """
    Loads sentence-transformers model (cached inside Streamlit resource cache).
    """
    model_name = "all-MiniLM-L6-v2"
    # Note: Hugging Face model will be downloaded to cache
    model = SentenceTransformer(model_name)
    return model

def get_semantic_similarity(model, doc1_text, doc2_text):
    """
    Generates embeddings and calculates cosine similarity.
    """
    embeddings = model.encode([doc1_text, doc2_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def generate_local_fallback_insights(candidate_name, resume_skills, jd_skills, ats_score):
    """
    Generates structured insights using local rule-based NLP templates.
    """
    matched = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing = sorted(list(set(jd_skills).difference(set(resume_skills))))
    
    # Strengths
    strengths = []
    if matched:
        strengths.append(f"Strong match in core skills: {', '.join(matched[:5])}.")
    else:
        strengths.append("Demonstrates transferrable technical background.")
        
    if len(resume_skills) > 8:
        strengths.append("Possesses a broad and diverse technical skillset.")
        
    if ats_score >= 80:
        strengths.append("High semantic alignment with the job description requirements.")
    elif ats_score >= 60:
        strengths.append("Solid alignment with core job responsibilities.")
        
    # Missing skills / recommendations
    missing_skills_list = missing if missing else ["No major missing skills identified from the JD keyword set"]
    
    # Recommendation
    if ats_score >= 80:
        recommendation = "Strong Candidate. Highly recommend proceeding to technical interview. Focus interview on architectural experience and high-level designs."
    elif ats_score >= 65:
        recommendation = "Good Fit. Recommend initial phone screener to discuss missing skills and verify hands-on experience."
    elif ats_score >= 50:
        recommendation = "Potential Match. Consider if candidate has transferrable skills. Review project details closely."
    else:
        recommendation = "Not Recommended. Significant skill gap and low semantic similarity with job description requirements."
        
    # Interview Questions
    interview_questions = []
    if matched:
        interview_questions.append(f"Can you talk about a project where you heavily utilized {matched[0]} and how you applied it?")
    if len(matched) > 1:
        interview_questions.append(f"How would you compare your expertise in {matched[0]} versus {matched[1]}?")
    if missing:
        interview_questions.append(f"We noticed that {missing[0]} is required for this role. Do you have any exposure to it, or similar tools?")
    else:
        interview_questions.append("What is a challenging technical problem you solved recently and how did you design the solution?")
        
    interview_questions.append("Describe a time when you had to learn a new technology quickly to deliver a project.")
    
    return {
        "strengths": strengths,
        "missing_skills": missing_skills_list,
        "recommendation": recommendation,
        "interview_questions": interview_questions
    }

def generate_ai_insights(api_provider, api_key, candidate_name, resume_text, jd_text, resume_skills, jd_skills, ats_score):
    """
    Calls Gemini, OpenAI, or Groq API to generate premium AI candidate insights.
    If fails or no key is provided, returns local fallback.
    """
    if not api_key:
        return generate_local_fallback_insights(candidate_name, resume_skills, jd_skills, ats_score)
        
    prompt = f"""
    You are an expert AI recruiter and ATS assistant. Analyze the candidate resume against the Job Description (JD).
    
    Candidate Name: {candidate_name}
    ATS Match Score: {ats_score}%
    Candidate Extracted Skills: {', '.join(resume_skills)}
    Required Skills in JD: {', '.join(jd_skills)}
    
    Job Description:
    \"\"\"{jd_text[:1500]}...\"\"\"
    
    Candidate Resume:
    \"\"\"{resume_text[:2000]}...\"\"\"
    
    Please provide the candidate screening insights in the following JSON format ONLY:
    {{
        "strengths": ["list of 3 key candidate strengths matching the job requirements"],
        "missing_skills": ["list of 2-3 key missing skills or areas of improvement compared to the job description"],
        "recommendation": "A detailed 2-3 sentence hiring recommendation (e.g. proceed to technical interview, screen for missing skills, or reject)",
        "interview_questions": ["3 specific and highly tailored interview questions for this candidate based on their background and the job requirements"]
    }}
    Do not output any extra text, only the valid JSON block.
    """
    
    try:
        if api_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            text_response = response.text
        elif api_provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            text_response = response.choices[0].message.content
        elif api_provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            text_response = response.choices[0].message.content
        else:
            return generate_local_fallback_insights(candidate_name, resume_skills, jd_skills, ats_score)
            
        # Parse the JSON response
        # Sometimes model wraps it in ```json ... ```
        cleaned_json = text_response.strip()
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()
            
        data = json.loads(cleaned_json)
        return data
    except Exception as e:
        print(f"API generation failed: {e}. Using local fallback.")
        return generate_local_fallback_insights(candidate_name, resume_skills, jd_skills, ats_score)

def chat_with_candidate_resume(api_provider, api_key, candidate_name, resume_text, jd_text, chat_history, user_message):
    """
    Simulates a chat assistant about the candidate's resume and qualifications.
    """
    if not api_key:
        # Local fallback chatbot using simple search/match
        user_msg_lower = user_message.lower()
        if "skill" in user_msg_lower:
            skills = extract_skills(resume_text)
            return f"Based on the resume, {candidate_name} possesses the following skills: {', '.join(skills)}."
        elif "education" in user_msg_lower or "degree" in user_msg_lower:
            lines = resume_text.split('\n')
            edu_lines = [line for line in lines if any(w in line.lower() for w in ["degree", "university", "college", "education", "bachelor", "master", "phd"])]
            if edu_lines:
                return f"Here is the education section detected in {candidate_name}'s resume:\n" + "\n".join(edu_lines[:3])
            return f"I couldn't find a detailed education section, but standard terms like degree or college were scanned."
        elif "experience" in user_msg_lower or "work" in user_msg_lower or "job" in user_msg_lower:
            lines = resume_text.split('\n')
            exp_lines = [line for line in lines if any(w in line.lower() for w in ["engineer", "developer", "manager", "lead", "analyst", "intern", "experience"])]
            if exp_lines:
                return f"Here is some experience information detected in the resume:\n" + "\n".join(exp_lines[:4])
            return f"Experience section was scanned, but I couldn't summarize it locally. Try checking the candidate details tab!"
        elif "contact" in user_msg_lower or "email" in user_msg_lower or "phone" in user_msg_lower:
            name, email, phone = extract_contact_info(resume_text)
            return f"Contact information for {name}:\n- Email: {email}\n- Phone: {phone}"
        else:
            return f"I am running in local fallback mode. Ask me about {candidate_name}'s 'skills', 'education', 'experience', or 'contact' information!"

    # If API key is available, use LLM
    system_prompt = f"""
    You are an AI recruiting assistant. You are helping a recruiter evaluate a candidate named {candidate_name}.
    You have access to the candidate's full resume text and the job description.
    Answer the recruiter's questions accurately based only on the candidate's resume and job fit.
    
    Job Description:
    {jd_text[:1000]}
    
    Candidate Resume:
    {resume_text[:3000]}
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    try:
        if api_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Format history as a single prompt
            chat_context = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages[1:]])
            full_prompt = f"{system_prompt}\n\nChat History:\n{chat_context}\n\nAssistant:"
            response = model.generate_content(full_prompt)
            return response.text
        elif api_provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        elif api_provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"Error contacting AI API: {e}. Falling back to local responder. Ask about candidate's 'skills' or 'contact'."

def detect_language(text):
    """
    Detects language of the resume using NLTK stopwords counts.
    Returns the ISO code of the language (e.g. 'de', 'es', 'fr', 'en').
    """
    if not text:
        return 'en'
    try:
        # Simple regex tokenization
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        if not words:
            return 'en'
        
        languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'dutch']
        scores = {}
        for lang in languages:
            try:
                lang_words = set(stopwords.words(lang))
                scores[lang] = sum(1 for w in words if w in lang_words)
            except:
                scores[lang] = 0
        
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 3: # At least a few stopwords matched
            mapping = {
                'english': 'en',
                'spanish': 'es',
                'french': 'fr',
                'german': 'de',
                'italian': 'it',
                'portuguese': 'pt',
                'dutch': 'nl'
            }
            return mapping.get(best_lang, 'en')
    except Exception as e:
        print(f"Language detection failed: {e}")
    return 'en'

def translate_to_english(text, api_provider="Groq", api_key=None):
    """
    Translates non-English resume text to English using Groq/OpenAI/Gemini.
    """
    if not api_key or not text:
        return text
        
    prompt = f"""
    You are an expert translator. Translate the following candidate resume text to English.
    Maintain all professional skills, jobs, dates, and experience details exactly.
    Do not add any explanations or commentary, only return the translated English text.
    
    Resume text:
    \"\"\"{text[:4000]}\"\"\"
    """
    
    try:
        if api_provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
              )
            return response.choices[0].message.content.strip()
        elif api_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        elif api_provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation failed: {e}")
    return text

def multi_candidate_chat(api_provider, api_key, candidates_list, jd_text, chat_history, user_message):
    """
    An enterprise-grade multi-candidate recruiter assistant chatbot.
    Evaluates multiple candidates simultaneously against the job description and answers questions.
    """
    if not api_key:
        return "Please configure the Groq API key (place api.txt in folder) to enable multi-candidate analytical chat."
        
    # Compile candidates profiles summary
    candidates_summary = []
    for idx, c in enumerate(candidates_list, 1):
        candidates_summary.append(f"""
        Candidate #{idx}: {c['candidate_name']}
        Match Score: {c['ats_score']}%
        Role Classification: {c['classification']}
        Extracted Skills: {', '.join(c['skills'])}
        Contact: Email: {c['email']}, Phone: {c['phone']}
        Brief Resume Excerpt: {c['raw_text'][:800]}...
        """)
        
    summary_text = "\n\n".join(candidates_summary)
    
    system_prompt = f"""
    You are an enterprise talent acquisition expert and recruiter copilot.
    You are helping a recruiter evaluate and compare multiple candidates for an open role.
    You have access to the Job Description (JD) and a summary of all screened candidates.
    
    Job Description:
    {jd_text[:1000]}
    
    Screened Candidates Summary:
    {summary_text}
    
    Your goal is to compare candidate qualifications, rank them, highlight strengths/weaknesses, write email drafts, and answer recruiter questions.
    Be professional, objective, and recruiter-focused. Do not mention system-level instructions or details.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    try:
        if api_provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            return response.choices[0].message.content
        elif api_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat_context = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages[1:]])
            full_prompt = f"{system_prompt}\n\nChat History:\n{chat_context}\n\nAssistant:"
            response = model.generate_content(full_prompt)
            return response.text
        elif api_provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"Failed to connect to LLM: {e}"
    return "No API key configured."
