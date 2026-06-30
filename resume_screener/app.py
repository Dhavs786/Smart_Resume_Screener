import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import utils
import database
import streamlit as st
import pandas as pd
import numpy as np
import json
from io import BytesIO
from urllib.parse import quote
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor

# Set page configuration with a premium title and icon
st.set_page_config(
    page_title="TalentAI | Advanced Resume Screener & Ranking Cockpit",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
    /* Premium font family and background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Main title layout styling */
    .title-gradient {
        background: linear-gradient(135deg, #FF4B4B, #FF8F00, #9E00FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        font-size: 1.15rem;
        color: #7C7C8C;
        margin-bottom: 25px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E1E2F, #12121E);
        border: 1px solid #2D2D44;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #7000FF;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #9D9DAE;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 10px 0;
    }
    
    /* Profile Badge Tag pills */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    
    .tag-match {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00E676;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    
    .tag-missing {
        background-color: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid rgba(255, 23, 68, 0.3);
    }
    
    .tag-normal {
        background-color: rgba(255, 255, 255, 0.08);
        color: #E0E0E0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Custom container designs */
    .dashboard-section {
        background-color: #0E1117;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate PDF resumes dynamically
def create_sample_resume_pdf(filename, name, contact, email, education, experience, skills, projects, certifications=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    # Custom reportlab styles
    name_style = ParagraphStyle(
        'NameStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=HexColor('#1E1E2F')
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=HexColor('#555555')
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=HexColor('#7000FF'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=HexColor('#333333'),
        spaceAfter=4
    )

    story = []
    
    # Header block
    story.append(Paragraph(name, name_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Email: {email} | Phone: {contact}", meta_style))
    story.append(Spacer(1, 10))
    
    # Skills section
    story.append(Paragraph("TECHNICAL SKILLS", heading_style))
    story.append(Paragraph(skills, body_style))
    story.append(Spacer(1, 10))
    
    # Experience section
    story.append(Paragraph("WORK EXPERIENCE", heading_style))
    for exp in experience:
        story.append(Paragraph(exp, body_style))
    story.append(Spacer(1, 10))
    
    # Education section
    story.append(Paragraph("EDUCATION", heading_style))
    story.append(Paragraph(education, body_style))
    story.append(Spacer(1, 10))
    
    # Projects section
    story.append(Paragraph("PROJECTS", heading_style))
    for proj in projects:
        story.append(Paragraph(proj, body_style))
        
    if certifications:
        story.append(Spacer(1, 10))
        story.append(Paragraph("CERTIFICATIONS", heading_style))
        story.append(Paragraph(certifications, body_style))
        
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

@st.cache_resource
def load_embedding_model():
    return utils.get_sentence_transformer_model()

# Initialize session state variables
if "resumes_processed" not in st.session_state:
    st.session_state.resumes_processed = {}
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "jd_input" not in st.session_state:
    st.session_state.jd_input = ""
if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []
if "job_presets" not in st.session_state:
    st.session_state.job_presets = {
        "Python Machine Learning Engineer": """We are seeking a Python Machine Learning Engineer to design and implement end-to-end ML workflows.
Requirements:
- Strong programming skills in Python and SQL.
- Practical experience with Machine Learning models, Deep Learning, and Natural Language Processing (NLP).
- Hands-on expertise with PyTorch, TensorFlow, Scikit-Learn, Pandas, and NumPy.
- Exposure to cloud technologies like AWS or Docker.
- Experience building REST APIs using FastAPI or Flask is a plus.""",
        
        "Data Scientist": """We are looking for a Data Scientist to analyze complex datasets and extract business insights.
Requirements:
- Proficient in Python, SQL, and R.
- Advanced knowledge of Machine Learning, statistical modeling, and data analytics.
- Proficiency in BI tools like Power BI and Tableau.
- Experience with Pandas, NumPy, Scikit-Learn, and Jupyter notebooks.
- Strong communication skills to present findings to stakeholders.""",
        
        "Full-Stack Web Developer": """Join our team as a Full-Stack Web Developer to build scalable web applications.
Requirements:
- Strong frontend development with Javascript, TypeScript, React, HTML, and CSS.
- Backend proficiency with Node.js, Express, FastAPI, or Django.
- Experience with databases like PostgreSQL, MySQL, and MongoDB.
- Git, CI/CD, Docker deployment workflows.
- Ability to build modern responsive web UI/UX."""
    }

# ----------------- SIDEBAR -----------------
# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135682.png", width=70)
    st.markdown("<h2 style='margin-top:0;'>Dharav Antani Cockpit</h2>", unsafe_allow_html=True)
    st.markdown("Automated candidate screening and talent matching engine to instantly identify the best fit for your open roles.")
    st.markdown("---")
    
    # LLM Settings (Automatic backend detection, no UI selector/input)
    api_provider = "Groq"
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Auto-detect Groq key in local api.txt file if env var is empty
    if not api_key:
        api_paths = ["api.txt", "../api.txt", "resume_screener/api.txt"]
        for path in api_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        k = f.read().strip()
                        if k:
                            api_key = k
                            break
                except Exception as e:
                    pass
                
    if api_key:
        st.markdown("""
        <div style="background-color:rgba(0, 230, 118, 0.05); border:1px solid #00E676; padding:15px; border-radius:10px; margin-bottom:15px;">
            <div style="font-weight:700; color:#00E676; font-size:0.95rem;">⚡ AI Insights Enabled</div>
            <div style="font-size:0.8rem; color:#A0A0B0; margin-top:5px;">Connected to Groq Cloud LLM engine for premium candidate profiles and interview generation.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:rgba(255, 143, 0, 0.05); border:1px solid #FF8F00; padding:15px; border-radius:10px; margin-bottom:15px;">
            <div style="font-weight:700; color:#FF8F00; font-size:0.95rem;">⚠️ Local Engine Active</div>
            <div style="font-size:0.8rem; color:#A0A0B0; margin-top:5px;">Place api.txt containing a Groq API key in the folder to activate AI insights and profile reviews.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("### 💾 Session Persistence")
    
    # Save active session if it exists
    if st.session_state.resumes_processed:
        with st.form("save_session_form"):
            save_name = st.text_input("Save active run as:", placeholder="e.g. Senior Python Developer")
            save_btn = st.form_submit_button("Save Run to History")
            if save_btn:
                if save_name.strip():
                    with st.spinner("Saving screening run..."):
                        # Load embedding model to generate vectors for persistent search
                        model = load_embedding_model()
                        job_id = database.save_screening_run(
                            save_name.strip(),
                            st.session_state.resumes_processed["jd_text"],
                            st.session_state.resumes_processed["candidates"],
                            model=model
                        )
                        st.success(f"Session saved successfully!")
                        st.rerun()
                else:
                    st.error("Please enter a name for the saved run.")
                    
    # Load previously saved sessions
    saved_runs = database.load_saved_runs()
    if saved_runs:
        st.markdown("##### Reload Saved Session:")
        run_options = {f"{r['title']} ({r['created_at'].split(' ')[0]})": r['id'] for r in saved_runs}
        selected_run_label = st.selectbox("Select past run", list(run_options.keys()))
        
        col_load, col_del = st.columns(2)
        with col_load:
            if st.button("Reload Run", use_container_width=True):
                run_id = run_options[selected_run_label]
                job_data, candidates = database.load_run_details(run_id)
                st.session_state.resumes_processed = {
                    "jd_text": job_data["description"],
                    "jd_skills": utils.extract_skills(job_data["description"]),
                    "candidates": candidates
                }
                # Pre-fill active JD input
                st.session_state.jd_input = job_data["description"]
                st.success("Session reloaded!")
                st.rerun()
                
        with col_del:
            if st.button("Delete Run", use_container_width=True):
                run_id = run_options[selected_run_label]
                database.delete_run(run_id)
                st.warning("Session deleted.")
                st.rerun()
    else:
        st.caption("No saved runs found in database yet.")
        
    st.markdown("---")
    st.caption("Version 1.3.0 | © 2026 Dharav Antani")

# ----------------- MAIN LAYOUT -----------------
st.markdown("<h1 class='title-gradient'>Dharav Antani Resume Ranking Cockpit</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Next-Gen semantic parsing and AI screening engine for modern recruiting.</div>", unsafe_allow_html=True)

# Create tabs for application features
tab_dashboard, tab_multijob, tab_search, tab_copilot, tab_howitworks = st.tabs([
    "🎯 Screening Dashboard", 
    "⚖️ Multi-Job Alignment",
    "🔍 Talent Pool Search",
    "💬 Recruiter Copilot",
    "📖 User Guide & FAQs"
])

# ----------------- TAB: SCREENING DASHBOARD -----------------
with tab_dashboard:
    col_jd, col_upload = st.columns([1, 1])
    
    with col_jd:
        st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
        st.markdown("### 📋 Job Description (JD)")
        
        jd_text = st.text_area("Enter Job Description Requirements here", value="", height=220, placeholder="Paste details of the open job role, required skills, and experience criteria here...")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_upload:
        st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload Candidate Resumes")
        uploaded_files = st.file_uploader("Upload resumes in PDF format (multiple files supported)", type="pdf", accept_multiple_files=True)
        st.markdown("<p style='font-size:0.85rem; color:#7C7C8C;'>Note: Upload multiple resumes to rank them in order of job fitness.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Process and Screen button
    st.markdown("<br/>", unsafe_allow_html=True)
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
    with col_btn_2:
        screen_button = st.button("🚀 EXECUTE SEMANTIC CANDIDATE SCREENING", use_container_width=True)
        
    if screen_button:
        if not jd_text.strip():
            st.error("Please enter a Job Description to compare candidate resumes against.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF candidate resume.")
        else:
            with st.status("Initializing Screening Assistant...") as status:
                status.write("Loading matching profiles database...")
                model = load_embedding_model()
                
                status.write("Analyzing job description requirements...")
                clean_jd, jd_tokens = utils.preprocess_text(jd_text)
                jd_skills = utils.extract_skills(jd_text)
                
                processed_results = []
                
                for idx, file in enumerate(uploaded_files):
                    status.write(f"[{idx+1}/{len(uploaded_files)}] Scanning resume: {file.name}...")
                    raw_text = utils.extract_text_from_pdf(file)
                    
                    # Language detection & translation fallback
                    detected_lang = utils.detect_language(raw_text)
                    if detected_lang != 'en':
                        status.write(f"[{idx+1}/{len(uploaded_files)}] Detected resume in non-English ({detected_lang.upper()}). Translating...")
                        translated_text = utils.translate_to_english(raw_text, api_provider=api_provider, api_key=api_key)
                        
                        status.write(f"[{idx+1}/{len(uploaded_files)}] Processing candidate profile...")
                        clean_text, resume_tokens = utils.preprocess_text(translated_text)
                        
                        status.write(f"[{idx+1}/{len(uploaded_files)}] Extracting candidate skills...")
                        resume_skills = utils.extract_skills(translated_text)
                    else:
                        status.write(f"[{idx+1}/{len(uploaded_files)}] Processing candidate profile...")
                        clean_text, resume_tokens = utils.preprocess_text(raw_text)
                        
                        status.write(f"[{idx+1}/{len(uploaded_files)}] Extracting candidate skills...")
                        resume_skills = utils.extract_skills(raw_text)
                    
                    status.write(f"[{idx+1}/{len(uploaded_files)}] Categorizing candidate experience...")
                    classification = utils.classify_resume(resume_skills)
                    
                    status.write(f"[{idx+1}/{len(uploaded_files)}] Evaluating job description alignment...")
                    similarity = utils.get_semantic_similarity(model, clean_jd, clean_text)
                    
                    status.write(f"[{idx+1}/{len(uploaded_files)}] Generating overall fit match score...")
                    ats_score, breakdown = utils.calculate_ats_score(
                        raw_text, jd_text, resume_skills, jd_skills, similarity
                    )
                    
                    candidate_name, email, phone = utils.extract_contact_info(raw_text)
                    
                    processed_results.append({
                        "file_name": file.name,
                        "candidate_name": candidate_name if candidate_name != "Candidate" else file.name.split(".pdf")[0].replace("_", " ").title(),
                        "email": email,
                        "phone": phone,
                        "raw_text": raw_text,
                        "clean_text": clean_text,
                        "skills": resume_skills,
                        "classification": classification,
                        "ats_score": ats_score,
                        "similarity": similarity,
                        "breakdown": breakdown
                    })
                    
                # Store in session state
                st.session_state.resumes_processed = {
                    "jd_text": jd_text,
                    "jd_skills": jd_skills,
                    "candidates": processed_results
                }
                status.update(label="Screening complete! Top candidates ranked below.", state="complete")

    # Render ranking table if results are loaded
    if st.session_state.resumes_processed:
        jd_skills = st.session_state.resumes_processed["jd_skills"]
        candidates = st.session_state.resumes_processed["candidates"]
        
        # Sort candidates by ATS score descending
        candidates_sorted = sorted(candidates, key=lambda x: x["ats_score"], reverse=True)
        
        st.markdown("## 📊 Candidate Talent Ranking")
        st.markdown("Use the filters below to dynamically narrow down your candidate search in real-time.")
        
        # Filtering Dashboard
        st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Real-Time Candidate Filters")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            score_filter = st.slider("Minimum Match Score (%)", min_value=0, max_value=100, value=0, step=5)
        with col_f2:
            all_classes = sorted(list(set(c.get("classification", "General Technical Professional") for c in candidates_sorted)))
            class_filter = st.multiselect("Filter by Role Classification", options=all_classes, default=all_classes)
        with col_f3:
            search_query = st.text_input("Search Candidate Name", value="", placeholder="Type candidate name...")
        st.markdown("</div>", unsafe_allow_html=True)
            
        # Apply filters
        candidates_filtered = [
            c for c in candidates_sorted
            if c["ats_score"] >= score_filter
            and c.get("classification", "General Technical Professional") in class_filter
            and (search_query.lower() in c["candidate_name"].lower() if search_query else True)
        ]
        
        if not candidates_filtered:
            st.warning("⚠️ No candidates match the current filter criteria. Try adjusting your filters.")
        else:
            # Display main stats cockpit based on filtered data
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Total Filtered</div>
                    <div class='metric-value'>{len(candidates_filtered)} / {len(candidates_sorted)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Top Candidate</div>
                    <div class='metric-value'>{candidates_filtered[0]["candidate_name"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                avg_score = round(sum(c["ats_score"] for c in candidates_filtered)/len(candidates_filtered), 1)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Average Match Score</div>
                    <div class='metric-value'>{avg_score}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                above_threshold = sum(1 for c in candidates_filtered if c["ats_score"] >= 70)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Top Candidates (>=70%)</div>
                    <div class='metric-value'>{above_threshold}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # Interactive Candidate Table dataframe
            table_data = []
            for rank, cand in enumerate(candidates_filtered, 1):
                matched_skills_count = len(set(cand["skills"]).intersection(set(jd_skills)))
                linkedin_query = cand['candidate_name'].strip()
                linkedin_url = f"https://www.linkedin.com/search/results/people/?keywords={quote(linkedin_query)}"
                table_data.append({
                    "Rank": rank,
                    "Candidate Name": cand["candidate_name"],
                    "Role Classification": cand.get("classification", "General Technical Professional"),
                    "Match Score": f"{cand['ats_score']}%",
                    "Skills Match": f"{matched_skills_count} / {len(jd_skills)}",
                    "Email": cand["email"],
                    "Phone": cand["phone"],
                    "LinkedIn Profile": linkedin_url
                })
                
            df_ranking = pd.DataFrame(table_data)
            st.dataframe(
                df_ranking, 
                column_config={
                    "LinkedIn Profile": st.column_config.LinkColumn(
                        "LinkedIn Profile",
                        help="Search for candidate on LinkedIn",
                        display_text="Find on LinkedIn"
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Download reports option (includes full filtered report with classifications)
            csv_buffer = BytesIO()
            df_ranking.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Filtered Rankings to CSV",
                data=csv_buffer.getvalue(),
                file_name="candidate_rankings.csv",
                mime="text/csv"
            )
            
        st.markdown("---")
        
        # Select Candidate for Detail deep-dive
        st.markdown("## 🔍 Candidate Premium Evaluation Panel")
        names_list = [c["candidate_name"] for c in candidates_sorted]
        selected_name = st.selectbox("Select Candidate to view deep-dive analysis:", names_list)
        
        # Get active candidate data
        cand_data = next(c for c in candidates_sorted if c["candidate_name"] == selected_name)
        st.session_state.selected_candidate = cand_data
        
        # Display selected candidate deep-dive
        if cand_data:
            col_profile_header, col_score_wheels = st.columns([2, 3])
            
            with col_profile_header:
                # Dynamic LinkedIn Search query using candidate name and top skill
                linkedin_query = cand_data['candidate_name'].strip()
                linkedin_url = f"https://www.linkedin.com/search/results/people/?keywords={quote(linkedin_query)}"
                st.markdown(f"### 👤 {cand_data['candidate_name']} [🔗 Find on LinkedIn]({linkedin_url})")
                st.write(f"**Filename:** `{cand_data['file_name']}`")
                
                # Contact info cards
                st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.03); border:1px solid #2D2D44; padding:15px; border-radius:10px;">
                    <p style="margin:5px 0;">📧 <strong>Email:</strong> <a href="mailto:{cand_data['email']}">{cand_data['email']}</a></p>
                    <p style="margin:5px 0;">📞 <strong>Phone:</strong> {cand_data['phone']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Extracted Resume Skills vs Job Requirements")
                # Show skills match tags
                matched_s = set(cand_data["skills"]).intersection(set(jd_skills))
                missing_s = set(jd_skills).difference(set(cand_data["skills"]))
                other_s = set(cand_data["skills"]).difference(set(jd_skills))
                
                st.markdown("**Matched Required Skills:**")
                if matched_s:
                    for skill in sorted(list(matched_s)):
                        st.markdown(f"<span class='tag tag-match'>{skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("None")
                    
                st.markdown("**Missing Required Skills:**")
                if missing_s:
                    for skill in sorted(list(missing_s)):
                        st.markdown(f"<span class='tag tag-missing'>{skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("None! 100% skill requirements matched.")
                    
                st.markdown("**Other Extracted Resume Skills:**")
                if other_s:
                    for skill in sorted(list(other_s)):
                        st.markdown(f"<span class='tag tag-normal'>{skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("None")
                    
            with col_score_wheels:
                st.markdown("#### 🎯 Score Card Analysis")
                
                # Display individual progress bars
                b = cand_data["breakdown"]
                
                # Overall score indicator
                status_color = "red"
                if cand_data["ats_score"] >= 80:
                    status_color = "#00E676"
                elif cand_data["ats_score"] >= 60:
                    status_color = "#FFD600"
                elif cand_data["ats_score"] >= 45:
                    status_color = "#FF8F00"
                else:
                    status_color = "#FF1744"
                    
                st.markdown(f"""
                <div style="background-color:rgba(112, 0, 255, 0.05); border: 2px solid #7000FF; border-radius:15px; padding:20px; text-align:center; margin-bottom:15px;">
                    <div style="font-size:1.1rem; font-weight:600; text-transform:uppercase; color:#B088FF; letter-spacing:1px;">Overall Job Fit Match</div>
                    <div style="font-size:3.5rem; font-weight:900; color:{status_color}; margin:10px 0;">{cand_data['ats_score']}%</div>
                    <div style="font-size:0.9rem; color:#A0A0B0;">Weighted match score evaluating skills, experience alignment, and profile completeness.</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Score Breakdown
                st.markdown("##### Role Requirements Alignment")
                sim_pct = round(cand_data['similarity']*100, 0)
                sim_label = "Strong Fit" if sim_pct >= 75 else "Moderate Fit" if sim_pct >= 50 else "Basic Alignment"
                st.progress(float(cand_data["similarity"]), text=f"{sim_pct}% ({sim_label})")
                
                st.markdown("##### Key Technical Skills")
                st.progress(b["skills_match_score"]/100.0, text=f"{int(b['skills_match_score'])}% of required skills present")
                
                st.markdown("##### Work & Education Background")
                st.progress(b["experience_education_score"]/100.0, text=f"Profile completeness: {'Complete' if b['experience_education_score'] > 50 else 'Partial'}")
                
                st.markdown("##### Recruitment Details Completeness")
                st.progress(b["formatting_contact_score"]/100.0, text=f"Contact channels: {'Verified' if b['formatting_contact_score'] > 50 else 'Incomplete'}")
                
            st.markdown("---")
            
            # Fetch AI Insights
            with st.spinner("Generating premium insights and recommendations..."):
                ai_data = utils.generate_ai_insights(
                    api_provider,
                    api_key,
                    cand_data["candidate_name"],
                    cand_data["raw_text"],
                    jd_text,
                    cand_data["skills"],
                    jd_skills,
                    cand_data["ats_score"]
                )
                
            # Create three subtabs for analysis
            tab_ai_feedback, tab_questions, tab_chat = st.tabs([
                "✨ AI Candidate Summary", 
                "❓ Tailored Interview Questions", 
                "💬 Chat with Candidate's Resume"
            ])
            
            with tab_ai_feedback:
                st.markdown("### 🏆 AI Screening Report")
                
                col_strengths, col_gaps = st.columns(2)
                with col_strengths:
                    st.markdown("<div style='background-color:rgba(0, 230, 118, 0.05); padding:15px; border-radius:10px; border: 1px solid rgba(0, 230, 118, 0.2); height: 100%;'>", unsafe_allow_html=True)
                    st.markdown("#### 💪 Key Strengths")
                    for strength in ai_data.get("strengths", []):
                        st.markdown(f"- {strength}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_gaps:
                    st.markdown("<div style='background-color:rgba(255, 23, 68, 0.05); padding:15px; border-radius:10px; border: 1px solid rgba(255, 23, 68, 0.2); height: 100%;'>", unsafe_allow_html=True)
                    st.markdown("#### ⚠️ Skill Gaps & Weaknesses")
                    for gap in ai_data.get("missing_skills", []):
                        st.markdown(f"- {gap}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("<div style='background-color:rgba(112, 0, 255, 0.05); padding:20px; border-radius:10px; border: 1px solid rgba(112, 0, 255, 0.2);'>", unsafe_allow_html=True)
                st.markdown("#### 🎯 Hiring Recommendation")
                st.write(ai_data.get("recommendation", "Review candidate portfolio closely."))
                st.markdown("</div>", unsafe_allow_html=True)
                
            with tab_questions:
                st.markdown("### ❓ AI-Generated Candidate Interview Questions")
                st.markdown("These questions are specifically engineered to evaluate the candidate's core strengths and drill into their detected skill gaps:")
                
                for idx, question in enumerate(ai_data.get("interview_questions", []), 1):
                    st.markdown(f"""
                    <div style="background-color:rgba(255,255,255,0.02); border-left:4px solid #7000FF; padding:12px; margin-bottom:10px; border-radius:0 8px 8px 0;">
                        <strong>Question {idx}:</strong> {question}
                    </div>
                    """, unsafe_allow_html=True)
                    
            with tab_chat:
                col_title, col_clear = st.columns([4, 1])
                with col_title:
                    st.markdown(f"### 💬 Ask Questions about {cand_data['candidate_name']}'s Resume")
                with col_clear:
                    if st.button("🗑️ Clear Chat", key=f"clear_{cand_data['candidate_name']}", use_container_width=True):
                        st.session_state.chat_histories[cand_data["candidate_name"]] = []
                        st.rerun()
                        
                st.markdown(f"Interact directly with this candidate's resume parser. Ask things like: *'Which machine learning projects did they build?'* or *'What is their work history?'*")
                
                # Chat Interface
                cand_id = cand_data["candidate_name"]
                if cand_id not in st.session_state.chat_histories:
                    st.session_state.chat_histories[cand_id] = []
                    
                chat_history = st.session_state.chat_histories[cand_id]
                
                # Render previous messages
                for msg in chat_history:
                    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                        st.write(msg["content"])
                
                # Suggestions/Prompt Starters
                st.markdown("<p style='font-size:0.85rem; color:#A0A0B0; margin-bottom:5px;'>💡 Prompt Starters:</p>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                candidate_suggestion = None
                with col_s1:
                    if st.button("📝 Summarize Resume", key=f"s1_{cand_id}", use_container_width=True):
                        candidate_suggestion = "Summarize the candidate's core profile and top strengths."
                with col_s2:
                    if st.button("🛠️ Check Tech Fit", key=f"s2_{cand_id}", use_container_width=True):
                        candidate_suggestion = "Check how well the candidate fits the technical requirements of the job description."
                with col_s3:
                    if st.button("❓ Key Gaps", key=f"s3_{cand_id}", use_container_width=True):
                        candidate_suggestion = "Identify the major gaps or missing skills between this resume and the job description."
                with col_s4:
                    if st.button("✉️ Draft Invite", key=f"s4_{cand_id}", use_container_width=True):
                        candidate_suggestion = "Draft a professional email inviting this candidate to a first-round interview."
                        
                # Message input
                user_msg = st.chat_input("Ask a question about the candidate's resume...", key=f"chat_input_{cand_id}")
                
                active_cand_input = None
                if user_msg:
                    active_cand_input = user_msg
                elif candidate_suggestion:
                    active_cand_input = candidate_suggestion
                    
                if active_cand_input:
                    with st.chat_message("user", avatar="👤"):
                        st.write(active_cand_input)
                    chat_history.append({"role": "user", "content": active_cand_input})
                    
                    with st.spinner("Analyzing resume content..."):
                        response = utils.chat_with_candidate_resume(
                            api_provider,
                            api_key,
                            cand_data["candidate_name"],
                            cand_data["raw_text"],
                            jd_text,
                            chat_history[:-1],
                            active_cand_input
                        )
                        
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(response)
                    chat_history.append({"role": "assistant", "content": response})
                    st.session_state.chat_histories[cand_id] = chat_history
                    st.rerun()

# ----------------- TAB: MULTI-JOB COMPARISON -----------------
with tab_multijob:
    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    st.markdown("### 🔀 Candidate Fitness Across Different Job Profiles")
    st.markdown("Compare candidates side-by-side against multiple target job profiles to find their optimal placement.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not st.session_state.resumes_processed:
        st.info("Please execute semantic screening on the main 'Screening Dashboard' first with candidate resumes.")
    else:
        # We have processed resumes. Let's define multiple job descriptions.
        st.markdown("#### Define Up To 3 Different Job Roles")
        
        col_role1, col_role2, col_role3 = st.columns(3)
        with col_role1:
            title_1 = st.text_input("Role 1 Title", value="Machine Learning Engineer")
            desc_1 = st.text_area("Role 1 JD", value=st.session_state.job_presets["Python Machine Learning Engineer"], height=200)
        with col_role2:
            title_2 = st.text_input("Role 2 Title", value="Data Scientist")
            desc_2 = st.text_area("Role 2 JD", value=st.session_state.job_presets["Data Scientist"], height=200)
        with col_role3:
            title_3 = st.text_input("Role 3 Title", value="Full Stack Developer")
            desc_3 = st.text_area("Role 3 JD", value=st.session_state.job_presets["Full-Stack Web Developer"], height=200)
            
        if st.button("⚖️ Run Comparative Placement Matrix"):
            with st.spinner("Evaluating candidates across all roles..."):
                model = load_embedding_model()
                
                # Preprocess JDs
                jd1_clean, jd1_tokens = utils.preprocess_text(desc_1)
                jd2_clean, jd2_tokens = utils.preprocess_text(desc_2)
                jd3_clean, jd3_tokens = utils.preprocess_text(desc_3)
                
                skills_1 = utils.extract_skills(desc_1)
                skills_2 = utils.extract_skills(desc_2)
                skills_3 = utils.extract_skills(desc_3)
                
                comp_rows = []
                for cand in st.session_state.resumes_processed["candidates"]:
                    # Score for Role 1
                    sim_1 = utils.get_semantic_similarity(model, jd1_clean, cand["clean_text"])
                    score_1, _ = utils.calculate_ats_score(cand["raw_text"], desc_1, cand["skills"], skills_1, sim_1)
                    
                    # Score for Role 2
                    sim_2 = utils.get_semantic_similarity(model, jd2_clean, cand["clean_text"])
                    score_2, _ = utils.calculate_ats_score(cand["raw_text"], desc_2, cand["skills"], skills_2, sim_2)
                    
                    # Score for Role 3
                    sim_3 = utils.get_semantic_similarity(model, jd3_clean, cand["clean_text"])
                    score_3, _ = utils.calculate_ats_score(cand["raw_text"], desc_3, cand["skills"], skills_3, sim_3)
                    
                    # Best Fit Role
                    scores = {title_1: score_1, title_2: score_2, title_3: score_3}
                    best_role = max(scores, key=scores.get)
                    
                    comp_rows.append({
                        "Candidate Name": cand["candidate_name"],
                        f"{title_1} Score": f"{score_1}%",
                        f"{title_2} Score": f"{score_2}%",
                        f"{title_3} Score": f"{score_3}%",
                        "Optimal Alignment": best_role,
                        "Top Fit Score": f"{scores[best_role]}%"
                    })
                    
                df_compare = pd.DataFrame(comp_rows)
                st.markdown("### 📊 Cross-Job Placement Matrix")
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
                # Highlight matrix findings
                st.markdown("#### 💡 Placement Insights")
                for row in comp_rows:
                    st.write(f"- **{row['Candidate Name']}** aligns best with the **{row['Optimal Alignment']}** profile ({row['Top Fit Score']} match).")

# ----------------- TAB: TALENT POOL SEARCH -----------------
with tab_search:
    st.markdown("### 🔍 Semantic Talent Pool Search")
    st.markdown("Search through all candidates stored historically in the local SQLite database using natural language.")
    
    db_search_query = st.text_input("Enter search criteria (role, skills, projects)", placeholder="e.g. FastAPI backend developer with Docker experience")
    
    if db_search_query.strip():
        model = load_embedding_model()
        with st.spinner("Searching talent pool semantically..."):
            search_results = database.semantic_search_all_candidates(model, db_search_query.strip(), top_n=5)
            
        if not search_results:
            st.info("No candidates with valid semantic indexes found in the database. Save some screening runs to populate the pool.")
        else:
            st.markdown(f"#### Top Matches for: *\"{db_search_query}\"*")
            
            # Format display table
            results_data = []
            for idx, res in enumerate(search_results, 1):
                results_data.append({
                    "Rank": idx,
                    "Candidate Name": res["name"],
                    "Match Relevancy": f"{round(res['score']*100, 1)}%",
                    "Role Classification": res["classification"],
                    "Skills": ", ".join(res["skills"]),
                    "Email": res["email"],
                    "Phone": res["phone"],
                    "Original Position Run": res["job_title"]
                })
            df_search = pd.DataFrame(results_data)
            st.dataframe(df_search, use_container_width=True, hide_index=True)
            
            # LinkedIn links helper list
            st.markdown("##### Direct LinkedIn Profiles lookup:")
            for res in search_results:
                linkedin_keywords = res['name'].strip()
                st.markdown(f"- **{res['name']}** ({res['classification']}) — [🔗 Find Profile on LinkedIn](https://www.linkedin.com/search/results/people/?keywords={quote(linkedin_keywords)})")

# ----------------- TAB: AI RECRUITER COPILOT -----------------
with tab_copilot:
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("### 💬 Enterprise AI Recruiter Copilot")
    with col_clear:
        if st.button("🗑️ Clear Copilot", key="clear_copilot", use_container_width=True):
            st.session_state.copilot_messages = []
            st.rerun()
            
    st.markdown("This chatbot has access to all candidates in the active screening pool. Ask it to compare qualifications, summarize profiles, or draft emails.")
    
    if not st.session_state.resumes_processed:
        st.warning("⚠️ No active screening data. Please run a candidate screening dashboard analysis first.")
    else:
        active_candidates = st.session_state.resumes_processed["candidates"]
        active_jd = st.session_state.resumes_processed["jd_text"]
        
        # Display chat messages
        for msg in st.session_state.copilot_messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.write(msg["content"])
                
        # Suggestions/Prompt Starters
        st.markdown("<p style='font-size:0.85rem; color:#A0A0B0; margin-bottom:5px;'>💡 Prompt Starters:</p>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        copilot_suggestion = None
        with col_c1:
            if st.button("⚖️ Compare Top 3", key="cop_s1", use_container_width=True):
                copilot_suggestion = "Compare the qualifications, strengths, and weaknesses of the top 3 candidates side by side."
        with col_c2:
            if st.button("📊 Skills Comparison", key="cop_s2", use_container_width=True):
                copilot_suggestion = "Compare the matching and missing skills of all candidates in the active pool."
        with col_c3:
            if st.button("📧 Invite Top Fit", key="cop_s3", use_container_width=True):
                copilot_suggestion = "Draft a professional recruitment email inviting the top-ranked candidate to an interview."
        with col_c4:
            if st.button("💡 Summarize Pool", key="cop_s4", use_container_width=True):
                copilot_suggestion = "Summarize the key strengths and alignment patterns of the active candidate pool."
                
        # Chat input
        copilot_input = st.chat_input("Ask about candidate comparisons, summaries, or interview logistics...")
        
        active_input = None
        if copilot_input:
            active_input = copilot_input
        elif copilot_suggestion:
            active_input = copilot_suggestion
            
        if active_input:
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.write(active_input)
            st.session_state.copilot_messages.append({"role": "user", "content": active_input})
            
            # Generate assistant response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing active candidate pool..."):
                    assistant_response = utils.multi_candidate_chat(
                        api_provider=api_provider,
                        api_key=api_key,
                        candidates_list=active_candidates,
                        jd_text=active_jd,
                        chat_history=st.session_state.copilot_messages[:-1],
                        user_message=active_input
                    )
                    st.write(assistant_response)
            st.session_state.copilot_messages.append({"role": "assistant", "content": assistant_response})
            st.rerun()

# ----------------- TAB: USER GUIDE & FAQS -----------------
with tab_howitworks:
    st.markdown("### 📖 Recruiter User Guide & FAQs")
    st.markdown("Welcome to TalentAI! This guide helps you navigate the features and understand how candidate matching works under the hood.")
    
    col_guide1, col_guide2 = st.columns(2)
    with col_guide1:
        st.markdown("""
        #### 🎯 How the Match Score is Calculated
        Rather than simple keyword searching, TalentAI uses advanced language understanding to score candidates:
        * **50% Role Alignment**: Measures how closely the candidate's projects and experience match the overall responsibilities described in the Job Description.
        * **30% Key Skills Match**: Verifies the presence of exact mandatory technical skills specified in the requirements.
        * **10% Profile Completeness**: Confirms that key sections like work history and educational details are clearly structured.
        * **10% Contact Completeness**: Verifies that standard recruiter contact points (email, phone, and name) are present.
        
        #### 💾 Saving & Reloading Runs
        To store a candidate list across app reloads, use the **Session Persistence** panel in the sidebar:
        1. Type a session title and click **Save Run to History**.
        2. Load any past run instantly from the select box dropdown.
        """)
        
    with col_guide2:
        st.markdown("""
        #### ⚖️ Using the Multi-Job Alignment Matrix
        Sometimes a candidate is a weak fit for one role but perfect for another:
        1. Open the **Multi-Job Alignment** tab.
        2. Define up to three different open positions (e.g. Developer, Data Analyst, Manager).
        3. Click **Run Comparative Placement Matrix** to see a side-by-side comparison of where each candidate aligns best.
        
        #### 🔍 Searching Candidates Semantically
        Go to the **Talent Pool Search** tab and describe your ideal profile in natural language (e.g., *"React frontend dev with Docker experience"*). The system will search and rank all past candidates in the database.
        
        #### 🤖 Customizing AI Insights
        To unlock premium candidate feedback and direct resume chat:
        1. Obtain an API Key (e.g. from Groq, Gemini, or OpenAI).
        2. Input the key in the sidebar panel.
        3. The system will automatically generate custom summaries, key strengths, tailored interview questions, and activate a chat box to answer candidate-specific questions.
        """)
