import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

try:
    import utils
    print("utils.py imported successfully!")
except Exception as e:
    print("Error importing utils.py:", e)
    sys.exit(1)

# Test PDF generation and extraction
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from io import BytesIO
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("John Doe", styles['Heading1']), Paragraph("Email: john.doe@example.com | Phone: +1234567890", styles['Normal']), Paragraph("Skills: Python, Machine Learning, SQL", styles['Normal'])]
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Test extract text
    pdf_file = BytesIO(pdf_bytes)
    text = utils.extract_text_from_pdf(pdf_file)
    print("Extracted text successfully!")
    print("Text Preview:", text[:150])
    
    # Test Name extraction
    name, email, phone = utils.extract_contact_info(text)
    print(f"Extracted Contact - Name: {name}, Email: {email}, Phone: {phone}")
    
    # Test preprocess text
    clean, tokens = utils.preprocess_text(text)
    print("Preprocessed tokens:", tokens)
    
    # Test skill extraction
    skills = utils.extract_skills(text)
    print("Extracted skills:", skills)
    
    # Test embeddings
    print("Loading Sentence Transformer model...")
    model = utils.get_sentence_transformer_model()
    sim = utils.get_semantic_similarity(model, "Python and Machine learning role", clean)
    print("Semantic similarity:", sim)
    
    # Test ATS Score
    score, breakdown = utils.calculate_ats_score(text, "Python Machine Learning", skills, ["Python", "Machine Learning"], sim)
    print("ATS Score:", score)
    print("Breakdown:", breakdown)
    
    # Test SQLite Database Persistence
    print("Initializing Database test...")
    import database
    database.init_db()
    
    cand_data = {
        "candidate_name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1234567890",
        "raw_text": text,
        "clean_text": clean,
        "skills": skills,
        "classification": "Machine Learning Engineer",
        "ats_score": score,
        "similarity": sim,
        "breakdown": breakdown
    }
    
    print("Saving test run...")
    job_id = database.save_screening_run("Test Python Role", "Python Machine Learning JD", [cand_data], model=model)
    print(f"Run saved successfully with Job ID: {job_id}")
    
    print("Loading test run...")
    job_data, candidates = database.load_run_details(job_id)
    print(f"Loaded Job: {job_data['title']}, Candidates count: {len(candidates)}")
    assert job_data['title'] == "Test Python Role"
    assert len(candidates) == 1
    assert candidates[0]['candidate_name'] == "John Doe"
    
    print("Testing semantic talent search...")
    search_res = database.semantic_search_all_candidates(model, "Python expert", top_n=3)
    print(f"Search results count: {len(search_res)}")
    assert len(search_res) > 0
    print(f"Top matched candidate: {search_res[0]['name']} (Score: {search_res[0]['score']})")
    
    # Clean up test run
    database.delete_run(job_id)
    print("Test run deleted from database.")
    
    print("ALL TESTS PASSED SUCCESSFULLY!")
except Exception as e:
    print("Test failed with error:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
