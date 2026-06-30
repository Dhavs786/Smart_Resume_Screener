import sqlite3
import json
import numpy as np
import os
from datetime import datetime

DB_PATH = "talent_screener.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite tables for storing screening history and candidate profiles.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create jobs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    
    # Create candidates table with vector embedding blob
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        name TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        raw_text TEXT,
        clean_text TEXT,
        skills TEXT, -- JSON string
        classification TEXT,
        match_score REAL,
        similarity_score REAL,
        breakdown TEXT, -- JSON string
        embedding BLOB, -- binary serialized float32 numpy array
        created_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()

def save_screening_run(job_title, job_description, candidates_list, model=None):
    """
    Saves a screening run (Job and associated processed candidates) to the database.
    If a SentenceTransformer model is provided, it generates and stores the candidate embeddings.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Insert job details
    cursor.execute(
        "INSERT INTO jobs (title, description, created_at) VALUES (?, ?, ?)",
        (job_title, job_description, now_str)
    )
    job_id = cursor.lastrowid
    
    # Insert each candidate
    for cand in candidates_list:
        skills_json = json.dumps(cand.get("skills", []))
        breakdown_json = json.dumps(cand.get("breakdown", {}))
        
        # Generate embedding if model is present and clean text exists
        emb_blob = None
        if model is not None and cand.get("clean_text"):
            try:
                emb = model.encode(cand["clean_text"])
                # Ensure it is float32 numpy array
                emb_arr = np.array(emb, dtype=np.float32)
                emb_blob = emb_arr.tobytes()
            except Exception as e:
                print(f"Error generating embedding for DB storage: {e}")
                
        cursor.execute("""
        INSERT INTO candidates (
            job_id, name, email, phone, raw_text, clean_text, 
            skills, classification, match_score, similarity_score, 
            breakdown, embedding, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            cand["candidate_name"],
            cand.get("email", "Not Found"),
            cand.get("phone", "Not Found"),
            cand.get("raw_text", ""),
            cand.get("clean_text", ""),
            skills_json,
            cand.get("classification", "General Technical Professional"),
            cand.get("ats_score", 0.0),
            cand.get("similarity", 0.0),
            breakdown_json,
            emb_blob,
            now_str
        ))
        
    conn.commit()
    conn.close()
    return job_id

def load_saved_runs():
    """
    Returns all saved screening runs (jobs) from the database.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM jobs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def load_run_details(job_id):
    """
    Returns the Job details and the list of candidates belonging to the job.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get Job info
    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job_row = cursor.fetchone()
    if not job_row:
        conn.close()
        return None, []
        
    job_data = dict(job_row)
    
    # Get Candidates info
    cursor.execute("SELECT * FROM candidates WHERE job_id = ?", (job_id,))
    cand_rows = cursor.fetchall()
    
    candidates = []
    for r in cand_rows:
        cand_dict = dict(r)
        # Deserialize JSONs
        try:
            cand_dict["skills"] = json.loads(cand_dict["skills"])
        except:
            cand_dict["skills"] = []
            
        try:
            cand_dict["breakdown"] = json.loads(cand_dict["breakdown"])
        except:
            cand_dict["breakdown"] = {}
            
        # Re-map DB column names to streamlit keys
        candidates.append({
            "candidate_name": cand_dict["name"],
            "file_name": cand_dict["name"] + ".pdf",
            "email": cand_dict["email"],
            "phone": cand_dict["phone"],
            "raw_text": cand_dict["raw_text"],
            "clean_text": cand_dict["clean_text"],
            "skills": cand_dict["skills"],
            "classification": cand_dict["classification"],
            "ats_score": cand_dict["match_score"],
            "similarity": cand_dict["similarity_score"],
            "breakdown": cand_dict["breakdown"]
        })
        
    conn.close()
    return job_data, candidates

def delete_run(job_id):
    """
    Deletes a screening run and its cascade candidates.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    cursor.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()

def semantic_search_all_candidates(model, query_text, top_n=5):
    """
    Performs cosine similarity search against all stored candidates in the database.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Select candidates that have valid embeddings
    cursor.execute("""
    SELECT c.name, c.email, c.phone, c.classification, c.skills, c.clean_text, c.embedding, j.title as job_title 
    FROM candidates c
    JOIN jobs j ON c.job_id = j.id
    WHERE c.embedding IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    # Embed the query
    query_vector = np.array(model.encode(query_text), dtype=np.float32)
    
    results = []
    for r in rows:
        db_emb_bytes = r["embedding"]
        if not db_emb_bytes:
            continue
            
        db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(query_vector, db_emb)
        norm_q = np.linalg.norm(query_vector)
        norm_db = np.linalg.norm(db_emb)
        if norm_q > 0 and norm_db > 0:
            score = float(dot_product / (norm_q * norm_db))
        else:
            score = 0.0
            
        # Parse skills
        try:
            skills = json.loads(r["skills"])
        except:
            skills = []
            
        results.append({
            "name": r["name"],
            "email": r["email"],
            "phone": r["phone"],
            "classification": r["classification"],
            "skills": skills,
            "job_title": r["job_title"],
            "score": score
        })
        
    # Sort descending by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_n]
