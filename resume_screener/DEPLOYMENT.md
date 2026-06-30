# 🚀 Dharav Antani Resume Screener Deployment Guide (Non-AWS)

This guide walks you through the steps to deploy the Dharav Antani Resume Screener Cockpit to **Streamlit Community Cloud**, **Render**, or **Google Cloud Run**, including setting up persistent database volumes.

---

## 🎨 Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit Community Cloud is the easiest and most performant way to host the application for free.

### Step 1: Prepare Repository
1. Push your source code (with `.streamlit/config.toml`, `requirements.txt`, etc.) to your GitHub repository.
2. Ensure you have the `.gitignore` configured to omit local databases or api key txt files.

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app** in the top right.
3. Select your repository, branch, and specify `resume_screener/app.py` as the main file path.
4. Click **Deploy**.

### Step 3: Configure Environment Variables & Database Persistence
Streamlit Community Cloud does not support persistent local storage disks, but you can configure the app to read/write database entries seamlessly:
1. In your Streamlit App Dashboard, click **Settings** (gear icon) in the bottom right corner.
2. Select **Secrets**.
3. Add your Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_key_here"
   ```
4. Click **Save**.

---

## 🐋 Option 2: Render (Docker Container with Persistent Database)
Render allows you to host web applications inside Docker containers, which is excellent for persistent volumes and isolated environments.

### Step 1: Connect Render to GitHub
1. Go to [Render](https://render.com/) and sign up.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.

### Step 2: Configure Build Settings
Render will automatically detect the `Dockerfile` inside your repository.
- **Runtime**: `Docker`
- **Plan**: `Starter` (recommended for Sentence Transformer embeddings, requiring 1GB+ RAM).

### Step 3: Mount a Persistent Disk for your SQLite Database
To ensure your screening history and candidate profiles are not lost when the container restarts:
1. Navigate to your Web Service dashboard on Render.
2. Click on **Disks** in the sidebar.
3. Click **Add Disk**:
   - **Name**: `db-volume`
   - **Mount Path**: `/data`
   - **Size**: `1 GB` (more than enough for thousands of candidate rows)
4. Click **Save**.

### Step 4: Define Environment Variables
Navigate to the **Environment** tab on Render and add the following:
- Key: `PORT` | Value: `8501`
- Key: `KMP_DUPLICATE_LIB_OK` | Value: `TRUE`
- Key: `GROQ_API_KEY` | Value: `gsk_your_actual_groq_key_here`
- Key: `DATABASE_PATH` | Value: `/data/talent_screener.db` (This points your database file to the mounted persistent volume!)

Render will automatically deploy and hook your application up to the persistent disk!

---

## ☁️ Option 3: Google Cloud Run (Serverless Docker with Google Cloud Storage / Volume Mounts)
Google Cloud Run is a fully managed serverless platform that automatically scales containerized applications.

### Step 1: Build the Docker Image
Build the container image using Google Cloud Build:
```bash
gcloud builds submit --tag gcr.io/[PROJECT_ID]/resume-screener
```

### Step 2: Create a Cloud Storage bucket or Volume Mount for Persistence
Cloud Run allows mounting Cloud Storage buckets as local directories:
1. Create a GCS bucket: `gs://[PROJECT_ID]-screener-db`
2. Grant the Cloud Run service account access to read/write to the bucket.

### Step 3: Deploy to Cloud Run
Deploy the container with a volume mount:
```bash
gcloud run deploy resume-screener \
    --image gcr.io/[PROJECT_ID]/resume-screener \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --update-env-vars GROQ_API_KEY=gsk_your_actual_groq_key_here,DATABASE_PATH=/mnt/db/talent_screener.db \
    --add-volume name=db-bucket,type=cloud-storage,bucket=[PROJECT_ID]-screener-db \
    --add-volume-mount volume=db-bucket,mount-path=/mnt/db
```

Once the deployment completes, Cloud Run will provide a secure HTTPS URL for your recruiter cockpit with full database persistence across scale-to-zero periods!
