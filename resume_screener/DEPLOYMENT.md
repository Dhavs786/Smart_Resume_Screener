# 🚀 TalentAI Deployment Guide (Non-AWS)

This guide walks you through the steps to deploy the TalentAI Resume Screener Cockpit to **Streamlit Community Cloud**, **Render**, or **Google Cloud Run**.

---

## 🎨 Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit Community Cloud is the easiest and most performant way to host Streamlit applications for free.

### Step 1: Prepare Repository
1. Commit and push the `resume_screener/` source code to a public or private GitHub repository.
2. Ensure your repository structure has `app.py`, `utils.py`, `database.py`, and `requirements.txt` at the root directory of the project folder.

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app** in the top right.
3. Select your repository, branch, and specify `app.py` as the main file path.
4. Click **Deploy**.

### Step 3: Configure Environment variables
To enable Groq AI insights in production without exposing keys in the codebase:
1. In your Streamlit App Dashboard, click **Settings** (gear icon) in the bottom right corner.
2. Select **Secrets**.
3. Add your Groq API key in the secrets textbox:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_key_here"
   ```
4. Save. The application will restart and auto-detect this secret.

---

## 🐋 Option 2: Render (Docker Container Deployment)
Render allows you to host web applications inside Docker containers, which is excellent for isolation and custom environments.

### Step 1: Connect Render to GitHub
1. Go to [Render](https://render.com/) and sign up.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository containing the TalentAI code.

### Step 2: Configure Build Settings
Render will automatically detect the `Dockerfile` inside the repository.
- **Runtime**: `Docker`
- **Plan**: `Starter` (or free tier if CPU demands are low, though 1GB+ RAM is recommended for Sentence Transformers).

### Step 3: Define Environment variables
In the Render Web Service settings, navigate to the **Environment** tab and add:
- Key: `PORT` | Value: `8501`
- Key: `KMP_DUPLICATE_LIB_OK` | Value: `TRUE`

For AI insights, place a text file containing the Groq key or set up a secret file path matching `api.txt`.

---

## ☁️ Option 3: Google Cloud Run (Serverless Docker)
Google Cloud Run is a fully managed serverless platform that automatically scales containerized applications.

### Step 1: Build the Docker Image
Build the container image using Google Cloud Build:
```bash
gcloud builds submit --tag gcr.io/[PROJECT_ID]/talentai-cockpit
```

### Step 2: Deploy to Cloud Run
Deploy the container:
```bash
gcloud run deploy talentai-cockpit \
    --image gcr.io/[PROJECT_ID]/talentai-cockpit \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi
```

Once the deployment completes, Cloud Run will provide a secure HTTPS URL for your recruiter cockpit.
