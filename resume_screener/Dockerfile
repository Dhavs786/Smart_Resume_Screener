# Use a stable, official Python slim image
FROM python:3.10-slim

# Set environment variables to optimize Python within the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Install basic build system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency specifications and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache NLTK resources to avoid runtime download overhead
RUN python -c "import nltk; \
    nltk.download('stopwords'); \
    nltk.download('punkt'); \
    nltk.download('punkt_tab'); \
    nltk.download('wordnet'); \
    nltk.download('omw-1.4'); \
    nltk.download('averaged_perceptron_tagger')"

# Pre-cache the SentenceTransformer model to prevent slow container startups
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the application on startup
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
