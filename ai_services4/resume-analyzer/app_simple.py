"""
Resume Analyzer - Simplified Version for Quick Testing
Works without scikit-learn and spacy dependencies
"""

import os
import time
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Basic PDF processing
import PyPDF2
import pdfplumber
from io import BytesIO

app = FastAPI(
    title="Resume Analyzer - Simplified",
    description="Basic resume analysis without heavy ML dependencies",
    version="1.0.0"
)

# CORS configuration - Updated for Codespaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "http://localhost:5178", 
        "http://127.0.0.1:5178",
        "https://noxious-spell-q7qvvw9p66rp357v-5173.app.github.dev"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes) -> str:
    """Extract text from PDF using multiple methods"""
    text = ""
    
    # Method 1: PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
    
    # Method 2: pdfplumber if PyPDF2 fails
    if not text.strip():
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}")
    
    return text

def simple_keyword_analysis(resume_text: str, jd_text: str) -> dict:
    """Simple keyword-based analysis without ML"""
    # Extract keywords (simple word frequency)
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    
    # Find common words (excluding common stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    resume_keywords = resume_words - stop_words
    jd_keywords = jd_words - stop_words
    
    matched_keywords = resume_keywords.intersection(jd_keywords)
    missing_keywords = jd_keywords - resume_keywords
    
    # Calculate simple match score
    if len(jd_keywords) > 0:
        match_score = (len(matched_keywords) / len(jd_keywords)) * 100
    else:
        match_score = 0
    
    return {
        "match_score": round(match_score, 2),
        "matched_keywords": list(matched_keywords)[:20],  # Limit to 20
        "missing_keywords": list(missing_keywords)[:20],  # Limit to 20
        "keyword_coverage": round((len(matched_keywords) / len(jd_keywords)) * 100, 2) if len(jd_keywords) > 0 else 0
    }

@app.get("/")
async def root():
    return {
        "service": "Resume Analyzer - Simplified",
        "version": "1.0.0",
        "status": "ready",
        "algorithms": {
            "similarity": "Keyword-based",
            "pdf": "PyPDF2 + pdfplumber"
        },
        "endpoints": [
            "POST /analyze-skills",
            "POST /test-connection"
        ]
    }

@app.get("/test-connection")
async def test_connection():
    """Test connection endpoint"""
    return {
        "status": "connected",
        "message": "Resume Analyzer is running",
        "timestamp": time.time()
    }

@app.post("/analyze-skills")
async def analyze_skills(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...)
):
    """Analyze resume against job description using simple keyword matching"""
    try:
        print("🎯 /analyze-skills called (Simplified)")
        start_time = time.time()
        
        # Extract text from PDFs
        resume_text = extract_text_from_pdf(await resume_file.read())
        jd_text = extract_text_from_pdf(await jd_file.read())
        
        print(f"📄 Resume length: {len(resume_text)} chars")
        print(f"📄 JD length: {len(jd_text)} chars")
        
        # Perform simple analysis
        analysis_result = simple_keyword_analysis(resume_text, jd_text)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "success": True,
            "match_score": analysis_result["match_score"],
            "keyword_coverage": analysis_result["keyword_coverage"],
            "matched_keywords": analysis_result["matched_keywords"],
            "missing_keywords": analysis_result["missing_keywords"],
            "improvement_tips": [
                f"Add these keywords: {', '.join(analysis_result['missing_keywords'][:10])}",
                "Consider adding more relevant keywords from the job description",
                "Expand your resume to cover more job requirements"
            ],
            "processing_time_ms": round(processing_time, 2),
            "algorithm": "Keyword-based Matching",
            "message": "Analysis completed successfully (Simplified Version)"
        }
        
        print(f"✅ Analysis completed in {processing_time:.2f}ms")
        return result
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simplified Resume Analyzer...")
    uvicorn.run(app, host="0.0.0.0", port=8000)