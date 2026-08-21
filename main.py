import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import PyPDF2
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ==============================
# CONFIG
# ==============================
# In-memory processing (no upload folder needed)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables. Please check your .env file.")
genai.configure(api_key=api_key)

app = Flask(__name__)
CORS(app)  # Enable CORS
# No local uploads folder needed for in-memory processing

# ==============================
# HOME ROUTE (UI)
# ==============================
@app.route("/")
def home():
    return render_template("Index.html")

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ==============================
# COMBINED ATS ANALYSIS (LLM)
# ==============================
def analyze_resume_and_jd(resume_text, jd_text):
    prompt = f"""
You are an expert Applicant Tracking System.
I will provide you with a Resume and a Job Description.
I need you to analyze them and provide 3 sections separated exactly by "===SECTION_SEPARATOR===".

Section 1: Parsed Resume
Extract: Skills, Experience summary, Education, Tools & technologies (in bullet points)

===SECTION_SEPARATOR===
Section 2: Parsed Job Description
Extract: Required skills, Responsibilities, Preferred qualifications (in bullet points)

===SECTION_SEPARATOR===
Section 3: ATS Match Result
Provide: Match percentage (0-100), Matching skills, Missing skills, Strengths, Improvement suggestions

Resume:
{resume_text}

Job Description:
{jd_text}
"""
    # Using Gemini 1.5 Flash as the default lightweight/fast model for parsing
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
        )
    )
    
    content = response.text
    sections = content.split("===SECTION_SEPARATOR===")
    
    if len(sections) == 3:
        return sections[0].strip(), sections[1].strip(), sections[2].strip()
    else:
        # Fallback if the model didn't perfectly use the separator
        return "Failed to parse resume section cleanly.", "Failed to parse JD section cleanly.", content

# ==============================
# API ROUTE (PDF UPLOAD)
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # Extract resume text directly from memory
    resume_text = extract_text_from_pdf(resume_file)

    # Parse and Match using single LLM call
    parsed_resume, parsed_jd, ats_result = analyze_resume_and_jd(resume_text, jd_text)

    return jsonify({
        "parsed_resume": parsed_resume,
        "parsed_job_description": parsed_jd,
        "ats_result": ats_result
    })

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8080)
