# app.py
import os
import ast
import sqlite3
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pymupdf  # PyMuPDF
import docx2txt
from pathlib import Path
import re
import pandas as pd
import io
import warnings
import uuid
import shutil
from google.api_core.exceptions import ResourceExhausted
import time
from datetime import datetime 
from dashboard import render_dashboard
from huggingface_hub import login


st.set_page_config(page_title="EasyHire")
st.title("EasyHire")
st.caption("Curriculum Vitae parser and matcher that helps companies without HR staff simplify the hiring process")

login(token=st.secrets["HUGGINGFACE"]["HUGGINGFACE_HUB_TOKEN"])

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
warnings.filterwarnings("ignore")



VECTOR_DIR = "chroma_store"
UPLOAD_DIR = "temp_files"
DB_PATH = "CV_data.db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state once when the app starts
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.submitted = False
    st.session_state.job_desc = ""
    st.session_state.uploaded_files = []
    st.session_state.CV_texts = {}
    st.session_state.analysis_results = []
    st.session_state.match_percentages = {}

#selected_tab = st.sidebar.selectbox("Choose a section", ["📤 Upload & Match", "📊 Dashboard"], key="selected_tab")

# Setup DB
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS CVs (
    filename TEXT PRIMARY KEY,
    candidate_name TEXT,
    match_percent TEXT,
    word_count INTEGER,
    skills TEXT,
    match_explanation TEXT,
    tools TEXT
)''')
conn.commit()

def clear_CV_cache(flag=1):
    c.execute("DELETE FROM CVs")
    conn.commit()
    # Properly close Chroma DB before deleting
    #if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
        #try:
            #chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
            #chroma_client.reset()
        #except Exception as e:
            #if not flag:
                #st.warning(f"Warning while releasing ChromaDB lock: {e}")
    # Now safely remove directories
    shutil.rmtree(VECTOR_DIR, ignore_errors=True)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 🧹 Reset session state to avoid using deleted files
    st.session_state.uploaded_files = []
    st.session_state.CV_texts = {}
    st.session_state.analysis_results = []
    st.session_state.job_desc = ""

    if not flag:
        st.success("All cached CV data cleared.")

    st.rerun()
      

# Utility Functions

def save_uploaded_file(file):
    file_path = os.path.join(UPLOAD_DIR, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def extract_CV_chunks_from_path(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    return splitter.split_documents(documents)

def create_or_load_vector_store(filename, chunks):
    store_path = os.path.join(VECTOR_DIR, filename.replace(".", "_"))
    if not os.path.exists(store_path):
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
    return vectorstore

def retrieve_matching_chunks(job_description, vectorstore, k=3):
    return vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(job_description)

def get_genai_response(prompt, retrieved_chunks, job_description):
    context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
    model = genai.GenerativeModel("gpt-3.5-turbo")
    response = model.generate_content([
        prompt,
        f"Relevant CV Info:\n{context}",
        f"Job Description:\n{job_description}"
    ])
    return response.text

def extract_text(file_bytes, file_type):
    if file_type == "application/pdf":
        return "\n".join([page.get_text() for page in pymupdf.open(stream=io.BytesIO(file_bytes), filetype="pdf")])
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(io.BytesIO(file_bytes))
    elif file_type == "text/plain":
        return file_bytes.decode("utf-8")
    return ""

def extract_required_skills(job_description):
    model = genai.GenerativeModel("gpt-3.5-turbo")
    prompt = f"""Extract a list of key skills required for the following job description. Just list them as a comma-separated string.\n\nJob Description:\n{job_description}"""
    response = model.generate_content(prompt)
    #print([skill.strip() for skill in response.text.split(",")])
    return [skill.strip() for skill in response.text.split(",")]

def count_matching_skills(CV_text, dynamic_skills_list):
    matched_skills = set()
    CV_lower = CV_text.lower()
    for skill in dynamic_skills_list:
        skill_lower = skill.lower()
        pattern = re.escape(skill_lower)
        if re.search(pattern, CV_lower):
            matched_skills.add(skill)
    #print(len(matched_skills), list(matched_skills))
    return len(matched_skills), list(matched_skills)

def get_candidate_names(text):
    prompt = """Extract the following information:

1.  **Candidate Name:** Identify and extract the full name of the candidate.

2.  **Tools:** Identify and list all the technical tools, software, programming languages, and technologies mentioned in the CV. Provide these as a comma-separated string.

3.  **Experience:** Extract the work experience of the candidate. For each role, identify the duration (in years) and the organization. Present this information as a Python dictionary where the keys are the duration in years (calculate the difference between start and end dates, round to the nearest whole number) and the values are the name of the organization. If specific dates are not available, make your best reasonable estimate of the duration in years.

4.  **Interests:** Identify any hobbies, interests, or personal pursuits mentioned in the CV. Extract these as a Python list of individual words.

Provide the extracted information as a Python dictionary that can be directly evaluated using Python's `eval()` function. The dictionary should have the following structure:

{
  "candidate_name": "...",
  "tools": "tool1, tool2, tool3, ...",
  "experience": {
    "yyyy1": "organization1",
    "yyyy2": "organization2",
    # ...
  },
  "interests": ["word1", "word2", "word3", ...]
}

"""
    model = genai.GenerativeModel("gpt-3.5-turbo")
    return model.generate_content([prompt, text]).text.strip()


# Prompts
prompt_analysis = """
You are a Technical HR Recruiter. Your task is to analyze the following CV against the provided job description and highlight the key aspects in a crisp and to-the-point manner.

Highlight candidate Name.

Your analysis should be structured into the following four sections:

1.  **Profile Summary:** Provide a concise (1-2 sentences) summary of the candidate's core experience and key skills relevant to the job description.

2.  **Strengths:** List key strengths of the candidate in bullet points, focusing on direct relevance to the job description's requirements and preferences. Use specific technologies, tools, and action verbs.

3.  **Weaknesses:** Identify potential weaknesses or areas for development in bullet points. Frame these objectively and professionally, focusing on gaps compared to the job description.

4.  **Missing Important Skills:** List critical skills explicitly mentioned in the job description that are not evident in the CV in bullet points. Be specific about the missing skills.

Ensure your analysis is concise, apt, and directly related to the information provided in the job description and CV. Avoid unnecessary elaboration or subjective opinions.
"""
prompt_match = """
You are an ATS (Applicant Tracking System). Analyze the following CV against the provided job description and provide a structured output.

Highlight candidate Name.

Your output should include the following three sections:

1.  **Matching Percentage:** Calculate and return a numerical percentage representing the overall match between the CV and the job description based on keyword presence and relevance.

2.  **Missing or Unmatched Keywords:** List the keywords and key phrases explicitly mentioned in the job description that are either absent or not prominently featured (unmatched) in the CV.

3.  **Final Thoughts:** Provide a brief (1-2 sentences) overall assessment of the candidate's suitability based on the keyword analysis.
"""
#promp_match_percentage = """Get the match percentage with the job description from this CV text. Just the match percentage digit. No need to include %.  Dont print any other words"""

def display_tabs():    

    job_desc = st.session_state.job_desc
    uploaded_files = st.session_state.uploaded_files
    CV_texts = st.session_state.CV_texts
    analysis_results = st.session_state.analysis_results

    #print("inside diplay_tabs:- ", job_desc[:20] , analysis_results )

    # UI Tabs
    tab1, tab2 = st.tabs(["Individual Analysis", "Dashboard"])

    with tab1:
        selected_file = st.selectbox("Select a CV for analysis:", list(CV_texts.keys()))
        if selected_file and job_desc:
            text = CV_texts[selected_file]
            file_path = os.path.join(UPLOAD_DIR, selected_file)
            chunks = extract_CV_chunks_from_path(file_path)
            vectorstore = create_or_load_vector_store(selected_file, chunks)
            retrieved = retrieve_matching_chunks(job_desc, vectorstore)

            col1, col2, col3 = st.columns(3)
            response = ""

            with col1:
                if st.button("📝 Analyze CV"):
                    st.session_state.show_question_input = False
                    with st.spinner("Analyzing..."):
                        response = get_genai_response(prompt_analysis, retrieved, job_desc)

            with col2:
                if st.button("📊 Match Percentage"):
                    with st.spinner("Evaluating..."):
                        st.session_state.show_question_input = False
                        c.execute("SELECT match_explanation FROM CVs WHERE filename = ?", (selected_file,))
                        response = c.fetchone()[0]  # ✅ direct from DB
                        #st.write(explanation)

            with col3:
                if st.button("🤔 Ask a Question"):
                    st.session_state.show_question_input = True

            if st.session_state.get("show_question_input"):
                query = st.text_input("Enter your question about this CV:")
                if st.button("Submit Question"):
                    if query:
                        with st.spinner("Getting answer..."):
                            qa_prompt = f"""
                            You are "CVBot," an intelligent assistant for an ATS system. You have been provided with structured data extracted from a candidate's CV. 
                            Your goal is to answer user questions about this candidate accurately and efficiently.

                            **User Question:** "{query}"

                            Respond to the user's question based on the provided data. Be direct and informative. 
                            If the question is ambiguous, ask for clarification. 
                            If the information is not available, state "The CV does not contain information about that." or a similar polite refusal.
                            """
                            response = get_genai_response(qa_prompt, retrieved, job_desc)

            st.write(response)
        else:
            st.info("Please select a CV and paste a job description.")

    with tab2:
        #st.subheader("📊 CV Dashboard")
        if analysis_results:            
            df = pd.DataFrame(analysis_results)
            df["Parsed Match %"] = df["Job Match Percentage"].apply(lambda x: int(re.search(r'\d+', x).group(0)) if isinstance(x, str) and re.search(r'\d+', x) else 0)
            df["Experience"] = df["Experience"].apply(lambda exp: "\n".join([f"{k}: {v}" for k, v in exp.items()]) if isinstance(exp, dict) else exp)
            df["Interests"] = df["Interests"].apply(lambda interests: ", ".join(interests) if isinstance(interests, list) else interests)
            df_sorted = df.sort_values(by="Parsed Match %", ascending=False).drop(columns=["Parsed Match %"])
            render_dashboard(df_sorted)
        else:
            st.info("Upload CVs to view dashboard analysis.")



# Sidebar Inputs
job_desc = st.sidebar.text_area("Paste the Job Description here:", height=250)
uploaded_files = st.sidebar.file_uploader("Upload CVs", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if st.sidebar.button("Submit JD and CV"):        
    #print("After submitting button start:- ", job_desc[:20] , st.session_state.analysis_results )
    if not job_desc or not uploaded_files:
        st.warning("Please provide job description and upload at least one CV.")
    else:          
        time.sleep(1)
        #print("current Job desc:-  ", job_desc[:50])             
        st.session_state.submitted = True
        st.session_state.job_desc = job_desc
        st.session_state.uploaded_files = uploaded_files
        st.session_state.CV_texts = {}
        st.session_state.analysis_results = []
        dynamic_skills = extract_required_skills(job_desc) if job_desc else []
        progress_bar = st.progress(0, text="Processing CVs...")
        total_files = len(uploaded_files)
        # Preprocess and store if not already in DB
        for idx, file in enumerate(uploaded_files):
            progress_bar.progress((idx + 1) / total_files, text=f"Processing {file.name} ({idx + 1}/{total_files})")
            saved_path = save_uploaded_file(file)
            file_bytes = Path(saved_path).read_bytes()
            text = extract_text(file_bytes, file.type)
            st.session_state.CV_texts[file.name] = text
            # Always recompute fresh
            word_count = len(text.split())
            skill_count,skill_list = count_matching_skills(text, dynamic_skills)
            # Only reuse genai results if already cached
            c.execute("SELECT * FROM CVs WHERE filename = ?", (file.name,))
            result = c.fetchone()
            if result:
                name, match_percent,match_response,tools = result[1], result[2],result[5],result[6]
            else:
                info = get_candidate_names(text)[10:-3]
                print(info,type(info))
                info = ast.literal_eval(info)
                print(info,type(info))
                name = info.get("candidate_name", "N/A")
                tools = info.get("tools", "")
                experience = info.get("experience", {})
                interests = info.get("interests", [])                
                time.sleep(5.2)
                chunks = extract_CV_chunks_from_path(saved_path)
                vectorstore = create_or_load_vector_store(file.name, chunks)
                retrieved = retrieve_matching_chunks(job_desc, vectorstore)
                match_response = get_genai_response(prompt_match, retrieved, job_desc)
                time.sleep(5.2)
                match_percent = re.search(r'\d{1,3}%', match_response)
                match_percent = match_percent.group(0) if match_percent else "0%"
                c.execute("INSERT INTO CVs VALUES (?, ?, ?, ?, ?, ?,?)", (
            file.name, name, match_percent, word_count, ", ".join(skill_list), match_response,tools))
                conn.commit()
                time.sleep(5.2)
            
            st.session_state.analysis_results.append({
                "Filename": file.name,
                "Candidate Name": name.upper(),
                "Job Match Percentage": match_percent,
                "Word Count": word_count,
                "Detected Skills": skill_list,
                "Skill Match Count": skill_count,
                "Tools":tools,
                "Experience":experience,
                "Interests":interests
            })
            
    #print("After submitting button start and before calling display tab:- ", job_desc[:20] , st.session_state.analysis_results )
    progress_bar=""
    if st.session_state.submitted and st.session_state.analysis_results:
        display_tabs()
elif st.session_state.submitted and st.session_state.analysis_results:
    #print("After submitted and analysis is already aailable:- ", st.session_state.job_desc[:20] , st.session_state.analysis_results )  
    display_tabs()
else:
    print("After when job desc and CV not uploaded:- ", st.session_state.job_desc[:20] , st.session_state.analysis_results )
    st.info("Please select CV and paste a job description.")


# Clear DB button
with st.sidebar:
    if st.button("Reset"):
        clear_CV_cache(flag=0)

    

    