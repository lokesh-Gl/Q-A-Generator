import os
import io
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import csv

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
from dotenv import load_dotenv, find_dotenv
os.environ['REPORTLAB_TTF_CACHE'] = '/tmp'  
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Configure logging
logger = get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Load environment variables
load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found!")

# Model configurations - Using valid Groq models
QUESTION_GENERATION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ANSWER_GENERATION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Temperature settings for reproducibility
QUESTION_TEMPERATURE = 0.3
ANSWER_TEMPERATURE = 0.3

# Streamlit page configuration
st.set_page_config(
    page_title="Q&A Generator",
    page_icon="ðŸ“„",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

custom_css = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    if 'questions_list' not in st.session_state:
        st.session_state.questions_list = []

    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []

    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = ""

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = ""

initialize_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_validate_pdf(uploaded_file) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Load and validate PDF file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (success, content, error_message)
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            return False, None, "PDF is empty or could not be read."

        # Combine all pages
        content = "\n\n".join([page.page_content for page in pages])

        # Clean up
        os.unlink(tmp_path)

        return True, content, None

    except Exception as e:
        logger.error(f"PDF loading error: {str(e)}")
        return False, None, f"Error loading PDF: {str(e)}"


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                           chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: Input text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Text splitting error: {str(e)}")
        raise


@st.cache_resource
def initialize_embeddings():
    """Initialize and cache embedding model."""
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding initialization error: {str(e)}")
        raise


def create_vector_store(chunks: List[str]) -> Optional[FAISS]:
    """
    Create FAISS vector store from text chunks.

    Args:
        chunks: List of text chunks

    Returns:
        FAISS vector store or None if error
    """
    try:
        documents = [Document(page_content=chunk) for chunk in chunks]
        embeddings = initialize_embeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Vector store creation error: {str(e)}")
        return None


def generate_questions(text: str, num_questions: int = 10) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Generate questions from text using LLM.

    Args:
        text: Input text
        num_questions: Number of questions to generate

    Returns:
        Tuple of (success, questions_list, error_message)
    """
    try:
        # Initialize LLM
        question_llm = ChatGroq(
            model=QUESTION_GENERATION_MODEL,
            temperature=QUESTION_TEMPERATURE,
            api_key=GROQ_API_KEY
        )

        # Create prompt template
        prompt_template = f"""You are an expert at extracting the MOST IMPORTANT questions from any text.

Your task:
- Generate ONLY the Top {num_questions} most important questions.
- No MCQs, no True/False, no short/long labels.
- Just write the {num_questions} most important and meaningful questions.
- Questions must fully cover the key ideas in the text.
- Do NOT include answers.
- Number the questions (1., 2., 3., etc.)

Text to analyze:
{{text}}

Produce ONLY the Top {num_questions} important questions numbered:"""

        question_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['text']
        )

        # Create chain
        question_chain = LLMChain(llm=question_llm, prompt=question_prompt)

        # Generate questions
        result = question_chain.run(text=text)
        
        # Extract questions from result
        questions_list = extract_questions_as_list(result)

        return True, questions_list, None

    except Exception as e:
        logger.error(f"Question generation error: {str(e)}")
        return False, None, f"Error generating questions: {str(e)}"


# ---------- REPLACEMENT: generate_questions + helpers + fallback ----------
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize  # you may need to `pip install nltk` and download punkt
import math

def extract_questions_as_list(questions_text: str, desired_n: int = 10) -> List[str]:
    """
    Robustly extract individual questions from raw LLM text.
    Handles:
     - Numbered lines: "1. What is...?"
     - Bulleted lines: "- What is...?" or "* What...?"
     - Plain sentences where each sentence is a question.
    Returns up to desired_n questions.
    """
    if not questions_text:
        return []

    # Normalize whitespace
    text = questions_text.strip()
    # Replace common bullet characters with newline
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Split using newlines first
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    questions = []

    for line in lines:
        # If it starts with a number like "1." or "1)", remove numbering
        m = re.match(r'^\s*\d+[\.\)]\s*(.*)', line)
        if m:
            candidate = m.group(1).strip()
        else:
            candidate = line

        # If the line contains multiple questions separated by "?" keep each
        parts = re.split(r'\?\s*', candidate)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # restore question mark if missing
            if not p.endswith('?'):
                p = p + '?'
            # heuristics: length and must contain at least a verb/noun (rough check)
            if len(p) > 20:  # avoid tiny fragments
                questions.append(p)
            elif len(questions) < desired_n and len(p) > 10:
                questions.append(p)

        if len(questions) >= desired_n:
            break

    # Last resort: if still empty, try splitting into sentences and taking those ending with '?'
    if not questions:
        sents = re.split(r'(?<=[\.\?\!])\s+', text)
        for s in sents:
            s = s.strip()
            if s.endswith('?') and len(s) > 20:
                questions.append(s)
            if len(questions) >= desired_n:
                break

    return questions[:desired_n]


def fallback_generate_questions_tfidf(text: str, num_questions: int = 10) -> List[str]:
    """
    Deterministic fallback: score sentences by TF-IDF and pick top N sentences that look like questions.
    If not enough question-like sentences, convert top informative sentences into question form using heuristics.
    """
    try:
        # Sentence split (nltk punkt required). If not available, naive split:
        try:
            sents = sent_tokenize(text)
        except Exception:
            sents = re.split(r'(?<=[\.\?\!])\s+', text)

        # Filter out very short sentences
        sents = [s.strip() for s in sents if len(s.strip()) > 30]

        if not sents:
            return []

        # Compute TF-IDF sentence scores by treating sentences as documents
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
        X = vectorizer.fit_transform(sents)
        scores = X.sum(axis=1).A1  # sum TF-IDF weights per sentence
        ranked_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))

        questions = []
        for idx in ranked_idx:
            sent = sents[idx]
            # If sentence already is a question, keep it
            if sent.endswith('?'):
                q = sent
            else:
                # Try a simple conversion: if sentence has "is/are/was/were/can/will/should/must/could"
                m = re.search(r'\b(is|are|was|were|can|will|should|must|could|do|does|did|has|have|had|may)\b', sent, flags=re.I)
                if m:
                    # naive inversion: put the modal/auxiliary at front
                    aux = m.group(1)
                    before = sent[:m.start()].strip()
                    after = sent[m.end():].strip()
                    q = f"{aux.capitalize()} {before} {after}".strip()
                    if not q.endswith('?'):
                        q = q + '?'
                else:
                    # fallback: make "What is ..." or "How does ..." prefixes intelligently using length
                    if len(sent.split()) < 12:
                        q = f"What is {sent.strip().rstrip('.')}?"
                    else:
                        q = f"How does {sent.strip().rstrip('.')}?"
            # Basic cleanup
            q = re.sub(r'\s+', ' ', q).strip()
            if len(q) > 20:
                questions.append(q)
            if len(questions) >= num_questions:
                break

        # If still short, pad by slicing other top sentences converted similarly
        if len(questions) < num_questions:
            for idx in ranked_idx:
                if idx >= len(sents):
                    continue
                candidate = sents[idx]
                candidate_q = candidate if candidate.endswith('?') else candidate.rstrip('.') + '?'
                if candidate_q not in questions:
                    questions.append(candidate_q)
                if len(questions) >= num_questions:
                    break

        return questions[:num_questions]

    except Exception as e:
        logger.error(f"Fallback TF-IDF question generation error: {str(e)}")
        return []


def generate_questions(text: str, num_questions: int = 10) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Generate questions from text using LLM with a strict prompt, with retry and fallback.
    Returns (success, questions_list, error_message)
    """
    try:
        # Strong, unambiguous prompt. Explicitly forbid clarifying questions or meta responses.
        strict_prompt = (
            "SYSTEM: You are an expert question extractor. You MUST produce exactly the top "
            f"{num_questions} MOST IMPORTANT questions about the provided text. DO NOT ask "
            "any clarifying questions, do NOT return answers or commentary, and do NOT return "
            "any text other than a numbered list of questions. Example format:\n\n"
            "1. First important question?\n"
            "2. Second important question?\n\n"
            "Now analyze the text and produce ONLY the numbered questions.\n\n"
            "TEXT:\n{text}\n\nProduce ONLY the numbered questions.\n"
        )

        # Initialize LLM once
        question_llm = ChatGroq(
            model=QUESTION_GENERATION_MODEL,
            temperature=QUESTION_TEMPERATURE,
            api_key=GROQ_API_KEY,
            # If ChatGroq supports max_tokens / stop keywords pass them - safe to include if supported
            # max_tokens=512,
        )

        # Use a direct run with the strict prompt
        prompt_template = PromptTemplate(
            template=strict_prompt,
            input_variables=['text']
        )

        question_chain = LLMChain(llm=question_llm, prompt=prompt_template)

        # 1) Primary attempt
        result = question_chain.run(text=text)

        # 2) Validate result
        questions_list = extract_questions_as_list(result, desired_n=num_questions)

        # If the result contains assistant-y text or not enough questions, retry once with even more explicit instruction
        if (not questions_list) or len(questions_list) < max(3, math.ceil(0.6 * num_questions)) or re.search(r"haven'?t asked|please go ahead|can't|cannot|ask me", (result or ""), flags=re.I):
            logger.info("LLM output invalid or insufficient; retrying with stricter instruction.")
            retry_prompt = strict_prompt + (
                "\nRETRY: You failed to produce the required output. This is a strict retry: produce ONLY the numbered questions, nothing else.\n"
            )
            retry_template = PromptTemplate(template=retry_prompt, input_variables=['text'])
            retry_chain = LLMChain(llm=question_llm, prompt=retry_template)
            result_retry = retry_chain.run(text=text)
            questions_list = extract_questions_as_list(result_retry, desired_n=num_questions)

        # If still invalid, use deterministic fallback TF-IDF
        if not questions_list or len(questions_list) < 2:
            logger.info("Using deterministic TF-IDF fallback to generate questions.")
            questions_list = fallback_generate_questions_tfidf(text, num_questions)

        # Final guard: ensure we have up to num_questions entries
        if not questions_list:
            return False, None, "Unable to generate questions (LLM failed and fallback returned nothing)."

        # Trim/pad to exact num_questions
        questions_list = questions_list[:num_questions]
        return True, questions_list, None

    except Exception as e:
        logger.error(f"Question generation error: {str(e)}")
        # fallback attempt
        try:
            questions_list = fallback_generate_questions_tfidf(text, num_questions)
            if questions_list:
                return True, questions_list, None
        except Exception:
            pass
        return False, None, f"Error generating questions: {str(e)}"
def clean_answer_text(text: str) -> str:
    import re
    if not text:
        return ""

    # Fix decimals like "1 . 25" -> "1.25"
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)

    # Collapse multiple spaces (but keep single spaces)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Normalize multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()




def generate_answer(question: str, retriever) -> Optional[str]:
    """
    Generate a structured answer with guaranteed line breaks.
    """

    TEMPLATE = """
You are writing a well-structured, readable answer.

Structure the answer as:
- Short introduction paragraph (2–3 sentences).
- Numbered list of 3–4 key points.
- Optional short closing paragraph (1–2 sentences).

Rules:
- Use plain text only (no markdown).
- Each numbered point must start with "1.", "2.", etc. at the beginning of a new line.
- Subpoints (a., b.) are optional; if used, put them on their own line and indent with 3 spaces.
- Do not include meta-instructions like "First point written as 2–4 lines".
- Do not break words or insert spaces inside words.
- Let line wrapping be handled by the viewer (do not try to force fixed line lengths).

CONTEXT:
{context}

QUESTION:
{question}

Write the answer now in the described structure.
"""


    try:
        # Retrieve relevant docs
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = ChatGroq(
            model=ANSWER_GENERATION_MODEL,
            temperature=ANSWER_TEMPERATURE,
            api_key=GROQ_API_KEY
        )

        final_prompt = TEMPLATE.format(context=context, question=question)
        result = llm.invoke(final_prompt)
        answer = clean_answer_text(result.content)

        # Clean symbols
        answer = re.sub(r"[#*•▪►`_]+", " ", answer)

        # Ensure double newlines between sections
        answer = re.sub(r"\n{2,}", "\n\n", answer)

        return answer.strip()

    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        return f"Error generating answer: {str(e)}"

def generate_qa_pairs(questions: List[str], vector_store) -> List[Dict[str, str]]:
    """
    Generate Q&A pairs for all questions.

    Args:
        questions: List of questions
        vector_store: FAISS vector store

    Returns:
        List of Q&A dictionaries
    """
    qa_pairs = []
    retriever = vector_store.as_retriever()

    for idx, question in enumerate(questions, 1):
        answer = generate_answer(question, retriever)
        qa_pairs.append({
            "Question_No": idx,
            "Question": question,
            "Answer": answer
        })

    return qa_pairs


def create_csv_export(qa_pairs: List[Dict[str, str]]) -> bytes:
    """Create CSV export from Q&A pairs."""
    output = io.StringIO()
    if qa_pairs:
        fieldnames = ['Question_No', 'Question', 'Answer']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_pairs)

    return output.getvalue().encode('utf-8')


def create_txt_export(qa_pairs: List[Dict[str, str]]) -> bytes:
    """Create TXT export from Q&A pairs."""
    output = []
    output.append("=" * 80)
    output.append("Q&A GENERATOR - EXPORT RESULTS")
    output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    output.append("")

    for qa in qa_pairs:
        output.append(f"Question {qa['Question_No']}: {qa['Question']}")
        output.append("-" * 80)
        output.append(f"{qa['Answer']}")
        output.append("")
        output.append("")

    return "\n".join(output).encode('utf-8')


def create_pdf_export(qa_pairs):
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
            ListFlowable, ListItem
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        buffer = io.BytesIO()

        # PAGE MARGINS
        LEFT = RIGHT = 0.75 * inch
        TOP = 1.0 * inch
        BOTTOM = 0.75 * inch

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
        )

        def draw_page(canvas, doc):
            canvas.saveState()
            
            # BORDER aligned perfectly with margins
            canvas.setStrokeColor(colors.HexColor("#0077cc"))
            canvas.setLineWidth(1.5)
            canvas.rect(
                LEFT, 
                BOTTOM, 
                letter[0] - LEFT - RIGHT, 
                letter[1] - TOP - BOTTOM
            )

            # Header
            canvas.setFont("Helvetica-Bold", 12)
            canvas.setFillColor(colors.HexColor("#004d99"))
            canvas.drawString(LEFT + 0.2*inch, letter[1] - TOP + 0.4*inch, "PDF Q&A Generator Report")

            # Footer
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.gray)
            canvas.drawCentredString(letter[0] / 2.0, 0.5 * inch, f"Page {doc.page}")

            canvas.restoreState()

        styles = getSampleStyleSheet()

        # Title Style
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=24,
            textColor=colors.HexColor("#004d99"),
            spaceAfter=25
        )
        goal_style = ParagraphStyle(
            "GoalStyle",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            spaceAfter=4,
            leftIndent=0,
            textColor=colors.HexColor("#003366"),
        )

        bullet_style = ParagraphStyle(
            "BulletStyle",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            leftIndent=5,
            spaceAfter=2,
            textColor=colors.HexColor("#222222"),
        )


        # Question style
        question_style = ParagraphStyle(
            "Question",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            textColor=colors.HexColor("#2b7cff"),
            spaceAfter=8,
            spaceBefore=12,
        )

        # Answer paragraph style
        answer_para_style = ParagraphStyle(
            "AnswerParagraph",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )

        # List item style
        list_item_style = ParagraphStyle(
            "ListItem",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            leftIndent=0,
            spaceAfter=8,
        )

        # Subpoint style
        subpoint_style = ParagraphStyle(
            "Subpoint",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            leftIndent=20,
            spaceAfter=6,
        )

        elements = []

        # COVER PAGE
        #elements.append(Spacer(1, 1 * inch))
        #elements.append(Paragraph("PDF Question & Answer Report", title_style))
        #elements.append(Paragraph(
           # f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
           # styles["Normal"]
        #))
        #elements.append(PageBreak())

        # Q&A CONTENT
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["Question"]
            answer = qa["Answer"]

            # Add question
            elements.append(Paragraph(f"Q{i}: {question}", question_style))

            # Parse and format answer
            parsed_elements = parse_answer_to_elements(
                answer, 
                answer_para_style, 
                bullet_style
            )
            
            elements.extend(parsed_elements)
            elements.append(Spacer(1, 0.15 * inch))

        doc.build(elements, onFirstPage=draw_page, onLaterPages=draw_page)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"PDF Export Error: {e}")
        return None


def parse_answer_to_elements(answer_text, para_style, bullet_style):
    from reportlab.platypus import Paragraph, ListFlowable, ListItem
    import re

    elements = []
    current_paragraph = []
    bullet_items = []

    lines = answer_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading numbers (e.g., "1 ...", "2 ...")
        line_no_num = re.sub(r'^\d+\s*\.?\s*', '', line)

        # Optional: Remove letter labels too, e.g. "a.", "b."
        line_no_label = re.sub(r'^[a-zA-Z]\.\s*', '', line_no_num)

        bullet_items.append(ListItem(Paragraph(line_no_label, bullet_style)))

    # Optionally: handle paragraphs before bullet points (e.g. intro text)
    if current_paragraph:
        elements.append(Paragraph(' '.join(current_paragraph), para_style))

    if bullet_items:
        elements.append(ListFlowable(bullet_items, bulletType='bullet', leftIndent=5,bulletIndent=0 ))

    return elements


# ============================================================================
# MAIN APP UI
# ============================================================================

def render_header():
    col1, col2 = st.columns([0.85, 0.15])

    with col1:
        st.markdown(
            """
            <div style='text-align: center; padding-top: 10px; padding-bottom: 10px;'>
                <h1 style='font-size: 55px; margin-bottom: 0;'>Q&A Generator</h1>
                <p style='font-size: 25px; margin-top: 5px;'>
                    Generate intelligent questions and answers from your PDF documents using AI
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if st.session_state.processing_complete:
            st.markdown(
                """
                <div style="
                    margin-top: 45px;
                    background: linear-gradient(135deg, #7f5eff, #9f78ff);
                    color: white;
                    border-radius: 10px;
                    padding: 10px 20px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: 600;
                ">
                    Ready to Export
                </div>
                """,
                unsafe_allow_html=True
            )

def render_sidebar():
    """Render sidebar configuration."""
    with st.sidebar:
        st.header("Question Configuration")

        num_questions = st.slider(
            "Number of Questions to Generate",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
            help="Select how many important questions to generate from the PDF"
        )

        st.markdown("---")
        st.subheader("Session Status")

        if st.session_state.pdf_processed:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PDF Uploaded", "yes")
            with col2:
                st.metric("Questions Generated", len(st.session_state.questions_list))
        else:
            st.info("Upload a PDF to begin")

        return num_questions


def render_upload_section():
    """Render PDF upload section."""
    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to extract questions and answers"
    )

    return uploaded_file


def render_processing_section(uploaded_file, num_questions):
    """Render processing section."""
    st.header("Process Document")

    if uploaded_file:
        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            process_btn = st.button(
                "Generate Questions & Answers",
                use_container_width=True,
                type="primary",
                key="process_btn"
            )

        with col2:
            if st.session_state.processing_complete:
                reset_btn = st.button(
                    "Reset",
                    use_container_width=True,
                    key="reset_btn"
                )
                if reset_btn:
                    st.session_state.pdf_processed = False
                    st.session_state.questions_list = []
                    st.session_state.qa_pairs = []
                    st.session_state.pdf_content = ""
                    st.session_state.vector_store = None
                    st.session_state.processing_complete = False
                    st.session_state.pdf_filename = ""
                    st.rerun()

        if process_btn:
            with st.spinner("Reading PDF..."):
                success, content, error = load_and_validate_pdf(uploaded_file)

            if not success:
                st.error(f"PDF Error: {error}")
                return

            st.session_state.pdf_content = content
            st.session_state.pdf_filename = uploaded_file.name

            with st.spinner("Splitting text into chunks..."):
                chunks = split_text_into_chunks(content)
                st.session_state.vector_store = create_vector_store(chunks)

            if st.session_state.vector_store is None:
                st.error("Error creating vector store")
                return

            with st.spinner(f"Generating {num_questions} questions..."):
                success, questions_list, error = generate_questions(content, num_questions)

            if not success:
                st.error(f"Question Generation Error: {error}")
                return

            st.session_state.questions_list = questions_list if questions_list else []

            with st.spinner("Generating answers (this may take a moment)..."):
                st.session_state.qa_pairs = generate_qa_pairs(
                    st.session_state.questions_list,
                    st.session_state.vector_store
                )

            st.session_state.pdf_processed = True
            st.session_state.processing_complete = True

            st.success("Processing complete!")
            st.rerun()


def render_results_section():
    """Render results display section."""
    if not st.session_state.processing_complete or not st.session_state.qa_pairs:
        return

    st.header("Generated Q&A Pairs")

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(st.session_state.qa_pairs))
    with col2:
        filename = st.session_state.pdf_filename
        display_name = filename[:30] + "..." if len(filename) > 30 else filename
        st.metric("File Name", display_name)
    with col3:
        st.metric("Processing Time", "~2-3 min (approx)")

    st.markdown("---")

    # Tabbed view
    tab1, tab2, tab3 = st.tabs(["Preview", "Detailed View", "Table View"])

    with tab1:
        st.subheader("Quick Preview")
        for qa in st.session_state.qa_pairs[:3]:  # Show only first 3
            with st.expander(f"Q{qa['Question_No']}: {qa['Question'][:60]}..."):
                st.write(f"**Question:** {qa['Question']}")
                st.write(f"**Answer:** {qa['Answer']}")

    with tab2:
        st.subheader("Detailed View")
        for qa in st.session_state.qa_pairs:
            col1, col2 = st.columns([0.1, 0.9])
            st.write(f"**Q{qa['Question_No']}: {qa['Question']}**")
            st.write(qa['Answer'])
            st.markdown("---")

    with tab3:
        st.subheader("Table View")
        df = pd.DataFrame(st.session_state.qa_pairs)
        st.dataframe(df, use_container_width=True, height=400)


def render_export_section():
    """Render export section."""
    if not st.session_state.processing_complete or not st.session_state.qa_pairs:
        return

    st.header("Export Results")
    st.markdown("Download your generated Q&A pairs in your preferred format")

    col1, col2, col3 = st.columns(3)

    # CSV Export
    with col1:
        csv_data = create_csv_export(st.session_state.qa_pairs)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TXT Export
    with col2:
        txt_data = create_txt_export(st.session_state.qa_pairs)
        st.download_button(
            label="Download as TXT",
            data=txt_data,
            file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # PDF Export
    with col3:
        pdf_data = create_pdf_export(st.session_state.qa_pairs)
        if pdf_data:
            st.download_button(
                label="Download as PDF",
                data=pdf_data,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width='strech'
            )
        else:
            st.warning("PDF export not available (reportlab not installed)")


def main():
    """Main application function."""
    # Render components
    render_header()
    num_questions = render_sidebar()

    uploaded_file = render_upload_section()
    render_processing_section(uploaded_file, num_questions)
    render_results_section()
    render_export_section()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <center>
        <small>
        Q&A Generator | Powered by LangChain, Groq, and Streamlit<br>
        Version 1.0.0 | 2025
        </small>
        </center>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

