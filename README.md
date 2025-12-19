📖 AI-Powered PDF Q&A Generator

An end-to-end AI-driven Question & Answer Generator that extracts key knowledge from PDF documents and automatically generates important questions with well-structured answers. Built using Streamlit, LangChain, Groq LLMs, FAISS, and HuggingFace embeddings, this project is designed for students, educators, researchers, and professionals who want instant insights from large PDF files.

⸻

🚀 Features
	•	📂 Upload any PDF document
	•	🧠 Automatically generate important questions from the content
	•	✍️ Generate clear, structured answers using Retrieval-Augmented Generation (RAG)
	•	🔍 Semantic search using FAISS vector store
	•	🔁 Robust fallback logic (LLM + TF-IDF) for reliable question generation
	•	📊 Interactive UI with preview, detailed view, and table view
	•	📤 Export results in CSV, TXT, and PDF formats
	•	🎨 Clean, professional Streamlit UI with custom styling

⸻

🧩 System Architecture

PDF Upload
   ↓
PDF Loader (LangChain)
   ↓
Text Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings (HuggingFace BGE)
   ↓
FAISS Vector Store
   ↓
Question Generation (Groq LLM)
   ↓
Answer Generation (RAG + Groq LLM)
   ↓
Preview & Export (CSV / TXT / PDF)


⸻

🛠️ Tech Stack
	•	Frontend: Streamlit
	•	LLM Provider: Groq
	•	Models Used:
	•	meta-llama/llama-4-scout-17b-16e-instruct
	•	Framework: LangChain
	•	Embeddings: BAAI/bge-small-en-v1.5
	•	Vector Database: FAISS
	•	PDF Processing: PyPDFLoader, PyPDF
	•	Export: CSV, TXT, PDF (ReportLab)

⸻

⚙️ Installation & Setup

1️⃣ Clone the Repository

``` bash
git clone https://github.com/your-username/pdf-qa-generator.git
cd pdf-qa-generator
```

2️⃣ Create Virtual Environment (Recommended)

``` bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```
3️⃣ Install Dependencies
``` bash

pip install -r requirements.txt
```
4️⃣ Set Environment Variables

Create a .env file in the root directory:
``` bash
GROQ_API_KEY=your_groq_api_key_here
```
5️⃣ Run the Application
``` bash
streamlit run app.py
```

⸻

🧠 How It Works

🔹 Step 1: PDF Processing
	•	PDF is loaded and validated
	•	Text is extracted and cleaned

🔹 Step 2: Chunking & Embeddings
	•	Text is split into overlapping chunks
	•	Each chunk is embedded using HuggingFace BGE embeddings

🔹 Step 3: Vector Store Creation
	•	FAISS stores embeddings for fast semantic retrieval

🔹 Step 4: Question Generation
	•	Groq LLM generates important questions
	•	If LLM fails, a TF-IDF-based fallback ensures reliability

🔹 Step 5: Answer Generation (RAG)
	•	Relevant chunks are retrieved from FAISS
	•	Groq LLM generates structured answers using context

🔹 Step 6: Export
	•	Results can be downloaded as CSV, TXT, or PDF

⸻

📊 Output Formats
	•	CSV – Easy to analyze and store
	•	TXT – Simple readable format
	•	PDF – Professionally styled report with borders, headers & footers

⸻

🎯 Use Cases
	•	📚 Exam preparation & study notes
	•	🧑‍🏫 Teaching & question paper creation
	•	📑 Research paper understanding
	•	🏢 Corporate document analysis
	•	🧠 Knowledge extraction from large documents

⸻

🔐 Error Handling & Reliability
	•	Validates PDF integrity
	•	Retries LLM calls on malformed outputs
	•	Falls back to deterministic TF-IDF question generation
	•	Session-state management for smooth user experience

⸻

📌 Project Highlights
	•	Uses Retrieval-Augmented Generation (RAG)
	•	Deterministic + LLM hybrid approach
	•	Clean separation of UI, logic, and utilities
	•	Production-ready export pipeline

⸻

📈 Future Enhancements
	•	🔎 Page-level answer citations
	•	🌐 Multilingual support
	•	🧾 DOCX export
	•	☁️ Cloud deployment (AWS / GCP)
	•	👤 User authentication & history

⸻

⭐ Acknowledgements
	•	LangChain
	•	Groq
	•	HuggingFace
	•	Streamlit

