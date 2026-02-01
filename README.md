# 🎓 Placement Companion Chatbot

An intelligent **Agentic RAG-based conversational AI system** designed to help M.Tech students at MSIS, MAHE access placement-related information through natural language queries.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Key Features

- **Dual RAG Modes**
  - **Basic RAG**: Fast semantic search for simple queries
  - **Agentic RAG**: Multi-step reasoning for complex questions

- **11 Specialized Tools**
  - Vector Search, Company Comparison, Eligibility Check
  - Trend Analysis, Skill Demand, Personalized Recommendations
  - And 5 more analytics tools

- **16 Query Type Classifications**
  - From simple company info to complex statistical analysis

- **Quality Assurance System**
  - Built-in Critic component that evaluates answers before delivery
  - Iterative refinement (up to 3 cycles) for complex queries

- **Conversational Memory**
  - Remembers context across chat turns
  - Resolves pronouns and references intelligently

## 🏗️ Architecture

```
User Query → Input Validation → Query Routing → Processing → Response
                                      ↓
                        ┌─────────────────────────┐
                        │    Basic RAG            │
                        │    or                   │
                        │    Agentic RAG          │
                        │    (Plan-Execute-Eval)  │
                        └─────────────────────────┘
```

### Agentic RAG Components

1. **Planner**: Analyzes query and creates execution plan
2. **Executor**: Runs specialized tools step-by-step
3. **Critic**: Evaluates quality and decides (Accept/Refine/Replan)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Vector Database** | Pinecone (Serverless) |
| **Embedding Model** | Alibaba GTE-Qwen2-7B-Instruct (384D) |
| **LLM** | Qwen2.5-72B via HuggingFace Router |
| **Reranker** | Cross-Encoder MS-MARCO-MiniLM-L6-v2 |
| **OCR** | DeepSeek OCR |
| **Classification** | DeBERTa-v3-large (Zero-shot) |
| **Language** | Python 3.10+ |

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- API keys (Pinecone, HuggingFace)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manjus2003/placement_chatbot.git
   cd placement_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a file named `environment.env` in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   HF_TOKEN=your_huggingface_token
   PINECONE_INDEX_NAME=placement-companion-v5
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_ui_v2.py
   ```

5. **Access the UI**
   
   Open your browser and navigate to: `http://localhost:8501`

## 🚀 Usage

### Basic Queries
```
User: "What is Amazon's CTC?"
Bot: [Returns salary information with sources]
```

### Complex Queries
```
User: "Compare Amazon and Google salaries and tell me if I'm eligible with 8.5 CGPA in CSE"
Bot: [Executes multi-step plan]
     1. Searches Amazon data
     2. Searches Google data
     3. Compares packages
     4. Checks eligibility
     5. Generates comparison table
```

### Follow-up Questions
```
User: "What about their interview process?"
Bot: [Remembers context and provides interview info for Amazon & Google]
```

## 📂 Project Structure

```
placement_chatbot/
├── agentic/                          # Agentic RAG module
│   ├── agentic_rag.py               # Main orchestrator
│   ├── planner.py                   # Query planning
│   ├── executor.py                  # Tool execution
│   ├── critic.py                    # Quality evaluation
│   ├── query_analyzer.py            # Routing logic
│   ├── memory_resolver.py           # Context resolution
│   ├── entity_extractor.py          # Entity extraction
│   └── tools/                       # 11 specialized tools
│       ├── vector_search.py
│       ├── company_extractor.py
│       ├── comparison.py
│       ├── eligibility.py
│       ├── answer_generator.py
│       ├── trend_analyzer.py
│       ├── branch_stats.py
│       ├── skill_demand.py
│       ├── company_cluster.py
│       ├── recommendation_engine.py
│       └── sql_query.py
├── streamlit_ui_v2.py               # Web interface
├── query_helper.py                  # Basic RAG
├── llm_reasoner.py                  # Answer refinement
├── input_validator.py               # Security validation
├── feedback_collector.py            # User feedback
├── chunks_generation.py             # Data chunking
├── embeddings.py                    # Embedding generation
├── pinecone_upsert.py              # Vector DB upload
├── deepseek_ocr.py                 # OCR processing
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🔧 Data Processing Pipeline

### Offline Phase (One-time setup)

1. **Document Collection**: Gather PDFs, DOCX, PPTX files
2. **OCR Processing**: Extract text from scanned documents
3. **Chunking**: Split into 512-token chunks with overlap
4. **Classification**: Label sections (Eligibility, Compensation, etc.)
5. **Embedding**: Generate 384D vectors
6. **Upload**: Push to Pinecone vector database

### Real-time Phase (Per query)

1. **Validation**: Security checks, sanitization
2. **Routing**: Determine Basic vs Agentic mode
3. **Processing**: Execute search/tools
4. **Quality Check**: Critic evaluation
5. **Response**: Format and display with sources

## 🎯 Query Types Supported

| Category | Examples |
|----------|----------|
| **Single Company Info** | "What is Intel's CTC?" |
| **Multi-Company Comparison** | "Compare Amazon vs Google" |
| **Statistical Analysis** | "Average CTC of all companies" |
| **Eligibility Check** | "Am I eligible with 8.5 CGPA?" |
| **Interview Prep** | "Google interview questions" |
| **Timeline Info** | "When is Amazon visiting?" |
| **Trend Analysis** | "CTC trends over years" |
| **Branch Analysis** | "CSE vs ECE placements" |
| **Skill Analysis** | "Most demanded skills" |
| **Personalized Recommendations** | "Best companies for me" |

## 🔒 Security Features

- Input length validation (3-2000 characters)
- SQL injection prevention
- XSS attack filtering
- Rate limiting for abuse prevention
- Sanitized user inputs

## 📊 Performance Metrics

- **Simple queries**: ~2 seconds response time, 95% accuracy
- **Complex queries**: ~4 seconds response time, 82% accuracy
- **Data coverage**: 50+ companies, 3 years (2023-2025)
- **Vector database**: ~5000 chunks, 384D embeddings

## 🤝 Contributing

This is an academic project developed for MSIS, MAHE. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Manjus2003**
- GitHub: [@Manjus2003](https://github.com/Manjus2003)
- Institution: MSIS, MAHE

## 🙏 Acknowledgments

- MSIS Placement Cell for data access
- HuggingFace for model hosting
- Pinecone for vector database
- Streamlit for UI framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for MSIS, MAHE**
