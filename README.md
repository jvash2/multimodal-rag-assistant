# Multimodal RAG Assistant with Context Memory

A professional-grade **Retrieval-Augmented Generation (RAG)** application designed to bridge the gap between static documents and interactive AI. This project features a robust memory system and the ability to process diverse data sources in real-time.

##  Key Features
- **Contextual Memory:** Maintains conversation history for coherent, multi-turn dialogues.
- **Multimodal Inputs:** Seamlessly processes information from:
  - Local text files (`.txt`).
  - Web content via direct URL extraction.
- **Session Management:** Includes a dedicated chat history reset functionality.
- **RAG Architecture:** Efficiently retrieves relevant document chunks to provide grounded, hallucination-free answers.

##  Tech Stack
- **Language:** Python 3.10+
- **AI Orchestration:** (e.g., LangChain / LlamaIndex)
- **Vector Database:** (e.g., ChromaDB / FAISS)
- **LLM:** (e.g., GPT-4 / Gemini Pro)
- **Environment Management:** Python-dotenv

##  Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name

2. Create a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
4. Environment Variables:
Create a .env file in the root directory and add your API keys:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   # Or for Gemini
   GOOGLE_API_KEY=your_api_key_here
   # For Groq ( i used this one)
   GROQ_API_KEY=your_api_key_here

Usage
  Run the main application:
     ```bash
        
        python main.py

To load a file: Follow the on-screen prompts to provide the path to your .txt file.

To load a URL: Input the URL when prompted to include web data in the knowledge base.

To reset: Use the internal reset command to clear the conversation memory.

License
This project is open-source and available under the MIT License.

   
