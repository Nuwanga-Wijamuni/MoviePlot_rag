## 🚀 How to Run

1. **Clone the repository**
   ```powershell
   git clone https://github.com/Nuwanga-Wijamuni/MoviePlot_rag.git
   cd MoviePlot_rag
   ```

2. **Create and activate a Python virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Add your Groq API key to `.env`**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Prepare the dataset**
   ```powershell
   python subset.py
   ```

6. **Build the vector store**
   ```powershell
   python vectorstore.py
   ```

7. **Run the query and answer generation**
   ```powershell
   python query_generationllama.py
   ```

8. **(Optional) Evaluate retrieval metrics**
   ```powershell
   python retrieval_metrics.py
   ```
