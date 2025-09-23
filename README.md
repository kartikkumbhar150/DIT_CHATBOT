Step 1: Setup .env file in /backend
Below is the .env file
GROQ_API_KEY=your_groq_api_key_here
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./faiss.index
PORT=5000


Step 2 : Setup your api from groq ai

Step 3 : create a folder named data inside backend

Step 4 : put you files in it.

Step 5 : In Terminal cd backend

Step 6 : install the requirements : pip install -r requirements.txt

Step 7 : python scraper.py

Step 8 : python embedding_indexer.py --build

Step 9 : python app.py
