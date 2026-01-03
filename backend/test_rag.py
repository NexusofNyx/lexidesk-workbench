import sys
import os
from pathlib import Path

# Add backend to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from lexidesk_chatbot.api.router import qa, QARequest

def test_chatbot():
    print("--- Starting Chatbot Test ---")
    question = "What is the document about?"
    print(f"Question: {question}")
    
    try:
        req = QARequest(question=question, top_k=3)
        response = qa(req)
        
        print("\n--- Response ---")
        print(f"Answer: {response['answer']}")
        print("\n--- Sources ---")
        for i, source in enumerate(response['sources']):
            print(f"Source {i+1}: {source['text'][:100]}...")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_chatbot()
