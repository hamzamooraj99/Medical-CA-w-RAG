import os

# Check if the FAISS index exists
path = os.path.abspath(r".\dataset\nhsInform\faiss_index.bin")
print("Resolved FAISS path:", path)
print("Exists:", os.path.exists(path))

# C:\Users\hamza\Documents\Heriot-Watt\Y4\F20CA\Medical-CA-w-RAG\dataset\nhsInform\faiss_index.bin