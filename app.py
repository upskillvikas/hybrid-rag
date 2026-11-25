import os
import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from rag_pipeline import rag_pipeline

api = FastAPI()

@api.post("/rag")
def get_response(query: str = Form(...), file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        answer = rag_pipeline(tmp_path, query)
        return {"answer": answer}
    finally:
        os.remove(tmp_path)