from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from uuid import uuid4
from typing import Dict
import uvicorn
import logging
import time
import io
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and tokenizer
model_name = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI()

# In-memory storage to track task statuses
tasks = {}

class FileData:
    """
    Class to store uploaded file data.
    """
    def __init__(self, content: bytes, filename: str, content_type: str):
        self.content = content
        self.filename = filename
        self.content_type = content_type

def create_task_id() -> str:
    """
    Generate a unique task identifier.
    """
    return str(uuid4())

@app.get("/status/{task_id}")
async def check_status(task_id: str) -> Dict[str, str]:
    """
    Check the status of a given task.
    """
    task = tasks.get(task_id)
    if task:
        return task
    return {"status": "not found"}

def extract_text_from_pdf(file_data: FileData) -> str:
    """
    Extract text from a PDF file based on its binary content.
    """
    try:
        if '.' not in file_data.filename:
            logger.error("Filename has no extension.")
            raise ValueError("Filename has no extension.")

        file_extension = file_data.filename.split('.')[-1].lower()
        logger.info(f"Processing file: {file_data.filename} with extension: {file_extension}")

        if file_extension != 'pdf':
            logger.error(f"Unsupported file format: {file_extension}")
            raise ValueError("Unsupported file format (only PDF files are supported)")

        # Use BytesIO to read binary content
        with io.BytesIO(file_data.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            
            if not text.strip():
                raise ValueError("No text found in the PDF.")
            
            if len(text) < 50:
                raise ValueError("The extracted text is too short (minimum 50 words required).")
            
            return text

    except Exception as e:
        logger.error(f"Error processing file {file_data.filename}: {e}")
        raise ValueError(f"Error processing file: {str(e)}")

def process_file_sync(task_id: str, file_data: FileData) -> None:
    """
    Synchronous version of file processing for use in background tasks.
    """
    try:
        logger.info(f"Starting processing of task {task_id}")
        
        start_time = time.time()
        text = extract_text_from_pdf(file_data)
        logger.info(f"Text extraction time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Extracted text length: {len(text)} characters")

        # Tokenization
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

        # Summarization
        summary_start_time = time.time()
        summary_ids = model.generate(inputs, max_length=250, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info(f"Summary generation time: {time.time() - summary_start_time:.2f} seconds")

        tasks[task_id] = {"status": "completed", "summary": summary}
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/summarize")
async def summarize_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, str]:
    """
    API endpoint to summarize a PDF file.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read PDF content
        content = await file.read()
        logger.info(f"File read, size: {len(content)} bytes")
        
        # Store file information
        file_data = FileData(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )
        
        task_id = create_task_id()
        tasks[task_id] = {"status": "processing"}
        logger.info(f"Created task {task_id}, adding to background tasks")
        
        background_tasks.add_task(process_file_sync, task_id, file_data)
        
        return {"task_id": task_id}
    
    except Exception as e:
        logger.error(f"Error in /summarize endpoint: {e}")
        return {"error": str(e)}, 500

# Launch the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
