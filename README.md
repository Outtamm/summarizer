PDF Summarizer Application
Overview
This project is a web-based application that automatically generates summaries of PDF documents using Natural Language Processing (NLP). The application consists of two main components:

A FastAPI backend that handles PDF processing and text summarization using the LaMini-Flan-T5-248M model
A Streamlit frontend that provides a user-friendly interface for uploading PDFs and displaying summaries

The application efficiently handles the upload, processing, and summarization of PDF documents through an asynchronous task management system, allowing users to track the progress of their summarization requests in real-time.
Features

Upload PDF documents through a user-friendly web interface
Extract text content from PDF files
Generate concise summaries using the LaMini-Flan-T5-248M model
Real-time progress tracking of summarization tasks
Background processing for handling large documents without blocking the main application
Downloadable summary results
Error handling and comprehensive logging

Technical Architecture
Backend (FastAPI)
The backend is implemented as a RESTful API using FastAPI, providing endpoints for:

Uploading PDF files for summarization (/summarize)
Checking the status of ongoing summarization tasks (/status/{task_id})

The backend handles file processing, text extraction, and summary generation in background tasks to ensure responsiveness.
Frontend (Streamlit)
The frontend is built with Streamlit, offering:

A file upload interface
Progress tracking with visual indicators
Display of generated summaries
Download options for saving results

Text Summarization
The application uses the LaMini-Flan-T5-248M model from MBZUAI, a lightweight yet powerful text summarization model based on the T5 architecture.
Dependencies

Python 3.8+
FastAPI: A modern, fast web framework for building APIs
Streamlit: An open-source app framework for Machine Learning and Data Science projects
Transformers: Hugging Face's library for state-of-the-art NLP models
PyPDF2: A pure-python PDF library
Uvicorn: ASGI server for FastAPI
Torch: PyTorch for running the summarization model
Pydantic: Data validation and settings management
Requests: HTTP library for making API calls

Installation
1. Clone the repository
bashCopygit clone https://github.com/yourusername/pdf-summarizer.git
cd pdf-summarizer
2. Create a virtual environment
bashCopy# Using venv
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (Linux/macOS)
source venv/bin/activate
3. Install dependencies
bashCopypip install fastapi streamlit transformers torch pydantic uvicorn pypdf2 requests
4. Download model (optional)
The first time you run the application, it will download the LaMini-Flan-T5-248M model automatically. If you want to pre-download it:
bashCopypython -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; tokenizer = T5Tokenizer.from_pretrained('MBZUAI/LaMini-Flan-T5-248M'); model = T5ForConditionalGeneration.from_pretrained('MBZUAI/LaMini-Flan-T5-248M')"
Usage
1. Start the FastAPI backend
bashCopypython api.py
This will start the backend server on http://localhost:8000
2. Start the Streamlit frontend (in a new terminal)
bashCopystreamlit run streamlit_app.py
This will start the frontend application and open it in your default web browser at http://localhost:8501
3. Using the application

Upload a PDF document using the file uploader
Click "Generate Summary"
Wait for the processing to complete (progress is shown in real-time)
View and download the generated summary

API Documentation
Endpoints
POST /summarize
Upload a PDF file for summarization.
Request:

Content-Type: multipart/form-data
Body: file (PDF document)

Response:
jsonCopy{
  "task_id": "34ac790b-a568-450b-8133-481e6435cf7d"
}
GET /status/{task_id}
Check the status of a summarization task.
Parameters:

task_id: UUID of the task to check

Response (processing):
jsonCopy{
  "status": "processing"
}
Response (completed):
jsonCopy{
  "status": "completed",
  "summary": "This is the generated summary text..."
}
Response (failed):
jsonCopy{
  "status": "failed",
  "error": "Error message describing what went wrong"
}
Project Structure
Copypdf-summarizer/
├── api.py                  # FastAPI backend application
├── streamlit_app.py        # Streamlit frontend application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
How It Works
1. PDF Processing Flow

User uploads a PDF file through the Streamlit interface
The file is sent to the FastAPI backend
The backend creates a unique task ID and stores the task in memory
The backend reads the file content immediately to prevent it from being closed
A background task is started to process the file asynchronously
The task ID is returned to the frontend

2. Text Extraction

The PDF file is processed using PyPDF2 to extract text content
The text is cleaned and prepared for the summarization model

3. Summarization

The extracted text is tokenized and fed to the LaMini-Flan-T5-248M model
The model generates a concise summary with specified parameters:

Maximum length: 250 tokens
Minimum length: 50 tokens
Length penalty: 2.0
Beam search: 4 beams
Early stopping: Enabled



4. Result Handling

The frontend polls the backend for task status updates
When processing is complete, the summary is displayed to the user
The user can download the summary as a text file

Performance Considerations

The application uses background tasks to prevent blocking the main thread
Large PDFs may take longer to process depending on:

The complexity and length of the document
The available computational resources
The size of the PDF file


The LaMini-Flan-T5-248M model is a smaller version of T5, chosen for its balance of performance and resource requirements

Troubleshooting
Common Issues

"Cannot enter context with closed file" or "read of closed file" errors

These errors occur when FastAPI attempts to access a file that has been closed
Solution: Our implementation reads the entire file content before passing it to background tasks


PDF text extraction failures

Some PDFs may be scanned documents or have security features that prevent text extraction
Solution: Error handling in the code catches these issues and returns appropriate error messages


Slow processing times

Summarization can be computationally intensive, especially for large documents
Solution: Background task processing and status updates keep the application responsive



Logs
The application uses Python's logging module to provide detailed information about its operation:

INFO level logs for normal operation tracking
ERROR level logs for exception handling
All logs are printed to the console by default

Future Improvements

Database Integration

Replace in-memory task storage with a persistent database
Allow users to access their previous summaries


Multiple Model Support

Add options for different summarization models
Allow users to select models based on speed vs. quality preferences


Advanced PDF Handling

Support for password-protected PDFs
Better handling of scanned documents via OCR integration


User Authentication

Add user accounts and authentication
Secure API endpoints


Deployment Options

Docker containerization
Cloud deployment instructions (AWS, GCP, Azure)



License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

MBZUAI for the LaMini-Flan-T5-248M model
Hugging Face for the Transformers library
FastAPI and Streamlit teams for their excellent frameworks
