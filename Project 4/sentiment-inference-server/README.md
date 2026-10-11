Sentiment Inference Server

A small project where I put a pre-trained machine-learning model behind a web API, so other programs can send it text and get a sentiment (positive/negative) back.

I wanted to learn how a model goes from "a file on my laptop" to something a real system can call. I didn't train the model. I used DistilBERT from Hugging Face (distilbert-base-uncased-finetuned-sst-2-english) and focused on serving it.

How it works
The server (FastAPI) starts and loads the model once into memory.
A client sends text to an endpoint as JSON.
The server runs the model and returns a label, a confidence score and how long it took.

Loading the model once at startup matters because loading it on every request would be very slow.

Endpoints
GET /health checks the server is up and the model is loaded
POST /predict classifies one piece of text
POST /predict/batch classifies a list of texts in one go

Example:

bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this film"}'

Output I got when I ran it:

"5 passed in 13.14s"
Running it
bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000

The first start downloads the model (about 270 MB). Then open http://localhost:8000/docs to try it in the browser.

Tests (they use a fake model, so they run without the download):

bash
python -m pytest -q
Things I learned / problems I hit
Installing everything into my Anaconda setup caused package version conflicts (numpy and scikit-learn), so I used a separate virtual environment to fix it.

Ideas for improving it
Add GitHub Actions to run the tests automatically
Try a different model or add GPU support
Add logging of request times