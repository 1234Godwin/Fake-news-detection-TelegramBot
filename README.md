# 📰 Fake News Detection Telegram Bot

A machine learning-powered Telegram bot that detects fake news in Nigerian news articles.
It uses a fine-tuned transformer model to classify whether a news article is True or Fake/Misleading, helping users quickly verify information.

## 🚀 Features

🤖 Telegram Bot integration for real-time news verification

🧠 Transformer-based NLP model (fine-tuned on Nigerian fact-check datasets: Dubawa, AfricaCheck)

📊 Data preprocessing, EDA, and model training notebooks included

🌐 FastAPI backend for serving model predictions & Telegram bot endpoints

☁️ Deployable on Railway or any cloud platform


## ⚙️ Installation

Clone the repo and set up the environment:

git clone https://github.com/1234Godwin/Fake-news-detection-TelegramBot.git
cd Fake-news-detection-TelegramBot


Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

🤖 Run Locally
1️⃣ Start FastAPI app
uvicorn API_folder.main:app --reload


API will run at: http://127.0.0.1:8000

2️⃣ Start Telegram Bot

Update your Telegram Bot Token in .env or as an environment variable, then run:

python API_folder/telegram_bot.py

## ☁️ Deployment (Railway)

Push project to GitHub

Create a new Railway project → Deploy from GitHub Repo

Set required environment variables (e.g., TELEGRAM_TOKEN, MODEL_PATH)

Railway automatically installs dependencies from requirements.txt and runs Procfile

## 📊 Dataset

Dubawa Fact-Check articles

AfricaCheck verified claims

Labels normalized into:

True

Fake/Misleading

## 🔮 Future Improvements

Add multilingual support (e.g., Hausa, Igbo, Yoruba)

Improve model accuracy with larger transformer models

Expand dataset with more Nigerian fact-check sources

## 👨‍💻 Author

Chiemelie Onu

🌍 AI/ML Engineer

LinkedIn: https://www.linkedin.com/in/chiemelieonu/
GitHub: https://github.com/1234Godwin
