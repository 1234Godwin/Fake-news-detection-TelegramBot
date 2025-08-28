# 📰 Fake News Detection Telegram Bot

A machine learning-powered Telegram bot that detects fake news in Nigerian news articles.
It uses a fine-tuned transformer model to classify whether a news article is True or Fake/Misleading, helping users quickly verify information.

## 🚀 Features

🤖 Telegram Bot integration for real-time news verification

🧠 Transformer-based NLP model (fine-tuned on Nigerian fact-check datasets: Dubawa, AfricaCheck)

📊 Data preprocessing, EDA, and model training notebooks included

🌐 FastAPI backend for serving model predictions & Telegram bot endpoints

☁️ Deployable on Railway or any cloud platform

## 🏁 Milestones Achieved

Data acquisition pipeline: Built Scrapy + Playwright spiders to scrape Dubawa and other fact-check sites; normalized labels to True and Fake/Misleading. 

EDA & preprocessing: Added notebooks for cleaning, exploration, and feature preparation to ensure consistent inputs for training. 

Model fine-tuning: Trained a transformer-based classifier on Nigerian fact-check data; integrated it behind a FastAPI prediction endpoint. 

Bot integration: Wired the Telegram bot to the API for real-time verification flows. 

Containerization & deployability: Included Dockerfile and Procfile with instructions for Railway deployment. 

## 🌍 Importance

Combats misinformation at scale: Automated triage (“True” vs “False”) helps journalists, researchers, and the public prioritize which claims need deeper review. 

Bridges research and practice: Moves transformer-based NLP from notebooks into a production chat interface, demonstrating a practical pathway from model to user impact. 

Open, extensible pipeline: Because data collection, EDA, training, and serving are all in one repo, others can extend sources (more Nigerian fact-checkers), swap models, or add languages (Hausa/Igbo/Yoruba). 

Keeps pace with dynamic websites: Using scrapy-playwright means you can ingest content from sites that rely on JavaScript rendering—critical for modern media pages. 


## ⚙️ Installation

Clone the repo and set up the environment:

git clone https://github.com/1234Godwin/Fake-news-detection-TelegramBot.git
cd Fake-news-detection-TelegramBot


## 📊 Dataset

Dubawa Fact-Check articles

AfricaCheck verified claims

Labels normalized into: True and False

## 🔮 Future Improvements

Add multilingual support (e.g., Hausa, Igbo, Yoruba)

Improve model accuracy with larger transformer models

Expand dataset with more Nigerian fact-check sources

## 👨‍💻 Author

Chiemelie Onu

🌍 AI/ML Engineer

LinkedIn: https://www.linkedin.com/in/chiemelieonu/
GitHub: https://github.com/1234Godwin
