import os
import re
import torch
import logging
import asyncio
import aiohttp
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# ---------------- Logging ---------------- #
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("fake-news-api-bot")

# ---------------- Config ---------------- #
HF_REPO = os.getenv("HF_REPO", "Chiemelie/finetuned-fake-news-bot")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# API self-call base (so bot talks to this same process)
PORT = os.getenv("PORT", "8000")
API_BASE = os.getenv("API_BASE", f"http://127.0.0.1:{PORT}")
API_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

# ---------------- Model ---------------- #
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSequenceClassification] = None

def load_model() -> bool:
    """Always load model from Hugging Face Hub"""
    global tokenizer, model
    try:
        logger.info(f"Loading model & tokenizer from Hugging Face Hub: {HF_REPO}")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN", None)
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN", None)
        )
        model.eval()
        logger.info("✅ Model & tokenizer loaded from Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model from Hugging Face Hub: {e}")
        return False

# ---------------- FastAPI ---------------- #
class InputRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Text to classify")

class OutputResponse(BaseModel):
    prediction: str
    confidence: float
    text_length: int

app = FastAPI(
    title="Nigerian Fake News Detection API",
    description="FastAPI + Telegram Bot (Hub-only model)",
    version="1.0.0"
)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[^\w\s\.\,\!\?\-']", "", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text

@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.warning("⚠️ Model not loaded. Endpoints will return 503 until it loads successfully.")

@app.get("/")
async def root():
    return {
        "message": "Fake News Detection API + Telegram Bot",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict", response_model=OutputResponse)
async def predict(req: InputRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        text = clean_text(req.text)
        if len(text) < 10:
            raise HTTPException(status_code=400, detail="Text too short after cleaning")
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        conf = float(probs[0][pred_class].item())
        prediction = "true" if pred_class == 1 else "false"
        logger.info(f"Prediction={prediction} | Confidence={conf:.3f} | Length={len(text)}")
        return OutputResponse(prediction=prediction, confidence=conf, text_length=len(text))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ---------------- Telegram Bot ---------------- #
class FakeNewsBot:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("Check Status", callback_data="status")],
            [InlineKeyboardButton("Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "👋 Welcome to the Nigerian Fake News Detection Bot!\n"
            "Send me any news text and I'll tell you if it's likely to be fake or not.",
            reply_markup=reply_markup
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "🤖 *How to use this bot:*\\n\\n"
            "1️⃣ Simply send any news text \\(headline or full article\\).\\n"
            "2️⃣ The bot will analyze and respond with the prediction.\\n"
            "3️⃣ Use /status to check if the service is healthy.\\n\\n"
            "⚠️ Note: This is an AI\\-powered prediction, always verify with trusted sources\\."
        )
        await update.message.reply_text(help_text, parse_mode="MarkdownV2")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = await self.check_api_health()
        await update.message.reply_text(status)

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        await update.message.reply_text("Analyzing the news text... ⏳")
        prediction = await self.predict_fake_news(text)
        await update.message.reply_text(prediction, disable_web_page_preview=True)

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if query.data == "status":
            status = await self.check_api_health()
            await query.edit_message_text(status)
        elif query.data == "help":
            help_text = (
                "🤖 *How to use this bot:*\\n\\n"
                "1️⃣ Simply send any news text \\(headline or full article\\).\\n"
                "2️⃣ The bot will analyze and respond with the prediction.\\n"
                "3️⃣ Use /status to check if the service is healthy.\\n\\n"
                "⚠️ Note: This is an AI\\-powered prediction, always verify with trusted sources\\."
            )
            await query.edit_message_text(help_text, parse_mode="MarkdownV2")

    async def predict_fake_news(self, text: str) -> str:
        try:
            async with self.session.post(API_URL, json={"text": text}, timeout=45) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pred = data.get("prediction", "unknown")
                    conf = data.get("confidence", 0.0)
                    return f"📰 Prediction: *{pred}*\nConfidence: *{conf:.3f}*" \
                           if pred != "unknown" else "⚠️ Empty response from API."
                else:
                    return f"⚠️ API error: HTTP {resp.status}"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"❌ Error contacting API: {e}"

    async def check_api_health(self) -> str:
        try:
            async with self.session.get(HEALTH_URL, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return "✅ API is healthy and running!" if data.get("status") == "healthy" else "⚠️ API reports unhealthy."
                else:
                    return f"⚠️ API health check failed: HTTP {resp.status}"
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return f"❌ API health check error: {e}"

async def run_telegram():
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("Please set TELEGRAM_TOKEN environment variable")
        return
    async with aiohttp.ClientSession() as session:
        bot = FakeNewsBot(session)
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", bot.start_command))
        application.add_handler(CommandHandler("help", bot.help_command))
        application.add_handler(CommandHandler("status", bot.status_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text_message))
        application.add_handler(CallbackQueryHandler(bot.handle_callback_query))
        logger.info("🤖 Telegram bot starting (long polling)...")
        await application.run_polling(close_loop=False)

# ---------------- Run Both ---------------- #
if __name__ == "__main__":
    import uvicorn
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop()
    loop.create_task(run_telegram())
    logger.info(f"🚀 Starting FastAPI on 0.0.0.0:{PORT} | API_BASE={API_BASE}")
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
