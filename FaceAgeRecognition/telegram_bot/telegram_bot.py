import os
import io
import logging
from typing import Optional
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


def conv_block(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class AgeCNN(nn.Module):

    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),

            conv_block(128, 256),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x).squeeze(1)
        x = torch.sigmoid(x)  # normalized age in [0,1]
        return x

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "best_agecnn.pt")

IMG_SIZE = int(os.environ.get("IMG_SIZE", "200"))
MAX_AGE = float(os.environ.get("MAX_AGE", "100"))

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: str) -> nn.Module:
    model = AgeCNN(dropout=0.4).to(DEVICE)
    ckpt = torch.load(weights_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


MODEL = load_model(WEIGHTS_PATH)


def predict_age_years(pil_img: Image.Image) -> float:
    pil_img = pil_img.convert("RGB")
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        age_norm = MODEL(x).item()  # [0..1]

    age_years = age_norm * MAX_AGE
    return float(age_years)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("age-bot")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Пришли фото (лицо крупно, без сильного наклона) — оценю возраст."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n"
        "/start — начать\n"
        "/help — помощь\n\n"
        "Просто отправь фото или картинку файлом."
    )


async def _handle_image_bytes(update: Update, data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        age = predict_age_years(img)
        await update.message.reply_text(f"Оценка возраста: ~{age:.1f} лет")
    except Exception as e:
        logger.exception("Failed to process image: %s", e)
        await update.message.reply_text(
            "Не смог обработать изображение 😕\n"
            "Попробуй другое фото (лицо ближе/чётче, меньше фон)."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    data = await tg_file.download_as_bytearray()
    await _handle_image_bytes(update, data)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return

    mime = doc.mime_type or ""
    if not mime.startswith("image/"):
        await update.message.reply_text("Пришли файл-картинку (jpg/png).")
        return

    tg_file = await doc.get_file()
    data = await tg_file.download_as_bytearray()
    await _handle_image_bytes(update, data)


def main():
    token = os.environ.get("BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN. Укажи его в переменной окружения.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot started. Device=%s, Weights=%s", DEVICE, WEIGHTS_PATH)
    app.run_polling()


if __name__ == "__main__":
    main()