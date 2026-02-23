import asyncio
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction


load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

class Config:
    MODEL_NAME = 'efficientnet_b3'
    NUM_CLASSES = 7
    IMAGE_SIZE = 224
    DEVICE = 'cpu'
    CHECKPOINT_PATH = './models/best_model.pth'


config = Config()

CLASS_NAMES = {
    0: '🔴 МЕЛАНОМА',
    1: '🟢 Невус (безопасно)',
    2: '🟡 Базалиома',
    3: '🟡 Актинический кератоз',
    4: '🟢 Дерматоз',
    5: '🟢 Дерматофиброма',
    6: '🟡 Сосудистое поражение'
}

DESCRIPTIONS = {
    0: 'ОПАСНО! Это может быть МЕЛАНОМА. Немедленно обратись к дерматологу!',
    1: 'Обычная безопасная родинка. Наблюдай, но поводов для беспокойства нет.',
    2: 'Это может быть базалиома. Требует осмотра дерматолога.',
    3: 'Это может быть актинический кератоз. Требует осмотра дерматолога.',
    4: 'Обычное доброкачественное поражение. Наблюдай.',
    5: 'Доброкачественное поражение. Наблюдай.',
    6: 'Это может быть сосудистое поражение. Требует осмотра дерматолога.'
}


def create_model(model_name, num_classes, pretrained=True):
    if model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    return model


class MelanomaModel(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


def load_model(config):
    backbone = create_model(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False)
    model = MelanomaModel(backbone, config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    return model


model = load_model(config)

def predict_image(image_pil, model, config):
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE + 20),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image_pil).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    result = {
        'class_id': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'description': DESCRIPTIONS[predicted_class],
        'confidence': float(probabilities[predicted_class].cpu().numpy()) * 100,
        'all_probs': {CLASS_NAMES[i]: float(p.cpu().numpy()) * 100 for i, p in enumerate(probabilities)}
    }

    return result

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """ <b>Привет! Я бот для анализа родинок</b>

Отправь мне фото родинки, и я расскажу что это такое.

<b>Команды:</b>
/start - помощь
/info - информация о классах

⚠<b>ВАЖНО:</b>
Я <b>НЕ врач</b> и это <b>НЕ медицинский диагноз</b>!
Используй меня только как вспомогательный инструмент.
При любых сомнениях обратись к дерматологу!""",
        parse_mode='HTML'
    )


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """📋 <b>Классы которые я распознаю:</b>

🔴 <b>Меланома</b> - ОПАСНО! Немедленно к врачу!
🟡 <b>Базалиома</b> - требует осмотра дерматолога
🟡 <b>Актинический кератоз</b> - требует осмотра
🟡 <b>Сосудистое поражение</b> - требует осмотра
🟢 <b>Невус</b> - обычная безопасная родинка
🟢 <b>Дерматофиброма</b> - доброкачественная родинка
🟢 <b>Дерматоз</b> - доброкачественное поражение

ℹ️ <b>О модели:</b>
Обучена на HAM10000 датасете (10,000+ изображений)
Точность: ~85-90%

⚠️ <b>ДИСКЛЕЙМЕР:</b>
Это автоматическое предсказание модели ИИ, а не медицинский диагноз!""",
        parse_mode='HTML'
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        image = Image.open(BytesIO(photo_bytes)).convert('RGB')

        print(f"Обрабатываю фото от {update.effective_user.username or update.effective_user.first_name}")
        result = predict_image(image, model, config)

        message = f"""<b>РЕЗУЛЬТАТ АНАЛИЗА</b>

<b>Диагноз:</b> {result['class_name']}
<b>Уверенность:</b> {result['confidence']:.1f}%

<b>Описание:</b>
{result['description']}

<b>📊 Вероятности по всем классам:</b>
"""
        for class_name, prob in result['all_probs'].items():
            bar_length = 15
            filled = int(bar_length * prob / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            message += f"{class_name}: {bar} {prob:.1f}%\n"

        message += f"""
<b>⚠ВНИМАНИЕ:</b>
Это автоматическое предсказание модели ИИ.
<b>НЕ ЯВЛЯЕТСЯ медицинским диагнозом!</b>
Обратись к дерматологу для точного диагноза."""

        await update.message.reply_text(message, parse_mode='HTML')

    except Exception as e:
        print(f"Ошибка: {e}")
        await update.message.reply_text(
            f"""<b>Ошибка при обработке фото:</b>

Убедись что:
Отправил именно ФОТО (не скриншот)
Качество фото хорошее
Видна сама родинка
Размер файла не слишком большой

Попробуй еще раз!""",
            parse_mode='HTML'
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """Отправь мне <b>ФОТО родинки</b>, и я проанализирую! 

Команды:
/start - помощь
/info - информация""",
        parse_mode='HTML'
    )


def main():
    print(f"Устройство: {config.DEVICE}")
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Токен загружен: {'да' if TELEGRAM_BOT_TOKEN else 'нет'}")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))

    app.run_polling()


if __name__ == '__main__':
    main()
