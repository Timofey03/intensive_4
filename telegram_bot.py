import logging
import joblib
import numpy as np
from difflib import get_close_matches
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Определяем класс для stats.pkl
class TextStatsExtractor:
    def __init__(self):
        pass

    def transform(self, texts):
        features = []
        for text in texts:
            length = len(text)
            digits = sum(c.isdigit() for c in text)
            words = len(text.split())
            features.append([length, digits, words])
        return features

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Загрузка модели и трансформеров
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")
stats = joblib.load("stats.pkl")

# Категории
categories = [
    "Вопрос решен",
    "Нравится качество выполнения заявки",
    "Нравится качество работы сотрудников",
    "Нравится скорость отработки заявок",
    "Понравилось выполнение заявки"
]

# Индивидуальные пороги
thresholds = {
    "Вопрос решен": 0.75,
    "Нравится качество выполнения заявки": 0.6,
    "Нравится качество работы сотрудников": 0.45,
    "Нравится скорость отработки заявок": 0.6,
    "Понравилось выполнение заявки": 0.6,
}

# Словарь ключевых слов
keyword_map = {
    "Нравится качество работы сотрудников": ["мастер", "специалист", "работник", "работала", "работал", "сотрудник", "вежливый", "грамотный", "хороший"],
    "Нравится скорость отработки заявок": ["быстро", "оперативно", "сразу", "недолго", "мгновенно", "скорость"],
    "Нравится качество выполнения заявки": ["хорошо", "качественно", "аккуратно", "выполнено", "чисто", "работа сделана"],
    "Понравилось выполнение заявки": ["заявка", "выполнена", "выполнено", "всё сделали", "сделали"],
}

# Ключевые слова, которые автоматически считаются осмысленными (даже если короткие)
base_meaningful_words = {
    "спасибо", "благодарю", "быстро", "оперативно", "мастер", "скорость", "вежливо", "чисто", "аккуратно"
}

def correct_typos(text):
    known_words = set([
        "работа", "мастер", "быстро", "качество", "заявка", "выполнено", "вежливо", "спасибо",
        "оперативно", "сотрудник", "грамотно", "чисто", "аккуратно", "вопрос", "решено"
    ])
    words = text.split()
    corrected = []

    for word in words:
        matches = get_close_matches(word, known_words, n=1, cutoff=0.8)
        corrected.append(matches[0] if matches else word)

    return ' '.join(corrected)

def preprocess(text):
    text = text.strip()
    X_tfidf = tfidf.transform([text])
    X_stats = stats.transform([text])
    X_stats_scaled = scaler.transform(X_stats)

    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_stats_scaled)])
    return X_combined

def is_meaningful(text):
    text_clean = text.lower().strip()

    # Если одно слово и оно есть в базе коротких осмысленных слов
    if len(text_clean.split()) == 1 and text_clean in base_meaningful_words:
        return True

    if len(text_clean) < 3:
        return False

    vowels = set("аеёиоуыэюяaeiou")
    return any(ch in vowels for ch in text_clean)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет! Отправь комментарий, и я классифицирую его по категориям.")

def handle_message(update: Update, context: CallbackContext):
    user_text = update.message.text.strip()

    # Исправляем опечатки
    user_text = correct_typos(user_text)

    if not is_meaningful(user_text):
        update.message.reply_text("Пожалуйста, отправьте осмысленный комментарий.")
        return

    # Снижение фильтрации на длину, но всё ещё отбрасываем слишком короткие бессмысленные сообщения
    if len(user_text.split()) <= 1 and len(user_text) < 3:
        update.message.reply_text("Комментарий слишком короткий для определения категории.")
        return

    X_input = preprocess(user_text)

    try:
        proba = model.predict_proba(X_input)
        result = []
        for cat, p in zip(categories, proba[0]):
            prob = p if isinstance(p, float) else p[1] if len(p) == 2 else p
            if prob >= thresholds.get(cat, 0.6):
                result.append(cat)
    except AttributeError:
        prediction = model.predict(X_input)[0]
        result = [cat for cat, pred in zip(categories, prediction) if pred == 1]

    # Ручное определение по ключевым словам
    manual_result = []
    lower_text = user_text.lower()
    for cat, keywords in keyword_map.items():
        if any(word in lower_text for word in keywords):
            manual_result.append(cat)

    # Объединение результатов, убираем дубликаты
    all_results = list(set(result + manual_result))

    if all_results:
        response = "Комментарий относится к категориям:\n" + "\n".join(f"• {r}" for r in all_results)
    else:
        response = "Комментарий не подходит ни под одну из категорий."

    update.message.reply_text(response)

def main():
    TOKEN = "7773493595:AAFHOqgfqe3obTPMhEpq-2TQ0DRlQT3aRgU"  # <-- замени на свой токен при необходимости
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
