import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# Налаштування
MODEL_NAME = "dima806/bird_species_image_detection"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
DB_FILE = BASE_DIR / "birds.db"
CSV_FILE = BASE_DIR / "bird_analysis.csv"
CHART_FILE = BASE_DIR / "bird_chart.png"

INPUT_DIR.mkdir(exist_ok=True)

bird_model = None


# Створення бази даних
def create_database():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS birds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                predicted_species TEXT,
                confidence REAL,
                analysis_date TEXT,
                description TEXT,
                source TEXT,
                top_three TEXT
            )
        """)


# Завантаження AI-моделі
def load_model():
    global bird_model
    if bird_model is None:
        print("\nЗавантаження моделі Hugging Face...")
        bird_model = pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=-1
        )
        print("Модель готова до роботи!")
    return bird_model


# Отримання інформації з Wikipedia
def get_bird_info(species):
    title = species.replace("_", " ").replace("-", " ").title()
    params = urlencode({
        "action": "query",
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
        "format": "json",
        "redirects": 1
    })
    url = "https://en.wikipedia.org/w/api.php?" + params
    try:
        request = Request(
            url,
            headers={"User-Agent": "AIBirdLabStudent/1.0"}
        )
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" not in page:
            description = page.get("extract", "").strip()
            if description:
                description = " ".join(description.split())
                return description[:500], page.get("fullurl", "")
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    return "Довідкова інформація недоступна.", ""


# Аналіз фотографії
def analyze_image(image_path):
    if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Підтримуються лише JPG, JPEG та PNG.")
    if not image_path.is_file():
        raise FileNotFoundError("Фотографію не знайдено!")
    try:
        with Image.open(image_path) as photo:
            image = photo.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("Неможливо відкрити зображення!")
    model = load_model()
    print("\nАналіз:", image_path.name)
    predictions = model(image, top_k=3)
    if not predictions:
        raise ValueError("Модель не повернула результат!")
    best = predictions[0]
    species = str(best["label"])
    confidence = float(best["score"])
    description, source = get_bird_info(species)
    print("\n     AI RESULT")
    for i, item in enumerate(predictions, 1):
        print(
            f"{i}. {item['label']} — "
            f"{item['score'] * 100:.2f}%"
        )
    print("\nНайімовірніший вид:", species)
    print("Впевненість:", f"{confidence * 100:.2f}%")
    print("Опис:", description)
    if source:
        print("Джерело:", source)
    # Зберігаємо результат у SQLite
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO birds (
                image_name, predicted_species, confidence,
                analysis_date, description, source, top_three
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            image_path.name,
            species,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description,
            source,
            json.dumps(predictions, ensure_ascii=False)
        ))
    print("\nРезультат збережено в базі даних!")
    export_csv()
    return species


# Вибір фотографії
def choose_image():
    images = sorted(
        file for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print("\nДоступні фотографії:")
    for i, image in enumerate(images, 1):
        print(f"{i}. {image.name}")
    choice = input("\nВведіть номер або шлях до фото: ").strip().strip('"')
    if choice.isdigit():
        number = int(choice)
        if 1 <= number <= len(images):
            return images[number - 1]
        print("Невірний номер!")
        return None
    path = Path(choice)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        print("Файл не знайдено!")
        return None
    return path


# Аналіз однієї фотографії
def analyze_one():
    path = choose_image()
    if path is not None:
        analyze_image(path)


# Масовий аналіз фотографій
def analyze_all():
    images = sorted(
        file for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        print("\nУ папці input немає фотографій!")
        return
    print("\nЗнайдено фотографій:", len(images))
    success = 0
    for image_path in images:
        try:
            analyze_image(image_path)
            success += 1
        except Exception as error:
            print("\nПомилка:", image_path.name, error)
    print("\n     ANALYSIS REPORT")
    print("Усього фотографій:", len(images))
    print("Успішно:", success)
    print("Помилок:", len(images) - success)


# Експорт у CSV
def export_csv():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("""
            SELECT image_name, predicted_species,
                   confidence, analysis_date, description
            FROM birds
            ORDER BY id
        """).fetchall()
    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_name",
            "predicted_species",
            "confidence",
            "analysis_date",
            "description"
        ])
        writer.writerows(rows)


# Перегляд історії
def show_history():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("""
            SELECT id, image_name, predicted_species,
                   confidence, analysis_date
            FROM birds
            ORDER BY id DESC
        """).fetchall()
    print("\n     HISTORY")
    if not rows:
        print("Історія порожня.")
        return
    for row in rows:
        print(
            f"ID: {row[0]} | Фото: {row[1]} | "
            f"Вид: {row[2]} | "
            f"{row[3] * 100:.2f}% | {row[4]}"
        )


# Пошук за назвою
def search_bird():
    name = input("\nВведіть назву птаха: ").strip()
    if not name:
        print("Введіть назву для пошуку!")
        return
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("""
            SELECT id, image_name, predicted_species,
                   confidence, description, source
            FROM birds
            WHERE predicted_species LIKE ?
        """, ("%" + name + "%",)).fetchall()
    if not rows:
        print("Записів не знайдено!")
        return
    for row in rows:
        print("\nID:", row[0])
        print("Фото:", row[1])
        print("Вид:", row[2])
        print(f"Впевненість: {row[3] * 100:.2f}%")
        print("Опис:", row[4])
        if row[5]:
            print("Джерело:", row[5])


# Видалення запису
def delete_record():
    try:
        bird_id = int(input("\nВведіть ID для видалення: "))
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.execute(
                "DELETE FROM birds WHERE id = ?",
                (bird_id,)
            )
            deleted = cursor.rowcount
        if deleted:
            export_csv()
            print("Запис видалено!")
        else:
            print("Запис із таким ID не знайдено!")
    except ValueError:
        print("Введіть правильний числовий ID!")


# Статистика
def show_statistics():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("""
            SELECT predicted_species, confidence
            FROM birds
        """).fetchall()
    if not rows:
        print("\nНемає результатів для статистики!")
        return
    names = [row[0] for row in rows]
    most_common = Counter(names).most_common(5)
    highest = max(rows, key=lambda x: x[1])
    lowest = min(rows, key=lambda x: x[1])
    print("\n     STATISTICS")
    print("Усього аналізів:", len(rows))
    print("Різних визначених видів:", len(set(names)))
    print("\nНайчастіші види:")
    for name, count in most_common:
        print(f"- {name}: {count}")
    print("\nНайвища впевненість:")
    print(f"{highest[0]} — {highest[1] * 100:.2f}%")
    print("\nНайнижча впевненість:")
    print(f"{lowest[0]} — {lowest[1] * 100:.2f}%")


# Порівняння двох записів
def compare_birds():
    show_history()
    try:
        first = int(input("\nID першого запису: "))
        second = int(input("ID другого запису: "))
        if first == second:
            print("Оберіть два різні записи!")
            return
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute("""
                SELECT id, predicted_species, confidence, description
                FROM birds
                WHERE id IN (?, ?)
            """, (first, second)).fetchall()
        if len(rows) != 2:
            print("Один із записів не знайдено!")
            return
        print("\n    COMPARISON")
        for row in rows:
            print(f"\nID: {row[0]}")
            print("Вид:", row[1])
            print(f"Впевненість: {row[2] * 100:.2f}%")
            print("Опис:", row[3])
    except ValueError:
        print("Введіть правильні ID!")


# Графік класифікації
def create_chart():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("""
            SELECT predicted_species
            FROM birds
        """).fetchall()
    if not rows:
        print("\nНемає даних для графіка!")
        return
    counts = Counter(row[0] for row in rows)
    names = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(11, 6))
    plt.barh(names, values)
    plt.xlabel("Кількість результатів")
    plt.title("AI Bird Lab — розподіл визначених видів")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.show()
    plt.close()
    print("\nГрафік збережено:", CHART_FILE)


# Інформація про модель
def model_info():
    print("\n      MODEL INFORMATION")
    print("Модель:", MODEL_NAME)
    print("Посилання: https://huggingface.co/" + MODEL_NAME)
    print("Завдання: класифікація видів птахів за фото")
    print("Вхід: зображення JPG, JPEG або PNG")
    print("Вихід: три найімовірніші класи та confidence")
    print("Бібліотеки: transformers, torch, pillow")
    print("Довідкова інформація: Wikipedia (за наявності)")


# Головне меню
def show_menu():
    print("\n          AI BIRD LAB")
    print("1. Аналіз однієї фотографії")
    print("2. Аналіз усіх фотографій")
    print("3. Історія аналізів")
    print("4. Пошук птаха")
    print("5. Видалити запис")
    print("6. Статистика")
    print("7. Порівняти два записи")
    print("8. Побудувати графік")
    print("9. Інформація про модель")
    print("0. Вихід")


# Запуск програми
def main():
    create_database()
    export_csv()
    while True:
        show_menu()
        choice = input("\nОберіть дію: ").strip()
        try:
            if choice == "1":
                analyze_one()
            elif choice == "2":
                analyze_all()
            elif choice == "3":
                show_history()
            elif choice == "4":
                search_bird()
            elif choice == "5":
                delete_record()
            elif choice == "6":
                show_statistics()
            elif choice == "7":
                compare_birds()
            elif choice == "8":
                create_chart()
            elif choice == "9":
                model_info()
            elif choice == "0":
                print("\nДо побачення!")
                break
            else:
                print("Невірний вибір!")
        except Exception as error:
            print("\nПомилка:", error)


if __name__ == "__main__":
    main()
