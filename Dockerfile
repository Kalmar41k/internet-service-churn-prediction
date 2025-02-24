# Вибираємо базовий образ для Python
FROM python:3.9-slim

# Оновлюємо pip
RUN pip install --upgrade pip

# Копіюємо файл requirements.txt в контейнер
COPY requirements.txt /app/requirements.txt

# Встановлюємо всі залежності з requirements.txt
RUN pip install -r /app/requirements.txt

# Робоча директорія
WORKDIR /app
COPY . /app

# Вказуємо команду для запуску додатку
CMD ["streamlit", "run", "app.py"]