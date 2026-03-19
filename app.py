from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import google.generativeai as genai
import re

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels EXACTLY as given
with open("labels.txt") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f]

# 🔹 Clean Gemini response
def clean_text(text):
    text = re.sub(r'[#*]', '', text)
    return text.strip()

# 🔹 Gemini setup
genai.configure(api_key="AIzaSyCJB35Bv7cTJH6Q-d50VfDJQqXlfNdFnn0")
model_gemini = genai.GenerativeModel("gemini-3-flash-preview")

def get_solution_from_ai(disease, language):
    prompt = f"""
    A farmer's crop has {disease}.

    Explain in {language} language:
    - What is this disease
    - Why it happens
    - How to cure it
    - Preventive measures

    Keep it simple and clear.
    """

    response = model_gemini.generate_content(prompt)
    return clean_text(response.text)   # ✅ CLEAN HERE

# 🔹 Prediction function
def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    return labels[index]   # NO lower/strip

# 🔹 Home route
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Upload route
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    language = request.form.get('language', 'en')

    path = os.path.join("static", file.filename)
    file.save(path)

    result = predict_image(path)
    print("Prediction:", result)  # Debug

    # Language mapping
    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi"
    }
    selected_lang = lang_map.get(language, "English")

    # 🔥 LOGIC (EXACT MATCH)
    if result == "Not_a_Plant":
        solution = {
            "English": "❌ This is not a plant image. Please upload a crop leaf image.",
            "Hindi": "❌ यह पौधे की तस्वीर नहीं है। कृपया फसल की पत्ती की तस्वीर अपलोड करें।",
            "Marathi": "❌ ही वनस्पतीची प्रतिमा नाही. कृपया पिकाच्या पानाची प्रतिमा अपलोड करा."
        }[selected_lang]

    elif result == "Heathly Plant":
        solution = {
            "English": "✅ Your plant is healthy. No disease detected.",
            "Hindi": "✅ आपका पौधा स्वस्थ है। कोई बीमारी नहीं मिली।",
            "Marathi": "✅ तुमचा रोप निरोगी आहे. कोणताही रोग आढळला नाही."
        }[selected_lang]

    elif result == "Fungel infection disease" or result == "Bacteria infection disease":
        solution = get_solution_from_ai(result, selected_lang)

    else:
        solution = "⚠️ Unable to detect properly."

    return render_template(
        'index.html',
        result=result,
        solution=solution,
        image=path,
        selected_lang=language
    )

# chat route 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get("message")

    prompt = f"""
    User is a farmer asking: {user_msg}

    Give:
    - Current price estimate in India
    - Suggest what to buy
    - Simple advice

    Keep answer short.
    """

    response = model_gemini.generate_content(prompt)

    return {"response": response.text}

if __name__ == "__main__":
    app.run(debug=True)