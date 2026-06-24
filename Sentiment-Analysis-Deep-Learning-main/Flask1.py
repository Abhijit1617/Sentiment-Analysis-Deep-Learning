from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# === Database Setup ===
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

# === Load Model, Tokenizer, and Label Encoder ===
MODEL_PATH = r"C:\Users\gr064\OneDrive\Desktop\Sentiment Analysis\tf_env\sentiment_model.h5"
TOKENIZER_PATH = r"C:\Users\gr064\OneDrive\Desktop\Sentiment Analysis\tf_env\tokenizer.pkl"
LABEL_ENCODER_PATH = r"C:\Users\gr064\OneDrive\Desktop\Sentiment Analysis\tf_env\label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(LABEL_ENCODER_PATH, 'rb') as handle:
    label_encoder = pickle.load(handle)

# === Routes ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session['user'] = user.email
            flash("Login successful!", "success")
            return redirect(url_for('sentiment'))
        else:
            flash("Invalid credentials!", "error")
            return redirect(url_for('login'))

    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required!", "error")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("User already exists!", "error")
            return redirect(url_for('register'))

        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        session['user'] = new_user.email
        flash("Registration successful!", "success")
        return redirect(url_for('sentiment'))

    return render_template("register.html")

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if 'user' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    sentiment_result = None
    if request.method == 'POST':
        text = request.form.get("text")

        if not text or text.strip() == "":
            flash("Please enter text for analysis!", "error")
            return redirect(url_for('sentiment'))

        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=100)

        prediction = model.predict(padded_sequences)
        predicted_label = np.argmax(prediction, axis=1)
        sentiment_result = label_encoder.inverse_transform(predicted_label)[0]

    return render_template("sentiment.html", sentiment=sentiment_result)

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for('login'))

# === Main ===
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


