# Sentiment Analysis Through Deep Learning  
A deep learning-based sentiment analysis system built using **Convolutional Neural Networks (CNN)** and deployed through a **Flask web application**.  
The model predicts **Positive**, **Negative**, and **Neutral** sentiments from text comments.

---

## 🚀 Features
- 🔒 **User Login & Registration (Flask + SQLite)**  
- 🧠 **CNN-based Deep Learning Model** for sentiment classification  
- 🗂 **1 Million Sentence Dataset** (cleaned & preprocessed)  
- 📝 **Tokenizer + LabelEncoder saved for inference**  
- 🌐 **Web interface for real-time sentiment prediction**  
- 📊 **Training graphs, confusion matrix & evaluation metrics**  

---

## 📁 Project Structure
```
sentiment-analysis-deep-learning/
│
├── app.py
├── SentimentAnalysis.ipynb
├── YoutubeCommentsDataSet.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── SentimentCNN_Custom_MultiClass.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│
├── instance/
│   ├── database.db       # auto-created
│   ├── users.db          # auto-created
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── sentiment.html
│
└── static/
    ├── css/
    ├── js/
    ├── images/
```

---

## 🧠 Deep Learning Model
- Model Type: **CNN (Conv1D, GlobalMaxPooling1D, Dense Layers)**  
- Text preprocessing:  
  - Tokenization  
  - Padding  
  - Lowercasing  
  - Removing unwanted characters  
- Encoded using `LabelEncoder`  
- Trained on **1,000,000 labeled sentences**

---

## 📊 Evaluation
- Confusion Matrix  
- Classification Report  
- Accuracy, Precision, Recall, F1-score  
- Training & Validation curves stored in notebook  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/sentiment-analysis-deep-learning.git
cd sentiment-analysis-deep-learning
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web App
```bash
python app.py
```

Open the browser →  
👉 http://127.0.0.1:5000/

---

## 🛠 Tech Stack
- **Python**
- **Flask**
- **TensorFlow / Keras**
- **Pandas, NumPy**
- **Sklearn**
- **HTML, CSS, JavaScript**
- **SQLite Database**

---

## 📌 Future Enhancements
- Deploy on AWS/GCP  
- Add BERT or Transformer-based models  
- Add Admin Dashboard  
- Provide API endpoints for mobile apps  

---

## 👨‍💻 Author  
**Abhijit Zade**
