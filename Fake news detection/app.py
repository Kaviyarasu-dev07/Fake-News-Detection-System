from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import re
import os
import secrets
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
# HARDCODED SECRET KEY: Crucial for cloud platforms like Render so that 
# rotating Gunicorn server workers don't accidentally log you out every 5 seconds!
app.secret_key = "fnds_secure_production_key_2026" 

# --- AUTHENTICATION LOGIC ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple hardcoded credentials loop
        if username == 'admin' and password == '1234':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials! Please try again.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


# --- MACHINE LEARNING CORE ---
def load_models():
    """Load existing models built externally by train_model.py"""
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(vectorizer_path, 'rb') as v_file:
            vectorizer = pickle.load(v_file)
        return model, vectorizer
    else:
        print("WARNING: Models not found! Please run 'python train_model.py' first.")
        return None, None

model, vectorizer = load_models()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_real_keywords(text_vec):
    """
    Extract actual top influencing words from the TF-IDF vector matrix.
    Works dynamically for models that use text vectorization maps.
    """
    if vectorizer is None: return []
    try:
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
            
        dense_array = text_vec.todense().tolist()[0]
        word_scores = [(feature_names[i], score) for i, score in enumerate(dense_array) if score > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return [w[0] for w in word_scores[:8]]
    except Exception:
        return []

def predict_single(text):
    """Core prediction logic wrapped for easy multi-use (Compare/URL)"""
    if not text.strip(): return None, 0, "", []
    p_text = preprocess_text(text)
    if not p_text: return None, 0, "", []
    
    if model is None or vectorizer is None: 
        return "Error", 0, "Wait! You need to run 'python train_model.py' to build the AI engine before predicting.", []
    
    try:
        vec = vectorizer.transform([p_text])
    except Exception as e:
        return "Error", 0, f"Vectorization failed: {e}", []
    
    pred_class = model.predict(vec)[0]
    
    # Strictly mapping 0 = Fake, 1 = Real ensuring prediction accuracy
    label = "Real" if pred_class == 1 else "Fake"
    
    conf = 0.0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vec)[0]
        
        class_list = list(model.classes_)
        fake_idx = class_list.index(0) if 0 in class_list else 0
        real_idx = class_list.index(1) if 1 in class_list else 1
        
        prob_fake = probs[fake_idx]
        prob_real = probs[real_idx]
        
        conf_target = prob_real if pred_class == 1 else prob_fake
        conf = round(conf_target * 100, 2)
        
    kw = extract_real_keywords(vec)
    exp = f"Top Influencing Words mathematically identified by TF-IDF: {', '.join(kw)}" if kw else "Prediction based on general sentence structure."
    
    return label, conf, exp, kw

def fetch_url_content(url):
    """REAL URL NEWS EXTRACTION: Use requests + BeautifulSoup"""
    if not url.startswith('http'):
        return "Error: Invalid URL Format. Must start with http/https.", False
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        paras = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paras])
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 50:
            return "Error: No significant article content found on this page.", False
            
        return text[:3000], True
        
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed. The website might be offline or deploying automated bot protection. Details: {e}", False
    except Exception as e:
        return f"Error: Failed to parse content. Details: {str(e)}", False


# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    # Authentication Lock
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    if 'history' not in session: session['history'] = []
    if 'analytics' not in session: session['analytics'] = {'Real': 0, 'Fake': 0}
    
    if model is None:
        return render_template('index.html', mode='single', error="CRITICAL: model.pkl not found! Please open your terminal and run 'python train_model.py' to initialize the Neural matrices.", history=[], analytics={'Real': 0, 'Fake': 0})
        
    return render_template('index.html', mode='single', history=session['history'], analytics=session['analytics'])

@app.route('/predict', methods=['POST'])
def predict():
    # Authentication Lock
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    if 'history' not in session: session['history'] = []
    if 'analytics' not in session: session['analytics'] = {'Real': 0, 'Fake': 0}
        
    mode = request.form.get('mode', 'single')
    analytics = session['analytics']
    history = session['history']
    
    if mode == 'single':
        news_text = request.form.get('news_text', '')
        if not news_text: 
            return render_template('index.html', error="Text cannot be empty.", mode=mode, history=history, analytics=analytics)
            
        label, conf, exp, kw = predict_single(news_text)
        
        if label and label != "Error":
            analytics[label] += 1
            history.insert(0, {'text': news_text[:50]+'...', 'prediction': label, 'confidence': conf})
        elif label == "Error":
            return render_template('index.html', error=exp, mode=mode, history=history, analytics=analytics)
            
        session['analytics'] = analytics
        session['history'] = history[:5]
        session.modified = True
        
        return render_template('index.html', 
                                mode=mode, text1=news_text,
                                result1={'label': label, 'conf': conf, 'exp': exp, 'kw': kw},
                                history=session['history'], analytics=session['analytics'])
                                
    elif mode == 'url':
        url = request.form.get('url_input', '')
        if not url:
             return render_template('index.html', error="URL cannot be empty.", mode=mode, history=history, analytics=analytics)
             
        extracted_text, success = fetch_url_content(url)
        
        if not success:
            return render_template('index.html', error=extracted_text, mode=mode, history=history, analytics=analytics)
            
        label, conf, exp, kw = predict_single(extracted_text)
        
        if label and label != "Error":
            analytics[label] += 1
            history.insert(0, {'text': f"(URL) {url[:40]}...", 'prediction': label, 'confidence': conf})
        elif label == "Error":
            return render_template('index.html', error=exp, mode=mode, history=history, analytics=analytics)
            
        session['analytics'] = analytics
        session['history'] = history[:5]
        session.modified = True
        
        return render_template('index.html', 
                                mode=mode, url=url, text_extracted=extracted_text, text1=extracted_text,
                                result1={'label': label, 'conf': conf, 'exp': exp, 'kw': kw},
                                history=session['history'], analytics=session['analytics'])

    elif mode == 'compare':
        text1 = request.form.get('compare_text_1', '')
        text2 = request.form.get('compare_text_2', '')
        
        if not text1 or not text2:
             return render_template('index.html', error="Both texts required for comparison.", mode=mode, history=history, analytics=analytics)
             
        label1, conf1, exp1, kw1 = predict_single(text1)
        label2, conf2, exp2, kw2 = predict_single(text2)
        
        if label1 == "Error" or label2 == "Error":
             return render_template('index.html', error=exp1, mode=mode, history=history, analytics=analytics)
        
        if label1 and label1 != "Error":
            analytics[label1] += 1
            history.insert(0, {'text': text1[:40]+'...', 'prediction': label1, 'confidence': conf1})
        if label2 and label2 != "Error":
            analytics[label2] += 1
            history.insert(0, {'text': text2[:40]+'...', 'prediction': label2, 'confidence': conf2})
            
        session['analytics'] = analytics
        session['history'] = history[:5]
        session.modified = True
        
        return render_template('index.html', 
                                mode=mode, text1=text1, text2=text2,
                                result1={'label': label1, 'conf': conf1, 'exp': exp1, 'kw': kw1},
                                result2={'label': label2, 'conf': conf2, 'exp': exp2, 'kw': kw2},
                                history=session['history'], analytics=session['analytics'])

if __name__ == '__main__':
    app.run(debug=True)
