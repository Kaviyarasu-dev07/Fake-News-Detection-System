import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 1. ⚖️ LOAD AND EXPLICITLY BALANCE DATASET
def prepare_dataset():
    file_path = "dataset.csv"
    if os.path.exists(file_path):
        print(f"Loading external dataset from {file_path}...")
        df = pd.read_csv(file_path)
        # Map labels if needed assuming a general set. For now assume columns are 'text' and 'label'
    else:
        print("No external dataset found. Generating explicitly mapped massive synthetic dataset...")
        # Repetitive dataset mapping ensures we safely pass min_df=2 requirements
        real_texts = [
            "the stock market index closed higher today due to strong economy.",
            "scientists report that the economy is growing and stocks are higher.",
            "strong economy factors influence the stock market positively.",
            "the president announced a new health policy based on scientists research.",
            "health policy reform announced by the president today.",
            "scientists published new data revealing stock market trends.",
            "the economy is stable according to the new health policy data.",
            "nasa confirms new space mission data released today.",
            "space mission data shows higher stock market value.",
            "the president validates the nasa space mission policy."
        ] * 10  # Multiplied out to simulate volume

        fake_texts = [
            "aliens landed and gave me a miracle cure for all diseases.",
            "miracle cure discovered by simple trick aliens landed.",
            "shocking fake news aliens predict the future.",
            "shocking trick to lose weight simple miracle cure.",
            "fake news media hiding the truth about aliens.",
            "the earth is flat and scientists are hiding the truth shocking.",
            "flat earth trick discovered by local man fake news.",
            "the president is actually aliens hiding the truth.",
            "miracle cure for flat earth believers shocking trick.",
            "aliens landed fake news shocking truth."
        ] * 10
        
        data = {
            'text': real_texts + fake_texts,
            'label': [1] * len(real_texts) + [0] * len(fake_texts)
        }
        df = pd.DataFrame(data)

    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # 🎯 FIX: EQUAL SAMPLING OF DATASET (Addresses Bias)
    min_class_size = df['label'].value_counts().min()
    df_real = df[df['label'] == 1].sample(n=min_class_size, random_state=42)
    df_fake = df[df['label'] == 0].sample(n=min_class_size, random_state=42)
    df_balanced = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n--- DATASET BALANCED ---")
    print(f"Class distribution mapping (0=Fake, 1=Real):\n{df_balanced['label'].value_counts()}")
    return df_balanced['clean_text'], df_balanced['label']

X, y = prepare_dataset()

# 2. 🧠 IMPROVED TEXT VECTORIZATION
print("\n--- VECTORIZING TEXT ---")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), # Capture both words and paired phrases (bigrams)
    min_df=2,           # Ignore words that appear entirely standalone
    max_df=0.7,         # Cut words dominating more than 70% of dataset
    max_features=5000,  # Cap token memory
    stop_words='english'
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42, stratify=y)


# 3. 🤖 MODEL TRAINING & COMPARISON
print("\n--- TRAINING MULTIPLE MODELS ---")
# Logistic regression mapping constraints: max_iter=1000, class_weight='balanced'
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
nb = MultinomialNB()

log_reg.fit(X_train, y_train)
nb.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_pred_nb = nb.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_nb = accuracy_score(y_test, y_pred_nb)

print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(f"Multinomial Naive Bayes Accuracy: {acc_nb:.4f}")

if acc_nb > acc_lr:
    print("=> Multinomial Naive Bayes performed better. Adopting NB as optimal model.")
    best_model = nb
    y_pred_best = y_pred_nb
else:
    print("=> Logistic Regression holds strong. Adopting LR as optimal model.")
    best_model = log_reg
    y_pred_best = y_pred_lr


# 4. 📊 PROPER EVALUATION
print("\n========== MODEL EVALUATION ==========")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report (0=Fake, 1=Real):")
print(classification_report(y_test, y_pred_best, target_names=["Fake News (0)", "Real News (1)"]))


# 5 & 6. 🧪 ADD TESTING SAMPLES AND FIX LABEL MAPPING
print("\n========== TESTING SANITY SAMPLES ==========")
test_samples = [
    "nasa launches new observation satellite into orbit successfully", # Real
    "shocking trick to get free money aliens landed overnight miracle" # Fake
]
test_vec = vectorizer.transform([preprocess_text(t) for t in test_samples])
test_preds = best_model.predict(test_vec)
test_probs = best_model.predict_proba(test_vec)

# Explicitly grabbing correct numeric index placements
class_list = list(best_model.classes_)
fake_idx = class_list.index(0)
real_idx = class_list.index(1)

for i, text in enumerate(test_samples):
    label = "Real News" if test_preds[i] == 1 else "Fake News"
    prob_fake = test_probs[i][fake_idx] * 100
    prob_real = test_probs[i][real_idx] * 100
    print(f"Text Submitted: '{text}'")
    print(f"Predicted Output Index: {test_preds[i]} => {label}")
    print(f"Probability - Fake: {prob_fake:.2f}% | Real: {prob_real:.2f}%\n")


# 7. SAVE ARTIFACTS
print("\nSaving robust algorithm matrices...")
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
    
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    
print("✅ Success! Generated rigidly balanced model.pkl and vectorizer.pkl.")
