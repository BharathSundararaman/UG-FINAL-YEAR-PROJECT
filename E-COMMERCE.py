# --------------------------------------
# ALL MODELS: BERT, RF, CNN, Word2Vec, SVM
# --------------------------------------

# ✅ Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from gensim.models import Word2Vec
from google.colab import drive

# ✅ Mount Google Drive
drive.mount('/content/drive')

# ✅ Load dataset
dataset_path = '/content/drive/My Drive/info_acc.csv'  # Update path if needed
df = pd.read_csv(dataset_path)
print("Dataset Loaded. Shape:", df.shape)

# ✅ Helper function: Evaluation
def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} Results:")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ✅ Prepare Data
features = ['Price', 'DescriptionLength', 'InStock', 'ProductRating']
target = 'Accuracy'
text_field = 'ProductName'  # For BERT and Word2Vec

X = df[features]
y = df[target]

# Scale numeric features for NN & SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# --------------------------------------
# 🌟 1. RANDOM FOREST
# --------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))

# Feature importance
print("\n🌿 Random Forest Feature Importance:")
for f, imp in zip(features, rf.feature_importances_):
    print(f"{f}: {imp:.4f}")

# --------------------------------------
# 🌟 2. CONVOLUTIONAL NEURAL NETWORK (CNN)
# --------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models

# Reshape tabular data for CNN input: (samples, features, 1)
X_train_cnn = X_scaled[:len(y_train)].reshape(-1, X_train.shape[1], 1)
X_test_cnn = X_scaled[len(y_train):].reshape(-1, X_train.shape[1], 1)

# Build CNN model
model_cnn = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)  # Output layer for regression
])

# Compile the CNN model
model_cnn.compile(optimizer='adam', loss='mean_squared_error')

# Train the CNN model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_cnn.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test),
              epochs=100, batch_size=32, callbacks=[callback], verbose=0)

# Predict and evaluate CNN
y_pred_cnn = model_cnn.predict(X_test_cnn).flatten()
results.append(evaluate_model(y_test, y_pred_cnn, "CNN"))

# --------------------------------------
# 🌟 3. WORD2VEC + LINEAR REGRESSION
# --------------------------------------
sentences = [str(text).split() for text in df[text_field]]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
def vectorize(text):
    tokens = text.split()
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X_w2v = np.vstack(df[text_field].apply(vectorize))
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train_w2v, y_train_w2v)
y_pred_w2v = lr.predict(X_test_w2v)
results.append(evaluate_model(y_test_w2v, y_pred_w2v, "Word2Vec + Linear Regression"))

# --------------------------------------
# 🌟 4. SVM REGRESSOR
# --------------------------------------
svm = SVR(kernel='rbf')
svm.fit(X_scaled[:len(y_train)], y_train)
y_pred_svm = svm.predict(X_scaled[len(y_train):])
results.append(evaluate_model(y_test, y_pred_svm, "SVM"))

# --------------------------------------
# 🌟 5. BERT REGRESSOR
# --------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(df[text_field].astype(str).tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
input_ids, attention_masks = encoded['input_ids'], encoded['attention_mask']
y_tensor = torch.tensor(y.values, dtype=torch.float32)

train_size = int(0.8 * len(y))
train_dataset = TensorDataset(input_ids[:train_size], attention_masks[:train_size], y_tensor[:train_size])
test_dataset = TensorDataset(input_ids[train_size:], attention_masks[train_size:], y_tensor[train_size:])

model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, problem_type="regression")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert.to(device)
optimizer = AdamW(model_bert.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset))
loss_fn = nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
model_bert.train()
for epoch in range(3):  # Fewer epochs for demo; increase for better results
    for batch in train_loader:
        optimizer.zero_grad()
        ids, masks, labels = [b.to(device) for b in batch]
        outputs = model_bert(input_ids=ids, attention_mask=masks)
        logits = outputs.logits.view(-1)
        loss = loss_fn(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluate BERT
model_bert.eval()
test_loader = DataLoader(test_dataset, batch_size=16)
y_preds_bert, y_true_bert = [], []
with torch.no_grad():
    for batch in test_loader:
        ids, masks, labels = [b.to(device) for b in batch]
        outputs = model_bert(input_ids=ids, attention_mask=masks)
        y_preds_bert.extend(outputs.logits.view(-1).cpu().numpy())
        y_true_bert.extend(labels.cpu().numpy())
results.append(evaluate_model(y_true_bert, y_preds_bert, "BERT"))

# --------------------------------------
# 📋 Summary Table
# --------------------------------------
results_df = pd.DataFrame(results)
print("\n🔽 Model Comparison:")
print(results_df.sort_values(by="RMSE"))
