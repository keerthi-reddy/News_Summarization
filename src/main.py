#2669320
import os
from preprocess import preprocess_data, load_data, fit_tokenizer, save_tokenizer, load_tokenizer 
from model import build_encoder_decoder, train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
# Paths
train_path = "cnn_dailymail/train.csv"
test_path = "cnn_dailymail/test.csv"
validation_path = "cnn_dailymail/validation.csv"
tokenizer_path = "tokenizer_w/tokenizer_w50.pkl"
checkpoint_path = "checkpoints/encoder_decoder_model_w50.h5"

batch_size=16
epochs=1

# Load datasets
train_df, test_df, val_df = load_data(train_path, test_path, validation_path)

train_df = train_df.sample(frac=0.5, random_state=42)  # Use 50% of the training data
print("50% of data")
train_df.reset_index(drop=True, inplace=True)
print("train_df shape: ",train_df.shape)
print("test_df shape: ",test_df.shape)
                     
# Display basic information
print("Train Dataset:")
print(train_df.info())
print("\nTest Dataset:")
print(test_df.info())
print("\nValidation Dataset:")
print(val_df.info())

print("\nNull Values in Train Dataset:")
print(train_df.isnull().sum())
print("\nNull Values in Test Dataset:")
print(test_df.isnull().sum())
print("\nNull Values in Validation Dataset:")
print(val_df.isnull().sum())

# Check for duplicates in the 'id' field
print("\nDuplicate IDs in Train Dataset:", train_df['id'].duplicated().sum())
print("Duplicate IDs in Test Dataset:", test_df['id'].duplicated().sum())
print("Duplicate IDs in Validation Dataset:", val_df['id'].duplicated().sum())

print("Duplicates in Dataset: ",train_df.duplicated().sum())
train_df.dropna(inplace = True)

# Preprocess datasets
train_df = preprocess_data(train_df)
val_df = preprocess_data(val_df)

# Tokenizer
if not os.path.exists(tokenizer_path):
    tokenizer = fit_tokenizer(train_df['article'], train_df['highlights'])
    save_tokenizer(tokenizer, tokenizer_path)
else:
    tokenizer = load_tokenizer(tokenizer_path)
print("Len of tok.word_index",len(tokenizer.word_index))
print("count of tok.document_count",tokenizer.document_count)

# Tokenize
train_df['article'] = tokenizer.texts_to_sequences(train_df['article'])
train_df['highlights'] = tokenizer.texts_to_sequences(train_df['highlights'])
val_df['article'] = tokenizer.texts_to_sequences(val_df['article'])
val_df['highlights'] = tokenizer.texts_to_sequences(val_df['highlights'])

# Pad sequences
max_article_len = 400
max_summary_len = 50
x_train = pad_sequences(train_df['article'], maxlen=max_article_len, padding='post')
y_train = pad_sequences(train_df['highlights'], maxlen=max_summary_len, padding='post')
x_val = pad_sequences(val_df['article'], maxlen=max_article_len, padding='post')
y_val = pad_sequences(val_df['highlights'], maxlen=max_summary_len, padding='post')
print("Converted training articles and summaries to sequences")

# Build model
print("Model training started")
input_vocab_size = len(tokenizer.word_index) + 1
output_vocab_size = len(tokenizer.word_index) + 1
model = build_encoder_decoder(input_vocab_size, output_vocab_size, max_article_len, max_summary_len)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
if not os.path.exists(checkpoint_path):
    train_model(model, x_train, y_train, x_val, y_val, checkpoint_path, batch_size, epochs)
    print("Training completed")
else:
    model.load_weights(checkpoint_path)

# Evaluate model
predictions, references, bleu_score = evaluate_model(model, x_val, y_val, tokenizer,11490)
print("Sample Predictions:")
for pred, ref in zip(predictions[:5], references[:5]):
    print(f"Prediction: {pred}")
    print(f"Reference: {' '.join(ref[0])}\n")

print(f"BLEU Score: {bleu_score}")
