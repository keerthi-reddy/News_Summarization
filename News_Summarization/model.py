from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, dot, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def build_encoder_decoder(input_vocab_size, output_vocab_size, max_length_input, max_length_output):
    # Encoder
    encoder_inputs = Input(shape=(max_length_input,))
    encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=128)(encoder_inputs)
    encoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_length_output,))
    decoder_embedding = Embedding(input_dim=output_vocab_size, output_dim=128)(decoder_inputs)
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_outputs], axes=[2, 1])

    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, context])
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(model, x_train, y_train, x_test, y_test, checkpoint_path, batch_size=16, epochs=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')
    print("checkpoint done")
    history = model.fit(
        [x_train, y_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([x_test, y_test], y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

def evaluate_model(model, x_test, y_test, tokenizer, num_samples=5):
    rev_tok = {idx: word for word, idx in tokenizer.word_index.items()}

    x_test_subset = x_test[:num_samples]
    y_test_padded_subset = y_test[:num_samples]

    predictions = model.predict([x_test_subset, y_test_padded_subset], batch_size=16)
    predicted_tokens_np = np.argmax(predictions, axis=-1)

    predicted_summaries = []
    references = []

    for sample in predicted_tokens_np:
        predicted_summary = ' '.join([rev_tok.get(token, '<unknown>') for token in sample if token not in {0, tokenizer.word_index.get('start'), tokenizer.word_index.get('end')}])
        predicted_summaries.append(predicted_summary)

    for i in range(len(y_test_padded_subset)):
        reference_summary = ' '.join([rev_tok.get(token, '<unknown>') for token in y_test_padded_subset[i] if token not in {0, tokenizer.word_index.get('start'), tokenizer.word_index.get('end')}])
        references.append([reference_summary.split()])
    

    bleu_score = corpus_bleu(references, [pred.split() for pred in predicted_summaries])
    return predicted_summaries, references, bleu_score
