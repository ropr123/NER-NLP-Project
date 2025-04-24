import streamlit as st
from transformers import BertTokenizerFast
from nltk.corpus import words, wordnet
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Layer
from tensorflow.keras.optimizers import Adam

# --- Your Custom CRF Layer ---
class CRF(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trans = self.add_weight(
            name="trans",
            shape=(self.output_dim, self.output_dim),
            initializer="uniform",
            trainable=True
        )
        super(CRF, self).build(input_shape)

    def call(self, logits):
        return logits  # Keep logits; decode externally if needed

    def loss_fn(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def accuracy_fn(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

# --- Model Reconstruction Function ---
def build_bilstm_crf_model(vocab_size, tag_size, embedding_matrix, max_len=900, embedding_dim=100):
    input_layer = Input(shape=(max_len,))
    model = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )(input_layer)
    model = Bidirectional(
        LSTM(units=64, return_sequences=True)
    )(model)
    logits = TimeDistributed(
        Dense(tag_size)
    )(model)
    crf = CRF(tag_size)
    output = crf(logits)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(0.001), loss=crf.loss_fn, metrics=[crf.accuracy_fn])
    return model

# --- Load Saved Components ---
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)

idx2tag = {i: tag for tag, i in tag2idx.items()}

embedding_matrix = np.load("embedding_matrix.npy")

# --- Rebuild and Load Model ---
model = build_bilstm_crf_model(
    vocab_size=len(word2idx),
    tag_size=len(tag2idx),
    embedding_matrix=embedding_matrix,
    max_len=900,
    embedding_dim=100
)

model.load_weights("model_weights.h5")

print("✅ Model loaded with weights and ready for inference.")


nltk_words = set(words.words())
# Extract additional words from WordNet
wordnet_words = set(lemma.name().replace('_', ' ') for synset in wordnet.all_synsets() for lemma in synset.lemmas())
# Combine both word sets
english_vocab = nltk_words.union(wordnet_words)
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
# Identify missing words
missing_words = [word for word in english_vocab if tokenizer.tokenize(word) == ['[UNK]']]
print(f"Total Missing Words: {len(missing_words)}")


def combine_subwords(tokens):
    """
    Combines subword tokens marked with '##' into complete words.
    
    Args:
    tokens (list): List of tokenized words from a tokenizer.

    Returns:
    list: List of cleaned words without subword fragmentation.
    """
    combined_tokens = []
    current_word = ""

    for token in tokens:
        if token.startswith("##"):
            # Append to the previous word (removing the '##')
            current_word += token[2:]
        else:
            # Add the previous complete word to the list if it exists
            if current_word:
                combined_tokens.append(current_word)
            # Start a new word
            current_word = token
        
    # Append the last word
    if current_word:
        combined_tokens.append(current_word)

    return combined_tokens

def predict_tags_from_tokens(tokens, model, word2idx, idx2tag, max_len=900):
    """
    Predicts NER tags for a list of tokens using the trained model.

    Args:
        tokens (list): List of word tokens (e.g., ["worked", "at", "Google"])
        model: Trained BiLSTM-CRF model
        word2idx (dict): Word → index mapping
        idx2tag (dict): Tag index → tag label
        max_len (int): Max sequence length used during training

    Returns:
        list: List of (token, predicted_tag) tuples
    """
    # Step 1: Convert tokens to indices
    token_ids = [word2idx.get(token, word2idx["UNK"]) for token in tokens]
    
    # Step 2: Pad sequence
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_input = pad_sequences([token_ids], maxlen=max_len, padding='post')

    # Step 3: Predict
    logits = model.predict(padded_input)
    pred_ids = np.argmax(logits, axis=-1)[0][:len(tokens)]

    # Step 4: Convert to tags
    pred_tags = [idx2tag[i] for i in pred_ids]
    return list(zip(tokens, pred_tags))


# Dummy NER prediction function — replace with your actual model
def predict_ner(text):
    tokens = combine_subwords(tokenizer.tokenize(text))
    return predict_tags_from_tokens(tokens, model, word2idx, idx2tag)




# Label mappings (entity code → readable label)
entity_display = {
    "EDU": "Education",
    "HSK": "Hard Skill",
    "YOE": "Years of Experience",
    "ORG": "Organization",
    "JOB": "Job Title",
    "LOC": "Location",
    "O": "Other"
}

# Entity type to color (shared for both B- and I- tags)
entity_colors = {
    "EDU": "#feca57",   # Education
    "HSK": "#1dd1a1",   # Hard Skill
    "YOE": "#ff6b6b",   # Years of Experience
    "ORG": "#54a0ff",   # Organization
    "JOB": "#5f27cd",   # Job Title
    "LOC": "#ff9ff3",   # Location
    "O": None           # No color for non-entities
}


st.title("NER Annotation Viewer")

# Session state setup
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = []

# Submit callback
def handle_submit():
    text = st.session_state.user_input
    if text.strip():
        st.session_state.last_prediction = predict_ner(text)
        st.session_state.user_input = ""  # Clear the input

# Input text area + button
st.text_area("Enter text to annotate:", key="user_input", height=150)
st.button("Submit", on_click=handle_submit)

# Fetch prediction if available
token_label_pairs = st.session_state.last_prediction

if token_label_pairs:
    # Filter selector
    entity_types = list(entity_display.keys())
    selected_entity = st.radio(
        "Filter by Entity Type:",
        ["All"] + [entity_display[e] for e in entity_types if e != "O"],
        horizontal=True
    )

    def entity_matches(label, selected):
        if selected == "All":
            return True
        for code, name in entity_display.items():
            if name == selected and label.endswith(code):
                return True
        return False

    # Generate HTML with shared color per entity
    html = ""
    for token, label in token_label_pairs:
        entity_type = label.split("-")[-1] if "-" in label else "O"
        color = entity_colors.get(entity_type)
        if entity_matches(label, selected_entity) and color:
            html += f'<span style="background-color:{color}; padding:2px; border-radius:4px; margin-right:2px">{token}</span> '
        else:
            html += f"{token} "

    st.markdown("### Annotated Output:")
    st.markdown(f"<div style='line-height: 2.0; font-size: 18px'>{html}</div>", unsafe_allow_html=True)

