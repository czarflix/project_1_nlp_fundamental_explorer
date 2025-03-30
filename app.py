import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

# --- NLTK Data Download ---
# Streamlit message to inform the user about downloading necessary resources
st.info("Checking and downloading required NLTK resources...")


# Function to safely download an NLTK resource if it's missing
def download_nltk_resource(resource_name):
    """
    Checks if an NLTK resource exists; if not, it downloads the resource.
    :param resource_name: Name of the NLTK resource to check/download.
    :return: True if the resource is available or successfully downloaded, False otherwise.
    """
    try:
        nltk.data.find(f"{resource_name}")  # Check if resource exists
        return True
    except LookupError:
        st.info(f"Downloading NLTK resource: {resource_name}")
        try:
            nltk.download(resource_name, quiet=True)
            return True
        except Exception as e:
            st.error(f"Error downloading {resource_name}: {str(e)}")
            return False


# List of required resources for different NLP operations
resources = [
    "punkt",  # Tokenization
    "stopwords",  # Stopword removal
    "wordnet",  # Lemmatization
    'averaged_perceptron_tagger',  # POS tagging
    "maxent_ne_chunker",  # Named Entity Recognition
    "words"  # Required by NER chunker
]

# Download each required resource, ensuring they are available
all_success = True
for resource in resources:
    if not download_nltk_resource(resource):
        all_success = False  # Flag error if any download fails

# Display download status
if all_success:
    st.success("All NLTK resources successfully checked/downloaded")
else:
    st.warning("Some NLTK resources may not have downloaded correctly")

# --- Initializing NLP Tools ---
porter = PorterStemmer()  # Stemming tool
lemmatizer = WordNetLemmatizer()  # Lemmatization tool
stop_words = set(stopwords.words("english"))  # Stopword list

# --- Streamlit UI Setup ---
st.title("NLP Fundamentals Explorer")
st.write("""
Enter some text below to see fundamental NLP preprocessing techniques applied using NLTK.
This demonstrates how raw text is transformed before being used in more complex models.
""")

# --- User Input Section ---
user_text = st.text_area("Enter Text Here:",
                         "Dr. Smith went to the U.S.A. on Friday to buy some delicious apples for $5.50.")

# --- NLP Processing Logic ---
if user_text:  # Only process if input text is provided
    st.divider()  # UI separator for better readability

    # --- 1. Display Original Text ---
    st.subheader("1. Original Text")
    st.write(user_text)
    st.divider()

    # --- 2. Tokenization ---
    st.subheader("2. Tokenization")
    st.write("Breaking text into individual words or symbols (tokens).")

    try:
        tokens = word_tokenize(user_text)  # Apply tokenization
        st.write(tokens)
    except Exception as e:
        st.error(f"Error in tokenization: {str(e)}")
        # If tokenization fails, use a simple fallback tokenizer
        st.warning("Using fallback tokenizer (basic split by space).")
        tokens = user_text.split()
        st.write(tokens)

    st.divider()

    # --- 3. Stopword Removal ---
    st.subheader("3. Stopword Removal")
    st.write("Removing common words that don't add much value to the meaning of the text (e.g., 'the', 'is', 'and').")

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    st.write(filtered_tokens)
    st.divider()

    # --- 4. Stemming ---
    st.subheader("4. Stemming")
    st.write("""Stemming reduces words to their root form by removing suffixes 
             (e.g., 'running' -> 'run'). It can sometimes create non-dictionary words.""")

    stemmed_tokens = [porter.stem(tkn) for tkn in filtered_tokens]
    st.write(stemmed_tokens)
    st.divider()

    # --- 5. Lemmatization ---
    st.subheader("5. Lemmatization")
    st.write("""Lemmatization converts words to their base (dictionary) form, considering meaning 
             (e.g., 'better' -> 'good', 'running' -> 'run'). It is more accurate than stemming.""")

    lemmatized_tokens = [lemmatizer.lemmatize(tkn) for tkn in filtered_tokens]
    st.write(lemmatized_tokens)
    st.divider()

    # --- 6. Part of Speech (POS) Tagging ---
    st.subheader("6. Part of Speech (POS) Tagging")
    st.write("""Assigns grammatical roles to words (noun, verb, adjective, etc.).
                Uses the original tokenized text.""")

    try:
        pos_tagged_tokens = pos_tag(tokens)  # Apply POS tagging
        st.write(pos_tagged_tokens)

        # Explanation of common POS tags
        st.info("""Common POS Tags:
        - **NNP**: Proper Noun (Singular)
        - **NN**: Noun (Singular)
        - **VBD**: Verb (Past Tense)
        - **VBG**: Verb (Gerund/Present Participle)
        - **JJ**: Adjective
        - **IN**: Preposition/Subordinating Conjunction
        - **DT**: Determiner
        - **.**: Punctuation""")

    except Exception as e:
        st.error(f"Error in POS tagging: {str(e)}")
        # Fallback: Assign "UNK" (unknown) POS tag
        pos_tagged_tokens = [(token, "UNK") for token in tokens]
        st.write(pos_tagged_tokens)

    st.divider()

    # --- 7. Named Entity Recognition (NER) ---
    st.subheader("7. Named Entity Recognition (NER)")
    st.write("""Identifies and categorizes named entities (e.g., Person, Organization, Location).
             Requires POS-tagged tokens.""")

    try:
        # Apply NER chunking on POS-tagged tokens
        ner_tree = ne_chunk(pos_tagged_tokens)

        st.write("NER Tree Structure (NLTK's output):")
        st.text(ner_tree)  # Show tree structure as text output

        # Extract and display named entities from the tree
        named_entities = []
        for subtree in ner_tree:
            if hasattr(subtree, "label"):  # Check if it's a named entity subtree
                entity = " ".join([token for token, pos in subtree.leaves()])
                label = subtree.label()
                named_entities.append((entity, label))

        # Display results
        if named_entities:
            st.write("Detected Named Entities:")
            st.write(named_entities)
        else:
            st.write("No named entities detected.")

    except Exception as e:
        st.error(f"Error in Named Entity Recognition: {str(e)}")

    st.divider()
