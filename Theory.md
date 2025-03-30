# Project 1 - NLP Fundamentals Explorer

## Description
Develop a Streamlit application that applies and visualizes fundamental Natural Language Processing (NLP) preprocessing techniques on user-provided text.

## Topics Covered

### Core NLP concepts:
- **Tokenization**
- **Stemming**
- **Lemmatization**
- **Stopwords**
- **Part-of-Speech (POS) Tagging**
- **Named Entity Recognition (NER)**
- **Basic text representation ideas** (BoW, TF-IDF - conceptual)

## Why Preprocess Text?
1. **Reducing Noise** → Removing irrelevant information like punctuation and common words.
2. **Standardization** → Bringing words to a common base. Example: `running, ran, runs → run`.
3. **Feature Extraction** → Numerical conversion of words so they can be understood by an ML model.
4. **Improving Efficiency** → Smaller, standardized datasets are faster to process.

## Core NLP Concepts

### 1. Tokenization
Breaking down a stream of words into smaller units called tokens. These are usually words but can also be punctuation marks or sub-words depending on the tokenizer.

It is the fundamental first step as we need to work with individual words or symbols.

**Example:**
```plaintext
"Dr. Smith went to the U.S.A. today." → ['Dr.', 'Smith', 'went', 'to', 'the', 'U.S.A.', 'today', '.']
```
**Tool:** `nltk.word_tokenize()`

---

### 2. Stopwords
Words that don't carry significant meaning for analysis tasks like topic modeling or sentiment analysis.

Removing them reduces the size of the dataset and helps focus on important words.

**Example:** `[a, is, in, on, and, an]`

**Caution:** Stopwords are sometimes necessary for recognizing writing styles and other analyses, so removing them depends on the task.

**Tool:** `nltk.corpus.stopwords.words('english')` (provides a list of English stopwords)

---

### 3. Stemming
A crude process of chopping off prefixes or suffixes from a word to get its root form (stem). The resultant word may or may not make sense.

We do this to group multiple words together based on the stem.

**Example:**
```plaintext
"running", "runner", "runs" → "run"
"studies", "studying" → "studi" (not a dictionary word)
```
**Tools:**
1. `nltk.stem.PorterStemmer()` → More commonly used, less aggressive.
2. `nltk.stem.LancasterStemmer()` → Less commonly used, more aggressive.

---

### 4. Lemmatization
A more sophisticated way of reducing words to their base or dictionary form. It considers the context of the word before converting it to a lemma and checks its validity using a vocabulary like WordNet.

**Why Context Matters:**
- The word *meeting* could be a noun (e.g., *"The meeting was long."*) or a verb (e.g., *"He is meeting her later."*).
- Lemmatization, if aware of POS, would return "meeting" (noun) for the first case and "meet" (verb) for the second.

**Comparison to Stemming:**
- The purpose is the same as stemming (to group words together), but it produces actual words, making the output more interpretable.
- Preferred over stemming if computational costs are not a major issue.

**Example:**
```plaintext
"running" → "run", "studies" → "study", "better" → "good" (if context is provided)
```
**Tool:** `nltk.stem.WordNetLemmatizer()`

---

### 5. Part of Speech (POS) Tagging
Assigning a grammatical category like noun, verb, adjective, or adverb to a token in each sentence.

Crucial for removing ambiguity (e.g., *book* as a noun vs. *book* as a verb). It helps improve lemmatization and Named Entity Recognition (NER).

**Example:**
```plaintext
[('Dr.', 'NNP'), ('Smith', 'NNP'), ('went', 'VBD'), ('to', 'TO'), ('the', 'DT'), ('U.S.A.', 'NNP'), ('today', 'NN'), ('.', '.')]
```
**Tagging Legend:**
- **NNP** - Proper Noun
- **VBD** - Verb Past Tense
- **TO** - 'to'
- **DT** - Determiner
- **.** - Period

**Tool:** `nltk.pos_tag()` (requires tokenized input)

---

### 6. Named Entity Recognition (NER)
Identifies and categorizes named entities into predefined categories like *Person Names*, *Organizations*, *GPE (Geo-Political Entities)*, *Locations*, *Dates*, and *Monetary Values*.

Crucial for extracting key information and understanding the text’s meaning. Useful for information retrieval, question answering, and knowledge graph creation.

**Example:**
```plaintext
"[PERSON Dr. Smith] went to the [GPE U.S.A.] [DATE today]."
```
**Tool:** `nltk.ne_chunk()` (requires POS-tagged input and returns a tree structure)

---

### 7. Bag-of-Words (BoW) - *Conceptual*
A simple way to represent text numerically by ignoring grammar and word order, focusing only on word frequency in a document.

**Purpose:** Converts text into vectors that a machine learning model can understand.

**Example:**
```plaintext
Doc 1: "The cat sat."
Doc 2: "The dog sat."
Vocabulary: {"The", "cat", "sat", "dog"}
BoW Vector Doc 1: [1, 1, 1, 0] (assuming order: The, cat, sat, dog)
BoW Vector Doc 2: [1, 0, 1, 1]
```

---

### 8. Term Frequency-Inverse Document Frequency (TF-IDF) - *Conceptual*
An improvement over BoW that scores words based on how often they appear in a document (TF) but penalizes them based on their frequency across all documents (IDF).

**Purpose:**
- Gives higher importance to words that are frequent in one document but rare overall, making them more discriminative.
- Common words like *"the"* get low scores.

**Example:**
- In a collection of documents about pets, *"cat"* might have a high TF-IDF score in a document specifically about cats, while *"the"* would have a very low score across all documents.

---
