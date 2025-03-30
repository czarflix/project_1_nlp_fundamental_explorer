Here’s the formatted version of your text with clear headings and bullet points for better readability:  

---

### Why Preprocessing is Necessary  
- Raw text isn't directly usable by most algorithms.  
- Cleaning and structuring are essential before applying NLP techniques.  

### Core Techniques Covered  
- **Tokenization**: Splitting text into words or symbols (tokens).  
- **Stopword Removal**: Filtering out common words that add little value.  
- **Stemming & Lemmatization**: Reducing words to their root/base form.  
- **POS Tagging**: Assigning grammatical roles to words.  
- **Named Entity Recognition (NER)**: Identifying and categorizing named entities.  

### Stemming vs. Lemmatization  
- You can compare their outputs:  
  - **Stemming**: Cuts off word endings based on simple rules (may not always form real words).  
  - **Lemmatization**: Uses a dictionary-based approach to return meaningful base words.  

### NLTK Library Usage  
- You've worked with fundamental NLTK functions:  
  - `word_tokenize` for splitting text into words.  
  - `stopwords` for filtering common words.  
  - `PorterStemmer` & `WordNetLemmatizer` for word normalization.  
  - `pos_tag` for grammatical role identification.  
  - `ne_chunk` for extracting named entities.  

### Tool Interdependencies  
- NER often relies on POS tagging to identify meaningful entities accurately.  

### Conceptual Understanding  
- Basic idea of **Bag of Words (BoW)** and **TF-IDF**, which convert text into numerical vectors for machine learning.  

### Streamlit Basics  
- You’ve built a simple, interactive web application to demonstrate NLP concepts in action.  

---
