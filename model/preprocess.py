"""
preprocess.py
-------------
Text cleaning pipeline + TF-IDF vectorisation for EmotionSense.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


CONTRACTIONS = {
    "can't":"cannot","won't":"will not","n't":" not","i'm":"i am",
    "i've":"i have","i'll":"i will","i'd":"i would","it's":"it is",
    "that's":"that is","there's":"there is","they're":"they are",
    "we're":"we are","we've":"we have","you're":"you are","you've":"you have",
    "he's":"he is","she's":"she is","let's":"let us","don't":"do not",
    "doesn't":"does not","didn't":"did not","wasn't":"was not",
    "weren't":"were not","isn't":"is not","aren't":"are not",
    "haven't":"have not","hasn't":"has not","hadn't":"had not",
    "wouldn't":"would not","couldn't":"could not","shouldn't":"should not",
}

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "this","that","these","those","it","its","my","your","our","their",
    "his","her","we","they","you","he","she","me","him","us","them",
    "what","which","who","how","so","if","just","now","then","here",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r"http\S+|www\S+", "", text)      # URLs
    text = re.sub(r"@\w+|#\w+", "", text)           # mentions/hashtags
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


class TextVectorizer:
    """Wraps scikit-learn TF-IDF; outputs dense float32 arrays for the MLP."""

    def __init__(self, max_features=8000, ngram_range=(1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,   # log(1+tf)
            min_df=2,
            analyzer="word",
        )
        self.fitted = False

    def fit_transform(self, texts):
        cleaned = [clean_text(t) for t in texts]
        X = self.tfidf.fit_transform(cleaned).toarray().astype(np.float32)
        self.fitted = True
        return X

    def transform(self, texts):
        assert self.fitted, "Call fit_transform first."
        cleaned = [clean_text(t) for t in texts]
        return self.tfidf.transform(cleaned).toarray().astype(np.float32)

    @property
    def vocab_size(self):
        return len(self.tfidf.vocabulary_)
