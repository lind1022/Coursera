# Logistic Regression
## Vocabulary & Feature Extractions

For example, a tweet "I am happy because I am learning NLP. I hated the movie."

First create a vector V of unique words in the tweet, where V corresponds to the vocabulary size. For the tweet "I am happy because I am learning NLP", then put a 1 in the corresponding index for any word in the tweet, and a 0 otherwise.
`V=['I', 'am', 'happy', 'because', 'learning', 'NLP', 'hated', 'the', 'movie']`
Each word only appears once in the vector.

Feature extraction for tweet "I am happy because I am learning NLP". The number of features will equal to the size of entire vocabulary. **This will result to long training time and long prediction time.**
```
I -> 1
am -> 1
happy -> 1
because -> 1
learning -> 1
NLP -> 1
hated -> 0
the -> 0
movie -> 0
```

## Use word count as features in the logistic regression.
Given a word, keep track number of times the word appears in the positive/negative class.

Positive tweets:
- I am happy because I am learning NLP
- I am happy
Vocabulary for positive tweets will be:
```
I - 3
am - 3
happy - 2
because - 1
learning - 1
NLP - 1
sad - 0
not - 0
```

Negative tweets:
- I am sad, I am not learning NLP
- I am sad
Vecabulary for negative tweets will be:
```
I - 3
am - 2
happy - 0
because - 0
learning - 1
NLP - 1
sad - 2
not - 1
```
<img src="pics/voc_freq.png" width="500" height='300'>

## Feature Extraction with frequencies
Freqs: dictionary mapping from (word, class) to frequency.
<img src="pics/feat_extract.png">

Only 3 features, bias, sum of positive frequencies, sum of negative frequencies of the words from the vocabulary that appear on the tweet.

To extract the 3 features from the sentence: I am sad, I am not learning NLP.
Sum of positive features: 3(I) + 3(am) + 1(learning) + 1(NLP) = 8
Sum of negative features: 3(I) + 3(am) + 2(sad) + 1(not) + 1(learning) + 1(NLP) = 11
`X_m = [1, 8, 11]`

## Preprocessing
When preprocessing, do the following:
1. Eliminate handles (@abc) and URLs
2. Tokenise string into words
3. Remove stop words like "and, is, a, on etc."
4. Stemming or convert every word to its stem. Like dancer, dancing, danced, becomes 'danc'. Use porter stemmer to take care of this.
5. Convert all words to lower case.

Using **stemming** and **stop words**.

### Stop words and punctuation marks.
Remove them because they don't add meaning.
In practice, we compare tweets against 2 lists, "stop words" and "punctuation". Every word in the tweets also appears in the list should be eliminated.
Example: (in reality they are much larger)
- Stop words: "and", "is", "are", "at", "has", "for", "a"
- Punctuation: ",", ".", ":", "!"
However think about if punctuation add specific meaning to the NLP task

### Stemming
Stemming in NLP transforms any word to its base stem， the set of characters used to construct the word and its derivatives. "Tuning" for example， its stem is tun.






# end of script
