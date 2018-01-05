# AISpamNonSpam

Python 3.6.3

**Run**
```
>pip install -r requirements.txt
>python NaiveBayes.py spambase.data
```

**Output**
* A chart showing the number of non-spam and spam emails for both the training data and testing data
* A chart where each row is a different grouping of the spambase data for training and testing
    * Column one is the name of data
    * Column two is the probability of false positives (the algorithm predicts an email is spam when it's actually non-spam)
    * Column three is the probability of false negatives (predicting an email is non-spam when it's actually spam)
    * Column three is the total probability of wrong predictions
