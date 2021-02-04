# Sentiment Analysis


# Introduction
I was interesting to know the statistics of the books reviews reflecting the emotion with the starts. I analyzed the data of the collection of "Best Mystery & Thriller" from the Goodreads Choice Awards 2019 by the following steps: scraping each book reviews, investigating if these reviews corresponding with the stars and concluding my findings.


# Conclusion

The accuracy score is not very high, so I consider that they are either not very good models or they are not best fit in my data. As the examples shown below, some words like "not", "sure", and "better", "didn't", "like" may not be found together, which may cause mistake to compare with the scores.


df.iloc[[3751]]
df.at[3808,'comment']

df.at[3808,'comment']
df.iloc[[3738]]

df.at[3795,'comment']

The methods used in here is bag-of-words. It is simple to understand and implement, but it leads to a high dimensional feature vector due to large size of Vocabulary. To impove the score, we can use word embedding for text.

For further researchings, I would do more data collection to improve the accuracy of the models.









