"""
This is the Hello World of Spark - doing a word count.
Here we use Les Miserables, the txt version available from Project Gutenberg
"""
# Importing SparkContext from the pyspark module
import time
from pyspark import SparkContext

s = time.time()
# Defining a SparkContext locally
with (SparkContext.getOrCreate() as sc):
    miserables = sc.textFile('les_miserables.txt')
    print(miserables.take(10))

    # SHAPING
    # Create an RDD, called miserables_clean, which contains the text of the Les Misérables in lowercase and without punctuation using the methods map, lower and replace
    miserables_clean = miserables.map(lambda x: x.lower().replace(',',' '))

    # Create an RDD, called miserables_flat, including all words in a single dimension
    miserables_flat = miserables_clean.flatMap(lambda x: x.split(' '))
    print('Miserables flat top 10', miserables_flat.take(10))

    # MAP / REDUCE
    # From miserables_flat, create an RDD words containing all the couples (word, nb_occurences) using the methods map and reduceByKey
    words = miserables_flat.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
    words.sortBy(lambda x: x[1],ascending=False).take(10)

    # Create a list ordered in increasing order containing the couples: word, occurrence.

    words_sorted = words.sortBy(lambda x: x[1], ascending=False).collect()
    words_sorted[:10]
    print('Top 10 words in Les Miserables by number occurrences', words_sorted[:10])

    # We have some blank strings, should exclude

    # The above in a single command
    words_sorted_3 = miserables.flatMap(lambda x: x.split(' ')).map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],ascending=False).collect()
    print(words_sorted_3[:10])

e = time.time()
print(f'Done in {e-s:.2f}s')