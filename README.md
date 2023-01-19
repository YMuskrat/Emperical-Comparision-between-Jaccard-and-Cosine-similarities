# Emperical-Comparision-between-Jaccard-and-Cosine-similarities
## Overview
This script is used to perform text normalization and similarity calculation on a set of documents from the Reuters corpus. It contains functions for calculating the similarity between two documents represented as bags of words. The package includes implementations of Jaccard similarity and Cosine similarity measures, as well as functions to compute all-pairs similarities for a collection of documents.
## Objectives
The overall objective of this code is to analyze the performance of Jaccard similarity and Cosine similarity algorithms with different representations of the document using the Reuters corpus. The script aims to compare the running time of the algorithms and identify the most efficient one.

## Dependencies
- NLTK
- Pandas
- Matplotlib
- Numpy
- Scipy
- Timeit
- Multiprocessing
## Execution
To run the script, you will need to have the dependencies installed and download the 'punkt' and 'stopwords' resource from nltk by running the following commands:
```ruby
nltk.download('punkt')
nltk.download('stopwords')
```
After that, you can execute the script using your preferred method.

## Jaccard Similarity
Jaccard similarity is a measure of similarity between two sets, in this case, the sets of words in two documents. The Jaccard similarity is calculated as the size of the intersection of the two sets divided by the size of the union of the two sets.

In this package, documents are represented as Python dictionaries where the keys are the words in the document and the values are the frequencies of those words. The Jaccard similarity between two documents is calculated by first creating sets of the words in each document, then finding the intersection and union of the sets, and finally dividing the size of the intersection by the size of the union.

The theoretical worst-case running time of the Jaccard similarity measure applied to large documents represented as bags of words is O(n) where n is the number of unique words in the two sets being compared. This is because in order to calculate the intersection and union, we need to iterate through both sets and check for the presence of each word.
```ruby
def jaccard_similarity(doc1: dict, doc2: dict) -> float:
```

## Cosine Similarity
Cosine similarity is a measure of similarity between two non-zero vectors, in this case, the vectors representing the words in two documents. The Cosine similarity is calculated as the dot product of the two vectors divided by the product of the magnitudes of the vectors.

In this package, documents are represented as dense vectors, where there is a dimension for each of the V words in the English vocabulary. The cosine similarity between two documents is calculated by first computing the dot product of the two vectors and then dividing by the product of the magnitudes of the vectors.

The theoretical worst-case running time of the cosine similarity measure applied to large documents represented as dense vectors is O(d) where d is the number of dimensions in the vectors. This is because in order to calculate the dot product, we need to iterate through each dimension and multiply the corresponding values.

```ruby
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
```
## Sparse Cosine Similarity
A function that computes cosine similarity directly from sparse (dictionary) representations without converting them into dense (vector) representations is also provided. This function first creates a set of the words in each document and then iterates through both sets, counting the number of common words and the number of unique words. The cosine similarity is then calculated as the number of common words divided by the number of unique words.

The theoretical worst-case running time of this function is O(n^2) where n is the number of unique words in the two sets being compared. This is because we would need to iterate through both sets and check for the presence of each word, similar to the Jaccard similarity measure.
```ruby
def sparse_cosine_similarity(doc1: dict, doc2: dict) -> float:
```
## All-pairs Similarities
A function that computes all-pairs similarities for a collection of documents is provided. The function takes a list of dictionaries (the document collection) and a parameter specifying the similarity measure to be used (either 'jaccard' or 'cosine').

The theoretical worst-case running time for computing all-pairs similarities for a collection of documents is O(d^2) where d is the number of documents in the collection. This is because we would need to compare each document with every other document in the collection. The specific similarity measure used does not affect the running time.

```ruby
def all_pairs_similarities(docs: List[dict], measure: str) -> Dict[Tuple[int, int], float]:
```

## Parallel Computing
A function that implements all-pairs similarities for documents using some form of parallel computing, such as MapReduce, is also provided. This function allows to perform the computation of all-pairs similarities in parallel, reducing the time it takes to compute the similarities.

The number of parallel processes that gives optimal results for the implementation and computer should be investigated. It can be done by testing the function empirically for correctness and efficiency, and adjusting the number of processes accordingly.

```ruby
def parallel_all_pairs_similarities(docs: List[dict], measure: str, num_processes: int) -> Dict[Tuple[int, int], float]:
```

## Results
The script compares the performance of Jaccard similarity and Cosine similarity algorithms on the Reuters corpus. The results show that Jaccard similarity is better than Cosine similarity in general, due to the fact that it takes time of O(1) when searching for certain keys in the dictionary. The use of the dictionary in the cosine similarity resulted in similar results to the Jaccard algorithm.

For the cosine similarity, it was found that the dot product with numpy is faster than other representations, due to the fact that numpy is an optimized library for matrix multiplication. In terms of lists, the use of the get method is faster than iterating through the dictionary.

When calculating the all-pair similarity, the choice of the algorithm affected the average running time, but still the Jaccard algorithm outperformed the Cosine similarity, as expected even with small differences.

Finally, to reduce the time, the script used multiprocessing with the map-reduce technique to divide the work to multiple processes and let them work in parallel, which yielded faster results.

## Limitations
This script is designed to work with the Reuters corpus, and any modifications may be necessary to work with other corpora or text sources. Additionally, the script assumes that the user has access to the Reuters corpus, which requires a separate download. If running on google colab, keep in mind that, Google Colab does not support multiprocessing by default, so if the script is run on Google Colab, the multiprocessing functionality will not work and will need to be commented out or adapted to use the built-in Colab tools for parallel processing.
## Additional Information
The script includes the following functions:
- normalize(token_list) : This function takes in a list of tokens and returns the list after converting all tokens to lowercase and removing stopwords.
- filter_stopwords(token_list) : This function takes in a list of tokens and returns the list after removing stopwords.
- timeit(somefunc, *args, repeats=10, **kwargs) : This function takes in a function, arguments, and number of repeats and returns the mean execution time of the function over the number of repeats.
- Basic_Line_plot(xs, ys, label) : This function takes in a list of x-values, y-values and label and returns a line plot of x-values and y-values.
- log_plot(log_x, log_y, label) : This function takes in a list of log_x-values, log_y-values and label and returns a log-log plot of log_x-values and log_y-values. It also performs linear regression on the log-transformed data to analyze the trend of the algorithm's running time over the size of the input.
## Conclusion
In conclusion, the package provides a set of functions for measuring the similarity between documents represented as bags of words. The Jaccard similarity measure has a theoretical worst-case running time of O(n) where n is the number of unique words in the two sets being compared. On the other hand, The Cosine similarity measure has a theoretical worst-case running time of O(d) where d is the number of dimensions in the vectors. Additionally, the package includes a function for computing all-pairs similarities for a collection of documents, which has a theoretical worst-case running time of O(d^2) where d is the number of documents in the collection.

The package also offers a function for computing cosine similarity directly from sparse (dictionary) representations and a function that implements all-pairs similarities for documents using parallel computing. These functions can be useful when working with large sparse datasets. The running times of these functions will vary depending on the specific implementation and the computer being used. Therefore, it is important to test the functions empirically and estimate the constants for the running time for the specific implementation and computer.

Overall, this script provides a comprehensive analysis of the performance of Jaccard similarity and Cosine similarity algorithms on the Reuters corpus. The results show that Jaccard similarity is a more efficient algorithm than Cosine similarity. The script also includes functions to time the execution of any function, perform linear regression, and plot the running time over the size of the input, making it useful for analyzing the performance of other algorithms or text processing tasks.
