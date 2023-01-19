# Emperical-Comparision-between-Jaccard-and-Cosine-similarities
## Overview
This script is used to perform text normalization and similarity calculation on a set of documents from the Reuters corpus. It uses the NLTK library to tokenize, lowercase, and remove stopwords from each document, resulting in a list of dictionaries representing the bag of words representation of each document. Additionally, it includes functions to time the execution of any function, perform linear regression on the log-transformed data, and plot the running time over the size of the input.

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
```
nltk.download('punkt')
nltk.download('stopwords')
```
After that, you can execute the script using your preferred method.

## Results
The script compares the performance of Jaccard similarity and Cosine similarity algorithms on the Reuters corpus. The results show that Jaccard similarity is better than Cosine similarity in general, due to the fact that it takes time of O(1) when searching for certain keys in the dictionary. The use of the dictionary in the cosine similarity resulted in similar results to the Jaccard algorithm.

For the cosine similarity, it was found that the dot product with numpy is faster than other representations, due to the fact that numpy is an optimized library for matrix multiplication. In terms of lists, the use of the get method is faster than iterating through the dictionary.

When calculating the all-pair similarity, the choice of the algorithm affected the average running time, but still the Jaccard algorithm outperformed the Cosine similarity, as expected even with small differences.

Finally, to reduce the time, the script used multiprocessing with the map-reduce technique to divide the work to multiple processes and let them work in parallel, which yielded faster results.

## Limitations
This script is designed to work with the Reuters corpus, and any modifications may be necessary to work with other corpora or text sources. Additionally, the script assumes that the user has access to the Reuters corpus, which requires a separate download.
## Additional Information
The script includes the following functions:
- normalize(token_list) : This function takes in a list of tokens and returns the list after converting all tokens to lowercase and removing stopwords.
- filter_stopwords(token_list) : This function takes in a list of tokens and returns the list after removing stopwords.
- timeit(somefunc, *args, repeats=10, **kwargs) : This function takes in a function, arguments, and number of repeats and returns the mean execution time of the function over the number of repeats.
- Basic_Line_plot(xs, ys, label) : This function takes in a list of x-values, y-values and label and returns a line plot of x-values and y-values.
- log_plot(log_x, log_y, label) : This function takes in a list of log_x-values, log_y-values and label and returns a log-log plot of log_x-values and log_y-values. It also performs linear regression on the log-transformed data to analyze the trend of the algorithm's running time over the size of the input.
## Conclusion
This script provides a comprehensive analysis of the performance of Jaccard similarity and Cosine similarity algorithms on the Reuters corpus. The results show that Jaccard similarity is a more efficient algorithm than Cosine similarity, and the use of numpy and multiprocessing can further improve the performance. The script also includes functions to time the execution of any function, perform linear regression, and plot the running time over the size of the input, making it useful for analyzing the performance of other algorithms or text processing tasks.
