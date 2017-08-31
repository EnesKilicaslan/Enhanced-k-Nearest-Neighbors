# Implementation of Enhanced k-Nearest-Neighbors

## What is Enhanced K-Nearest Neighbour Algorithm:

Enhanced K-Nearest Neighbour algorithm is developed to classify multi label documents by ADReM group at University of Antwerpen.

####  *Algorithm*:
The Algorithm contains three main steps:
1. Find the k nearest neighbours according to BM25 similarity
2. Use the k nearest neighbours weighted vote on candidate classes
3. Decide the final predictions via the thresholding strategy.


![alt text][eknn]
[eknn]:  /Users/eneskilicaslan/Github/Enhanced-k-Nearest-Neighbors/images/eknn.png "Enhanced K-Nearest Neighbour" (https://nodesource.com/products/nsolid)

## About Project:

You can use this project either to find nearest neighbours according to BM25 similarity or to predict the label(s) of test documents.

**Implemented with two different ways:**

  1. **<u>Vector:</u>** fast on small datasets
  2. **<u>Inverted Index:</u>** efficient on large sparse datasets

The input data might be pre-processed or needs to pre-process, but you need to specify it by setting command line argument ** 'preprocess' ** to <u>true</u>.

## How To Compile And Run:

Because the project is implemented in standard c++, </br>
a UNIX like Operating System (e.g ***Ubuntu, MacOSx, ..**) and a c++ comiler (e.g  ***g++***) are needed.

Open a terminal in the directory where you downloaded the source code, then copy and paste the following command to the terminal.

```shell
g++ *.cpp -o exec
```

that command will create an executable named ***exec***, now you can use the executable wit *appropriate* command line arguments

*firstly*, </br>
**--help** command is available, and you can use it like following:

```shell
./exec --help
```

The output would list the command line arguments you can use and explains their meaning

*Secondly*,</br>
**k** , **train** , **test** arguments are mandatory arguments. So you need to set them correctly in order to use the executable.
- __**k:**__ is the decimal number to specify number of documents
- __**train:**__ is the path to a file or directory that contains training input
- __**test:**__ is the path to a file or directory that contains test input

Example usage:
```shell
./exec k=4 train="path/to/training/file" test="path/to/test/file"
```

*Lastly*, </br>
if you don't set **plabel** to ***_true_*** the application generates *k* number of nearest neighbours according to BM25 similarity, otherwise it will make a class ( label ) prediction by using _weighted voting_ and _thresholding_. </br>
As specified earlier, you can use two different type of data structure; one is vector and the other one is inverted index. In order to use vector (matrix) as a data structure, set **vector** to ***true***. If you don't set it, as a default the application will use inverted index. </br>
If you would like to see the data structure content in the file, set **save** to ***true***. This will save inverted index or vector (depending on what you are using to represent training documents) to a file. if you use vector, then  file name would be "train_vectors.txt"; otherwise "train_invertedindex.txt"
If your data needs preprocessing the set **preprocess** to ***true***. But train and test arguments must point to a directory each of which have to include .key and txt files for each document, like the dataset [here](https://drive.google.com/open?id=0BxSQJpmUf1flN0N0Mmpwc2ZTdDA).


### Some Running Examples:
I am assuming that you compiled the code without any error and have executable named **exec**

- Please download the data from [here](https://drive.google.com/open?id=0BxSQJpmUf1flN0N0Mmpwc2ZTdDA). This is actually NLM_500 dataset but I separated documents for test. Because the data needs preprocessing we need to use preprocess argument. I want to use sparse vector implementation and it would be great if I see that sparse vector in the file. I also want it to predict labels instead of nearest neighbors and I am using k=10. So the following command will do that :)

    ```shell
    ./exec k=10 train=NLM_500/train test=NLM_500/test preprocess=true vector=true save=true plabel=true
    ```

- This time I want to use inverted index with the same data, but I want it to retrieve the nearest neighbors according to BM25 similarity. I still need preprocessing but I don't want to save the inverted index So the following command is exactly what I am looking for:

    ```shell
    ./exec k=10 train=NLM_500/train test=NLM_500/test preprocess=true
    ```

- Surely, you can use a dataset which is already preprocessed like the data [here](https://drive.google.com/open?id=0B3lPMIHmG6vGRmEzVDVkNjBMR3c). This data is Wikipedia-500K which is really big for this kind of algorithms, so we must use inverted index implementation. Again I want to save the inverted indexes for train data sets to a file and as a result I expect prediction of labels. This time I am supplying path to files not directories for train and test arguments because of no need to preprocess. Lastly, this takes a couple of hours

    ```shell
    ./exec k=20 train=WikiLSHTC/wikiLSHTC_train.txt test=WikiLSHTC/wikiLSHTC_test.txt plabel=true save=true
    ```

****PS: I am assuming that the NLM_500 and WikiLSHTC folders are in the same directory with the executable****


## ToDo's:

- [x] read necessary materials to understand the concept

- [x] pre-processing code

- [x] implement BM25 similarity to compare two documents

- [x] implement k-NN

- [x] implement weighted vote scheme

- [x] implement thresholding strategy

- [x] add --help command

- [x] write documentation

- [x] use inverted index to make k-NN faster

- [x] running the implementation on the datasets and comparing results

- [ ] changing the implementation to multi-threaded

- [ ] writing report
