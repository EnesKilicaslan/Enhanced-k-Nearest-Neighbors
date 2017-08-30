#ifndef ENHANCEDKNNINVERTEDINDEX_H
#define ENHANCEDKNNINVERTEDINDEX_H

#include <iostream>
#include <vector>
#include <map>

#include "EnhancedKNNBase.h"

#define OUT_INVERTED_INDEX_FILE_NAME "train_invertedindex.txt"

class EnhancedKnnInvertedIndex: public EnhancedKNNBase{

public:
    /**
     * Constructor that takes k, train and test file paths
     *
     * @param k
     * @param trainFileName
     * @param testFileName
     * @param pLabel: if user asks to predict document labels or just needs to return nearest neighbours
     *
     */
    EnhancedKnnInvertedIndex(int k, const std::string &trainFileName, const std::string &testFileName, bool pLabel);

    void run(bool save);




private:

    void runTest();

    //fill the inverted indexes
    void fillInvertedIndexes();
    void fillTestInvertedIndex();

    void saveInvertedIndex() const;

    std::vector<std::string> enhancedKnn(int testDocumentIndex) const;

    //enhanced k nearest neighbor methods
    int n(long w) const ;
    double idf(long w) const;
    int f(long w, std::vector<std::pair<long, int> > const &s) const;
    double fPrime(long w, std::vector<std::pair<long, int> > const &s) const;
    double similarityBM25(std::vector< std::pair<long, int> > const &s1, std::vector< std::pair<long, int> > const &s2) const;

    void calculateLa(); //not a regular setter with an argument, it is first calculated in the method

    void printInformation() const; //prints some information about training document





    //data structures to keep training documents and words. AKA: Inverted Index
    /**
     * docToWords is actuall a map whose key is document id, value is
     * a set of words and their number of frequencies (occurrences)
     *  like following:
     *      doc1 -> (word1,3) - (word2,2) - (word3,5)
     *      doc2 -> (word2,1) - (word4,5)
     *      doc3 -> (word3,2) - (word5,9) - (word6,3)
     *      .
     *      .
     *      .
     *
     *
     *
     * wordToDoc maps a word to the documents the word appeared in
     *  like following:
     *      word1 -> doc1, doc5, doc88
     *      word2 -> doc4, doc43
     *      word3 -> doc90
     *      .
     *      .
     *      .
     */
    std::vector< std::vector < std::pair< long, int> > > docToWords;
    std::map< long , std::vector<long> > wordToDocs;
    std::vector< std::vector < std::string > > labels; //labels for each document in training set

    /*
     * test documents!
     * we keep only one inverted index which maps documents to the
     * (word, frequency) pairs. We don't really need an inverted index which
     * maps word to set of documents!
     */
    std::vector< std::vector < std::pair <long, int > > > testDocToWords;

};


#endif //ENHANCEDKNNINVERTEDINDEX_H
