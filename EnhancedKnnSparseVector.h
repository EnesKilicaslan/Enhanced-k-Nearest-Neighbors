#ifndef ENHANCEDKNNSPARSEVECTOR_H
#define ENHANCEDKNNSPARSEVECTOR_H

#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "EnhancedKNNBase.h"

#define OUT_VEC_FILE_NAME "train_vectors.txt"

class EnhancedKnnSparseVector: public  EnhancedKNNBase{

public:
    EnhancedKnnSparseVector(int k, const std::string &trainFileName, const std::string &testFileName, bool pLabel);
    void run(bool save);


private:

    void runTest();

    void fillVectors();
    void fillTestVectors();

    void saveVectors() const;

    double similarityBM25(std::vector<bool> const &s1, int len1, std::vector<bool> const &s2, int len2 ) const;

    int n(int wIndex) const; // directly takes index of the word
    double idf(int wIndex) const; // directly takes index of the word
    double fPrime(std::vector<bool > s, int len) const;// directly takes index of word

    // takes sparse vector that contains a notion for each word in the corpus
    // So its size must be the same as words vector variable field
    std::vector<std::string> enhancedKnn(int testDocumentIndex) const;






    std::vector< std::vector < std::string > > labels; //labels for each document
    std::vector< std::string > words; //column, all of the words in the corpus
    std::vector< std::vector <bool > > docs; //(0, vector<int>(0)); //row, vector for each document | each initalized to 0(false)
    std::vector< int > lengths; //lengths of the training documents, we have to keep this because docs.size() does not give correct result

    // Obviously, it contains zero values, which do not change the size of the documents

    std::vector< std::vector <bool > > testDocs;
    std::vector< int > testLengths;

};



#endif //ENHANCEDKNNSPARSEVECTOR_H
