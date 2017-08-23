#ifndef EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
#define EKNNV3_ENHANCEDKNNSPARSEVECTOR_H

#include <string>
#include <vector>
#include <map>
#include <algorithm>

#define OUT_VEC_FILE_NAME "train_vectors.txt"


class EnhancedKnnSparseVector {

public:
    EnhancedKnnSparseVector(int k, const std::string &trainFileName, const std::string &testFileName);

    void fillVectors();
    void saveVectors() const;

    double similarityBM25(std::vector<bool> const &s1, int len1, std::vector<bool> const &s2, int len2 ) const;

    // TODO will be private
    int n(int wIndex) const; // directly takes index of the word

    double idf(int wIndex) const; // directly takes index of the word

    double fPrime(std::vector<bool > s, int len) const;// directly takes index of word

    void fillTestVectors();

    // takes sparse vector that contains a notion for each word in the corpus
    // So its size must be the same as words vector variable field
    std::vector<std::string> enhancedKnn(int testDocumentIndex) const;
    void runTest();

    void printLenghts() const;
    void setLa(double la);

    void setPLabel(bool pLabel);

private:

    int docCounter;
    std::string trainFileName;
    std::string testFileName;

    std::vector< std::vector < std::string > > labels; //labels for each document
    std::vector< std::string > words; //column, all of the words in the corpus
    std::vector< std::vector <bool > > docs; //(0, vector<int>(0)); //row, vector for each document | each initalized to 0(false)
    std::vector< int > lengths; //lengths of the training documents

    std::vector< std::vector <bool > > testDocs;
    std::vector< int > testLengths;

    static const double K1, b; //constants for BM25 similarity
    long totalLenOfDocs; //total lenght of training documents
    double la;  //average lenght of training documents
    bool pLabel;

    int k; // this is the 'famous' k
    double alpha;
};


namespace EnesKilicaslanCommonOperations{

    template <typename T, typename A>
    int contains(std::vector<T, A> const *v, T x);

    template <typename T>
    std::string  numberToString(T pNumber);

    //https://stackoverflow.com/questions/18112773/sorting-a-vector-of-pairs
    bool pairCompare(const std::pair<int, double >& firstElem,
                     const std::pair<int, double >& secondElem);

    bool pairCompareStr(const std::pair<std::string, double >& firstElem,
                     const std::pair<std::string, double >& secondElem);


    int isDirectory(const char *path);

}


#endif //EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
