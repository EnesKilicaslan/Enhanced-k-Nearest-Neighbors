//
// Created by Enes Kılıçaslan on 15/07/17.
//

#ifndef EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
#define EKNNV3_ENHANCEDKNNSPARSEVECTOR_H

#include <string>
#include <vector>
#include <map>
#include <algorithm>


class EnhancedKnnSparseVector {

public:
    EnhancedKnnSparseVector(int k, const std::string &trainFileName, const std::string &testFileName);

    void fillVectors();
    void printVectors() const;

    double similarityBM25(std::vector<int> const &s1, std::vector<int> const &s2 ) const;

    int n(std::string w) const; // TODO will be private
    int n(int wIndex) const; // directly takes index of the word

    double idf(std::string w) const;// TODO will be private
    double idf(int wIndex) const; // directly takes index of the word

    double fPrime(std::string w, std::vector<int> s) const; // TODO will be private
    double fPrime(int  wIndex, std::vector<int> s) const;// directly takes index of word

    void fillTestVectors();

    // takes sparse vector that contains a notion for each word in the corpus
    // So its size must be the same as words vector variable field
    std::vector<std::string> enhancedKnn(const std::vector<int> & test) const;
    void runTest();

    void printLenghts() const;
    void setLa(double la);


private:

    int docCounter;
    std::string trainFileName;
    std::string testFileName;

    std::vector< std::vector < std::string > > labels; //labels for each document
    std::vector< std::string > words; //column, all of the words in the corpus
    std::vector< std::vector < int > > docs; //(0, vector<int>(0)); //row, vector for each document | each initalized to 0
    std::vector< int > lengths; //lengths of the training documents

    std::vector< std::vector <int> > testDocs;

    static const double K1, b; //constants for BM25 similarity
    long totalLenOfDocs; //total lenght of training documents
    double la;  //average lenght of training documents

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


}


#endif //EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
