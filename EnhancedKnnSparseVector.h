//
// Created by Enes Kılıçaslan on 15/07/17.
//

#ifndef EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
#define EKNNV3_ENHANCEDKNNSPARSEVECTOR_H

#include <string>
#include <vector>

class EnhancedKnnSparseVector {

public:
    EnhancedKnnSparseVector(const std::string &inputFileName);

    void fillVectors();
    void printVectors() const;

    double similarityBM25(std::vector<int> const &s1, std::vector<int> const &s2 ) const;

    int n(std::string w) const; // TODO will be private
    int n(int wIndex) const; // directly takes index of the word

    double idf(std::string w) const;// TODO will be private
    double idf(int wIndex) const; // directly takes index of the word

    double fPrime(std::string w, std::vector<int> s) const; // TODO will be private
    double fPrime(int  wIndex, std::vector<int> s) const;// directly takes index of word


    void printLenghts() const;

    void setLa(double la);


private:

    int docCounter;
    std::string inputFileName;

    std::vector< std::vector < std::string > > labels; //labels for each document
    std::vector< std::string > words; //column, all of the words in the corpus
    std::vector< std::vector < int > > docs; //(0, vector<int>(0)); //row, vector for each document | each initalized to 0
    std::vector< int > lengths; //lengths of the training documents

    static const double K1, b; //constants for BM25 similarity
    long totalLenOfDocs; //total lenght of training documents
    double la;  //average lenght of training documents
};


namespace EnesKilicaslanCommonOperations{

    template <typename T, typename A>
    int contains(std::vector<T, A> const *v, T x);

    template <typename T>
    std::string  numberToString(T pNumber);

}


#endif //EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
