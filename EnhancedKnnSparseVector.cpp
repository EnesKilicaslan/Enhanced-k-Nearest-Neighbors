//
// Created by Enes Kılıçaslan on 15/07/17.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <set>




#include "EnhancedKnnSparseVector.h"
using namespace std;


const double EnhancedKnnSparseVector::K1 = 0.25;
const double EnhancedKnnSparseVector::b  = 0.75;


EnhancedKnnSparseVector::EnhancedKnnSparseVector(int k, double alpha, const string &inputFileName)
        : inputFileName(inputFileName) {
    this->k = k;
    this->alpha = alpha;
    this->docCounter = 0;
    this->totalLenOfDocs = 0;
    this->la = 0.0;
}


void EnhancedKnnSparseVector::fillVectors() {

    ifstream input(inputFileName);
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {

        if(line == "\n")
            continue; //skip empty lines

        vector<string> labels_local;
        vector<int> docs_local(words.size(), 0);

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);
        //cout <<"debug 1" << endl;

        while( getline(ss_lab, tok_lab, ',')){
            labels_local.push_back(tok_lab);
        }

        labels.push_back(labels_local); //append labels
        docs.push_back(docs_local);
        lengths.push_back(0); //fist initialize lenght of the document to zero

        //parse word and occurance
        while(getline(ss_main, tok_main, ' ')){
            //cout <<"debug 2" << endl;

            long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance = stoi( tok_main.substr(pos + 1, 1) );
            int word_index = EnesKilicaslanCommonOperations::contains(&words, word);

            if (word_index >= 0){
                //cout << "index Debug " << word << " " << word_index << "docs counter " << doc_counter <<  endl;
                //cout << "lend docs: " << docs.size() << endl;
                docs[this->docCounter][word_index] = occurance;
                lengths[this->docCounter] += occurance;
                totalLenOfDocs += occurance;

            } else {
                words.push_back(word);

                //TODO make it with -1 for efficiency ! And use push back for occurances
                //!! Attention -- this makes really sparse matrix
                for( int i=0; i < docs.size(); ++i) //because this is new word, it never occurred before
                    docs[i].push_back(0);  // so make it 0 for the other documents

                docs[this->docCounter][words.size() - 1] = occurance; // Bag of words (not binary)
                lengths[this->docCounter] += occurance;
                totalLenOfDocs += occurance;
            }
        }

        //cout <<"debug 3" << endl;
        this->docCounter += 1;
    }

    setLa(totalLenOfDocs/docCounter); //calculate average document length once
}

void EnhancedKnnSparseVector::printVectors() const {

    for(vector<string>::const_iterator it = words.begin(); it != words.end(); ++it) {
        printf("%8s | ", (*it).c_str());
    }

    cout << endl;
    std::vector< std::vector<int> >::const_iterator row;
    std::vector<int>::const_iterator col;

    for (row = this->docs.begin(); row != this->docs.end(); ++row)
    {
        for (col = row->begin(); col != row->end(); ++col)
        {

            printf("%8s | ", EnesKilicaslanCommonOperations::numberToString(*col).c_str());
        }

        cout << endl;
    }

}

double EnhancedKnnSparseVector::similarityBM25(std::vector<int> const &s1, std::vector<int> const &s2 ) const{

    double result=0.0;

    if(s1.size() != s2.size())
        return  -1; // sizes must be the same

    for(int i=0; i<s1.size(); ++i){
        if(s1[i]>0 && s2[i] > 0) {
            /*cout << "ei: " << i << endl;
            cout << "idf: " << idf(i) << endl;
            cout << "s1:  " << fPrime(i, s1) << endl;
            cout << "s2:  " << fPrime(i, s2) << endl;*/
            result += fPrime(i,s1) *  fPrime(i,s2) * idf(i);
        }
    }

    return result;
}

/**
 *
 *                 (k1 + 1) f(w,s)
 * f'(w,s) = -----------------------------
 *                                |s|
 *              k1 + ( 1 - b + b ----- )
 *                                la
 */
double EnhancedKnnSparseVector::fPrime(std::string w, std::vector<int> s) const {
    int fws;

    int indx = EnesKilicaslanCommonOperations::contains(&words, w);
    //check if the word exist in our corpus
    //we are almost sure that it will exist, but security first!
    if (indx == -1) //TODO think about these return -1's again, 0 might be bettter
        return -1;

    fws = s[indx];

    return ((K1 + 1) * fws) / (K1 + (1 -b + b * s.size() / la ));
}

//this overloaded method takes directly index of the word
double EnhancedKnnSparseVector::fPrime(int indx, std::vector<int> s) const {
    int fws = s[indx];
    return ((K1 + 1) * fws) / (K1 + (1 -b + b * s.size() / la ));
}

/**
 * @ref: http://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/
 *
 * Trick: adding 1 before taking log, prevents us to get negeative result like said in referance
 *
 *
 *                    N - n(w) + 0.5
 *  idf(w) = log ( ------------------- + 1 )
 *                      n(w) + 0.5
**/
double EnhancedKnnSparseVector::idf(std::string w) const {
    int nw = n(w);
    cout << "idf done" << endl;
    return log(((docCounter - nw + 0.5) / (nw + 0.5)) + 1);
}

//this overloaded method takes directly index of the word
double EnhancedKnnSparseVector::idf(int wIndex) const {
    int nw = n(wIndex);
    double result = log10(((docCounter - nw + 0.5) / (nw + 0.5)) + 1);

    /*cout << "*********" << endl;
    cout << "nw: "<< nw << " N: " << docCounter << endl;
    cout << "result: " << result << endl;
    cout << "up: " << (docCounter - nw + 0.5) << endl;
    cout << "down: " << (nw + 0.5) << endl;
    cout << "*********" << endl;
    */
    return result;
}

int EnhancedKnnSparseVector::n(std::string w) const {
    int result = 0;

    int indx = EnesKilicaslanCommonOperations::contains(&words, w);
    //check if the word exist in our corpus
    //we are almost sure that it will exist, but security first!
    if (indx == -1)
        return -1;

    for(int i=0; i< docCounter; ++i) {
        int occ = docs[i][indx];
        if ( occ > 0)
        result += occ;
    }

    return result;
}

//this overloaded method takes directly index of the word
int EnhancedKnnSparseVector::n(int indx) const {
    int result = 0;

    for(int i=0; i< docCounter; ++i) {
        int occ = docs[i][indx];
        if ( occ > 0)
            result += 1;
    }

    return result;
}

void EnhancedKnnSparseVector::printLenghts() const {

    cout << "similarity: " <<  similarityBM25(docs[1], docs[2]) << endl;
}


// takes sparse vector that contains a notion for each word in the corpus
// So its size must be the same as words vector variable field
std::vector<std::string> EnhancedKnnSparseVector::enhancedKnn(const std::vector<int> & testDocument) const {

    std::vector< std::pair< int, double> > neighborSimPairs;
    set<string> candidateClasses;
    map<string , double> labelScores;


    if( testDocument.size() != words.size()) {
        cerr << "Error for test Document" << endl;
        exit(1);
    }


    //get the first k documents in order
    for(int i=0; i<this->k; ++i)
        neighborSimPairs.push_back(std::pair<int, double> (i,similarityBM25(testDocument, docs[i])) ) ;

    //sort documents
    std::sort(neighborSimPairs.begin(), neighborSimPairs.end(), EnesKilicaslanCommonOperations::pairCompare);

    // traverse all the documents in the training set
    // and replace the new document if there is lower similar document
    // in the k nearest neighbors
    for(int i=k; i<docCounter ; ++i){

        neighborSimPairs.push_back(std::pair<int, double> (i,similarityBM25(testDocument, docs[i])) ) ;
        std::sort(neighborSimPairs.begin(), neighborSimPairs.end(), EnesKilicaslanCommonOperations::pairCompare);

        //remove last element
        //we always keep first k documents!
        neighborSimPairs.erase(neighborSimPairs.end());
    }

    /*
     * k nearest neigbors are found!
     * now applying weighted voting on candidate classes
     * accourding to the following formula
     *
     *                     ---`
     * score(c|se) =       \  y(si, c)BM25(si, se)^a
     *                     /__,
    */

    //put candidate classes in a set to make them unique
    for (int j = 0; j < neighborSimPairs.size(); ++j) {

        int labIndex = neighborSimPairs[j].first;

        for (int i = 0; i < labels[labIndex].size() ; ++i)
            candidateClasses.insert(labels[labIndex][i]);
    }

    //calculate score for each candidate class
    /**
     * for each candidate class
     *      for each document in the k-nearest neighbors
     *          calculate score
     */
    for(set<string>::const_iterator it= candidateClasses.begin(); it != candidateClasses.end(); ++it){

        labelScores[*it] = 0.0;

        for (int j = 0; j < neighborSimPairs.size(); ++j) {

            int labIndex = neighborSimPairs[j].first;

            //if the document contains candidate class
            if (std::find(labels[labIndex].begin(), labels[labIndex].end(), *it) != labels[labIndex].end()){
                double sim = neighborSimPairs[j].second;

                labelScores[*it] += pow(sim,alpha);
            }
        }
    }







}

void EnhancedKnnSparseVector::setLa(double la) {
    this->la = la;
}

namespace EnesKilicaslanCommonOperations{

    template <typename T, typename A>
    int contains(vector<T, A> const *v, T x){
        typename vector<T, A>::const_iterator it = find(v->begin(), v->end(), x); //iterator

        if (it == v->end())
            return -1;
        else
            return it - v->begin();
    }

    template <typename T>
    string  numberToString(T pNumber)
    {
        ostringstream oOStrStream;
        oOStrStream << pNumber;
        return oOStrStream.str();
    }


    /**
     * Comparator for <id, similarity> (int, double) pair
     * @ref: https://stackoverflow.com/questions/18112773/sorting-a-vector-of-pairs
     * @return
     */
    bool pairCompare(const std::pair<int, double >& firstElem,
                     const std::pair<int, double >& secondElem) {
        return firstElem.second > secondElem.second;
    }



}

