//
// Created by Enes Kılıçaslan on 25/08/17.
//

#include "EnhancedKnnInvertedIndex.h"
#include "Common.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

EnhancedKnnInvertedIndex::EnhancedKnnInvertedIndex(int k, const std::string &trainFileName, const std::string &testFileName)
        : trainFileName(trainFileName), testFileName(testFileName) {
    this->alpha =0.3; //k/10;
    this->k = k;


    this->docCounter = 0;
    this->N =0;
    this->totalLenOfDocuments =0;
}

void EnhancedKnnInvertedIndex::fillInvertedIndexes() {
    ifstream input(trainFileName.c_str());
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {
        long docLen = 0;

        vector< pair <long, int> > local_doc;

        if(line == "\n")
            continue; //skip empty lines

        vector<string> labels_local;

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        while( getline(ss_lab, tok_lab, ',')){
            labels_local.push_back(tok_lab);
        }

        labels.push_back(labels_local); // append labels

        while(getline(ss_main, tok_main, ' ')) {

            unsigned long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss( tok_main.substr(pos + 1, 1));
            ss >> occurance;
            //cout << word << ":" << occurance << "-";

            if(occurance > 0){
                //filling index1 which maps doc to (word, frequency) pairs
                std::string::size_type sz;
                long word_dec = std::stol (word,&sz);
                local_doc.push_back(pair<long,int> (word_dec, occurance));

                //filling index2 which maps word to a set of documents
                this->wordToDocs[word_dec].push_back(docCounter);
            }

            docLen += occurance;
        }
        //keeping the words in inverted index gains us more efficiency
        std::sort(local_doc.begin(), local_doc.end(), EKCommonOperations::wordOccPairCompare);

        docToWords.push_back(local_doc);
        this->totalLenOfDocuments += docLen;
        this->docCounter +=1;
    }

    N = docToWords.size();
    setLa(); //N is set!
}

void EnhancedKnnInvertedIndex::printInvertedIndex() const {

    std::vector< std::vector <pair<long, int> > >::const_iterator it;

    for(it=docToWords.begin(); it!=docToWords.end(); ++it){

        vector<pair<long, int> >::const_iterator i;

        for (i=it->begin(); i != it->end(); ++i) {
            cout << i->first << " ";
        }

        cout << endl;
    }

    cout << "***********************" << endl;

    map< long , std::vector<long > >::const_iterator iter;

    for(iter= wordToDocs.begin(); iter != wordToDocs.end(); ++iter){
        std::cout << iter->first << " => ";

        vector<long>::const_iterator i;

        for ( i = iter->second.begin(); i < iter->second.end() ; ++i) {
            cout << *i << " ";
        }

        cout << endl;
    }

    cout << "N: " << N << endl;
    cout << "la: " << la << endl;
    cout << "idf(7253): " << idf(7253) << endl;
}


void EnhancedKnnInvertedIndex::fillTestInvertedIndex() {
    ifstream input(this->testFileName.c_str());
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {
        long docLen = 0;

        if (line == "\n")
            continue; //skip empty lines

        vector< pair <long, int> > local_doc;

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        //parse word and occurance
        while (getline(ss_main, tok_main, ' ')) {
            //cout <<"debug 2" << endl;

            unsigned long pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss(tok_main.substr(pos + 1, 1));
            ss >> occurance;

            if(occurance > 0){
                std::string::size_type sz;
                long word_dec = std::stol (word,&sz);
                local_doc.push_back(pair<long,int> (word_dec, occurance));
            }

            docLen += occurance;
        }

        //keeping the words in inverted index gains us more efficiency
        std::sort(local_doc.begin(), local_doc.end(), EKCommonOperations::wordOccPairCompare);

        testDocToWords.push_back(local_doc);
    }
}


int EnhancedKnnInvertedIndex::n(long w) const {
    return wordToDocs.at(w).size();
}

double EnhancedKnnInvertedIndex::idf(long w) const {
    return log((N - n(w) + 0.5) / n(w) + 0.5);
}

/**
 * returns term frequency of w in document s
 * @param w
 * @param s
 * @return
 */
int EnhancedKnnInvertedIndex::f(long w, vector<std::pair<long, int> > const &s) const {
    vector<std::pair<long, int> >::const_iterator it;

    for(it = s.begin(); it!=s.end(); ++it)
        if (it->first == w)
            return it->second;
    return 0;
}


void EnhancedKnnInvertedIndex::setPLabel(bool pLabel) {
    EnhancedKnnInvertedIndex::pLabel = pLabel;
}

//before using this method, N must be set!
void EnhancedKnnInvertedIndex::setLa(){
    this->la = totalLenOfDocuments / (double) N;
}
