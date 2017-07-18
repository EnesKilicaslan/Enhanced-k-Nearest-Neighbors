#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>
#include <sstream>

using namespace std;

template <typename T, typename A>
int contains(vector<T, A>  *v, T x);

template <typename T>
string number_to_string(T pNumber);


void print_verbose(vector<string>  *theWords,   vector< vector<int> > * theDocs);

int main() {
    int doc_counter = 0;

    //ifstream input("/Users/eneskilicaslan/CLionProjects/Extended-kNN/train");
    ifstream input("/Users/eneskilicaslan/CLionProjects/EKNNv2/train_head.txt");
    string line;
    getline( input, line ); // skip header


    vector<string> words;  //column, all of the words in the corpus
    vector< vector<int> > docs;//(0, vector<int>(0)); //row, vector for each document | each initalized to 0
    vector< vector<string> > labels; //labels for each document

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

        //parse word and occurance
        while(getline(ss_main, tok_main, ' ')){
            //cout <<"debug 2" << endl;

            long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance = stoi( tok_main.substr(pos + 1, 1) );
            int word_index = contains(&words, word);


            if (word_index >= 0){
                //cout << "index Debug " << word << " " << word_index << "docs counter " << doc_counter <<  endl;
                //cout << "lend docs: " << docs.size() << endl;
                docs[doc_counter][word_index] = occurance;

            } else {
                words.push_back(word);

                //TODO make it with -1 for efficiency
                //!! Attention -- this makes really sparse matrix
                for( int i=0; i < docs.size(); ++i) //because this is new word, it never occurred before
                    docs[i].push_back(0);  // so make it 0 for the other documents

                docs[doc_counter][words.size() - 1] = occurance; // Bag of words (not binary)
            }
        }

        //cout <<"debug 3" << endl;
        doc_counter += 1;
    }


    /*
    for(vector<string>::const_iterator i = words.begin(); i != words.end(); ++i) {
        // process i
        cout << *i << " "; // this will print all the contents of *features*
    }*/

    print_verbose(&words, &docs);

    //cout << "len: " << docs.size() << endl;
    return 0;
}

/**
 * Search in the vector
 * @param v vector
 * @param x element we are searching for
 * @return index of element if found, -1 otherwise
 *
template <typename T>
int contains(vector<T> v, T x){
    vector<T>::iterator it;
    it = find(v.begin(), v.end(), x); //iterator

    if (it == v.end())
        return -1;
    else
        return it - v.begin();
}*/
template <typename T, typename A>
int contains(vector<T, A>  *v, T x){
    typename vector<T, A>::iterator it = find(v->begin(), v->end(), x); //iterator

    if (it == v->end())
        return -1;
    else
        return it - v->begin();
}


void print_verbose(vector<string>  *theWords,   vector< vector<int> > * theDocs){

    for(vector<string>::iterator it = theWords->begin(); it != theWords->end(); ++it) {
        //  cout << *it << " ";
        printf("%8s | ", (*it).c_str());
    }

    cout << endl;
    std::vector< std::vector<int> >::const_iterator row;
    std::vector<int>::const_iterator col;

    for (row = theDocs->begin(); row != theDocs->end(); ++row)
    {
        for (col = row->begin(); col != row->end(); ++col)
        {

            printf("%8s | ", number_to_string(*col).c_str());
        }

        cout << endl;
    }
}

template <typename T>
string number_to_string(T pNumber)
{
    ostringstream oOStrStream;
    oOStrStream << pNumber;
    return oOStrStream.str();
}