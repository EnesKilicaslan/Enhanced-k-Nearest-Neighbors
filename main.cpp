#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>      /* printf */
#include <vector>
#include <map>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include "PreProcess.h"

//@Ref: stop words http://xpo6.com/list-of-english-stop-words/
//@Ref: stemming   https://github.com/OleanderSoftware/OleanderStemmingLibrary

#include "EnhancedKnnSparseVector.h"

using namespace std;

bool areArgumentsValid(map<string, string> const &arguments);
void printHelp();

int main(int argc, char** argv) {

    map<string, string> arguments;
    clock_t t;
    int  kValue;
    bool preProcess = false, save= false, pLabel= false;


    //** command line operations
    if(argc == 2 && strcmp(argv[1], "--help") == 0) {
        printHelp();
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        string command(argv[i]);

        string var, value;
        stringstream ss_command(command);

        getline(ss_command, var, '=');
        //cout <<  var << endl;
        getline(ss_command, value, '=');
        //cout <<  value << endl;

        if(value.empty() || var.empty()){
            cout << "Wrong paramater.. please see --help" << endl;
            return 1;
        }

        arguments[var] = value;
    }

    if (!areArgumentsValid(arguments)){
        cout << "Wrong paramater.. please see --help" << endl;
        return  1;
    }

    //if preprocess is asked by user
    if(arguments.count("preprocess") == 1)
        if(arguments["preprocess"] == "true")
            preProcess = true;

    //if preprocess is asked by user
    if(arguments.count("save") == 1)
        if(arguments["save"] == "true")
            save = true;

    //if preprocess is asked by user
    if(arguments.count("pLabel") == 1)
        if(arguments["pLabel"] == "true")
            pLabel = true;

    //** end of command line operations

    t = clock(); //** start timer to measure the time spent at the end
    std::cout << "***Welcome to Enhanced Knn***" << endl << endl;

    //the Data neeeds preprocessing
    if(preProcess) {

        cout << "Started Preprocessing.." << endl;
        //preprocess the train directory and then
        //create train file named "train_preprocessed"
        if(EnesKilicaslanCommonOperations::isDirectory(arguments["train"].c_str()) &&
                EnesKilicaslanCommonOperations::isDirectory(arguments["train"].c_str())){

            //preprocess the train directory and then
            //create train file named "pre_processed_train"
            PreProcess pp = PreProcess(arguments["train"].c_str(),
                                      "pre_processed_train",
                                       arguments["test"].c_str(),
                                       "pre_processed_test");
            pp.run();
            arguments["train"] = "pre_processed_train";
            arguments["test"] = "pre_processed_test";
            cout << "Preprocessing is done!" << endl << endl;
        }
        else{
            cerr << "You asked for preprocess the dataset but you use just a file"<<endl
                 << "instead of directory as train or test" << endl;
            return 1;
        }
    }

    stringstream ss(arguments["k"]);
    ss >> kValue;

    EnhancedKnnSparseVector eKnn( kValue, arguments["train"], arguments["test"]);

    eKnn.setPLabel(pLabel);
    cout << "Started filling train vectors.." << endl;
    eKnn.fillVectors();
    cout << "Filling train vectors is done!" << endl << endl;

    if(save) {
        cout << "Saving vectors to file named '" << OUT_VEC_FILE_NAME << "' " << endl;
        eKnn.saveVectors();
        cout << "Vectors are saved! you can check the file " << OUT_VEC_FILE_NAME << endl << endl;
    }

    cout << "Started filling test vectors.." << endl;
    eKnn.fillTestVectors();
    cout << "Filling test vectors is done!" << endl << endl;

    (pLabel) ? cout << "Making prediction.." << endl : cout << "Calculating nearest neighbours.." << endl ;
    eKnn.runTest();

    //** get time passed from the begining of the application
    //** and print it in minutes
    cout << "Operation Completed!" << endl;
    t = clock() - t;
    printf ("=> It took %ld minutes!\n",(((long)t)/CLOCKS_PER_SEC)/60);

    return 0;
}

bool areArgumentsValid(map<string, string> const &arguments){

    vector<string> argument_list;
    argument_list.push_back("train");
    argument_list.push_back("test");
    argument_list.push_back("result");
    argument_list.push_back("k");
    argument_list.push_back("export");
    argument_list.push_back("plabel");
    argument_list.push_back("vector");
    argument_list.push_back("save");
    argument_list.push_back("preprocess");

    for(map<string, string>::const_iterator it= arguments.begin(); it!=arguments.end(); ++it)
        if(find(argument_list.begin(), argument_list.end(), it->first) != argument_list.end())
            continue;
        else
            return false;


    if(arguments.count("train") == 1 && arguments.count("test") == 1 && arguments.count("k"))
        return true;
    else {
        return false;
    }
}

void printHelp(){

    cout << endl
         << "************************************************************************************" << endl
         << "**                            Welcome to Enhanced Knn                             **" << endl
         << "**                  This is an efficient implementation of eKNN                   **" << endl
         << "**                                                                                **" << endl
         << "**                         Implemented by Enes Kilicaslan                         **" << endl
         << "**               for Data Science Project at University of Antwerpen              **" << endl
         << "************************************************************************************" << endl << endl
         << "You can give Already Preprocessed file ( like the files from [3] ) " <<  endl
         << "\t\t\t Or \t\t\t" << endl
         << "Ask to preprocess a set of files ( like the files from [4] )" << endl
         << "The Followings are explanations of command line arguments:" << endl << endl
         << " --help:"<< endl << "\tprints the explanations of commands" << endl << endl
         << " k:"<< endl << "\ta number that represents number of files in the result (required)" << endl
         << "\tif you want to retrieve for example 5 documents then use k=5 " << endl << endl
         << " train:"<< endl << "\tspecifies the path of training file (required)" << endl
         << "\tif you don't set preprocess argument to true, this will be a path to the file" << endl
         << "\tif u set preprocess to true, the this is path to a directory which includes files" << endl <<endl
         << " test:"<< endl << "\tspecifies the path of testing file (required)"<< endl
         << "\tif you don't set preprocess argument to true, this will be a path to the file" << endl
         << "\tif u set preprocess to true, the this is path to a directory which includes files" << endl <<endl
         << " plabel:"<< endl << "\tstands for Predict Label. So if you set true, then it predicts label" << endl
         << "\tby using weighted vote and thresholding strategies" << endl
         << "\tdefault is false which means just retrieves the nearst documents "<< endl << endl
         << " vector:"<< endl << "\tif you want to use vector to keep track of the files, "<<endl
         << "\tset it to true.But be carreful while using this argument, because this creates" << endl
         << "\tvery sparse matrix and they are kept in main memory. If you test this with " << endl
         << "\ta big amounth of data set, it can easily overflow the memory. So use it if " << endl
         << "\tyou have moderate size of data set " << endl << endl
         << " save:" << endl << "\tin order to save sparse vector or inverted index to file" << endl
         << "\tset this argument to true" << endl << endl
         << " preprocess:" << endl << "\tif the data set that you use for train and test is not"  << endl
         << "\talready preprocessed, then set this option true to preprocess train and test" << endl
         << "\tdefault value for preprocess is false, meaningly wait for preprocessed dataset" << endl
         << "\tps: what we mean by preprocess is stemming and stop word removing" << endl << endl
         << "";
 }
