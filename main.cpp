#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>      /* printf */
#include <vector>
#include <map>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

#include "EnhancedKnnSparseVector.h"

using namespace std;

bool areArgumentsValid(map<string, string> const &arguments);

int main(int argc, char** argv) {

    std::cout << "Welcome to Enhanced Knn" << std::endl;
    map<string, string> arguments;

    clock_t t;
    t = clock();

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
        cout << "parameters are wrong, Check the parameters" << endl;
        return  1;
    }


    EnhancedKnnSparseVector eKnn(stoi(arguments["k"]), arguments["train"], arguments["test"]);

    eKnn.fillVectors();
    eKnn.printVectors();

    //eKnn.idf("60522");
    eKnn.printLenghts();
    cout << "TEST TIme" << endl;
    eKnn.fillTestVectors();
    cout << "Done" << endl;
    eKnn.runTest();
    t = clock() - t;
    printf ("It took %.3f minutes!\n",((float)t)/CLOCKS_PER_SEC/60);

    return 0;
}

bool areArgumentsValid(map<string, string> const &arguments){

    vector<string> argument_list;
    argument_list.push_back("train");
    argument_list.push_back("test");
    argument_list.push_back("result");
    argument_list.push_back("k");
    argument_list.push_back("export");

    for(map<string, string>::const_iterator it= arguments.begin(); it!=arguments.end(); ++it)
        if(find(argument_list.begin(), argument_list.end(), it->first) != argument_list.end())
            continue;
        else
            return false;


    if(arguments.count("train") && arguments.count("test") && arguments.count("k"))
        return true;
    else {
        return false;
    }
}