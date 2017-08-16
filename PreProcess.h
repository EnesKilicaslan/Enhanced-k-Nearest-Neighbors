//
// Created by Enes Kılıçaslan on 09/08/17.
//

#ifndef PREPROCESSOOP_PREPROCESS_H
#define PREPROCESSOOP_PREPROCESS_H

#define NUMBER_OF_STOP_WORDS 319

#include <string>
#include <set>
#include <unordered_map>
#include <vector>


class PreProcess {

public:
    PreProcess(const char *DIR_PATH);
    void run();
    static const char* const stopWords[NUMBER_OF_STOP_WORDS];


private:
    bool isStopWord(std::string s) const;
    int fillFileNames();
    void contructLibSVM();
    std::vector<std::string> split(const std::string &s) const;
    std::string trimContent(const std::string &c) const;


    const char* DIR_PATH;
    std::set<std::string> file_names;
    int wordCounter, labelCounter; //represents each word as a number
    std::unordered_map<std::string,int > words;
    std::unordered_map<std::string,int > labels;


};


#endif //PREPROCESSOOP_PREPROCESS_H
