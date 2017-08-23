#ifndef PREPROCESSOOP_PREPROCESS_H
#define PREPROCESSOOP_PREPROCESS_H

#define NUMBER_OF_STOP_WORDS 319

#include <string>
#include <set>
#include <unordered_map>
#include <vector>


class PreProcess {

public:
    PreProcess(const char *DIR_PATH, std::string output_file_name, const char *DIR_PATH_TEST, std::string output_file_name_test);
    void run();

    static const char* const stopWords[NUMBER_OF_STOP_WORDS];

private:
    bool isStopWord(std::string s) const;
    void writeToLibSvmFile(const std::set<std::string> &file_names, const std::string &fOutName, const char * DIR_PATH);



    int fillFileNames();
    void contructLibSVM();
    std::vector<std::string> split(const std::string &s) const;
    std::string trimContent(const std::string &c) const;

    const char* DIR_PATH_TRAIN;
    const char* DIR_PATH_TEST;
    std::set<std::string> file_names_train;
    std::set<std::string> file_names_test;
    int wordCounter, labelCounter; //represents each word as a number
    std::unordered_map<std::string,int > words;
    std::unordered_map<std::string,int > labels;
    std::string outFileName_train, outFileName_test ;


};


#endif //PREPROCESSOOP_PREPROCESS_H
