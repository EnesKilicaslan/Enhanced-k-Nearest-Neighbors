//
// Created by Enes Kılıçaslan on 09/08/17.
//
#include <iostream>
#include "dirent.h"
#include <vector>
#include <set>
#include <fstream>
#include <sstream>

#include "PreProcess.h"
#include "stemming/english_stem.h"


using namespace std;


const char* const PreProcess::stopWords[NUMBER_OF_STOP_WORDS] = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"};

PreProcess::PreProcess(const char *DIR_PATH) : DIR_PATH(DIR_PATH){
    this->wordCounter = 100;
    this->labelCounter = 100;
}

void PreProcess::run(){

    fillFileNames();
    contructLibSVM();

}

int PreProcess::fillFileNames() {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir (DIR_PATH)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL ) {

            if(strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name,  "..") == 0 || strcmp(ent->d_name, "") == 0)
                continue;

            string name(ent->d_name);
            std::size_t indx =0;
            indx = name.find(".");

            if(indx !=std::string::npos)
                file_names.insert(name.substr(0, indx));
            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("could not open directory");
        return EXIT_FAILURE;
    }

    return 0;
}

void PreProcess::contructLibSVM() {
    cout << " sizse: " << file_names.size() << endl;

    //iterate through file names
    set<string>::iterator it;
    for (it = file_names.begin(); it != file_names.end(); it++) {

        if(*it == "")
            continue;

        string file_path(DIR_PATH);
        if (file_path[file_path.length() - 1] != '/')
            file_path += "/";
        file_path += *it;

        string labels_path = file_path;
        labels_path += ".key";

        string contents_path = file_path;
        contents_path += ".txt";

        cout << "FILE:  ";
        cout << *it << endl;

        ifstream contents_file(contents_path.c_str());
        string content="", str="";

        // for key labels file

        // for txt contents file
        while (std::getline(contents_file, str))
        {
            content += str;
            content.push_back(' ');
        }

        content = trimContent(content);

        cout << content << endl;
        vector<string> splittedWords;

        splittedWords = split(content); //export words in string to vector as element!

        unordered_map<string, int > wordsCount; //keeps words in the current document and their counts
        for(int i=0; i< splittedWords.size(); ++i)
        {
            if(wordsCount.find(splittedWords[i]) == words.end())
                wordsCount.insert(make_pair(splittedWords[i], 1));
            else
                wordsCount[splittedWords[i]] += 1;
            //keeps words and their counts
        }

        /*
         * wordsCount map indeki her kelime icin
         *  words mapine bak,
         *    boyle bir kelime yoksa
         *      words e onu ekle ve yeni bir id numarasi al
         *
         *    dosyaya onun id numarasi ile count'unu yaz
         *
         *
         *
         *  for each word in map wordsCount
         *      if the word does not exist in map words
         *          add the word to the words and take new id
         *
         *      print the word's id and its count
         *
         *  TODO: Tabi bundan once dosyaya label lari yazilacak!
        */
        for (unordered_map<string, int>::iterator iter = wordsCount.begin(); iter != wordsCount.end(); ++iter ){

            if(words.find(iter->first) == words.end())
                words[iter->first] = wordCounter;

            cout << "(" << iter->first << ")" << words[iter->first] << ":" << iter->second << " ";
            wordCounter++;
        }


        string deneme;
        cout << "Enter: " << endl;

        //cout << "size: " << strlen(stopWords[NUMBER_OF_STOP_WORDS]);

        //for(int i=0; i< strlen(stopWords); ++i)
        //    cout << stopWords[i] << " ";

        cout << endl;
        cin >> deneme;
    }

}

vector<std::string> PreProcess::split(const std::string &s) const{
    string buf; // Have a buffer string
    stringstream ss(s); // Insert the string into a stream
    vector<string> tokens; // Create vector to hold our words

    while (ss >> buf) {
        if (buf.size() > 2 && !isStopWord(buf)) {
            //the word is not a stop word push it after stemming

            transform(buf.begin(), buf.end(), buf.begin(), ::tolower); //make all letters of the string lowercase
            wstring wBuf(buf.begin(), buf.end()); //convert it to wstring, because it is what StemEnglish class needs
            stemming::english_stem<> StemEnglish;

            StemEnglish(wBuf);
            string w(wBuf.begin(), wBuf.end());

            tokens.push_back(w);
        }
    }

    //for(int i=0; i< tokens.size(); ++i)
    //    cout << tokens[i] << endl ;

    return tokens;
}

std::string PreProcess::trimContent(const std::string &c) const{
    string res = "";

    for(int i=0; i< c.size(); ++i)
        if( c[i] == ' ' || isalpha(c[i]))
            res += c[i];

    return res;
}

bool PreProcess::isStopWord(std::string s) const {
    for(int i=0; i< NUMBER_OF_STOP_WORDS; ++i)
        if (strcmp(s.c_str(), stopWords[i]) == 0)
            return true;

    return false;
}
