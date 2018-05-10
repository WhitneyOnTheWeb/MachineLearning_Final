#!/usr/bin/python
import nltk
#nltk.download('all', halt_on_error=False)
from nltk.stem.snowball import SnowballStemmer
import string
import re

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    stem_list = []
    if len(content) > 1:
        ### remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text_string = content[1].translate(translator)

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words_list = text_string.split()
        stemmer = SnowballStemmer("english")
            
        for i in words_list:
            stem_list.append(stemmer.stem(i))
        
        words = ' '.join((' '.join(stem_list)).split())

    return words

    

def main():
    ff = open("../ud120-projects/text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()
