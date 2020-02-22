""" This module is python practice exercises to cover more advanced topics.
	Put the code for your solutions to each exercise in the appropriate function.
    Remove the 'pass' keyword when you implement the function.
	DON'T change the names of the functions!
	You may change the names of the input parameters.
	Put your code that tests your functions in the if __name__ == "__main__": section
	Don't forget to regularly commit and push to github.
    Please include an __author__ comment so I can tell whose code this is.
"""
__author__ = "Mitsuka"
__version__ = 4.1

import random
import re
from collections import Counter

# List Comprehension Practice

def even_list_elements(input_list):
    """ Use a list comprehension to return a new list that has 
        only the even elements of input_list in it.
    """
    newList = [x for x in input_list if x%2 == 0]
    return newList
    


def list_overlap_comp(list1, list2):
    """ Use a list comprehension to return a list that contains 
        only the elements that are in common between list1 and list2.
    """ 
    newList = [x for x in list1 if x in list2]
    return newList
    


def div7list():
    """ Use a list comprehension to return a list of all of the numbers 
        from 1-1000 that are divisible by 7.
    """
    newList = [x for x in range(1, 1001) if x % 7 == 0]
    return newList
    


def has3list():
    """ Use a list comprehension to return a list of the numbers from 
        1-1000 that have a 3 in them.
    """
    newList = [x for x in range(1,1001) if x % 10 == 3]
    return newList
    


def cube_triples(input_list):
    """ Use a list comprehension to return a list with the cubes
        of the numbers divisible by three in the input_list.
    """
    newList = [x**3 for x in input_list if x % 3 == 0]
    return newList


def remove_vowels(input_string):
    """ Use a list comprehension to remove all of the vowels in the 
        input string, and then return the new string.
    """
    vowels = 'aeiou'
    nonvowels = ''.join([l for l in input_string if not l in vowels])
    return nonvowels


def short_words(input_string):
    """ Use a list comprehension to return a list of all of the words 
        in the input string that are less than 4 letters.
    """
    newString = [x for x in input_string if len(x)  < 4]
    return newString


# Challenge problem for extra credit:

def div_1digit():
    """ Use a nested list comprehension to find all of the numbers from 
        1-1000 that are divisible by any single digit besides 1 (2-9).
    """
    newList = [x for x in range(1,1001) if [y for y in range(2,10) if x % y == 0]]  
    return newList

# More practice with Dictionaries, Files, and Text!
# Implement the following functions:

def longest_sentence(text_file_name):
    """ Read from the text file, split the data into sentences,
        and return the longest sentence in the file.
    """
    newList = []
    f = open(text_file_name, 'r')
    data = f.read()
    
    x = data.replace(';', '.')
    newList = x.split('.')
    
    longest = ""
    for i in newList:
        if len(i) > len(longest):
            longest = i

    return longest

def longest_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return the longest word in the file.
    """
    newList = []
    f = open(text_file_name, 'r')
    data = f.read()
    
    newList = data.split()

    longest = ""
    for i in newList:
        if len(i) > len(longest):
            longest = i

    return longest

def num_unique_words(text_file_name):
    """ Read from the text file, split the data into words,
        and return the number of unique words in the file.
        HINT: Use a set!
    """
    set1 = set()
    f = open(text_file_name, 'r')
    data = f.read()
    wordList = data.split()

    for word in wordList:
        set1.add(word)

    return len(set1)

def most_frequent_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return a tuple with the most frequently occuring word 
        in the file and the count of the number of times it appeared.
    """
    wordList = []
    f = open(text_file_name, 'r')
    data = f.read()
    
    largest = 0
    count = 0
    wordList = data.split()
    word = wordList[0]

    for i in wordList: 
        curr_frequency = wordList.count(i) 
        if(curr_frequency > largest): 
            largest = curr_frequency 
            word = i 
            count = count + 1
  
    tuple1 = (word, count) 
    return tuple1

def date_decoder(date_input):
    """ Accept a date in the "dd-MMM-yy" format (ex: 17-MAR-85 ) and 
        return a tuple in the form ( year, month_number, day).
        Create and use a dictionary suitable for decoding month names 
        to numbers. 
    """
    date = date_input.split('-')
    day=date[0]

    monthDict={'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY':5 , 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT' : 10, 'NOV': 11, 'DEC': 12}

    month = monthDict[date[1]]
    year=date[2]

    return day, month, year 

def isit_random(lowest, highest, num_tries):
    """ Create and return a dictionary that is a histogram of how many 
        times the random.randInt function returns each value in the  
        range from 'lowest' to 'highest'. Run the randInt function a
        total number of times equal to 'num_tries'.
    """
    hist = {}
    for i in range(num_tries):
      hist[i] = random.randint(lowest,highest)
    
    return hist
        
# Extra challenge problem: Surpassing Phrases!

"""
Surpassing words are English words for which the gap between each adjacent
pair of letters strictly increases. These gaps are computed without 
"wrapping around" from Z to A.

For example:  http://i.imgur.com/XKiCnUc.png

Write a function to determine whether an entire phrase passed into a 
function is made of surpassing words. You can assume that all words are 
made of only alphabetic characters, and are separated by whitespace. 
We will consider the empty string and a 1-character string to be valid 
surpassing phrases.

is_surpassing_phrase("superb subway") # => True
is_surpassing_phrase("excellent train") # => False
is_surpassing_phrase("porky hogs") # => True
is_surpassing_phrase("plump pigs") # => False
is_surpassing_phrase("turnip fields") # => True
is_surpassing_phrase("root vegetable lands") # => False
is_surpassing_phrase("a") # => True
is_surpassing_phrase("") # => True

You may find the Python functions `ord` (one-character string to integer 
ordinal) and `chr` (integer ordinal to one-character string) useful to 
solve this puzzle.

ord('a') # => 97
chr(97) # => 'a'
"""

# Using the 'words' file on haiku, which are surpassing words? As a sanity check, I expect ~1931 distinct surpassing words.

def is_surpassing_phrase(input_string):
    """ Returns true if every word in the input_string is a surpassing
        word, and false otherwise.
    """
    pass


# I have more funky  challenge problems if you need them!

if __name__ == "__main__":
    print(__author__ + "'s results:")

    #list = ['hi', 'bye', 'mitsuka', 'me']
    #list2 = ['bye', 'cool']
    #print(even_list_elements(list))
    #print(list_overlap_comp(list, list2))
    #print(div7list())
    #print(has3list())
    #list1 = [3, 5, 7, 8, 9]
    #print(cube_triples(list1))
    #sentence = "kittie"
    #print(remove_vowels(sentence))
    #print(short_words(list))
    #print(div_1digit())
    #print("the longest sentence is:" + longest_sentence('rj_prologue.txt'))
    print("longest word:", longest_word('rj_prologue.txt'))
    print("the longest sentence:", longest_sentence('rj_prologue.txt'))
    print("unique words: ", num_unique_words('rj_prologue.txt'))
    print("most frequent word:", most_frequent_word('rj_prologue.txt'))
    print(date_decoder("11-AUG-03"))
    print(isit_random(0, 10, 10))

   
    
