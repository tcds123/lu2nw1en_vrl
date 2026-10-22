# def anti_shuffle(s):
#     """
#     Write a function that takes a string and returns an ordered version of it.
#     Ordered version of string, is a string where all words (separated by space)
#     are replaced by a new word where all the characters arranged in
#     ascending order based on ascii value.
#     Note: You should keep the order of words and blank spaces in the sentence.

#     For example:
#     anti_shuffle('Hi') returns 'Hi'
#     anti_shuffle('hello') returns 'ehllo'
#     anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'
#     """
#     # Split the string into words and spaces
#     words = s.split(' ')
    
#     # Process each word to sort its characters
#     ordered_words = [''.join(sorted(word)) for word in words]
    
#     # Join the processed words back into a string
#     return ' '.join(ordered_words)

# # # Test cases
# # print(anti_shuffle('Hi'))  # Output: 'Hi'
# # print(anti_shuffle('hello'))  # Output: 'ehllo'
# # print(anti_shuffle('Hello World!!!'))
# print(anti_shuffle('\n\n123 456\n789\n'))

def anti_shuffle(s: str) -> str:
    """
    Returns an ordered version of the input string where each word's characters
    are sorted in ascending ASCII order. The order of words and spaces is preserved.

    :param s: Input string to be processed.
    :return: Ordered version of the input string.
    """
    return ' '.join(''.join(sorted(word)) for word in s.split(' '))

print(anti_shuffle('\n\n123 456\n789\n'))