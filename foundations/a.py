def find_alphabetically_first_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes first
    lexicographically. Words split by whitespace. Case-sensitive.
    """
    # BEGIN_YOUR_CODE
    words = text.split()
    if not words:
        return ""
    return min(words)
    # END_YOUR_CODE
    
    
print(find_alphabetically_first_word("the quick brown fox jumps over the lazy dog"))