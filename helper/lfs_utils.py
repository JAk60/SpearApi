def manual_trigWords(words):

    trigWord1 = {words}
    trigWord2 = set(words.replace(","," ").split())

    trigWord = trigWord1.union(trigWord2)
    trigWord = {element.lower() for element in trigWord}
    return trigWord
