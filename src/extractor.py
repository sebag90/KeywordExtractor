"""
The Extractor takes a corpus as input and processes
each file to extract possible candidates. It creates
a file occurrency dictionary with candidates and their
occurrence within the documents of the corpus
"""

import multiprocessing as mp
import string

import nltk

from src.utils import progress_bar


class Extractor:

    def __init__(
            self, min_sen, max_cap, min_tok,
            max_tok, not_paragraph, validation):
        self.min_sen = min_sen
        self.max_cap = max_cap
        self.min_tok = min_tok
        self.max_tok = max_tok
        self.not_paragraph = not_paragraph
        self.validation = validation
        self.validation_dictionary = set(nltk.corpus.words.words())
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

    @staticmethod
    def split_lists(list, n):
        """
        divide a list in n lists
        with equal number of elements
        """
        for i in range(0, n):
            yield list[i::n]

    def keep_paragraph(self, text):
        """
        decides if a paragraph should be kept.
        Conditions:
            EITHER:
                - paragraph has at least s sentences
            OR:
                *all following conditions must be true*
                - paragraph has at least c percent of capitalized words
                - tl minimum number of tokens
                - tu maxumim number of tokens
                - ends in punctuation

        If conditions are met, the paragraph can be used to extract keywords
        """
        sentences = nltk.sent_tokenize(text)
        keep = True

        # lenght of sentence must be higher than s
        if len(sentences) >= self.min_sen:
            return sentences

        # paragraph contains at least c percent of capizalized words
        tokens = nltk.word_tokenize(text)
        capitalized = [tok for tok in tokens if tok[0].isupper()]
        ratio = (len(capitalized) * 100) / len(tokens)
        if ratio < self.max_cap:
            keep = False

        # minimum number of tokens
        if len(tokens) < self.min_tok:
            keep = False

        # maximum number of tokens
        if len(tokens) > self.max_tok:
            keep = False

        # ends with punctation
        if tokens[-1] not in string.punctuation:
            keep = False

        if keep:
            return sentences

        return False

    def preprocess(self, sentences):
        """
        tokenizes and POS-tags sentences (a list of strings)
        and returns a tagged sentences
        """
        # tokenize
        tok_sents = [nltk.word_tokenize(sentence) for sentence in sentences]

        # POS-tag sentences
        tagged = [nltk.pos_tag(sent) for sent in tok_sents]

        return tagged

    def keep_candidate(self, tree):
        """
        this function decides if a candidate
        should be kept, it will remove:
            - proper nouns only
            - splitted words
            - stop words
            - words with punctuation (except -)
            - 2 identical words
            - too short candidates

        the function returns a candidate as a string
        """
        good_candidate = True

        words, tags = zip(*list(tree))

        set_tags = set(tags)

        # eliminate NPs if it consists of only proper nouns
        if len(set_tags) == 1 and set_tags.pop() in {"NNP", "NNPS"}:
            good_candidate = False

        # split - connected words
        no_dash = [word.replace("-", " ") for word in words]

        # collect every word individually
        splitted = list()
        for word in no_dash:
            splitted += word.split()

        # remove artifacts e.g. "th e"
        if "".join(splitted) in self.validation_dictionary:
            good_candidate = False

        # filter stopwords
        if len(set(words).intersection(self.stopwords)) > 0:
            good_candidate = False

        # filter punctuation
        for word in no_dash:
            for punct in string.punctuation:
                if punct in word:
                    good_candidate = False

        # a np is acceptable if ALL its components are actual english words
        if self.validation:
            splitted = set(splitted)
            if not splitted.issubset(self.validation_dictionary):
                good_candidate = False

        # avoid artifacts like "w h"
        if len(" ".join(words)) <= 3:
            good_candidate = False

        # 2 words must be different
        if len(set(splitted)) < 2:
            good_candidate = False

        if good_candidate:
            return " ".join(i.lower() for i in words)

        return False

    def extract_words(self, POS_sents):
        """
        This functions creates a generator that
        yields good candidates only given a list
        of of POS tagged sentences as an argument
        """

        grammar = "NP:{(<NN.*>|<JJ.*>|<VB(G|D|N)>)<NN.*>}"
        cp = nltk.RegexpParser(grammar)

        for sentence in POS_sents:
            parsed = cp.parse(sentence)
            NPs = list(parsed.subtrees(filter=lambda x: x.label() == "NP"))

            for tree in NPs:
                good_candidate = self.keep_candidate(tree)

                if good_candidate:
                    yield good_candidate

    def single(self, paths, verbose=False):
        """
        given a list of paths the function opens each file and extracts
        potential candidates from them. For each text the function
        calculates document frequency and saves it in self.candidates
        """
        errors = list()
        final = dict()

        for i, filepath in enumerate(paths):
            # process single file
            try:
                filedict = dict()
                with open(filepath, "r", encoding="utf-8") as infile:
                    for line in infile:
                        line = line.strip()
                        if len(line) > 0:

                            # extract sentences
                            if self.not_paragraph:
                                sentences = nltk.sent_tokenize(line)
                            else:
                                sentences = self.keep_paragraph(line)

                            # extract candidates from sentences
                            if sentences:
                                preprocessed = self.preprocess(sentences)
                                candidates = self.extract_words(preprocessed)

                                for candidate in candidates:
                                    if candidate not in filedict:
                                        filedict[candidate] = 0
                                    filedict[candidate] += 1

                # copy file results to final
                for candidate, frequency in filedict.items():
                    if candidate not in final:
                        final[candidate] = list()
                    final[candidate].append(frequency)

                if verbose:
                    progress_bar(
                        i+1, len(paths), prefix="Extracting", fixed_len=True
                    )

            except (OSError, UnicodeDecodeError):
                errors.append(filepath)

        return final, errors

    def join_results(self, results):
        """
        join results from multiprocessing
        """
        single = dict()
        errors = list()

        for result, error in results:
            errors += error
            for candidate, frequency in result.items():
                if candidate not in single:
                    single[candidate] = []
                single[candidate] += frequency

        return single, errors

    def multi(self, corpus):
        """
        extract candidates using multiprocessing
        """
        sublists = self.split_lists(corpus, len(corpus))

        results = list()

        with mp.Pool() as pool:
            for i, out in enumerate(pool.imap(self.single, sublists)):
                progress_bar(
                    i+1, len(corpus), prefix="Extracting", fixed_len=True
                )
                results.append(out)

        candidates, errors = self.join_results(results)
        return candidates, errors
