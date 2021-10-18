"""
The discriminator calculates for each candidate extracted
from the extractor domain relevance and domain consensus.
Given these and the alpha and theta value, it then
decides whether a word is a valid keyword or not. It finally
saves a list of accepted and rejected candidates
"""

import numpy as np

from src.ahoc_automaton import State
from src.utils import progress_bar


class Discriminator:

    def __init__(self, candidates, min_freq, clean_corpus=True):
        self.candidates = candidates
        self.matcher = None
        self.domain_frequency = {}
        self.reference_frequency = {}
        self.final_candidates = set()
        self.rejected_candidates = set()
        self.domain_relevance = {}
        self.domain_consensus = {}
        self.domain_conditional_probs = {}
        self.reference_conditional_probs = {}
        if clean_corpus:
            self.clean_corpus(min_freq)
        self.initialize_ahoc()

    def clean_corpus(self, min_freq):
        """
        If the user selected a minimum frequency for keywords,
        the discriminator will automatically remove all invalid
        entries to avoid confusion during calculations.
        This method also calculates the absolute frequency of
        each valid candidate in the domain corpus
        """

        # collect words with lower frequency than threshold
        words = list(self.candidates.keys())

        for word in words:
            abs_freq = sum(self.candidates[word])

            # eliminate invalid entries
            if abs_freq < min_freq:
                del self.candidates[word]

            # save absolute frequency of valid candidates
            else:
                self.domain_frequency[word] = abs_freq

    def initialize_ahoc(self):
        """
        This method initializes the Aho-Corasick automaton
        to calculate the absolute frequency of each candidate
        in the reference corpus
        """
        candidates = self.candidates.keys()
        self.matcher = State.create_automaton(candidates)

    def find_reference_frequence(self, text):
        """
        Given a string, this method finds matches with the
        Aho-Corasick automaton
        """
        self.matcher.find_match(text, True)

    @staticmethod
    def cond_prob(candidate, corpus, total_ocurrencies):
        """
        given a word, a corpus this method calculates
        the conditional probability of said word given
        the corpus
        """
        nominator = corpus[candidate]
        denominator = total_ocurrencies
        return nominator / denominator

    def calculate_domain_relevance(self, word):
        """
        given a word, calculates domain relevance
        domain relevance is defined as:

            DR = (CD domain) / (CD domain + CD reference)

        where CD is the conditional probability of the word
        in either the domain or the reference corpus
        """
        nominator = self.domain_conditional_probs[word]

        if word in self.reference_conditional_probs:
            reference_prob = self.reference_conditional_probs[word]
        else:
            reference_prob = 0

        denominator = self.domain_conditional_probs[word] + reference_prob

        return nominator / denominator

    def calculate_domain_consensus(self, word):
        """
        calculates domain consensus for a given word
        given a vector containing the occurrence of said
        word in each document (a column in a document-matrix
        with no zeros)
        """
        vec = np.array(self.candidates[word])
        PtD = vec / vec.sum()
        logged = PtD * np.log2(1 / PtD)
        consensus = logged.sum()

        return consensus

    def calculate_total(self):
        """
        calculates the total occurrence of every candidate
        in both the domain and reference corpus.
        This is needed to calculate the conditional probability
        """
        total_domain = 0
        for value in self.domain_frequency.values():
            total_domain += value

        total_reference = 0
        for value in self.reference_frequency.values():
            total_reference += value

        return total_domain, total_reference

    def calculate_dr_dc(self):
        """
        Calculate for each candidate domain relevance
        and consensus and saves them in the respective
        dictionary
        """
        # copy absolute frequency from aho-corasick automaton
        self.reference_frequency = self.matcher.counts

        candidates = self.candidates.keys()
        tot_candidates = len(candidates)

        total_domain, total_reference = self.calculate_total()

        for i, candidate in enumerate(candidates):

            # CONDITIONAL PROBABILITIES
            # domain corpus
            domain_prob = self.cond_prob(
                candidate, self.domain_frequency, total_domain
            )
            self.domain_conditional_probs[candidate] = domain_prob

            # reference corpus
            if candidate in self.reference_frequency:
                ref_prob = self.cond_prob(
                    candidate, self.reference_frequency, total_reference
                )
                self.reference_conditional_probs[candidate] = ref_prob

            # DOMAIN RELEVANCE
            dr = self.calculate_domain_relevance(candidate)
            self.domain_relevance[candidate] = dr

            # DOMAIN CONSENSUS
            consensus = self.calculate_domain_consensus(candidate)
            self.domain_consensus[candidate] = consensus

            progress_bar(
                i+1, tot_candidates, prefix='Calculating', fixed_len=True
            )

    def calculate_f_value(self, dom_rel, dom_cons, alpha):
        """
        this function calculates f-value based on
        alpha, theta, domain relevance and consensus
        """
        dr = alpha * dom_rel
        dc = (1 - alpha) * dom_cons
        f = dr + dc

        return f

    def generate_list(self, alpha, theta, verbose=True):
        """
        this function goes through each candidate and
        decides whether it is a good keyword or not.
        It then saves the candidate and its f-value
        accordingly
        """
        candidates = self.domain_relevance.keys()
        tot_candidates = len(candidates)

        for i, candidate in enumerate(candidates):
            dr = self.domain_relevance[candidate]
            consensus = self.domain_consensus[candidate]
            f_value = self.calculate_f_value(dr, consensus, alpha)

            if f_value < theta:
                self.rejected_candidates.add((candidate, f_value))
            else:
                self.final_candidates.add((candidate, f_value))

            if verbose:
                progress_bar(
                    i+1, tot_candidates, prefix='Generating', fixed_len=True
                )
