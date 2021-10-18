import unittest

from src.extractor import Extractor
from src.discriminator import Discriminator


class Test(unittest.TestCase):

    def test_candidates_extraction(self):
        expected = ['schooner america', 'international competition']

        text = (
            "The cup was originally awarded in 1851 by the Royal "
            "Yacht Squadron for a race around the Isle of Wight "
            "in the United Kingdom, which was won by the schooner "
            "America. Originally known as the 'RYS £100 Cup', "
            "the trophy was renamed the 'America's Cup' after the "
            "yacht and was donated to the New York Yacht Club (NYYC) "
            "under the terms of the Deed of Gift, which made the "
            "cup available for perpetual international competition."
        )

        extractor = Extractor(
            min_sen=2,
            max_cap=70,
            min_tok=5,
            max_tok=20,
            not_paragraph=False,
            validation=False
        )

        sents = extractor.keep_paragraph(text)
        tags = extractor.preprocess(sents)
        keyswords = extractor.extract_words(tags)

        self.assertListEqual(expected, list(keyswords))

    def test_domain_consensus(self):

        expected = [
            2.1180782093497093,
            1.9362600275315274,
            1.5219280948873624,
            1.9056390622295662,
            1.3609640474436813
        ]

        ext_input = {
            "w1": [1, 3, 1, 4, 2],
            "w2": [4, 2, 2, 3],
            "w3": [2, 1, 2],
            "w4": [3, 2, 1, 2],
            "w5": [5, 1, 4]
        }

        disc = Discriminator(
            candidates=ext_input,
            min_freq=0,
            clean_corpus=False
        )

        results = []
        for word in ext_input:
            dc = disc.calculate_domain_consensus(word)
            results.append(dc)

        self.assertListEqual(expected, results)

    def test_conditional_probs(self):
        domain_expected = [
            0.24444444444444444,
            0.24444444444444444,
            0.1111111111111111,
            0.17777777777777778,
            0.2222222222222222
        ]

        reference_expected = [
            0.18181818181818182,
            0.0,
            0.09090909090909091,
            0.45454545454545453,
            0.2727272727272727
        ]

        reference = {
            "w1": 2,
            "w2": 0,
            "w3": 1,
            "w4": 5,
            "w5": 3
        }

        ext_input = {
            "w1": [1, 3, 1, 4, 2],
            "w2": [4, 2, 2, 3],
            "w3": [2, 1, 2],
            "w4": [3, 2, 1, 2],
            "w5": [5, 1, 4]
        }

        disc = Discriminator(
            candidates=ext_input,
            min_freq=0,
            clean_corpus=True
        )

        disc.reference_frequency = reference
        total_domain, total_reference = disc.calculate_total()

        domain_results = list()
        reference_results = list()

        for candidate in ext_input:
            reference_prob = disc.cond_prob(
                candidate, disc.reference_frequency, total_reference
            )
            reference_results.append(reference_prob)

            domain_prob = disc.cond_prob(
                candidate, disc.domain_frequency, total_domain
            )
            domain_results.append(domain_prob)

        self.assertListEqual(domain_expected, domain_results)
        self.assertListEqual(reference_expected, reference_results)

    def test_domain_relevance(self):
        expected = [
            0.5734597156398105,
            1.0,
            0.5499999999999999,
            0.281150159744409,
            0.4489795918367347
        ]

        reference = {
            "w1": 2,
            "w2": 0,
            "w3": 1,
            "w4": 5,
            "w5": 3
        }

        ext_input = {
            "w1": [1, 3, 1, 4, 2],
            "w2": [4, 2, 2, 3],
            "w3": [2, 1, 2],
            "w4": [3, 2, 1, 2],
            "w5": [5, 1, 4]
        }

        disc = Discriminator(
            candidates=ext_input,
            min_freq=0,
            clean_corpus=True
        )

        disc.reference_frequency = reference
        total_domain, total_reference = disc.calculate_total()

        results = []

        for candidate in ext_input:
            reference_prob = disc.cond_prob(
                candidate, disc.reference_frequency, total_reference
            )
            disc.reference_conditional_probs[candidate] = reference_prob

            domain_prob = disc.cond_prob(
                candidate, disc.domain_frequency, total_domain
            )
            disc.domain_conditional_probs[candidate] = domain_prob

            dr = disc.calculate_domain_relevance(candidate)
            results.append(dr)

        self.assertListEqual(expected, results)


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)
