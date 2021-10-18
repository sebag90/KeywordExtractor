import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domain", metavar="DOMAIN", action="store",
        help="Path to the domain corpus"
    )

    parser.add_argument(
        "--reference", metavar="REF",
        action="store", help="Path to the reference Corpus"
    )

    parser.add_argument(
        "--min_sen", metavar="N", action="store",
        help="Minimum ammount of sentences per paragraph "
        "(Default: %(default)s)", default=2, type=int
    )

    parser.add_argument(
        "--max_cap", metavar="N", action="store", type=int,
        default=70, help="Max ammount of capitalized words "
        "per paragraph (Default: %(default)s)"
    )

    parser.add_argument(
        "--min_tok", metavar="N", action="store",
        type=int, default=5,
        help="Minimum ammount of tokens per paragraph "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--max_tok", metavar="N", action="store",
        type=int, default=20,
        help="Maximum ammount of sentences per paragraph "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--min_freq", metavar="N", action="store",
        type=int, default=25,
        help="Minimum absolute frequence of a candidate "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--alpha", metavar="N", action="store",
        type=float, default=0.99,
        help="Alpha value for the discriminator "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--theta", metavar="N", action="store",
        type=float, default=0.6,
        help="Theta value for the discriminator "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--not-paragraphed", action="store_true", default=False,
        help="Domain corpus has NOT one paragraph per line "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--validation", action="store_true", default=False,
        help="Validates candidates with a dictionary "
        "(Default: %(default)s)"
    )

    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Save domain relevance, consensus and rejected "
        "candidates (Default: %(default)s)"
    )

    parser.add_argument(
        "--single", action="store_true",
        help="disable multiprocessing"
    )

    args = parser.parse_args()
    return args
