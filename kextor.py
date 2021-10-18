import multiprocessing as mp
import os
from pathlib import Path

from nltk.corpus import reuters

from src.cli import parse_arguments
from src.extractor import Extractor
from src.discriminator import Discriminator
import src.utils as ut


def main():
    # collect arguments
    args = parse_arguments()
    path = args.domain
    min_sen = args.min_sen
    max_cap = args.max_cap
    min_tok = args.min_tok
    max_tok = args.max_tok
    min_freq = args.min_freq
    not_paragraph = args.not_paragraphed
    reference = args.reference
    alpha = args.alpha
    theta = args.theta
    validation = args.validation
    verbose = args.verbose
    single = args.single

    if not os.path.isdir(path):
        print("invalid PATH")
        return False

    errors = []
    # collect files
    files = ut.retrieve_files(path)

    if len(files) == 0:
        print("Empty directory")
        return False

    # EXTRACT CANDIDATES
    extractor = Extractor(
        min_sen,
        max_cap,
        min_tok,
        max_tok,
        not_paragraph,
        validation
    )

    if single:
        candidates, errors = extractor.single(files, verbose=True)
    else:
        candidates, errors = extractor.multi(files)

    # DISCRIMINATE CANDIDATES
    discriminator = Discriminator(candidates, min_freq)

    # read reference corpus, default reuters
    if reference is None:
        reference_files = reuters.fileids()
        for i, document in enumerate(reference_files):
            text = reuters.raw(document)
            discriminator.find_reference_frequence(text)
            ut.progress_bar(i+1, len(reference_files),
                            prefix="Counting", fixed_len=True)

    # reference was given by user, read every file
    else:
        reference_files = ut.retrieve_files(reference)
        for i, filepath in enumerate(reference_files):
            try:
                with open(filepath, "r", encoding="utf-8") as rfile:
                    for line in rfile:
                        line = line.strip()
                        if len(line) > 0:
                            discriminator.find_reference_frequence(line)
            except (OSError, UnicodeDecodeError):
                errors.append(filepath)

            ut.progress_bar(i+1, len(reference_files),
                            prefix="Counting", fixed_len=True)

    discriminator.calculate_dr_dc()
    discriminator.generate_list(alpha, theta)

    # SAVE OUTPUT
    # make sure output dir exists to avoid errors
    if not os.path.isdir("output"):
        os.mkdir("output")

    # remove already existing file
    output_path = Path("output/keywords.txt")
    if os.path.isfile(output_path):
        os.remove(output_path)

    # sort and save final candidates
    candidate_list = discriminator.final_candidates
    candidate_list = sorted(
        candidate_list,
        key=lambda x: x[-1],
        reverse=True
    )

    with open(output_path, "a", encoding="utf-8") as ofile:
        ofile.write(f"# alpha\t{alpha}\n")
        ofile.write(f"# theta\t{theta}\n")
        for word, f in candidate_list:
            ofile.write(f"{word}\t{round(f, 6)}\n")

    # ERROR
    if errors:
        print("\nThe following file(s) could not be opened:")
        for error in errors:
            print(f"\t- {error}")

    # save files for evaluation + rejected
    if verbose:
        paths = (
            Path("output/domain_consensus.txt"),
            Path("output/domain_relevance.txt"),
            Path("output/rejected.txt")
        )

        # remove old files
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)

        # sort list of rejected candidates
        candidate_list = discriminator.rejected_candidates
        candidate_list = sorted(
            candidate_list,
            key=lambda x: x[-1],
            reverse=True
        )

        rej_path = Path("output/rejected.txt")
        with open(rej_path, "a", encoding="utf-8") as ofile:
            ofile.write(f"# alpha\t{alpha}\n")
            ofile.write(f"# theta\t{theta}\n")
            for word, f in candidate_list:
                ofile.write(f"{word}\t{round(f, 6)}\n")

        dc_path = Path("output/domain_consensus.txt")
        with open(dc_path, "a", encoding="utf-8") as cf:
            for word, value in discriminator.domain_consensus.items():
                cf.write(f"{word}\t{round(value, 6)}\n")

        dr_path = Path("output/domain_relevance.txt")
        with open(dr_path, "a", encoding="utf-8") as cr:
            for word, value in discriminator.domain_relevance.items():
                cr.write(f"{word}\t{round(value, 6)}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
