"""Run the whole analysis: prepare -> syntax -> embed -> cluster -> topics.

    python -m poetry_nlp.pipeline            # everything
    python -m poetry_nlp.pipeline --skip syntax
"""
import argparse
import time

from . import cluster, config, embed, prepare, syntax, topics

STEPS = [("prepare", prepare), ("syntax", syntax), ("embed", embed),
         ("cluster", cluster), ("topics", topics)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=[n for n, _ in STEPS])
    args = ap.parse_args(argv)

    if not config.RAW_CSV.exists():
        raise SystemExit(f"Missing {config.RAW_CSV.relative_to(config.ROOT)}. "
                         "Run `python scripts/download_data.py` first (see README).")
    for name, module in STEPS:
        if name in args.skip:
            continue
        t0 = time.time()
        print(f"== {name}")
        module.main([])
        print(f"== {name} done in {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
