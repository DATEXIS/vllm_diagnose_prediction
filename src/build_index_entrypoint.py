import argparse
import logging
import yaml
from retriever import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rag = cfg.get("rag", {})
    if not rag.get("enabled"):
        logger.info("RAG not enabled, nothing to do.")
        return

    build_index(
        abstracts_path=rag["abstracts_path"],
        index_persist_dir=rag["index_persist_dir"],
        max_abstracts=rag.get("max_abstracts"),
    )


if __name__ == "__main__":
    main()
