"""Application entry point: run with python -m app.main."""
from app.pipeline.anpr_pipeline import ANPRPipeline


def main() -> None:
    ANPRPipeline().run()


if __name__ == "__main__":
    main()
