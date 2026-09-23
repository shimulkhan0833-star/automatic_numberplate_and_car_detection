"""Preview the configured video without saving output. Press Q to stop."""
from app.pipeline.anpr_pipeline import ANPRPipeline


def main() -> None:
    ANPRPipeline(save_output=False).run()


if __name__ == "__main__":
    main()
