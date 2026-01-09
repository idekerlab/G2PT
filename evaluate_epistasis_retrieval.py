from src.utils.analysis.epistasis_retrieval_evaluation import (
    EpistasisRetrievalEvaluator,
    build_arg_parser,
    build_config_from_args,
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    evaluator = EpistasisRetrievalEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {exc}")
        import traceback

        traceback.print_exc()
        raise
