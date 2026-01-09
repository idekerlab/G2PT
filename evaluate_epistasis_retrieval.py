from src.utils.analysis.epistasis_retrieval_evaluation import (
    build_arg_parser,
    build_evaluator_from_args,
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluator = build_evaluator_from_args(args)
    evaluator.evaluate(top_n_systems=args.top_n_systems)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {exc}")
        import traceback

        traceback.print_exc()
        raise
