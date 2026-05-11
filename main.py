import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python main.py train ...\n"
            "  python main.py evaluate ...\n"
            "  python main.py infer ...\n"
        )

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == 'train':
        from scripts.train_offline import main as train_main
        train_main()
        return

    if command == 'evaluate':
        from scripts.evaluate_methods import main as evaluate_main
        evaluate_main()
        return

    if command == 'infer':
        from scripts.infer import main as infer_main
        infer_main()
        return

    raise SystemExit(f"Unknown command: {command}")


if __name__ == '__main__':
    main()