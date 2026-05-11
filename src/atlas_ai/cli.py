from atlas_ai.system_info import get_system_info


def main() -> None:
    """Atlas AI command-line entry point."""
    info = get_system_info()

    print("Atlas AI — System Info")
    print("-" * 28)
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
