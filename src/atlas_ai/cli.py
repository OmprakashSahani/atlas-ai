from atlas_ai.profiler.system_profiler import get_system_profile


def main() -> None:
    """Atlas AI command-line interface."""

    profile = get_system_profile()

    print("Atlas AI — System Profile")
    print("-" * 32)

    for key, value in profile.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
