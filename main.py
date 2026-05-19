import argparse


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate or visualize Burgers PINN/NFD HDF5 data."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("generate", help="Create HDF5 data for visualization.")
    subparsers.add_parser("visualize", help="Visualize data from an HDF5 file.")
    args, rest = parser.parse_known_args()
    return args, rest


def main():
    args, rest = parseArgs()
    if args.command == "visualize":
        import visualize_data

        visualize_data.main(rest)
    else:
        import generate_data

        generate_data.main(rest)


if __name__ == "__main__":
    main()
