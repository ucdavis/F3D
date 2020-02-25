if __name__ == "__main__":
    import sys
    from .calculate_pi import main

    if len(sys.argv) > 1:
        raise ValueError("Too many values passed!")
    main(sys.argv[1])