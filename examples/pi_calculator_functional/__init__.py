from .calculate_pi import main

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise ValueError("Too many values passed!")
    main(sys.argv[1])

