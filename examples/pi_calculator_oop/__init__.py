if __name__ == "__main__":
    import sys
    from .calculate_pi import PiCalculator

    calculator = PiCalculator(sys.argv[1])
    calculator.calculate(sys.argv[0])