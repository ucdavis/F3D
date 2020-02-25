if __name__ == "__main__":
    import sys
    from .pi_calculator import PiCalculator

    calculator = PiCalculator(sys.argv[1])
    calculator.calculate(sys.argv[0])