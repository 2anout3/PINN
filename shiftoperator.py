from __future__ import annotations
import numpy as np
import numpy.polynomial as polynomial

class shift_op:
    def __init__(self, shift, obj = 1):
        self._min_power = shift
        if isinstance(obj, polynomial.Polynomial):
            self._poly = obj
        elif isinstance(obj, (list, tuple, np.ndarray)):
            self._poly = polynomial.Polynomial(obj)
        elif isinstance(obj, (float, int, np.float64, np.integer)):
            self._poly = polynomial.Polynomial([obj])
        else:
            raise TypeError(f"Unsupported type for alpha: {type(obj)}")

    def __add__(self, other: shift_op):
        min_power = np.min([self._min_power, other._min_power])
        diff = np.abs(self._min_power - other._min_power)
        if (self._min_power < other._min_power):
            first = self._poly
            second = shift_op._increase_power(other._poly, diff)
        elif (diff == 0):
            first = self._poly
            second = other._poly
        else:
            first = shift_op._increase_power(self._poly, diff)
            second = other._poly

        return shift_op(min_power, first + second)

    def __iadd__(self, other: shift_op):
        temp = self + other
        self._min_power = temp._min_power
        self._poly = temp._poly
        return self

    def __mul__(self, other: shift_op):
        min_power = self._min_power + other._min_power
        poly = self._poly * other._poly
        return shift_op(min_power, poly)

    def __rmul__(self, alpha: np.float64):
        if (not isinstance(alpha, (int, float, np.float64))):
            raise TypeError(f"Unsupported type for alpha: {type(obj)}")
        poly = alpha * self._poly
        min_power = self._min_power
        if (alpha == 0):
            min_power = 0
        return shift_op(min_power, poly)

    def __imul__(self, other: shift_op):
        temp = self * other
        self._min_power = temp._min_power
        self._poly = temp._poly
        return self

    def __repr__(self):
        return f"shift_op(shift={self._min_power}, poly={self._poly})"

    def apply_and_store(self, array: np.ndarray, output: np.ndarray):
        shift = self._min_power
        for c in self._poly.coef:
            if (c != 0):
                shift_op._add(shift, c, array, output)
            shift += 1

    @staticmethod
    def _add(shift, k: np.float64, array: np.ndarray, output: np.ndarray):
        n = len(output)
        shift = shift % len(array)
        if (shift == 0):
            output += k * array
            return

        left = array[shift:]
        right = array[:shift]

        lo = output[:n-shift]
        ro = output[n-shift:]

        # print(f"lo = {lo}")
        # print(f"ro = {ro}")
        # print(f"k = {k}")


        try:
            with np.errstate(over='raise', invalid='raise'):
                lo += k * left
                ro += k * right
        except FloatingPointError as _:
            # print(f"lo = {lo}")
            # print(f"ro = {ro}")
            print(f"k = {k}")

    @staticmethod
    def _increase_power(poly: polynomial.Polynomial, power):
        return polynomial.Polynomial(np.concatenate([np.zeros(power), poly.coef]))
