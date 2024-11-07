import numpy as np
import sympy as sym

def SimplifyAndRound(expr_array, threshold=1e-6):
  '''
  Simplify the given expression and round all small coefficients to zero.

  This function calls a sequence of:

  - round_small_coefficients
  - simplify
  - and round_small_coefficients (again), for any small coefficients created by the simplify operation

  Since simplify can be slow, this method can also be very slow.

  Args:
    expr_array: An expression, either a single expression or numpy ndarray of expressions
    threshold: Coefficients smaller than the threshold will be replaced with 0 (default 1e-6)

  Returns:
    result: The expression, or array of expressions, simplified and rounded 
  '''
  expr_array = RoundSmallCoefficients(expr_array, threshold)
  expr_array = sym.simplify(expr_array)
  expr_array = RoundSmallCoefficients(expr_array, threshold)
  return expr_array

def RoundSmallCoefficients(expr_array, threshold=1e-6):
  '''
  Round all coefficients with absolute value below the threshold to zero.
  This function will work for both symbolic expressions or numpy arrays of 
  symbolic expressions. 

  This function is reasonably fast.

  Args:
    expr_array: An expression, either a single expression or numpy ndarray of expressions
    threshold: Coefficients smaller than the threshold will be replaced with 0 (default 1e-6)

  Returns:
    result: The expression, or array of expressions, with small coefficients rounded to 0
  '''
  def custom_round(expr):
    if isinstance(expr, float) or isinstance(expr ,int):
      return sym.Number(expr)
    def round_number(n):
      if abs(n) < threshold:
        return 0
      return n
    result = expr.xreplace({term: round_number(term) for term in expr.atoms(sym.Number)})
    if result == 0:
      result = sym.Number(0)
    return result

  if isinstance(expr_array, np.ndarray):
    custom_round_vec = np.vectorize(custom_round)
    return custom_round_vec(expr_array)
  else:
    return custom_round(expr_array)

def SimplifyAndPrint(expr_array, round_only=False):
  '''
  Simplify and then pretty print the input symbolic expression
  Args:
    expr_array: A numpy array of symbolic expression
    round_only: Round small coefficients, but do not execute full simplification
  '''
  if round_only:
    result = RoundSmallCoefficients(expr_array)
  else:
    result = SimplifyAndRound(expr_array)
  if isinstance(expr_array, np.ndarray):
    result = sym.Matrix(result)
  sym.pprint(result)