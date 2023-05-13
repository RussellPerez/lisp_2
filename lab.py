"""
6.1010 Spring '23 Lab 12: LISP Interpreter Part 2
"""
#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    lines = source.split("\n")
    result = []

    for line in lines:
        started = False
        for char in line:
            if char == ";":
                break
            elif char == " ":
                started = False
            elif char in ["(", ")"]:
                result.append(char)
            elif not started:
                result.append(char)
                started = True
            else:
                result[-1] += char

    return result


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    if isinstance(tokens, str):
        return tokens
    elif len(tokens) == 1 and tokens[0] not in ["(", ")"]:
        return number_or_symbol(tokens[0])

    if syntax_error(tokens):
        raise SchemeSyntaxError

    def parse_recursion(index):
        if tokens[index] not in ["(", ")"]:
            return (number_or_symbol(tokens[index]), index)
        s_expression = []
        while tokens[index + 1] != ")":
            symbol, end_index = parse_recursion(index + 1)
            s_expression.append(symbol)
            index = end_index
        return (s_expression, index + 1)

    return parse_recursion(0)[0]


def syntax_error(tokens):
    """
    determines if a syntax error exists.
    Used for parsing
    """
    open_parens = 0
    close_parens = 0

    for token in tokens:
        if token == "(":
            open_parens += 1
        elif token == ")":
            close_parens += 1

    return open_parens != close_parens or tokens[-1] != ")"


######################
# Built-in Functions #
######################


class BuiltinFrame:
    """
    base class that holds the builtin functions like add and subtract
    """

    def __init__(self):
        self.scheme_builtins = {
            "+": sum,
            "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
            "*": lambda args: args[0] * self.scheme_builtins["*"](args[1:])
            if args
            else 1,
            "/": lambda args: args[0]
            if len(args) == 1
            else self.scheme_builtins["/"](args[:-1]) / args[-1],
            "#t": True,
            "#f": False,
            "equal?": lambda args: len(set(args)) <= 1,
            ">": lambda args: all(args[i] > args[i + 1] for i in range(len(args) - 1)),
            ">=": lambda args: all(
                args[i] >= args[i + 1] for i in range(len(args) - 1)
            ),
            "<": lambda args: all(args[i] < args[i + 1] for i in range(len(args) - 1)),
            "<=": lambda args: all(
                args[i] <= args[i + 1] for i in range(len(args) - 1)
            ),
            "and": all,
            "or": lambda args: True in args,
            "not": lambda args: not args[0],
            "car": lambda args: args[0].get_car(),
            "cdr": lambda args: args[0].get_cdr(),
            "nil": None,
            "list": lambda args: Pair(args[0], self.scheme_builtins["list"](args[1:]))
            if args
            else None,
            "list?": lambda args: (args[0] is None)
            or (
                isinstance(args[0], Pair)
                and self.scheme_builtins["list?"]([args[0].get_cdr()])
            ),
            "length": lambda args: 0
            if args[0] is None
            else 1 + self.scheme_builtins["length"]([args[0].get_cdr()]),
            "list-ref": lambda args: args[0].get_car()
            if args[1] == 0
            else self.scheme_builtins["list-ref"]([args[0].get_cdr(), args[1] - 1]),
            "append": self.append,
            "map": self.map,
            "filter": self.filter,
            "reduce": self.reduce,
            "begin": lambda args: args[-1],
        }

    def map(self, args):
        """
        function to map a function to set of values in a list
        """
        result = []
        pair = args[1]
        new_value = 0
        while pair is not None:
            if isinstance(args[0], UserDefinedFunction):
                new_frame = FunctionFrame(args[0].enclosing_frame)
                new_frame.bind(args[0], [pair.get_car()])
                new_value = evaluate(args[0].body, new_frame)
            else:
                new_value = args[0]([pair.get_car()])
            result.append(new_value)
            pair = pair.get_cdr()

        return self.scheme_builtins["list"](result)

    def filter(self, args):
        """
        function to filter the set of values in a list depending on the
        truth value in its function
        """
        truth_list = self.map(args)
        result = []
        pair = args[1]

        while pair is not None:
            if truth_list.get_car():
                result.append(pair.get_car())
            truth_list = truth_list.get_cdr()
            pair = pair.get_cdr()

        return self.scheme_builtins["list"](result)

    def reduce(self, args):
        """
        Funciton used to reduce the arguments
        """
        pair = args[1]
        first = args[2]
        while pair is not None:
            if isinstance(args[0], UserDefinedFunction):
                new_frame = FunctionFrame(args[0].enclosing_frame)
                new_frame.bind(args[0], [first, pair.get_car()])
                first = evaluate(args[0].body, new_frame)
            else:
                first = args[0]([first, pair.get_car()])
            pair = pair.get_cdr()

        return first

    def append(self, args):
        """
        Function used to append the arguments
        """
        if not args:
            return None
        elif args[0] is None:
            return self.append(args[1:])
        else:
            return Pair(args[0].get_car(), self.append([args[0].get_cdr()] + args[1:]))


##############
# Evaluation #
##############


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if frame is None:
        frame = GlobalFrame()

    try:
        return _extracted_from_evaluate_14(tree, frame)
    except (KeyError, SchemeNameError) as e:
        print(tree)
        raise SchemeNameError from e
    except Exception as exc:
        print(exc)
        raise SchemeEvaluationError from exc


def _extracted_from_evaluate_14(tree, frame):
    """
    used as main operations of evaluate
    """
    if isinstance(tree, (int, float)):
        return tree
    elif isinstance(tree, str):
        return frame.get(tree)
    elif tree[0] in ("define", "lambda", "if", "cons", "del", "let", "set!"):
        return special_form(tree, frame)
    
    return array_form(tree, frame)

def array_form(tree, frame):
    """
    Array_form
    """
    func = evaluate(tree[0], frame)
    if tree[0] in ("and", "or"):
        for expression in tree[1:]:
            if tree[0] == "and" and (not evaluate(expression, frame)):
                return False
            elif tree[0] == "or" and evaluate(expression, frame):
                return True
    args = [evaluate(expression, frame) for expression in tree[1:]]
    if isinstance(func, UserDefinedFunction):
        new_frame = FunctionFrame(func.enclosing_frame)
        new_frame.bind(func, args)
        return evaluate(func.body, new_frame)
    elif tree[0] in ("not", "car", "cdr") and len(args) != 1:
        raise SchemeEvaluationError
    elif tree[0] == "length" and not frame.get("list?")(args):
        raise SchemeEvaluationError
    return func(args)


def special_form(tree, frame):
    """
    returns value based on special form
    """
    if tree[0] == "define":
        if isinstance(tree[1], list):
            return frame.define(
                tree[1][0], evaluate(["lambda"] + [tree[1][1:]] + [tree[2]], frame)
            )
        return frame.define(tree[1], evaluate(tree[2], frame))
    elif tree[0] == "lambda":
        return UserDefinedFunction(tree[1], tree[2], frame)
    elif tree[0] == "if":
        return (
            evaluate(tree[2], frame)
            if evaluate(tree[1], frame)
            else evaluate(tree[3], frame)
        )
    elif tree[0] == "cons":
        if len(tree[1:]) != 2:
            raise SchemeEvaluationError
        return Pair(evaluate(tree[1], frame), evaluate(tree[2], frame))
    elif tree[0] == "del":
        try:
            return frame.delete(tree[1])
        except Exception as exc:
            raise SchemeNameError from exc
    elif tree[0] == "let":
        values = [evaluate(expression[1], frame) for expression in tree[1]]
        new_frame = Frame([expression[0] for expression in tree[1]], values, frame)
        return evaluate(tree[2], new_frame)
    elif tree[0] == "set!":
        value = evaluate(tree[2], frame)
        return frame.set(tree[1], value)


class Pair:
    """
    Pair class
    """

    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def get_car(self):
        return self.car

    def get_cdr(self):
        return self.cdr


class GlobalFrame(BuiltinFrame):
    """
    The global frame that is the frame where variables are set and defined
    """

    def __init__(self, variables=None):
        super().__init__()
        if variables is None:
            variables = {}
        self.variables = variables

    def define(self, variable, value):
        self.variables.update({variable: value})
        return value

    def get(self, variable):
        try:
            return self.variables[variable]
        except KeyError:
            return self.scheme_builtins[variable]

    def delete(self, variable):
        try:
            return self.variables.pop(variable)
        except Exception as e:
            raise SchemeNameError from e

    def set(self, var, value):
        if var in self.variables:
            self.variables[var] = value
        elif var in self.scheme_builtins:
            self.scheme_builtins[var] = value
        else:
            raise SchemeNameError
        return value


class FunctionFrame(BuiltinFrame):
    """
    Function frame that handles function execution
    """

    def __init__(self, frame):
        super().__init__()
        self.parent_frame = frame
        self.params = {}

    def bind(self, func, args):
        if len(func.params) != len(args):
            raise SchemeEvaluationError
        for i, param in enumerate(func.params):
            self.params.update({param: args[i]})

    def define(self, var, val):
        self.params[var] = val
        return val

    def get(self, variable):
        try:
            return self.params[variable]
        except KeyError:
            return self.parent_frame.get(variable)

    def delete(self, variable):
        try:
            return self.params.pop(variable)
        except Exception as e:
            raise SchemeNameError from e

    def set(self, var, value):
        try:
            if var in self.params:
                self.params[var] = value
            else:
                self.parent_frame.set(var, value)
        except Exception as exc:
            raise SchemeNameError from exc
        return value


class Frame(GlobalFrame):
    """
    Frame class
    """

    def __init__(self, variables, values, parent_frame):
        super().__init__()
        self.parent_frame = parent_frame
        self.local_vars = {}
        for i in range(len(variables)):
            self.local_vars[variables[i]] = values[i]

    def get(self, variable):
        try:
            return self.local_vars[variable]
        except KeyError:
            return self.parent_frame.get(variable)

    def set(self, var, value):
        try:
            if var in self.local_vars:
                self.local_vars[var] = value
            else:
                self.parent_frame.set(var, value)
        except Exception as exc:
            raise SchemeNameError from exc
        return value


class UserDefinedFunction:
    """
    Function class that is for user defuned functions
    """

    def __init__(self, params, body, enclosing_frame):
        self.body = body
        self.params = params
        self.enclosing_frame = enclosing_frame


def evaluate_file(file_name, frame=None):
    """
    Evaluate File of file
    """
    with open(file_name, "r") as file:
        return evaluate(parse(tokenize("".join(file.readlines()))), frame)


def result_and_frame(tree, frame=None):
    """
    to be used for the REPL function
    """
    if frame is None:
        frame = GlobalFrame()

    return (evaluate(tree, frame), frame)


def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print
    out the result. Repeat until user inputs "QUIT"

    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback

    _, frame = result_and_frame(["+"])  # make a global frame
    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, frame = result_and_frame(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    repl(True)
