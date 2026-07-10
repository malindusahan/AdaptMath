import ast
from typing import Any

import sympy as sp


class SafeMathExpressionParser:
    """
    Convert restricted mathematical text into a SymPy
    expression without evaluating arbitrary Python code.
    """

    def __init__(self) -> None:
        self.allowed_functions = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "log": sp.log,
            "exp": sp.exp,
            "abs": sp.Abs,
        }

        self.allowed_constants = {
            "pi": sp.pi,
            "E": sp.E,
        }

    def parse(
        self,
        expression: str,
    ) -> sp.Expr:
        if not isinstance(
            expression,
            str,
        ):
            raise TypeError(
                "Mathematical expression must be a string."
            )

        expression = expression.strip()

        if not expression:
            raise ValueError(
                "Mathematical expression cannot be empty."
            )

        if len(expression) > 5000:
            raise ValueError(
                "Mathematical expression is too long."
            )

        try:
            syntax_tree = ast.parse(
                expression,
                mode="eval",
            )

        except SyntaxError as exc:
            raise ValueError(
                "Invalid mathematical expression."
            ) from exc

        return self._convert(
            syntax_tree.body
        )

    def _convert(
        self,
        node: ast.AST,
    ) -> sp.Expr:

        # ====================================================
        # NUMBERS
        # ====================================================

        if isinstance(
            node,
            ast.Constant,
        ):
            value = node.value

            if isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    "Boolean values are not valid "
                    "mathematical numbers."
                )

            if isinstance(
                value,
                int,
            ):
                return sp.Integer(
                    value
                )

            if isinstance(
                value,
                float,
            ):
                return sp.Float(
                    value
                )

            raise ValueError(
                "Unsupported constant."
            )

        # ====================================================
        # VARIABLES / CONSTANTS
        # ====================================================

        if isinstance(
            node,
            ast.Name,
        ):
            name = node.id

            if name in self.allowed_constants:
                return self.allowed_constants[
                    name
                ]

            if name.startswith("_"):
                raise ValueError(
                    "Invalid variable name."
                )

            return sp.Symbol(
                name
            )

        # ====================================================
        # UNARY OPERATORS
        # ====================================================

        if isinstance(
            node,
            ast.UnaryOp,
        ):
            operand = self._convert(
                node.operand
            )

            if isinstance(
                node.op,
                ast.USub,
            ):
                return -operand

            if isinstance(
                node.op,
                ast.UAdd,
            ):
                return operand

            raise ValueError(
                "Unsupported unary operator."
            )

        # ====================================================
        # BINARY OPERATORS
        # ====================================================

        if isinstance(
            node,
            ast.BinOp,
        ):
            left = self._convert(
                node.left
            )

            right = self._convert(
                node.right
            )

            if isinstance(
                node.op,
                ast.Add,
            ):
                return left + right

            if isinstance(
                node.op,
                ast.Sub,
            ):
                return left - right

            if isinstance(
                node.op,
                ast.Mult,
            ):
                return left * right

            if isinstance(
                node.op,
                ast.Div,
            ):
                return left / right

            if isinstance(
                node.op,
                ast.Pow,
            ):
                return left**right

            raise ValueError(
                "Unsupported mathematical operator."
            )

        # ====================================================
        # APPROVED FUNCTIONS
        # ====================================================

        if isinstance(
            node,
            ast.Call,
        ):
            if not isinstance(
                node.func,
                ast.Name,
            ):
                raise ValueError(
                    "Only approved mathematical functions "
                    "may be called."
                )

            function_name = node.func.id

            if (
                function_name
                not in self.allowed_functions
            ):
                raise ValueError(
                    "Unsupported mathematical function: "
                    f"{function_name}"
                )

            if node.keywords:
                raise ValueError(
                    "Keyword arguments are not supported."
                )

            arguments = [
                self._convert(
                    argument
                )
                for argument
                in node.args
            ]

            function = (
                self.allowed_functions[
                    function_name
                ]
            )

            return function(
                *arguments
            )

        raise ValueError(
            "Expression contains unsupported syntax: "
            f"{type(node).__name__}"
        )


class SymPyMathVerifier:
    """
    Deterministic mathematical computation and
    verification tools for AdaptMath.

    This class performs mathematical operations only.

    It does NOT decide:
    - teaching strategy,
    - learner ability,
    - problem complexity,
    - workflow routing,
    - pedagogical adaptation.
    """

    def __init__(self) -> None:
        self.parser = (
            SafeMathExpressionParser()
        )

    # ========================================================
    # EVALUATE / SUBSTITUTE
    # ========================================================

    def evaluate(
        self,
        expression: str,
        substitutions: (
            dict[
                str,
                str | int | float,
            ]
            | None
        ) = None,
    ) -> str:
        parsed_expression = (
            self.parser.parse(
                expression
            )
        )

        parsed_substitutions = (
            self._parse_substitutions(
                substitutions
                or {}
            )
        )

        result = (
            parsed_expression.subs(
                parsed_substitutions
            )
        )

        result = sp.simplify(
            result
        )

        return str(
            result
        )

    # ========================================================
    # EXPAND
    # ========================================================

    def expand(
        self,
        expression: str,
    ) -> str:
        parsed_expression = (
            self.parser.parse(
                expression
            )
        )

        return str(
            sp.expand(
                parsed_expression
            )
        )

    # ========================================================
    # FACTOR
    # ========================================================

    def factor(
        self,
        expression: str,
    ) -> str:
        parsed_expression = (
            self.parser.parse(
                expression
            )
        )

        return str(
            sp.factor(
                parsed_expression
            )
        )

    # ========================================================
    # SYMBOLIC EQUIVALENCE
    # ========================================================

    def equivalent(
        self,
        left_expression: str,
        right_expression: str,
    ) -> bool | None:
        left = self.parser.parse(
            left_expression
        )

        right = self.parser.parse(
            right_expression
        )

        difference = sp.simplify(
            left - right
        )

        if difference == 0:
            return True

        result = difference.equals(
            0
        )

        if result is True:
            return True

        if result is False:
            return False

        return None

    # ========================================================
    # POLYNOMIAL DEGREE
    # ========================================================

    def polynomial_degree(
        self,
        expression: str,
        variable: str = "x",
    ) -> int | None:
        self._validate_variable_name(
            variable
        )

        x = sp.Symbol(
            variable
        )

        expression_value = (
            self.parser.parse(
                expression
            )
        )

        expression_value = sp.expand(
            expression_value
        )

        if expression_value == 0:
            return None

        try:
            polynomial = sp.Poly(
                expression_value,
                x,
            )

        except sp.PolynomialError as exc:
            raise ValueError(
                "Expression is not polynomial in "
                f"{variable}."
            ) from exc

        degree = polynomial.degree()

        if degree is sp.S.NegativeInfinity:
            return None

        return int(
            degree
        )

    # ========================================================
    # POLYNOMIAL IDENTITY
    # ========================================================

    def polynomial_identity(
        self,
        left_expression: str,
        right_expression: str,
    ) -> bool | None:
        left = self.parser.parse(
            left_expression
        )

        right = self.parser.parse(
            right_expression
        )

        difference = sp.expand(
            left - right
        )

        difference = sp.simplify(
            difference
        )

        if difference == 0:
            return True

        result = difference.equals(
            0
        )

        if result is True:
            return True

        if result is False:
            return False

        return None

    # ========================================================
    # COMPLETE POLYNOMIAL IDENTITY CONSTRAINTS
    # ========================================================

    def polynomial_identity_constraints(
        self,
        left_expression: str,
        right_expression: str,
        variable: str = "x",
    ) -> dict[str, Any]:
        self._validate_variable_name(
            variable
        )

        left = self.parser.parse(
            left_expression
        )

        right = self.parser.parse(
            right_expression
        )

        difference = sp.expand(
            left - right
        )

        return (
            self._constraints_from_difference(
                difference,
                variable,
            )
        )

    # ========================================================
    # COMMUTING POLYNOMIAL CHECK
    # ========================================================

    def polynomials_commute(
        self,
        p_expression: str,
        q_expression: str,
        variable: str = "x",
    ) -> bool | None:
        self._validate_variable_name(
            variable
        )

        x = sp.Symbol(
            variable
        )

        p = self.parser.parse(
            p_expression
        )

        q = self.parser.parse(
            q_expression
        )

        p_of_q = p.subs(
            x,
            q,
        )

        q_of_p = q.subs(
            x,
            p,
        )

        difference = sp.expand(
            p_of_q - q_of_p
        )

        difference = sp.simplify(
            difference
        )

        if difference == 0:
            return True

        result = difference.equals(
            0
        )

        if result is True:
            return True

        if result is False:
            return False

        return None

    # ========================================================
    # COMMUTING POLYNOMIAL CONSTRAINTS
    # ========================================================

    def polynomial_commutativity_constraints(
        self,
        p_expression: str,
        q_expression: str,
        variable: str = "x",
    ) -> dict[str, Any]:
        self._validate_variable_name(
            variable
        )

        x = sp.Symbol(
            variable
        )

        p = self.parser.parse(
            p_expression
        )

        q = self.parser.parse(
            q_expression
        )

        p_of_q = sp.expand(
            p.subs(
                x,
                q,
            )
        )

        q_of_p = sp.expand(
            q.subs(
                x,
                p,
            )
        )

        difference = sp.expand(
            p_of_q - q_of_p
        )

        result = (
            self._constraints_from_difference(
                difference,
                variable,
            )
        )

        result[
            "p_of_q"
        ] = str(
            p_of_q
        )

        result[
            "q_of_p"
        ] = str(
            q_of_p
        )

        return result

    # ========================================================
    # SOLVE EQUATION SYSTEM
    # ========================================================

    def solve_equations(
        self,
        equations: list[
            tuple[str, str]
        ],
        variables: list[str],
    ) -> list[
        dict[str, str]
    ]:
        if not equations:
            raise ValueError(
                "At least one equation is required."
            )

        if not variables:
            raise ValueError(
                "At least one variable is required."
            )

        if len(equations) > 50:
            raise ValueError(
                "Too many equations supplied."
            )

        if len(variables) > 20:
            raise ValueError(
                "Too many variables supplied."
            )

        symbols = self._create_symbols(
            variables
        )

        sympy_equations: list[
            sp.Equality
        ] = []

        for (
            left_expression,
            right_expression,
        ) in equations:

            left = self.parser.parse(
                left_expression
            )

            right = self.parser.parse(
                right_expression
            )

            difference = sp.simplify(
                left - right
            )

            if difference == 0:
                continue

            sympy_equations.append(
                sp.Eq(
                    left,
                    right,
                )
            )

        if not sympy_equations:
            return []

        solutions = sp.solve(
            sympy_equations,
            symbols,
            dict=True,
        )

        normalized_solutions: list[
            dict[str, str]
        ] = []

        for solution in solutions:
            normalized_solution: dict[
                str,
                str,
            ] = {}

            for symbol in symbols:
                if symbol in solution:
                    normalized_solution[
                        str(symbol)
                    ] = str(
                        sp.simplify(
                            solution[
                                symbol
                            ]
                        )
                    )

            normalized_solutions.append(
                normalized_solution
            )

        return normalized_solutions

    # ========================================================
    # GROEBNER SYSTEM REDUCTION
    # ========================================================

    def groebner_reduce_system(
        self,
        equations: list[
            tuple[str, str]
        ],
        variables: list[str],
        nonzero_variables: (
            list[str]
            | None
        ) = None,
        order: str = "grevlex",
    ) -> dict[str, Any]:
        """
        Reduce a polynomial equation system exactly.

        The fast path performs only mathematically exact
        transformations:

        1. remove factors already known to be nonzero,
        2. optionally branch on one exact product of linear
           factors,
        3. propagate linear equations only where division is
           justified by a coefficient proven to be nonzero.

        If this leaves exactly one complete consistent branch,
        a small Groebner basis canonicalizes those relations.

        If preprocessing cannot safely resolve the system, the
        original full Groebner strategy is used as a fallback.
        """

        if not equations:
            raise ValueError(
                "At least one equation is required."
            )

        if not variables:
            raise ValueError(
                "At least one variable is required."
            )

        if len(equations) > 50:
            raise ValueError(
                "Too many equations supplied."
            )

        if len(variables) > 20:
            raise ValueError(
                "Too many variables supplied."
            )

        allowed_orders = {
            "lex",
            "grlex",
            "grevlex",
        }

        if order not in allowed_orders:
            raise ValueError(
                "Unsupported Groebner order. "
                "Use lex, grlex, or grevlex."
            )

        nonzero_variables = (
            nonzero_variables
            or []
        )

        if len(nonzero_variables) > len(
            variables
        ):
            raise ValueError(
                "Too many nonzero variables supplied."
            )

        variable_symbols = (
            self._create_symbols(
                variables
            )
        )

        symbol_map = {
            name: symbol
            for name, symbol
            in zip(
                variables,
                variable_symbols,
            )
        }

        for variable in nonzero_variables:
            self._validate_variable_name(
                variable
            )

            if variable not in symbol_map:
                raise ValueError(
                    "Nonzero variable must also appear "
                    "in variables: "
                    f"{variable}"
                )

        polynomial_equations: list[
            sp.Expr
        ] = []

        for (
            left_expression,
            right_expression,
        ) in equations:

            left = self.parser.parse(
                left_expression
            )

            right = self.parser.parse(
                right_expression
            )

            difference = sp.expand(
                left - right
            )

            if difference != 0:
                polynomial_equations.append(
                    difference
                )

        if not polynomial_equations:
            return {
                "basis": [],
                "relations": [],
                "equations": [],
                "nonzero_variables":
                    list(
                        nonzero_variables
                    ),
                "order":
                    order,
                "preprocessing_used":
                    True,
                "fallback_used":
                    False,
            }

        nonzero_symbols = {
            symbol_map[
                variable
            ]
            for variable
            in nonzero_variables
        }

        preprocessed = (
            self._preprocess_polynomial_system(
                polynomial_equations,
                variable_symbols,
                nonzero_symbols,
                order,
            )
        )

        if preprocessed is not None:
            relation_text = [
                str(
                    sp.factor(
                        relation
                    )
                )
                for relation
                in preprocessed
            ]

            return {
                "basis":
                    relation_text,

                "relations":
                    relation_text,

                "equations": [
                    [
                        relation,
                        "0",
                    ]
                    for relation
                    in relation_text
                ],

                "nonzero_variables":
                    list(
                        nonzero_variables
                    ),

                "order":
                    order,

                "preprocessing_used":
                    True,

                "fallback_used":
                    False,
            }

        return self._groebner_fallback(
            polynomial_equations=(
                polynomial_equations
            ),
            variable_symbols=(
                variable_symbols
            ),
            symbol_map=(
                symbol_map
            ),
            nonzero_variables=(
                nonzero_variables
            ),
            order=order,
        )

    # ========================================================
    # FAST EXACT POLYNOMIAL PREPROCESSOR
    # ========================================================

    def _preprocess_polynomial_system(
        self,
        equations: list[sp.Expr],
        variable_symbols: list[
            sp.Symbol
        ],
        nonzero_symbols: set[
            sp.Symbol
        ],
        order: str,
    ) -> list[sp.Expr] | None:
        """
        Attempt an exact inexpensive reduction before the
        general Groebner computation.

        The optimization is conservative.

        If there are multiple surviving branches or unresolved
        nonlinear equations, None is returned and the caller
        uses the full Groebner fallback.
        """

        normalized: list[
            sp.Expr
        ] = []

        for expression in equations:
            value = (
                self._normalize_polynomial_equation(
                    expression,
                    nonzero_symbols,
                )
            )

            if value == 0:
                continue

            if self._equation_is_contradiction(
                value,
                nonzero_symbols,
            ):
                # An inconsistent polynomial system is
                # represented by the unit relation 1 = 0.
                return [
                    sp.Integer(1)
                ]

            normalized.append(
                value
            )

        branch = (
            self._find_linear_factor_branch(
                normalized,
                variable_symbols,
                nonzero_symbols,
            )
        )

        candidate_states: list[
            list[sp.Expr]
        ]

        if branch is None:
            candidate_states = [
                normalized
            ]

        else:
            (
                branch_index,
                branch_factors,
            ) = branch

            remaining = (
                normalized[
                    :branch_index
                ]
                + normalized[
                    branch_index + 1:
                ]
            )

            candidate_states = [
                [
                    factor
                ]
                + remaining
                for factor
                in branch_factors
            ]

        successful_relations: list[
            list[sp.Expr]
        ] = []

        for candidate in candidate_states:
            reduced = (
                self._propagate_linear_relations(
                    candidate,
                    variable_symbols,
                    nonzero_symbols,
                )
            )

            if reduced[
                "contradiction"
            ]:
                continue

            residual = reduced[
                "residual"
            ]

            relations = reduced[
                "relations"
            ]

            # If anything nonlinear or unresolved remains,
            # do not guess. Use the general fallback.
            if residual:
                return None

            successful_relations.append(
                relations
            )

        # Exactly one consistent branch is required for the
        # fast path. Multiple valid branches are delegated to
        # the general Groebner computation.
        if len(
            successful_relations
        ) != 1:
            return None

        relations = (
            successful_relations[
                0
            ]
        )

        if not relations:
            return []

        # By this point the expected fast-path relations are
        # linear. Canonicalizing this tiny system is cheap.
        for relation in relations:
            try:
                polynomial = sp.Poly(
                    relation,
                    *variable_symbols,
                )

            except sp.PolynomialError:
                return None

            if (
                polynomial.total_degree()
                > 1
            ):
                return [
                    sp.factor(
                        item
                    )
                    for item
                    in relations
                ]

        basis = sp.groebner(
            relations,
            *variable_symbols,
            order=order,
        )

        return [
            sp.factor(
                polynomial.as_expr()
            )
            for polynomial
            in basis.polys
        ]

    # ========================================================
    # LINEAR PROPAGATION
    # ========================================================

    def _propagate_linear_relations(
        self,
        equations: list[sp.Expr],
        variable_symbols: list[
            sp.Symbol
        ],
        nonzero_symbols: set[
            sp.Symbol
        ],
    ) -> dict[str, Any]:
        working = list(
            equations
        )

        relations: list[
            sp.Expr
        ] = []

        eliminated: set[
            sp.Symbol
        ] = set()

        while True:
            normalized: list[
                sp.Expr
            ] = []

            for expression in working:
                value = (
                    self._normalize_polynomial_equation(
                        expression,
                        nonzero_symbols,
                    )
                )

                if value == 0:
                    continue

                if self._equation_is_contradiction(
                    value,
                    nonzero_symbols,
                ):
                    return {
                        "contradiction":
                            True,

                        "residual":
                            [],

                        "relations":
                            relations,
                    }

                normalized.append(
                    value
                )

            working = normalized

            substitution = (
                self._find_safe_linear_substitution(
                    working,
                    variable_symbols,
                    nonzero_symbols,
                    eliminated,
                )
            )

            if substitution is None:
                return {
                    "contradiction":
                        False,

                    "residual":
                        working,

                    "relations":
                        relations,
                }

            (
                symbol,
                replacement,
                relation,
            ) = substitution

            eliminated.add(
                symbol
            )

            relations.append(
                relation
            )

            working = [
                sp.expand(
                    expression.subs(
                        symbol,
                        replacement,
                    )
                )
                for expression
                in working
            ]

    # ========================================================
    # SAFE LINEAR SUBSTITUTION
    # ========================================================

    def _find_safe_linear_substitution(
        self,
        equations: list[sp.Expr],
        variable_symbols: list[
            sp.Symbol
        ],
        nonzero_symbols: set[
            sp.Symbol
        ],
        eliminated: set[
            sp.Symbol
        ],
    ) -> (
        tuple[
            sp.Symbol,
            sp.Expr,
            sp.Expr,
        ]
        | None
    ):
        # Prefer eliminating ordinary variables before
        # variables carrying explicit nonzero constraints.
        candidates = [
            symbol
            for symbol
            in variable_symbols
            if (
                symbol
                not in nonzero_symbols
                and symbol
                not in eliminated
            )
        ] + [
            symbol
            for symbol
            in variable_symbols
            if (
                symbol
                in nonzero_symbols
                and symbol
                not in eliminated
            )
        ]

        for expression in equations:
            expanded = sp.expand(
                expression
            )

            for symbol in candidates:
                if (
                    symbol
                    not in expanded.free_symbols
                ):
                    continue

                try:
                    polynomial = sp.Poly(
                        expanded,
                        symbol,
                    )

                except sp.PolynomialError:
                    continue

                if polynomial.degree() != 1:
                    continue

                coefficient = (
                    expanded.coeff(
                        symbol,
                        1,
                    )
                )

                remainder = sp.simplify(
                    expanded
                    - (
                        coefficient
                        * symbol
                    )
                )

                if (
                    symbol
                    in remainder.free_symbols
                ):
                    continue

                coefficient = sp.factor(
                    coefficient
                )

                # Division is permitted only when the
                # coefficient is already proven nonzero.
                if not self._is_proven_nonzero(
                    coefficient,
                    nonzero_symbols,
                ):
                    continue

                replacement = sp.factor(
                    sp.cancel(
                        -remainder
                        / coefficient
                    )
                )

                # If the eliminated variable itself was known
                # nonzero, do not replace it by an expression
                # whose nonzero status is unknown.
                if (
                    symbol in nonzero_symbols
                    and not self._is_proven_nonzero(
                        replacement,
                        nonzero_symbols,
                    )
                ):
                    continue

                relation_numerator = (
                    sp.together(
                        symbol
                        - replacement
                    )
                    .as_numer_denom()[
                        0
                    ]
                )

                relation = sp.factor(
                    relation_numerator
                )

                return (
                    symbol,
                    replacement,
                    relation,
                )

        return None

    # ========================================================
    # EXACT LINEAR-FACTOR BRANCH
    # ========================================================

    def _find_linear_factor_branch(
        self,
        equations: list[sp.Expr],
        variable_symbols: list[
            sp.Symbol
        ],
        nonzero_symbols: set[
            sp.Symbol
        ],
    ) -> (
        tuple[
            int,
            list[sp.Expr],
        ]
        | None
    ):
        """
        Find one equation that is an exact product of
        multiple linear factors.

        For:

            (a - d) * (a + d) = 0

        the exact possibilities are:

            a - d = 0

        or:

            a + d = 0

        The fast path uses only one such branching stage.
        More complicated branching is left to the general
        Groebner fallback.
        """

        for (
            index,
            expression,
        ) in enumerate(
            equations
        ):
            factored = sp.factor(
                expression
            )

            if not factored.is_Mul:
                continue

            factors: list[
                sp.Expr
            ] = []

            for raw_factor in (
                factored.args
            ):
                factor = (
                    self._zero_factor_base(
                        raw_factor
                    )
                )

                if self._is_proven_nonzero(
                    factor,
                    nonzero_symbols,
                ):
                    continue

                if factor not in factors:
                    factors.append(
                        factor
                    )

            if len(factors) < 2:
                continue

            all_linear = True

            for factor in factors:
                try:
                    polynomial = sp.Poly(
                        factor,
                        *variable_symbols,
                    )

                except sp.PolynomialError:
                    all_linear = False
                    break

                if (
                    polynomial.total_degree()
                    != 1
                ):
                    all_linear = False
                    break

            if all_linear:
                return (
                    index,
                    factors,
                )

        return None

    # ========================================================
    # NORMALIZE EQUATION
    # ========================================================

    def _normalize_polynomial_equation(
        self,
        expression: sp.Expr,
        nonzero_symbols: set[
            sp.Symbol
        ],
    ) -> sp.Expr:
        value = sp.cancel(
            sp.together(
                expression
            )
        )

        (
            numerator,
            denominator,
        ) = value.as_numer_denom()

        # A denominator can be removed only when its
        # nonzero status is already established.
        if (
            denominator != 1
            and self._is_proven_nonzero(
                denominator,
                nonzero_symbols,
            )
        ):
            value = numerator

        value = sp.factor(
            sp.expand(
                value
            )
        )

        if value == 0:
            return sp.Integer(
                0
            )

        if self._is_proven_nonzero(
            value,
            nonzero_symbols,
        ):
            # value = 0 is impossible.
            return sp.Integer(
                1
            )

        if value.is_Mul:
            remaining_factors = [
                factor
                for factor
                in value.args
                if not (
                    self._is_proven_nonzero(
                        factor,
                        nonzero_symbols,
                    )
                )
            ]

            if not remaining_factors:
                return sp.Integer(
                    1
                )

            value = sp.factor(
                sp.Mul(
                    *remaining_factors
                )
            )

        return value

    # ========================================================
    # CONTRADICTION CHECK
    # ========================================================

    def _equation_is_contradiction(
        self,
        expression: sp.Expr,
        nonzero_symbols: set[
            sp.Symbol
        ],
    ) -> bool:
        if expression == 0:
            return False

        if not expression.free_symbols:
            return bool(
                sp.simplify(
                    expression
                )
                != 0
            )

        return (
            self._is_proven_nonzero(
                expression,
                nonzero_symbols,
            )
        )

    # ========================================================
    # PROVEN NONZERO CHECK
    # ========================================================

    def _is_proven_nonzero(
        self,
        expression: sp.Expr,
        nonzero_symbols: set[
            sp.Symbol
        ],
    ) -> bool:
        expression = sp.sympify(
            expression
        )

        if expression.is_number:
            return bool(
                expression != 0
            )

        if expression in nonzero_symbols:
            return True

        if expression.is_Mul:
            return all(
                self._is_proven_nonzero(
                    factor,
                    nonzero_symbols,
                )
                for factor
                in expression.args
            )

        if expression.is_Pow:
            (
                base,
                exponent,
            ) = (
                expression.as_base_exp()
            )

            if (
                exponent.is_integer
                and exponent != 0
            ):
                return (
                    self._is_proven_nonzero(
                        base,
                        nonzero_symbols,
                    )
                )

        return False

    # ========================================================
    # ZERO FACTOR BASE
    # ========================================================

    @staticmethod
    def _zero_factor_base(
        expression: sp.Expr,
    ) -> sp.Expr:
        if expression.is_Pow:
            (
                base,
                exponent,
            ) = (
                expression.as_base_exp()
            )

            if (
                exponent.is_integer
                and exponent.is_positive
            ):
                return base

        return expression

    # ========================================================
    # GENERAL GROEBNER FALLBACK
    # ========================================================

    def _groebner_fallback(
        self,
        polynomial_equations: list[
            sp.Expr
        ],
        variable_symbols: list[
            sp.Symbol
        ],
        symbol_map: dict[
            str,
            sp.Symbol
        ],
        nonzero_variables: list[str],
        order: str,
    ) -> dict[str, Any]:
        """
        Preserve the previous general-purpose Groebner
        behaviour when the exact fast path cannot safely
        resolve the system.
        """

        equations_with_constraints = list(
            polynomial_equations
        )

        auxiliary_symbols: list[
            sp.Symbol
        ] = []

        for (
            index,
            variable,
        ) in enumerate(
            nonzero_variables
        ):
            auxiliary_symbol = sp.Symbol(
                f"nonzero_aux_{index}"
            )

            auxiliary_symbols.append(
                auxiliary_symbol
            )

            equations_with_constraints.append(
                (
                    symbol_map[
                        variable
                    ]
                    * auxiliary_symbol
                )
                - 1
            )

        generators = (
            auxiliary_symbols
            + variable_symbols
        )

        basis = sp.groebner(
            equations_with_constraints,
            *generators,
            order=order,
        )

        basis_expressions = [
            sp.factor(
                polynomial.as_expr()
            )
            for polynomial
            in basis.polys
        ]

        auxiliary_set = set(
            auxiliary_symbols
        )

        relations = [
            sp.factor(
                expression
            )
            for expression
            in basis_expressions
            if not (
                expression.free_symbols
                & auxiliary_set
            )
        ]

        return {
            "basis": [
                str(
                    expression
                )
                for expression
                in basis_expressions
            ],

            "relations": [
                str(
                    expression
                )
                for expression
                in relations
            ],

            "equations": [
                [
                    str(
                        expression
                    ),
                    "0",
                ]
                for expression
                in relations
            ],

            "nonzero_variables":
                list(
                    nonzero_variables
                ),

            "order":
                order,

            "preprocessing_used":
                False,

            "fallback_used":
                True,
        }

    # ========================================================
    # INTERNAL: COEFFICIENT EXTRACTION
    # ========================================================

    def _constraints_from_difference(
        self,
        difference: sp.Expr,
        variable: str,
    ) -> dict[str, Any]:
        self._validate_variable_name(
            variable
        )

        x = sp.Symbol(
            variable
        )

        difference = sp.expand(
            difference
        )

        if difference == 0:
            return {
                "expanded_difference":
                    "0",

                "degree":
                    None,

                "identity_already_holds":
                    True,

                "constraints":
                    [],

                "equations":
                    [],
            }

        try:
            polynomial = sp.Poly(
                difference,
                x,
            )

        except sp.PolynomialError as exc:
            raise ValueError(
                "The identity difference is not polynomial "
                f"in {variable}."
            ) from exc

        degree = polynomial.degree()

        if degree is sp.S.NegativeInfinity:
            return {
                "expanded_difference":
                    "0",

                "degree":
                    None,

                "identity_already_holds":
                    True,

                "constraints":
                    [],

                "equations":
                    [],
            }

        degree_value = int(
            degree
        )

        coefficients = (
            polynomial.all_coeffs()
        )

        constraints: list[
            dict[str, Any]
        ] = []

        equations: list[
            list[str]
        ] = []

        for (
            index,
            coefficient,
        ) in enumerate(
            coefficients
        ):
            power = (
                degree_value
                - index
            )

            simplified_coefficient = (
                sp.factor(
                    sp.simplify(
                        coefficient
                    )
                )
            )

            coefficient_text = str(
                simplified_coefficient
            )

            already_zero = (
                simplified_coefficient
                == 0
            )

            constraints.append(
                {
                    "power":
                        power,

                    "coefficient":
                        coefficient_text,

                    "equation":
                        (
                            f"{coefficient_text} = 0"
                        ),

                    "already_zero":
                        already_zero,
                }
            )

            if not already_zero:
                equations.append(
                    [
                        coefficient_text,
                        "0",
                    ]
                )

        return {
            "expanded_difference":
                str(
                    difference
                ),

            "degree":
                degree_value,

            "identity_already_holds":
                False,

            "constraints":
                constraints,

            "equations":
                equations,
        }

    # ========================================================
    # SUBSTITUTION HELPERS
    # ========================================================

    def _parse_substitutions(
        self,
        substitutions: dict[
            str,
            str | int | float,
        ],
    ) -> dict[
        sp.Symbol,
        Any,
    ]:
        parsed: dict[
            sp.Symbol,
            Any,
        ] = {}

        for (
            variable,
            value,
        ) in substitutions.items():

            self._validate_variable_name(
                variable
            )

            symbol = sp.Symbol(
                variable
            )

            if isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    "Boolean substitution values "
                    "are not supported."
                )

            if isinstance(
                value,
                int,
            ):
                parsed_value = (
                    sp.Integer(
                        value
                    )
                )

            elif isinstance(
                value,
                float,
            ):
                parsed_value = (
                    sp.Float(
                        value
                    )
                )

            elif isinstance(
                value,
                str,
            ):
                parsed_value = (
                    self.parser.parse(
                        value
                    )
                )

            else:
                raise TypeError(
                    "Unsupported substitution value."
                )

            parsed[
                symbol
            ] = parsed_value

        return parsed

    # ========================================================
    # SYMBOL CREATION
    # ========================================================

    def _create_symbols(
        self,
        variables: list[str],
    ) -> list[sp.Symbol]:
        seen: set[
            str
        ] = set()

        symbols: list[
            sp.Symbol
        ] = []

        for variable in variables:
            self._validate_variable_name(
                variable
            )

            if variable in seen:
                raise ValueError(
                    "Duplicate variable supplied: "
                    f"{variable}"
                )

            seen.add(
                variable
            )

            symbols.append(
                sp.Symbol(
                    variable
                )
            )

        return symbols

    # ========================================================
    # VARIABLE VALIDATION
    # ========================================================

    @staticmethod
    def _validate_variable_name(
        variable: str,
    ) -> None:
        if not isinstance(
            variable,
            str,
        ):
            raise TypeError(
                "Variable name must be a string."
            )

        if not variable.isidentifier():
            raise ValueError(
                f"Invalid variable name: {variable}"
            )

        if variable.startswith("_"):
            raise ValueError(
                f"Invalid variable name: {variable}"
            )