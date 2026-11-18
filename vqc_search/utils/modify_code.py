import ast
from typing import Optional


def code_str_return_state(
    code_str: str,
    *,
    qnode_name: str = "circuit",
    replace_all_returns: bool = True,
    return_density_matrix: bool = False,
    density_matrix_wires_expr: Optional[str] = None,
) -> str:
    """
    Transform `code_str` so that the QNode function `qnode_name` returns the final
    quantum state (qml.state()) instead of its original return value.

    - By default replaces the return statement(s) inside `def circuit(...):`
      with `return qml.state()`.
    - Optionally returns density matrix via `qml.density_matrix(wires=...)`.

    Parameters
    ----------
    code_str:
        Source code defining at least one qml.qnode function.
    qnode_name:
        Name of the QNode function to rewrite (default: "circuit").
    replace_all_returns:
        If True, replace every `return ...` inside that function.
        If False, replace only the last return encountered.
    return_density_matrix:
        If True, rewrite to `return qml.density_matrix(wires=<expr>)`.
    density_matrix_wires_expr:
        Python expression string inserted as the `wires=` argument, e.g. "range(4)" or "[0,1,2,3]".
        Required if return_density_matrix=True.

    Returns
    -------
    Modified code string.

    Notes / Limitations
    -------------------
    - This is an AST rewrite: it does not attempt to preserve original formatting exactly.
    - If your circuit has multiple early returns and replace_all_returns=False, only the last return
      is replaced (which may leave earlier returns intact).
    - `qml.state()` typically requires a simulator with `shots=None`.

    Raises
    ------
    ValueError if the named function is not found, or if density_matrix_wires_expr is missing when needed.
    RuntimeError if parsing/unparsing fails.
    """
    if return_density_matrix and not density_matrix_wires_expr:
        raise ValueError(
            "density_matrix_wires_expr must be provided when return_density_matrix=True, "
            "e.g. density_matrix_wires_expr='range(4)'."
        )

    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise RuntimeError(f"code_str is not valid Python: {e}") from e

    class ReturnRewriter(ast.NodeTransformer):
        def __init__(self):
            self.in_target_func = False
            self.found_target = False
            self.returns_in_target = 0
            self._seen_returns = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            # Only rewrite the chosen function
            if node.name != qnode_name:
                return self.generic_visit(node)

            self.found_target = True
            prev = self.in_target_func
            self.in_target_func = True

            # First, rewrite children
            node = self.generic_visit(node)

            # If we only want to replace last return, undo rewrites except last
            if not replace_all_returns and self._seen_returns:
                # _seen_returns contains references to Return nodes after visit,
                # in visitation order. We replace only the last by editing the AST again:
                last_return = self._seen_returns[-1]
                for r in self._seen_returns[:-1]:
                    r.value = r._original_value  # type: ignore[attr-defined]
                # ensure last return is state (it already is)
                _ = last_return

            self.in_target_func = prev
            return node

        def visit_Return(self, node: ast.Return):
            if not self.in_target_func:
                return node

            # keep original for "replace only last"
            node._original_value = node.value  # type: ignore[attr-defined]
            self._seen_returns.append(node)

            if return_density_matrix:
                # Build: qml.density_matrix(wires=<expr>)
                wires_ast = ast.parse(density_matrix_wires_expr, mode="eval").body
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="qml", ctx=ast.Load()),
                        attr="density_matrix",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[ast.keyword(arg="wires", value=wires_ast)],
                )
            else:
                # Build: qml.state()
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="qml", ctx=ast.Load()),
                        attr="state",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )

            node.value = new_value
            self.returns_in_target += 1
            return node

    rewriter = ReturnRewriter()
    tree2 = rewriter.visit(tree)
    ast.fix_missing_locations(tree2)

    if not rewriter.found_target:
        raise ValueError(f"Function '{qnode_name}' was not found in code_str.")

    if rewriter.returns_in_target == 0:
        raise ValueError(
            f"Function '{qnode_name}' contains no return statement to rewrite."
        )

    try:
        # Python 3.9+
        new_code = ast.unparse(tree2)
    except Exception as e:
        raise RuntimeError(
            f"Failed to unparse modified AST: {type(e).__name__}: {e}"
        ) from e

    return new_code
