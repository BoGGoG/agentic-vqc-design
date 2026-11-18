import ast
import builtins
import hashlib
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np


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


class UnsafeCodeError(Exception):
    pass


def _assert_no_imports(code_str: str) -> None:
    """Reject import statements. We inject qml/np ourselves."""
    tree = ast.parse(code_str, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError(
                "Imports are not allowed in the provided code string."
            )


def create_pennylane_circuit(
    code_str: str,
    *,
    inject_np: bool = True,
    prefer_name: str = "circuit",
) -> Tuple[qml.QNode, Dict[str, Any]]:
    """
        Execute PennyLane code (without imports) in an isolated namespace and return a QNode plus metadata.

        Contract for `code_str`:
          - Must NOT contain any `import` statements.
          - Will have `qml` (and optionally `np`) available in its globals.
          - Should define at least one `@qml.qnode(dev)` function.

        Returns:
          (qnode, meta) where meta includes:
            - name: chosen QNode name
            - device: qml.Device
            - wires: list/int of wires (when available)
            - code_sha256: hash of the code string

        Example code_str (watch out for indentation!):
    dev = qml.device("default.qubit", wires=4, shots=None)

    @qml.qnode(dev)
    def circuit(params):
        for i in range(4):
            qml.RY(params[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    """
    _assert_no_imports(code_str)
    # Isolated namespace: only what we intentionally expose
    local_ns: Dict[str, Any] = {"qml": qml}
    if inject_np:
        local_ns["np"] = np  # typical agent code does "np.array(...)" etc.

    try:
        exec(code_str, local_ns)
    except Exception as e:
        raise RuntimeError(
            f"Failed to execute code string: {type(e).__name__}: {e}"
        ) from e

    # Collect QNodes
    qnodes = {name: obj for name, obj in local_ns.items() if isinstance(obj, qml.QNode)}
    if not qnodes:
        raise ValueError("No qml.QNode was defined by the provided code string.")

    # Pick a QNode: prefer `prefer_name`, else first by name
    if prefer_name in qnodes:
        qnode_name = prefer_name
    else:
        qnode_name = sorted(qnodes.keys())[0]
    qnode = qnodes[qnode_name]

    # Try to recover device & wires
    dev = getattr(qnode, "device", None)
    wires = getattr(dev, "wires", None) if dev is not None else None

    meta = {
        "name": qnode_name,
        "device": dev,
        "wires": wires,
        "code_sha256": hashlib.sha256(code_str.encode("utf-8")).hexdigest(),
    }
    return qnode, meta


def make_quantum_layer_custom_code(custom_VQC_code: str, weights_shape):
    circuit, meta = create_pennylane_circuit(
        code_str=custom_VQC_code,
        inject_np=True,
        prefer_name="circuit",
    )

    weight_shapes = {"weights": tuple(weights_shape)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return qlayer


def create_pennylane_circuit_jax(
    code_str: str,
    *,
    inject_np: bool = True,
    prefer_name: str = "circuit",
    interface: str = "jax",
    diff_method: str = "backprop",
) -> Tuple[qml.QNode, Dict[str, Any]]:
    """
    Execute PennyLane code (no imports) and return a JAX-configured QNode plus metadata.

    Contract for `code_str`:
      - Must NOT contain imports.
      - Must define a qml.Device named `dev` (or create a qnode with an internal device).
      - Must define at least one @qml.qnode(...) function.
      - Recommended signature: circuit(x, weights) for hybrid training.
    """
    _assert_no_imports(code_str)

    # Restrict builtins: treat code_str as trusted input anyway, but this reduces foot-guns.
    safe_builtins = {
        "range": builtins.range,
        "len": builtins.len,
        "min": builtins.min,
        "max": builtins.max,
        "sum": builtins.sum,
        "abs": builtins.abs,
        "float": builtins.float,
        "int": builtins.int,
        "bool": builtins.bool,
        "list": builtins.list,
        "tuple": builtins.tuple,
        "dict": builtins.dict,
        "enumerate": builtins.enumerate,
    }

    local_ns: Dict[str, Any] = {"qml": qml, "__builtins__": safe_builtins}
    if inject_np:
        # Many circuit snippets use np.*; mapping to jnp keeps things JAX-traceable.
        local_ns["np"] = jnp

    try:
        exec(code_str, local_ns)
    except Exception as e:
        raise RuntimeError(
            f"Failed to execute code string: {type(e).__name__}: {e}"
        ) from e

    qnodes = {name: obj for name, obj in local_ns.items() if isinstance(obj, qml.QNode)}
    if not qnodes:
        raise ValueError("No qml.QNode was defined by the provided code string.")

    qnode_name = prefer_name if prefer_name in qnodes else sorted(qnodes.keys())[0]
    qnode = qnodes[qnode_name]

    # Rewrap to enforce JAX interface/diff method (so code_str can stay backend-agnostic)
    dev = getattr(qnode, "device", None)
    if dev is None:
        raise ValueError("Could not recover device from QNode (qnode.device is None).")

    qnode_jax = qml.QNode(qnode.func, dev, interface=interface, diff_method=diff_method)

    wires = getattr(dev, "wires", None)
    meta = {
        "name": qnode_name,
        "device": dev,
        "wires": wires,
        "code_sha256": hashlib.sha256(code_str.encode("utf-8")).hexdigest(),
        "interface": interface,
        "diff_method": diff_method,
    }
    return qnode_jax, meta


if __name__ == "__main__":
    # Agent-provided snippet (NO imports; qml, np are injected)
    code_str = """
dev = qml.device("default.qubit", wires=4, shots=None)

@qml.qnode(dev)
def circuit(params):
    for i in range(4):
        qml.RY(params[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
"""

    qnode, meta = create_pennylane_circuit(code_str, inject_np=True)
    print("Loaded QNode:", meta["name"])
    print("Device:", meta["device"])
    print("Wires:", meta["wires"])
    print("Code hash:", meta["code_sha256"])

    theta = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    print("Output:", qnode(theta))
