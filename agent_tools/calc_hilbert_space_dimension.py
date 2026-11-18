from agent_tools._orchestral_compat import define_tool


def calc_hilbert_space_dimension(n_qubits: int) -> int:
    return 2**n_qubits


@define_tool
def CalcHilbertSpaceDimensionTool(n_qubits: int) -> int:
    """
    Calculate the dimension of the Hilbert space for a given number of qubits.
    """
    return calc_hilbert_space_dimension(n_qubits)


if __name__ == "__main__":
    print(calc_hilbert_space_dimension(3))  # Example usage
