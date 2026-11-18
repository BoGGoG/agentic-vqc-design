"""Use Orchestral's ``define_tool`` if Orchestral is installed, otherwise a no-op.

The agent sessions of the paper ran inside the Orchestral AI framework, which is not
part of this repository. Without Orchestral, the tool functions in this folder are
plain Python functions and can be called directly with keyword arguments, e.g.
``TrainCustomSimpleQNNTool(VQC_code=..., VQC_weights_shape=[...], q_enc_size=5,
q_out_size=5, epochs=7)``.
"""

try:
    from orchestral import define_tool  # noqa: F401
except ImportError:  # Orchestral not installed

    def define_tool(func):
        return func
