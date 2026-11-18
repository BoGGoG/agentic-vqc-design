"""How the agent sessions of the paper were launched (reference only, needs Orchestral).

This script needs the Orchestral AI framework (arXiv:2601.02577), which is NOT part of
this repository; the development version we used is not publicly released (see README.md,
"Orchestral"). It is our launcher from that time, with the tool imports pointed at this
repository's `agent_tools/` modules. In the committed launcher the Quanvolutional tool
was still imported from an older module; the Quanvolutional session used
`train_QuanvNN.py` (its tool calls pass `kernel_size`, `stride` and `VQC_output_dim`,
and its logs went to `QuanvNN_runs_log`), which is the version shipped here.

The agent got the tools below and Orchestral's default rich-UI system prompt plus the
current date; the exact system prompt of each session is the first message of its
transcript in `results/transcripts/`. The sessions ran in Orchestral's local web app,
where the LLM was chosen per session (Claude 3.7 Sonnet via the Anthropic API, Llama
3.3 70B via Groq) and where we typed the user messages that are in the transcripts.
API keys were read from environment variables by Orchestral; none are stored here.
"""

import os
import time

import app.server as app_server  # Orchestral's web app (part of Orchestral)
from orchestral import Agent
from orchestral.llm import Claude  # the LLM was switched per session in the web app
from orchestral.prompts import RICH_UI_SYSTEM_PROMPT

from agent_tools.calc_hilbert_space_dimension import CalcHilbertSpaceDimensionTool
from agent_tools.train_FullQuantum import TrainCustomFullQuantumQNN
from agent_tools.train_QuanvNN import TrainCustomQuanvNNTool
from agent_tools.train_SimpleQNN import TrainCustomSimpleQNNTool

tools = [
    CalcHilbertSpaceDimensionTool,
    TrainCustomQuanvNNTool,
    TrainCustomSimpleQNNTool,
    TrainCustomFullQuantumQNN,
]

system_prompt = f"{RICH_UI_SYSTEM_PROMPT}\n\nThe current date is {time.strftime('%Y-%m-%d')}"
agent = Agent(llm=Claude(), tools=tools, system_prompt=system_prompt)

if __name__ == "__main__":
    os.makedirs("workspace", exist_ok=True)
    app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True)
