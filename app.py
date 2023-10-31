import os
import autogen
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface
import memgpt.agent as agent       
import memgpt.system as system
import memgpt.utils as utils 
import memgpt.presets as presets
import memgpt.constants as constants 
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithEmbeddings, InMemoryStateManagerWithFaiss
import openai

config_list = [
    {
        "api_type": "open_ai",
        "api_base": "https://ekisktiz8hegao-5001.proxy.runpod.net/v1",
        "api_key": "NULL",
    },
]

llm_config = {"config_list": config_list, "seed": 42}

# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent

USE_MEMGPT = True

## api keys for the memGPT
openai.api_base="https://ekisktiz8hegao-5001.proxy.runpod.net/v1"
openai.api_key="NULL"


# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",  # needed?
    default_auto_reply="You are going to figure all out by your own. "
    "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation.",
)


interface = autogen_interface.AutoGenInterface()
persistence_manager=InMemoryStateManager()
persona = "I am a 10x engineer, trained in Python. I was the first engineer at Uber."
human = "Im a team manager at this company"
memgpt_agent=presets.use_preset(presets.DEFAULT_PRESET, model='gpt-4', persona=persona, human=human, interface=interface, persistence_manager=persistence_manager, agent_config=llm_config)


if not USE_MEMGPT:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber",
        human_input_mode="TERMINATE",
    )

else:
    # In our example, we swap this AutoGen agent with a MemGPT agent
    # This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
    print("\nMemGPT Agent at work\n")
    coder = memgpt_autogen.MemGPTAgent(
        name="MemGPT_coder",
        agent=memgpt_agent,
    )


# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    coder,
    message="Write a Function to print Numbers 1 to 10"
    )