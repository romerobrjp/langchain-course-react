from langchain.tools import tool
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from typing import Sequence, Union
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

@tool
def get_length_of_string(s: str)->int:
    """Returns the length of the input string."""
    print(f">> get_text_length enter with {s=}")
    s = s.strip("'\n").strip('"')
    return len(s)

def find_tool_by_name(tools: Sequence[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")

def main(input_str: str):
    print(">> Hello from langchain-course-react!")
    print(f">> Its length is: {get_length_of_string.invoke(input_str)}")

if __name__ == "__main__":
    main('lion')
    tools = [get_length_of_string]
    print(f">> Available tools: {[tool.name for tool in tools]}")
    template = """
    Answer the following questions as best as you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([tool.name for tool in tools]))

    llm = ChatOllama(model="qwen2.5", temperature=0, stop=["\nObservation", "Observation", "Observation:"])
    agent = { "input": lambda x: x["input"] } | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the word 'lion'?"})
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        tool_output = tool_to_use.invoke(tool_input)
        print(f">> Tool {tool_name} output: {tool_output}")
    elif isinstance(agent_step, AgentFinish):
        print(f">> Final answer: {agent_step.return_values}")
