from langchain.tools import tool
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

@tool
def get_length_of_string(s: str) -> int:
    """Returns the length of the input string."""
    print(f"get_text_length enter with {s=}")
    s = s.strip("'\n").strip('"')
    return len(s)

def main(str):
    print("Hello from langchain-course-react!")
    print(f"Its length is: {get_length_of_string.invoke(str)}")

if __name__ == "__main__":
    main('lion')
    tools = [get_length_of_string]
    print(f"Available tools: {[tool.name for tool in tools]}")
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

    response = agent.invoke({"input": "What is the length of the word 'lion'?"})
    print(response)
