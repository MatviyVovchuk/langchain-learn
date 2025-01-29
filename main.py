from typing import Union, List

import os
from dotenv import load_dotenv
# Importing the LangChain message schema:
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    AgentAction,
    AgentFinish
)

# Import ReActSingleInputOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser

# ChatOpenAI (from your local 'langchain_openai' module)
from langchain_openai import ChatOpenAI

# PromptTemplate
from langchain.prompts import PromptTemplate

# Tools
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters."""
    # Print debug message to show the incoming text
    print(f"get_text_length enter with {text=}")
    
    # Strip leading/trailing quotes and newline characters
    text = text.strip("'\n").strip('"')
    return len(text)


@tool
def check_connection_to_db(host: str) -> int:
    """Checks connection to the database and returns a connection status code."""
    # Print debug message to show the incoming host
    print(f"check_connection_to_db enter with {host=}")
    # This is a dummy value representing a successful connection
    return 200


@tool
def get_city_volume(city_name: str) -> float:
    """Calculates the volume in a given city."""
    # Print debug message to show the incoming city name
    print(f"get_city_volume enter with {city_name=}")
    # Return a dummy volume
    return 314125.23


@tool
def get_city_population(city_name: str) -> float:
    """Calculates the population in a given city."""
    # Print debug message to show the incoming city name
    print(f"get_city_population enter with {city_name=}")
    # Return a dummy population
    return 23523.23


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """
    Searches for a tool by name within a list of Tool objects.
    Raises a ValueError if the tool is not found.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    # Simple greeting to confirm code is running
    print("Hello ReAct LangChain!")
    
    # Create a list of available tools
    tools = [get_text_length, check_connection_to_db, get_city_volume, get_city_population]

    # ReAct template that instructs the LLM how to use tools
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the plain string input (no JSON)
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    # Create the base PromptTemplate with partial arguments for tools
    base_prompt_template = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # Create an instance of our custom ChatOpenAI LLM wrapper
    # (You can adjust the temperature and stop tokens as needed)
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation"],
        # model='deepseek-chat',
        # openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
        # openai_api_base='https://api.deepseek.com',
    )

    # Define a user question that will require multiple tool actions
    user_question = (
        "What is volume of city Lutsk? "
        "What is the length of 'DOGiiis'? "
        "Then check DB with host localhost. "
        "To answer you need to use more than 1 tool."
    )

    # 1) Format the system prompt using our base_prompt_template
    system_prompt_text = base_prompt_template.format(input=user_question)

    # 2) Create a list of initial conversation messages: System + Human
    conversation_messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_question),
    ]

    # Initialize the ReAct parser
    react_parser = ReActSingleInputOutputParser()

    # Print the fully rendered system prompt for debugging
    print(system_prompt_text)
    print('---')

    # Print the initial conversation messages for debugging
    print(conversation_messages)
    print('---')

    print("--- Start ReAct Loop ---")
    while True:
        # a) Invoke the LLM with the current conversation messages
        llm_response = llm.invoke(conversation_messages)
        
        # If we get an AIMessage, extract its content
        if isinstance(llm_response, AIMessage):
            llm_text = llm_response.content
        else:
            # Raise an error if we get an unexpected response type
            raise ValueError(f"Unexpected LLM return type: {type(llm_response)}")

        # Print the raw LLM output for inspection
        print("\n[LLM RAW OUTPUT]:")
        print(llm_text)

        # b) Parse the LLM output using the ReActSingleInputOutputParser
        agent_step = react_parser.parse(llm_text)

        # c) Check the type of the parsed step
        if isinstance(agent_step, AgentFinish):
            # If it's AgentFinish, the conversation is done; print final answer
            print(f"Final Answer: {agent_step.return_values['output']}")
            break

        elif isinstance(agent_step, AgentAction):
            # If it's an AgentAction, we need to call the corresponding tool
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            print(f"[Agent decided to call tool: {tool_name} with input={tool_input}]")

            # Find the specified tool by name
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input.strip()
            
            # Call the tool and get the observation
            observation = tool_to_use.func(tool_input)
            print(f"[Tool returned observation: {observation}]")

            # d) Append the Action and the Observation to the conversation as AI messages
            conversation_messages.append(
                AIMessage(content=f"Action: {tool_name}\nAction Input: {tool_input}")
            )
            conversation_messages.append(
                AIMessage(content=f"Observation: {observation}\nThought:")
            )

    print("--- ReAct Loop finished ---")
