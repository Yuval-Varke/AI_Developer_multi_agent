from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import subprocess, sys
from langchain.tools import tool
load_dotenv()
import os

model = ChatMistralAI(
    model="mistral-medium-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

coder_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
    You are experienced software developer.
    You write python code with proper understanding of the problem.
    You write code with proper comments and docstrings.
    Your mindset of edge cases and error handling while writing code is very good.
"""
)

planner_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
    You are an experienced software architect.
    You have a good understanding of software development lifecycle and project management.
    Your mindset is to break down complex problems into smaller, manageable tasks and to create a clear plan for solving the problem.
"""
)


@tool
def execute_code(code: str) -> str:
    """"Use this tool to execute python code and find out the code is working or not.
    
    Args:
        code (str): The python code to be executed.
    """

    result = subprocess.run(
        [sys.executable, "-c", code], 
        capture_output=True, 
        text=True,
        timeout=30
    )
    return str({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    })


tester_agent = create_agent(
    model=model,
    tools=[execute_code],
    system_prompt="""
    You are an experienced software tester.
    You have a good understanding of software testing methodologies and best practices.
    Your mindset is to write test cases that cover all possible scenarios and edge cases, and to execute the code to find out if it is working or not.
    """
)
