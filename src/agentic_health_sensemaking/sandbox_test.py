import docker
from dotenv import load_dotenv
from pathlib import Path
from tempfile import TemporaryDirectory
from pydantic_ai.agent import Agent
from pydantic import BaseModel, Field
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.ollama import OllamaModel

load_dotenv()
provider = OllamaProvider(
    base_url="http://localhost:11434/v1"
)

class CodeOutput(BaseModel):
    code: str = Field(description="The python function")

model = OllamaModel(
    model_name = "s1gnature/gemma4-mtp:e2b-q4_k_m",
    provider = provider
)

agent_tooled = Agent(
    model=model,
    system_prompt="You are a python programmer. Write the scripts that you're asked and use the tools to execute them. Then report the result",
    retries=3
)

agent = Agent(
    model=model,
    system_prompt="You are a python programmer. Only provide python code. No explanations, no comments. No 'here's your code'. No examples. Just pure python code",
    output_type=CodeOutput,
    retries=2
)


sandbox = docker.from_env()

def execute_python_script(code:str) -> str:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "script.py").write_text(code)

        output = sandbox.containers.run(
            image="code-sandbox",
            auto_remove=True,
            mem_limit="512m",
            network_mode="none",
            command="python /workspace/script.py",
            volumes={str(temp_path): {"bind": "/workspace", "mode": "rw"}}
        )

    return output.decode("utf-8")



response = agent.run_sync("Write a python script that multiplies two numbers and then it divides the result by 2 and the prints the result. The numbers are 4 and 3 ")
code: CodeOutput = response.output
print(code)

sandbox_output = execute_python_script(str(code.code))
print(sandbox_output)

