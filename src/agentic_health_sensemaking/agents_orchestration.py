import time
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic import BaseModel, Field
from agents_configuration import load_agents, SensemakingDeps
import docker

planner_agent, coding_agent, sensemaking_agent, presentation_agent = load_agents()

uid = "8391d8b451084771d0affcc26e34d773"
prompt = f"User {uid} said he was anxious in the morning of 12th June 2018. Why could that be?"
shared_memory = SensemakingDeps(user_query=prompt)

total_start_time = time.perf_counter()

# Planner Agent
t0 = time.perf_counter()
result = planner_agent.run_sync(prompt, retries=2)
planner_elapsed = time.perf_counter() - t0

shared_memory.uid = result.output.uid
shared_memory.action_plan = result.output.action_plan
print("PLAN AGENT")
print(shared_memory)
print("\n\n")

# Coding Agent
t0 = time.perf_counter()
result = coding_agent.run_sync("Follow the instructions", deps=shared_memory, retries=2)
coding_elapsed = time.perf_counter() - t0

print("CODE AGENT")
print(shared_memory)
print("\n\n")

# Sensemaking Agent
print("SENSEMAKER")
t0 = time.perf_counter()
result = sensemaking_agent.run_sync("Answer the user query", deps=shared_memory, retries=2)
sensemaking_elapsed = time.perf_counter() - t0

shared_memory.understanding = result.output.sensemaking
shared_memory.is_complete = result.output.is_complete

print(f"{shared_memory.understanding}\n {shared_memory.is_complete}")

total_elapsed = time.perf_counter() - total_start_time

print("\n" + "=" * 40)
print("EXECUTION TIME METRICS")
print("=" * 40)
print(f"Planner Agent:     {planner_elapsed:6.2f} s")
print(f"Coding Agent:      {coding_elapsed:6.2f} s")
print(f"Sensemaking Agent: {sensemaking_elapsed:6.2f} s")
print("-" * 40)
print(f"Total Elapsed:     {total_elapsed:6.2f} s")
print("=" * 40)
