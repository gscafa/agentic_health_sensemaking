import time
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic import BaseModel, Field
from agentic_health_sensemaking.agents_configuration import load_agents, SensemakingDeps, PlanModel
import docker
from agentic_health_sensemaking.base_helper_functions import (
    get_daily_summary,
    get_hourly_slice,
    get_ema_surveys,
    get_historical_weather,
)
from agentic_health_sensemaking.base_helper_functions import BASE_TOOLS

planner_agent, coding_agent, sensemaking_agent, presentation_agent = load_agents()

uid = "8391d8b451084771d0affcc26e34d773"
prompt = f"How physically active was user {uid} on 2018-09-11"
shared_memory = SensemakingDeps(user_query=prompt)



# Planner Agent
print("PLANNER AGENT")
result = planner_agent.run_sync(prompt)

plan_output: PlanModel = result.output

shared_memory.uid = uid
shared_memory.triangulation_instructions = plan_output.triangulation_instructions

for function in plan_output.retrieval_functions:
    
    exec_function = BASE_TOOLS[function.name]
    
    record = {
        "source": function.name,
        "date_range": str(function.params.get("start_date")) + "/" + str(function.params.get("end_date")),
        "records": exec_function(**function.params)
    }

    shared_memory.collected_data.append(record)

#print(plan_output.retrieval_functions)
#print(plan_output.triangulation_instructions)

print("\n\n")
#shared_memory.triangulation_instructions = None
if(shared_memory.triangulation_instructions and shared_memory.collected_data):

# Coding Agent

    print("CODING AGENT")
    result = coding_agent.run_sync("Proceed", deps=shared_memory)


    #print(shared_memory.collected_data)
    print(shared_memory)
    print("\n\n")
    
    


# Sensemaking Agent
print("SENSEMAKING AGENT")

result = sensemaking_agent.run_sync("Answer the user query", deps=shared_memory, retries=2)


shared_memory.understanding = result.output.sensemaking
shared_memory.is_complete = result.output.is_complete

print(f"{shared_memory.understanding}\n {shared_memory.is_complete}")

