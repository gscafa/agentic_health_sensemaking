from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.exceptions import ModelRetry
from pydantic import BaseModel, Field
from base_helper_functions import (
    get_daily_summary,
    get_hourly_slice,
    get_ema_surveys,
    get_historical_weather,
)

root_dir = Path(".").resolve()
load_dotenv(dotenv_path=root_dir / ".env")

models = {
"gemma4_e4b": "s1gnature/gemma4-mtp:e4b-q4_k_m",
"gemma4_e2b": "s1gnature/gemma4-mtp:e2b-q4_k_m",
"minicpm5_2b": "openbmb/minicpm5-2b:latest"
}


ollama_provider = OllamaProvider(
base_url="http://localhost:11434/v1"
)

model = OllamaModel(
model_name=models.get('gemma4_e2b'),
provider=ollama_provider
    )


data_dictionary = """### AVAILABLE DATA DICTIONARY & RETRIEVAL SURFACES

1. COLLECTION: daily_data (Helper: get_daily_summary(uid, start_date, end_date, fields=[...]))
   - Granularity: 1 record per day ('YYYY-MM-DD').
   - Keys: 'uid' , 'day', 'is_ios' (1=iOS, 0=Android).
   - Mobility & Activity (seconds/day, distance in meters):
     'step', 'loc_dist', 'act_on_foot', 'act_still', 'act_on_bike', 'act_in_vehicle'
   - Semantic Places (duration in seconds/day):
     'loc_home_dur', 'loc_study_dur', 'loc_workout_dur', 'loc_food_dur', 'loc_leisure_dur',
     'loc_social_dur', 'loc_self_dorm_dur', 'loc_other_dorm_dur', 'loc_health_dur', 'loc_worship_dur'
   - Phone & Social Interaction (durations in seconds):
     'unlock_duration', 'unlock_num', 'audio_convo_duration', 'other_playing_duration'
   - Phone Calls:
     'call_in_duration', 'call_in_num', 'call_out_duration', 'call_out_num', 'call_miss_num'
   - Sleep (seconds):
     'sleep_duration'

2. COLLECTION: hourly_data (Helper: get_hourly_slice(uid, start_date, end_date, start_hour, end_hour, fields=[...]))
   - Granularity: 24 records per day ('hour': 0..23).
   - Available fields:
     'step', 'loc_dist', 'act_on_foot', 'act_still', 'act_on_bike', 'act_in_vehicle',
     'unlock_duration', 'unlock_num', 'audio_convo_duration', 'other_playing_duration'

3. COLLECTION: ema_data (Helper: get_ema_surveys(uid, start_date, end_date, fields=[...]))
   - Granularity: Event-driven sparse self-report surveys.
   - Available fields:
     'pam' (Photographic Affect Meter), 'stress' (perceived stress), 'social_level',
     'phq4_score' (total anxiety/depression), 'phq4-1', 'phq4-2', 'phq4-3', 'phq4-4',
     'sse3-1', 'sse3-2', 'sse3-3', 'sse3-4'

4. EXTERNAL CONTEXT: Hanover, NH Weather (Helper: get_historical_weather(start_date, end_date, hourly=False))
   - Daily metrics: 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum', 'weather_code', 'wind_speed_10m_max'
   - Hourly metrics: 'temperature_2m', 'precipitation', 'snowfall', 'weather_code', 'wind_speed_10m'

STRICT RULES:
- Use ONLY exact column names listed above.
- Always instruct the coding agent to pass minimal 'fields=[...]' lists matching the query intent.
- Note: Continuous heart rate and raw GPS coordinates are NOT available in this dataset."""

class SensemakingDeps(BaseModel):
    uid: str = ""
    user_query: str = Field(description="The user request")
    action_plan: List[str] = Field(default_factory=list)
    iteration: int = 0
    collected_data: List[Dict[str, Any]] = Field(default_factory=list)
    understanding: str = ""
    is_complete: bool = Field(default=False, description="Wether the current sensemaking successfully satisfies the user query")


class PlanModel(BaseModel):
    uid: str = Field(description="The target user")
    action_plan: List[str] = Field(description="The steps necessary to answer the user query")

class SensemakingIteration(BaseModel):
    sensemaking: str = Field(description="The data interpretation")
    is_complete: bool = Field(description="Wether the current sensemaking successfully satisfies the user query")




def load_agents():

    
    planner_agent = Agent(
        model=model,
        system_prompt=f"""You are an expert in health sensemaking. Heres the data dictionary: {data_dictionary} .
        Your task is to understand the user request and formulate a detailed plan to answer it. First, define what base data to fetch via the tools. 
        Secondly, provide instructions of how the collected data should be programmatically triangulated and correlated to get better insights and facilitate sensemaking.
        Be concise, avoid redundant syntax and symbols. Only state instructions. Avoid mentioning the uid everytime you specify a tool call""",
        output_type=PlanModel
    )


    
    coding_agent = Agent(
        model=model,
        system_prompt="You are a data analytics coding expert in health sensemaking. Follow the instructions, retrieve and display the requested data. Don't interpret the data, only return the data obtained with the tools in plain JSON",
        deps_type=SensemakingDeps
    )

    @coding_agent.system_prompt
    def inject_shared_memory(ctx:RunContext[SensemakingDeps])-> str:
        return (
            f"Target uid: {ctx.deps.uid}\n"
            f"Action Plan: {ctx.deps.action_plan}"
        )

    @coding_agent.tool
    def fetch_daily_summary(
        ctx: RunContext[SensemakingDeps],
        uid: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> str:
        """Fetches daily summary records and stores them directly in shared memory."""
        records = get_daily_summary(uid=uid, start_date=start_date, end_date=end_date, fields=fields)
        ctx.deps.collected_data.append({
            "source": "daily_data",
            "range": f"{start_date} to {end_date}",
            "records": records,  
        })
        return f"SUCCESS: Saved {len(records)} daily records to shared memory."

    @coding_agent.tool
    def fetch_hourly_slice(
        ctx: RunContext[SensemakingDeps],
        uid: str,
        start_date: str,
        end_date: str,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        fields: Optional[List[str]] = None,
    ) -> str:
        """Fetches hourly time-series records and stores them directly in shared memory."""
        records = get_hourly_slice(
            uid=uid,
            start_date=start_date,
            end_date=end_date,
            start_hour=start_hour,
            end_hour=end_hour,
            fields=fields,
        )
        ctx.deps.collected_data.append({
            "source": "hourly_data",
            "range": f"{start_date} to {end_date}",
            "hours": f"{start_hour or 0}-{end_hour if end_hour is not None else 23}",
            "records": records,
        })
        return f"SUCCESS: Saved {len(records)} hourly records to shared memory."

    @coding_agent.tool
    def fetch_ema_surveys(
        ctx: RunContext[SensemakingDeps],
        uid: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> str:
        """Fetches EMA survey records and stores them directly in shared memory."""
        records = get_ema_surveys(uid=uid, start_date=start_date, end_date=end_date, fields=fields)
        ctx.deps.collected_data.append({
            "source": "ema_data",
            "range": f"{start_date} to {end_date}",
            "records": records,
        })
        return f"SUCCESS: Saved {len(records)} EMA records to shared memory."


    sensemaking_agent = Agent(
        model=model,
        system_prompt="""You are a sensemaking expert. Your role is to analyze a collection of health and behavioural data from a user. 
                Use that data to reason and find an answer to the user query. Interpret the data based on the request. Be concise. State only useful facts
                """,
        deps_type=SensemakingDeps,
        output_type=SensemakingIteration

    )

    @sensemaking_agent.system_prompt
    def s_inject_shared_memory(ctx:RunContext[SensemakingDeps]) -> str:
        return (
            f"Target uid: {ctx.deps.uid}\n"
            f"User Query: {ctx.deps.user_query}"
            f"Action Plan: {ctx.deps.action_plan}"
            f"Collected Data: {ctx.deps.collected_data}"
        )

    

    presentation_agent = Agent(
        model=model,
        system_prompt="",
        deps_type=SensemakingDeps
    )
   

    return planner_agent, coding_agent, sensemaking_agent, presentation_agent
