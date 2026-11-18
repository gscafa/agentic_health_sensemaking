from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_ai import Agent, RunContext, NativeOutput, ModelSettings
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.exceptions import ModelRetry
from pydantic import BaseModel, Field
import json
import docker
import tempfile

from agentic_health_sensemaking.base_helper_functions import (
    get_daily_summary,
    get_hourly_slice,
    get_ema_surveys,
    get_historical_weather,
)

root_dir = Path(".").resolve()
load_dotenv(dotenv_path=root_dir / ".env")

docker_sandbox = docker.from_env()

models = {
"gemma4_e4b": "s1gnature/gemma4-mtp:e4b-q4_k_m",
"gemma4_e2b": "s1gnature/gemma4-mtp:e2b-q4_k_m",
"minicpm5_2b": "openbmb/minicpm5-2b:latest"
}


ollama_provider = OllamaProvider(
base_url="http://localhost:11434/v1"
)

model = OllamaModel(
model_name=models.get('gemma4_e4b'),
provider=ollama_provider,
settings=ModelSettings(temperature=0.2)
    )


_data_dictionary = """### AVAILABLE DATA DICTIONARY & RETRIEVAL SURFACES

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
   - Sleep (hours):
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



_planner_system_prompt = (
    """You are a health sensemaking planner. Output ONLY valid JSON matching the PlanModel schema. Do NOT output markdown, headings, or prose.

"""
    + _data_dictionary
    + """

TASK:
1. Extract the target `uid` and date range from the user query.
2. Populate `retrieval_functions` with the exact helper function names and their keyword arguments inside `params` (including minimal `fields` lists when supported).
3. Set `triangulation_instructions`:
   - Set to `null` for simple factual lookups (e.g., retrieving a single day's steps, sleep, or survey score) where the raw fetched records directly answer the question.
   - Populate with concise Python calculation instructions ONLY when the query requires averages, totals, percentages, anomaly detection, or cross-dataset correlations.

EXAMPLE OUTPUT:
{
  "uid": "8391d8b451084771d0affcc26e34d773",
  "retrieval_functions": [
    {
      "name": "get_daily_summary",
      "params": {
        "uid": "8391d8b451084771d0affcc26e34d773",
        "start_date": "2018-06-24",
        "end_date": "2018-06-30",
        "fields": ["sleep_duration", "step", "loc_workout_dur"]
      }
    },
    {
      "name": "get_historical_weather",
      "params": {
        "start_date": "2018-06-24",
        "end_date": "2018-06-30",
        "hourly": false
      }
    }
  ],
  "triangulation_instructions": "Compute daily mean and min for sleep_duration and step. Correlate low-sleep or low-step days against precipitation_sum and temperature extremes."
}"""
)

_coding_agent_prompt = """You are a deterministic Python data analysis agent. Call `execute_python_code(code=...)` with a script fulfilling `Triangulation Instructions`.

### INPUT & SCHEMA
- Load `/workspace/input_data.json`: a list of `{"source": "<name>", "records": [...]}` blocks.
- Match `block["source"]` to `Available Sources & Schemas` in your context. Use ONLY listed column keys; handle empty `records` gracefully.
- NEVER hardcode mock data, write code comments, or use `try/except` blocks (let errors fail loudly).

### PROCESSING & TRIANGULATION
- Build `pandas.DataFrame`s and coerce numeric columns with `pd.to_numeric(..., errors="coerce")`.
- **Unmerged Baselines First**: Before merging, compute `count`, `mean`, `min`, `max` (with dates/keys), and `std` for every analyzed metric on its original DataFrame.
- **Safe Merging & Filtering**: Normalize temporal keys, then merge using `how="left"` anchored on the densest source (never `how="inner"`). NEVER call `merged_df.dropna()` across mixed sources; drop `NaN`s pairwise per calculation (`df[[a, b]].dropna()`).
- **Valid Correlations & Thresholds**: Compute correlations only when paired valid rows >= 5 and both `std() > 0` (otherwise record `None` with a reason). Define anomalies using sample-relative stats (`mean ± std` or quantiles), never hardcoded domain cutoffs.

### OUTPUT
- Replace `NaN`/`inf` with `None` and end with a single `print(json.dumps(result_dict, default=str))`. No other prints.
- After `execute_python_code` succeeds, reply strictly with: `COMPUTATION_COMPLETE`."""

_sensemaking_agent_prompt = """You are a sensemaking expert. Your role is to analyze a collection of health and behavioural data from a user. 
                Use that data to reason and find an answer to the user query. Interpret the data based on the request. Be concise. State only useful facts.
                Set 'is_complete' as False if you think that the current data retrieval and analysis are enough and not improvable.
                Set 'is_complete' as True if 
                    - you think the data is sufficient to answer the user query 
                    - you think the data is not sufficient but there is more data to fetch that could improve the analysis
                """
class RetrievalFunction(BaseModel):
    name: str = Field(description="Exact helper function name from the Data Dictionary")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments matching the function signature (e.g., uid, start_date, end_date, hourly, fields)",
    )

class SensemakingDeps(BaseModel):
    uid: str = ""
    user_query: str = Field(description="The user request")
    triangulation_instructions: Optional[str] = None
    iteration: int = 0
    collected_data: List[Dict[str, Any]] = Field(default_factory=list)
    computed_data: List[Dict[str, Any]] = Field(default_factory=list)
    understanding: str = ""
    is_complete: bool = Field(
        default=False,
        description="Whether the current sensemaking successfully satisfies the user query",
    )


class PlanModel(BaseModel):
    uid: str = Field(description="The target user")
    retrieval_functions: List[RetrievalFunction] = Field(
        default_factory=list,
        description="The list of filled python data retrieval functions that will be executed",
    )
    triangulation_instructions: Optional[str] = Field(
        default=None,
        description="Python calculation/correlation instructions, or null if raw fetched records directly answer the query",
    )

class SensemakingIteration(BaseModel):
    sensemaking: str = Field(description="The data interpretation")
    is_complete: bool = Field(description="Wether the current sensemaking successfully satisfies the user query")



def run_docker_sandbox(code: str, data: list) -> str:
    #print(f"CODE:\n {code}")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "input_data.json").write_text(json.dumps(data))
        (temp_path / "script.py").write_text(code)


        output = docker_sandbox.containers.run(
            image="code-sandbox",
            command="python /workspace/script.py",
            volumes={str(temp_path): {"bind": "/workspace", "mode": "rw"}},
            network_mode="none",
            mem_limit="512m",
            remove=True)
        
        return output.decode("utf-8")




def load_agents():



    
    planner_agent = Agent(
        model=model,
        system_prompt=_planner_system_prompt,
        output_type=NativeOutput(PlanModel),
        retries=3
    )






    
    coding_agent = Agent(
        model=model,
        system_prompt=_coding_agent_prompt,
        deps_type=SensemakingDeps,
        retries=3
    )

    @coding_agent.system_prompt
    def inject_coding_context(ctx: RunContext[SensemakingDeps]) -> str:
        data_profile = []
        for block in ctx.deps.collected_data:
            records = block.get("records", [])
            sample = records[:2]
            col_types = (
                {k: type(v).__name__ for k, v in records[0].items()}
                if records
                else {}
            )
            data_profile.append({
                "source": block.get("source"),
                "row_count": len(records),
                "column_types": col_types,
                "sample_rows": sample,
            })

        return (
            f"Target UID: {ctx.deps.uid}\n"
            f"Staged Data Profile in /workspace/input_data.json:\n{json.dumps(data_profile, default=str)}\n"
            f"Triangulation Instructions:\n{ctx.deps.triangulation_instructions}"
        )

    @coding_agent.tool
    def execute_python_code(ctx:RunContext[SensemakingDeps], code: str) -> str:
        """Runs Python code to analyze /workspace/input_data.json ."""

        print(f"CODE: \n {code}")
        try:
            stdout = run_docker_sandbox(code, ctx.deps.collected_data)
            parsed_data = json.loads(stdout)
        except docker.errors.ContainerError as e:
            error_message = e.stderr.decode("utf-8") if e.stderr else str(e)
            raise ModelRetry(f"The script failed its execution with error {error_message}. Retry")
        except json.JSONDecodeError:
            raise ModelRetry("Your script didn't return JSON output. Retry")

        ctx.deps.computed_data.append(parsed_data)
        return "Done"












    sensemaking_agent = Agent(
        model=model,
        system_prompt=_sensemaking_agent_prompt,
        deps_type=SensemakingDeps,
        output_type=SensemakingIteration,
        retries=3
    )

    @sensemaking_agent.system_prompt
    def inject_sensemaking_context(ctx:RunContext[SensemakingDeps]) -> str:
        return (
            f"Target uid: {ctx.deps.uid}\n"
            f"User Query: {ctx.deps.user_query}"
            f"Triangulation Instructions: {ctx.deps.triangulation_instructions}"
            f"Collected Data: {ctx.deps.collected_data}"
            f"Computed Data: {ctx.deps.computed_data}"
        )

    

    presentation_agent = Agent(
        model=model,
        system_prompt="",
        deps_type=SensemakingDeps
    )
   

    return planner_agent, coding_agent, sensemaking_agent, presentation_agent
