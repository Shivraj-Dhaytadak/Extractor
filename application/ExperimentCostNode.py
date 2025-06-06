# app.py

import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from typing import List, Literal, TypedDict, Optional, Dict, Any
import os
import json
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# -------------------------
# 1. LLM + Models
# -------------------------
gemini = ChatGoogleGenerativeAI(
    api_key="AIzaSyC3Fgt_Cx1PpWe8DEUA5sjpaIiWlTHOJNQ",
    model="gemini-2.0-flash"
)

class Relation(BaseModel):
    target: str
    description: str

class Service(BaseModel):
    name: str
    type: Literal["AWS service", "other"]
    description: str
    account_context: str
    count: int
    relations: List[Relation]

class Diagram(BaseModel):
    services: List[Service]

class PricingService(Service):
    cost: float
    explanation: Optional[str] = None 

class Cost(BaseModel):
    cost: float
    explanation: Optional[str] = None

class PricingState(TypedDict):
    queue: List[Service]
    completed: List[PricingService]


# -------------------------
# 2. Single compute_cost Logic
# -------------------------
def compute_cost(service: Service, config: Dict[str, Any]) -> Cost:
    """
    Generic cost computation function for any AWS service.
    - Looks up <service_name>.json for pricing details.
    - Passes `config` JSON to the LLM prompt so it can use user-provided configuration.
    """
    # Derive a filename from the service name (e.g., "AWS Lambda" -> "lambda.json")
    base_name = service.name.lower().replace(" ", "").replace("-", "").replace("_", "")
    json_filename = f"{base_name}.json"
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, json_filename)
    
    if not os.path.isfile(json_path):
        # If the JSON file isn't found, return a default Cost
        return Cost(cost=0.0, explanation=f"No pricing file found for '{service.name}' (expected '{json_filename}').")
    
    # Read pricing JSON
    with open(json_path, "r", encoding="utf-8") as f:
        pricing_contents = f.read()
    
    # Build the LLM prompt—include service name, user configuration, and pricing JSON
    user_config_str = json.dumps(config)
    prompt_messages = [
        {
            "type": "text",
            "text": (
                f"Given the AWS {service.name} service with configuration:\n"
                f"{user_config_str}\n\n"
                "Compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
                "Provide a detailed breakdown of the cost calculations. "
                "Include your internal chain-of-thought in the explanation."
            )
        },
        {
            "type": "text",
            "text": f"Use the following pricing details (JSON):\n{pricing_contents}"
        }
    ]
    lm_input = HumanMessage(content=prompt_messages)
    response = gemini.with_structured_output(Cost).invoke([lm_input])
    return response


def cost_node(state: PricingState, user_inputs: Dict[str, Dict[str, Any]]) -> PricingState:
    """
    Processes one Service at a time from state['queue'], using compute_cost().
    `user_inputs` maps service.name -> configuration dict.
    """
    if not state["queue"]:
        return state
    
    current = state["queue"][0]
    rest = state["queue"][1:]
    config = user_inputs.get(current.name, {})
    cost = compute_cost(current, config)
    
    new_pricing = PricingService(
        **current.model_dump(),
        cost=cost.cost,
        explanation=cost.explanation or "No explanation provided"
    )
    return PricingState(
        queue=rest,
        completed=state["completed"] + [new_pricing]
    )


# -------------------------
# 3. LangGraph Setup
# -------------------------
from functools import partial

# We will inject `user_inputs` via `partial` when constructing the RunnableLambda
graph = StateGraph(PricingState)
# Placeholder runnable; actual `user_inputs` will be bound later
graph.add_node("cost", RunnableLambda(lambda s, user_inputs: cost_node(s, user_inputs)))
graph.set_entry_point("cost")
graph.set_finish_point("cost")
graph.add_conditional_edges("cost", lambda s: END if not s["queue"] else "cost")
# Compile once—will re-bind `user_inputs` each time
cost_runner_template = graph.compile()


# -------------------------
# 4. Streamlit UI
# -------------------------
st.set_page_config(page_title="AWS Arch + Cost Analyzer", layout="centered")
st.title("📊 AWS Architecture & Cost Analyzer")

# 4.1 Upload and display architecture diagram
uploaded = st.file_uploader("Upload AWS diagram (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Please upload an architecture diagram.")
    st.stop()

img = Image.open(uploaded)
st.image(img, use_container_width=True)

# Convert image to base64 for LLM input
buffer = BytesIO()
img.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

# 4.2 Ask Gemini to extract services
msg = HumanMessage(content=[
    {"type": "text", "text": "Extract all AWS services from this image."},
    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
])

with st.spinner("Extracting services..."):
    diagram: Diagram = gemini.with_structured_output(Diagram).invoke([msg])
    services = [s for s in diagram.services if s.type == "AWS service"]
    st.success(f"Found {len(services)} AWS services.")

if not services:
    st.info("No AWS services found in the image.")
    st.stop()

# 4.3 Collect user inputs (configuration JSON) for each service
if "service_inputs" not in st.session_state:
    st.session_state.service_inputs = {}

st.subheader("🧩 Configure Detected AWS Services")
for service in services:
    saved_conf = st.session_state.service_inputs.get(service.name, {}).get("description", "")
    with st.expander(f"Configure {service.name}"):
        # Pre-populate text_area with previously saved JSON description (if any)
        description_json = st.text_area(
            f"{service.name} Configuration JSON",
            key=f"desc_{service.name}",
            value=saved_conf,
            placeholder='e.g. {"requests": 5000000, "memory_mb": 256, "duration_ms": 500}'
        )
        if st.button(f"Save {service.name} Configuration", key=f"save_{service.name}"):
            # Try to parse the JSON; if invalid, show error
            try:
                parsed = json.loads(description_json) if description_json.strip() else {}
                st.session_state.service_inputs[service.name] = {"description": parsed}
                st.success(f"Configuration for {service.name} saved.")
            except json.JSONDecodeError:
                st.error("Invalid JSON. Please correct and try again.")

# 4.4 Run cost analysis when ready
if st.button("Run Cost Analysis"):
    # Build a mapping: service.name -> config dict
    user_inputs: Dict[str, Dict[str, Any]] = {}
    for s in services:
        saved = st.session_state.service_inputs.get(s.name, {}).get("description", {})
        user_inputs[s.name] = saved if isinstance(saved, dict) else {}

    # Bind `user_inputs` into a fresh cost_node via partial, then compile
    bound_cost_node = partial(cost_node, user_inputs=user_inputs)
    graph = StateGraph(PricingState)
    graph.add_node("cost", RunnableLambda(bound_cost_node))
    graph.set_entry_point("cost")
    graph.set_finish_point("cost")
    graph.add_conditional_edges("cost", lambda s: END if not s["queue"] else "cost")
    cost_runner = graph.compile()

    # Initialize and invoke
    initial_state = PricingState(queue=services, completed=[])
    final_state = cost_runner.invoke(initial_state)

    # Display results
    df_state = pd.DataFrame(
        [
            {
                "Service": p.name,
                "Cost (Monthly USD)": f"${p.cost:.2f}",
                "Cost (Yearly USD)": f"${p.cost * 12:.2f}",
                "Explanation": p.explanation or "No explanation provided"
            }
            for p in final_state["completed"]
        ]
    )
    df_state.index = range(1, len(df_state) + 1)

    st.subheader("Cost Breakdown")
    st.dataframe(df_state, use_container_width=True)

    if final_state["completed"]:
        total_monthly = sum(p.cost for p in final_state["completed"])
        total_yearly = total_monthly * 12
        st.subheader("Total Cost Summary")
        st.write(f"• Monthly Total: **${total_monthly:.2f}**")
        st.write(f"• Yearly Total: **${total_yearly:.2f}**")
