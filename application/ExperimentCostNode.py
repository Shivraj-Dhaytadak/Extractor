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
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage , SystemMessage

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# -------------------------
# 1. LLM + Models
# -------------------------
gemini = ChatGoogleGenerativeAI(
    api_key="AIzaSyDPaPrhk1d9JDjWH-rnE4CdEWUgkWUgO7E",
    model="gemini-2.0-flash"
)
# gemini = ChatOllama(
#     model="gemma3:4b")

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

    def group_services(self) -> Dict[str, List[Service]]:
        """Group services by their type or other criteria."""
        grouped = {}
        for service in self.services:
            group_key = service.type  # Grouping by type, can be modified as needed
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(service)
        return grouped

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
# 2. Pricing Logic with user inputs
# -------------------------

def compute_cost_lambda(config: Dict[str, Any]) -> Cost:
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "lambda.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    user_desc = config.get("description", "10 million requests per month")
    msg_content = [
        {
            "type": "text",
            "text": (
                f"Given the AWS Lambda service with configuration: {user_desc}, "
                "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
                "Provide a detailed breakdown of the cost calculations."
            )
        },
        {
            "type": "text",
            "text": f"Use the following pricing details (JSON): {contents}"
        }
    ]
    msg_lambda = HumanMessage(content=msg_content)
    return gemini.with_structured_output(Cost).invoke([msg_lambda])


def compute_cost_s3(config: Dict[str, Any]) -> Cost:
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "s3.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    user_desc = config.get("description", "10 GB storage and 1000 GET requests per month")
    msg_content = [
        {
            "type": "text",
            "text": (
                f"Given the AWS S3 service with configuration: {user_desc}, "
                "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
                "Provide a detailed breakdown of the cost calculations."
            )
        },
        {
            "type": "text",
            "text": f"Use the following pricing details (JSON): {contents}"
        }
    ]
    msg_s3 = HumanMessage(content=msg_content)
    return gemini.with_structured_output(Cost).invoke([msg_s3])


def compute_cost_api_gateway(config: Dict[str, Any]) -> Cost:
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "apigateway.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    user_desc = config.get("description", "10 million API calls per month")
    msg_content = [
        {
            "type": "text",
            "text": (
                f"Given the AWS API Gateway service with configuration: {user_desc}, "
                "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
                "Provide a detailed breakdown of the cost calculations."
            )
        },
        {
            "type": "text",
            "text": f"Use the following pricing details (JSON): {contents}"
        }
    ]
    msg_api = HumanMessage(content=msg_content)
    return gemini.with_structured_output(Cost).invoke([msg_api])


def compute_cost(service: Service, config: Dict[str, Any]) -> Cost:
    name = service.name.lower()
    if "lambda" in name:
        return compute_cost_lambda(config)
    elif "s3" in name:
        return compute_cost_s3(config)
    elif "api gateway" in name:
        return compute_cost_api_gateway(config)
    else:
        return Cost(cost=0.0, explanation="Service not supported.")


def make_cost_node(user_inputs: Dict[str, Dict[str, Any]]):
    def cost_node(state: PricingState) -> PricingState:
        if not state["queue"]:
            return state
        current = state["queue"][0]
        rest = state["queue"][1:]
        config = user_inputs.get(current.name, {})
        cost = compute_cost(current, config)
        new = PricingService(**current.model_dump(),
                             cost=cost.cost,
                             explanation=cost.explanation or "No explanation provided")
        return PricingState(queue=rest, completed=state["completed"] + [new])
    return cost_node


# -------------------------
# 3. Streamlit UI
# -------------------------

st.set_page_config(page_title="AWS Arch + Cost Analyzer", layout="centered")

# Ensure navigation state and required session state keys are properly initialized
if 'page' not in st.session_state:
    st.session_state.page = 'extraction'
if 'grouped_services' not in st.session_state:
    st.session_state.grouped_services = {}
if 'service_inputs' not in st.session_state:
    st.session_state.service_inputs = {}
if 'current_group' not in st.session_state and st.session_state.grouped_services:
    st.session_state.current_group = list(st.session_state.grouped_services.keys())[0]

# Page: Extraction
if st.session_state.page == 'extraction':
    st.title("📊 AWS Architecture & Cost Analyzer")

    # 3.1 Upload and display diagram
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
    sysmsg = SystemMessage(content="You are an expert in AWS architecture diagrams. Your task is to extract all AWS services from the provided image and group them based on there diagram context. Each service should include its name, type, description, account context, and any relations to other services.")
    msg = HumanMessage(content=[
        {"type": "text", "text": "Extract all AWS services from this image."},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
    ])

    # 3.2 Extract services via Gemini
    with st.spinner("Extracting services..."):
        diagram: Diagram = gemini.with_structured_output(Diagram).invoke([sysmsg,msg])
        grouped_services = diagram.group_services()
        st.session_state.grouped_services = grouped_services
        st.success(f"Found {sum(len(v) for v in grouped_services.values())} AWS services across {len(grouped_services)} groups.")

    if not grouped_services:
        st.info("No AWS services found in the image.")
        st.stop()

    # Use session state to handle button click
    if st.button("Proceed to Service Configuration"):
        st.session_state.page = 'configuration'
        st.rerun()

# Page: Configuration
elif st.session_state.page == 'configuration':
    st.title("🧩 Configure Detected AWS Services")

    current_group = st.session_state.current_group
    services_in_group = st.session_state.grouped_services.get(current_group, [])

    st.subheader(f"Configure Services in Group: {current_group}")

    for idx, service in enumerate(services_in_group):
        with st.expander(f"Configure {service.name}"):
            description = st.text_area(
                f"{service.name} Configuration Description",
                key=f"desc_{service.name}_{idx}",  # Append index to ensure unique keys
                placeholder="e.g. 5 million requests, 256MB memory, 500ms average duration"
            )
            if st.button(f"Save {service.name} Configuration", key=f"save_{service.name}_{idx}"):
                st.session_state.service_inputs[f"{service.name}_{idx}"] = {"description": description or ""}
                st.success(f"Configuration for {service.name} saved.")

    # Add a file uploader for JSON configuration
    uploaded_config = st.file_uploader("Upload Service Configuration (JSON)", type="json")
    if uploaded_config:
        try:
            # Parse the uploaded JSON file
            config_data = json.load(uploaded_config)
            
            # Update session state with the configurations
            for service_name, config in config_data.items():
                for idx, service in enumerate(services_in_group):
                    if service.name == service_name:
                        key = f"{service.name}_{idx}"
                        st.session_state.service_inputs[key] = config
            st.success("Service configurations loaded successfully.")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid configuration file.")

    # Navigation between groups
    group_keys = list(st.session_state.grouped_services.keys())
    current_index = group_keys.index(current_group)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Group") and current_index > 0:
            st.session_state.current_group = group_keys[current_index - 1]
            st.rerun()
    with col2:
        if st.button("Next Group") and current_index < len(group_keys) - 1:
            st.session_state.current_group = group_keys[current_index + 1]
            st.rerun()

    # Button to proceed to cost analysis
    if current_index == len(group_keys) - 1:
        if st.button("Run Cost Analysis"):
            st.session_state.page = 'cost_analysis'
            st.rerun()

# Page: Cost Analysis
elif st.session_state.page == 'cost_analysis':
    st.title("💰 Cost Analysis")

    # Ensure each service has a config; if missing, default to empty description
    user_inputs: Dict[str, Dict[str, Any]] = {}
    for group, services in st.session_state.grouped_services.items():
        for idx, service in enumerate(services):
            key = f"{service.name}_{idx}"
            user_inputs[service.name] = st.session_state.service_inputs.get(key, {"description": ""})

    # Build and run LangGraph with the cost node factory
    cost_node_fn = make_cost_node(user_inputs)
    graph = StateGraph(PricingState)
    graph.add_node("cost", RunnableLambda(cost_node_fn))
    graph.set_entry_point("cost")
    graph.set_finish_point("cost")
    graph.add_conditional_edges("cost", lambda s: END if not s["queue"] else "cost")
    cost_runner = graph.compile(debug=True)

    initial_state = PricingState(queue=[s for services in st.session_state.grouped_services.values() for s in services], completed=[])
    final_state = cost_runner.invoke(initial_state)

    # Build DataFrame for display
    df_state = pd.DataFrame(
        [
            {
                "Service": s.name,
                "Cost (Monthly USD)": f"${s.cost:.2f}",
                "Cost (Yearly USD)": f"${s.cost * 12:.2f}",
                "Status": "Completed",
                "Explanation": s.explanation or "No explanation provided"
            }
            for s in final_state["completed"]
        ] +
        [
            {
                "Service": s.name,
                "Cost (Monthly USD)": "",
                "Cost (Yearly USD)": "",
                "Status": "Pending"
            }
            for s in final_state["queue"]
        ]
    )
    df_state.index = range(1, len(df_state) + 1)

    st.subheader("Final State Data")
    st.dataframe(df_state, use_container_width=True)

    if final_state["completed"]:
        st.subheader("Total Cost Summary")
        total_monthly = sum(s.cost for s in final_state["completed"])
        total_yearly = total_monthly * 12
        total_df = pd.DataFrame({
            "Cost Type": ["Monthly Total", "Yearly Total"],
            "Cost": [f"${total_monthly:.2f}", f"${total_yearly:.2f}"]
        })
        total_df.index = range(1, len(total_df) + 1)
        st.dataframe(total_df, use_container_width=True)
        with st.expander("View Detailed Cost Breakdown", expanded=False):
            st.subheader("Extracted AWS Services")
            df = pd.DataFrame([
                {
                    "Service": s.name,
                    "Cost (Monthly USD)": f"${s.cost:.2f}",
                    "Cost (Yearly USD)": f"${s.cost * 12:.2f}",
                }
                for s in final_state["completed"]
            ])
            df.index = range(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)
