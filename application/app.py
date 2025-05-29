# app.py

import streamlit as st
from PIL import Image
import base64
import time
from io import BytesIO
import pandas as pd
from typing import List, Literal , TypedDict , Optional
import os
from pydantic import BaseModel
from langchain_ollama.chat_models import ChatOllama
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
    cost : float
    explanation : Optional[str] = None

class PricingState(TypedDict):
    queue: List[Service]
    completed: List[PricingService]


# -------------------------
# 2. Pricing Logic
# -------------------------
def compute_cost_lambda():
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "lambda.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    msg_lambda = HumanMessage(content=[
    {"type": "text", "text": "Given the AWS Lambda service, compute the cost based on the following assumptions: 10 million requests per month "
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result."},
    # {"type": "text", "text": "1. 1 million requests per month\n2. 400,000 GB-seconds of compute time per month\n3. $0.20 per million requests\n4. $0.00001667 per GB-second"}
    { "type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations. use the json given : {contents}"},
    ])

    response = gemini.with_structured_output(Cost).invoke([msg_lambda])
    print(response.cost)
    return response

def compute_cost_s3():
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "s3.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    msg_s3 = HumanMessage(content=[
        {"type": "text", "text": "Given the AWS S3 service, compute the cost based 10 GB storage and 1000 Get requests per month."
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result."},
        {"type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations using the json given: {contents}"},
    ])
    response = gemini.with_structured_output(Cost).invoke([msg_s3])
    print(response.cost)
    return response

def compute_cost_api_gateway():
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "apigateway.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    msg_api = HumanMessage(content=[
        {"type": "text", "text": "You are an expert in AWS pricing. Given the AWS API Gateway service and a usage of 10 Million Messages per month, "
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result."},
        {"type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations using the json given: {contents}"},
    ])
    response = gemini.with_structured_output(Cost).invoke([msg_api])
    print(response.cost)
    return response

def compute_cost(service: Service):
    name = service.name.lower()
    if "lambda" in name:
        return compute_cost_lambda()
    elif "s3" in name:
        return compute_cost_s3()
    elif "api gateway" in name:
        return compute_cost_api_gateway()
    else:
        return 0.0

def cost_node(state: PricingState) -> PricingState:
    if not state["queue"]:
        return state
    current = state["queue"][0]
    rest = state["queue"][1:]
    cost = compute_cost(current)
    new = PricingService(**current.model_dump(), 
                          cost=cost.cost, 
                          explanation=cost.explanation or "No explanation provided")
    return PricingState(queue=rest, completed=state["completed"] + [new])

# -------------------------
# 3. LangGraph
# -------------------------
graph = StateGraph(PricingState)
graph.add_node("cost", RunnableLambda(cost_node))
graph.set_entry_point("cost")
graph.set_finish_point("cost")
graph.add_conditional_edges("cost", lambda s: END if not s["queue"] else "cost")
cost_runner = graph.compile()

# -------------------------
# 4. Streamlit UI
# -------------------------
st.set_page_config(page_title="AWS Arch + Cost Analyzer", layout="centered")
st.title("📊 AWS Architecture & Cost Analyzer")

uploaded = st.file_uploader("Upload AWS diagram (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Please upload an architecture diagram.")
    st.stop()

img = Image.open(uploaded)
st.image(img, use_container_width=True)

buffer = BytesIO(); img.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

msg = HumanMessage(content=[
    {"type": "text", "text": "Extract all AWS services from this image."},
    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
])
# Add a collapsible section
with st.expander("Additional Information (Assumptions and Data Points)", expanded=False):
    st.markdown("### Configuration Assumptions")
    st.write("""
    - **AWS Lambda**: 
    - **S3**: Object storage service offering industry-leading scalability, data availability, and security.
    - **API Gateway**: Fully managed service for creating, publishing, and managing APIs.
    """)
    st.markdown("### Example Data Points")
    st.write("""
    - **Region**: US East (N. Virginia)
    - **Currency**: USD
    - **Effective Date**: 2025-04-01
    """)
with st.spinner("Extracting services..."):
    
    diagram: Diagram = gemini.with_structured_output(Diagram).invoke([msg])
    services = [s for s in diagram.services if s.type == "AWS service"]
    st.success(f"Found {len(services)} AWS services.")

if not services:
    st.info("No AWS services found in the image.")
    st.stop()

# Run LangGraph
initial_state = PricingState(queue=services, completed=[])
final_state = cost_runner.invoke(initial_state)

# Build a DataFrame with all state data (both completed and pending)
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

# Display table and total cost summary inside an expander for detailed breakdown
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


