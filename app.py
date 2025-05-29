# app.py

import streamlit as st
from PIL import Image
import base64
import time
from io import BytesIO
import pandas as pd
from typing import List, Literal , TypedDict

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
    model="gemini-2.5-flash-preview-05-20"
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

class PricingState(TypedDict):
    queue: List[Service]
    completed: List[PricingService]

# -------------------------
# 2. Pricing Logic
# -------------------------
def compute_cost(service: Service) -> float:
    name = service.name.lower()
    if "lambda" in name:
        return 2.5  # dummy
    elif "s3" in name:
        return 1.1
    elif "api gateway" in name:
        return 0.9
    else:
        return 0.1

def cost_node(state: PricingState) -> PricingState:
    if not state["queue"]:
        return state
    current = state["queue"][0]
    rest = state["queue"][1:]
    cost = compute_cost(current)
    new = PricingService(**current.model_dump(), cost=cost)
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

# Display table and total cost summary inside an expander
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
                "Account Context": s.account_context,
                "Count": s.count
            }
            for s in final_state["completed"]
        ])
        df.index = range(1, len(df) + 1)  
        st.dataframe(df, use_container_width=True)

        
