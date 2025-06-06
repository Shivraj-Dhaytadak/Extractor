import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import os

from cost_logic import Service, Diagram, PricingService
from graph_agents import cost_runner
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit UI
st.set_page_config(page_title="AWS Arch + Cost Analyzer", layout="centered")
st.title("📊 AWS Architecture & Cost Analyzer")

# File uploader
uploaded = st.file_uploader("Upload AWS diagram (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Please upload an architecture diagram.")
    st.stop()

img = Image.open(uploaded)
st.image(img, use_container_width=True)

# Encode image to base64
buffer = BytesIO(); img.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

# LLM client (Gemini) for service extraction
gemini = ChatGoogleGenerativeAI(
    api_key="AIzaSyC3Fgt_Cx1PpWe8DEUA5sjpaIiWlTHOJNQ",
    model="gemini-2.0-flash"
)

msg = HumanMessage(content=[
    {"type": "text", "text": "Extract all AWS services from this image."},
    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
])

# Expandable assumptions
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

# Extract services\with st.spinner("Extracting services..."):
    diagram: Diagram = gemini.with_structured_output(Diagram).invoke([msg])
    services = [s for s in diagram.services if s.type == "AWS service"]
    st.success(f"Found {len(services)} AWS services.")

if not services:
    st.info("No AWS services found in the image.")
    st.stop()

# Session state for configurations
if 'service_inputs' not in st.session_state:
    st.session_state.service_inputs = {}

for service in services:
    with st.expander(f"Configure {service.name}"):
        with st.form(key=f"form_{service.name}"):
            region = st.selectbox("Region", ["us-east-1", "us-west-2"], key=f"region_{service.name}")
            usage = st.number_input("Monthly Usage (e.g., requests, GB)", min_value=0, key=f"usage_{service.name}")
            submit = st.form_submit_button("Save Configuration")
            if submit:
                st.session_state.service_inputs[service.name] = {
                    "region": region,
                    "usage": usage
                }
                st.success(f"Configuration for {service.name} saved.")

# Run LangGraph
initial_state = {"queue": services, "completed": []}
final_state = cost_runner.invoke(initial_state)

# Build DataFrame
df_state = pd.DataFrame([
    {
        "Service": s.name,
        "Cost (Monthly USD)": f"${s.cost:.2f}",
        "Cost (Yearly USD)": f"${s.cost * 12:.2f}",
        "Status": "Completed",
        "Explanation": s.explanation or "No explanation provided"
    }
    for s in final_state["completed"]
] + [
    {
        "Service": s.name,
        "Cost (Monthly USD)": "",
        "Cost (Yearly USD)": "",
        "Status": "Pending"
    }
    for s in final_state["queue"]
])
df_state.index = range(1, len(df_state) + 1)

st.subheader("Final State Data")
st.dataframe(df_state, use_container_width=True)

# Total cost summary
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
                "Cost (Yearly USD)": f"${s.cost * 12:.2f}"
            }
            for s in final_state["completed"]
        ])
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)