import streamlit as st
from PIL import Image
import base64
import json
import time
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Literal


API_KEY = 'AIzaSyC3Fgt_Cx1PpWe8DEUA5sjpaIiWlTHOJNQ'
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-2.5-flash-preview-05-20")


class relation(BaseModel):
    target: str
    description: str

class Services(BaseModel):
    name: str
    type: Literal['AWS service', 'other']
    description: str
    account_context: str
    count: int
    relations: List[relation]

class Group(BaseModel):
    name: str
    services: List[Services]

class Diagram(BaseModel):
    Groups: List[Group]


st.set_page_config(page_title="AWS Architecture Analyzer", layout="centered")
st.title("AWS Architecture Analyzer")
st.markdown("Upload an AWS architecture diagram and extract AWS service components and relationships.")

uploaded_file = st.file_uploader("Upload an AWS diagram (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Architecture Diagram", use_container_width=True)

    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_str}"

    
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract all the AWS services from this image and structure them in groups with their relationships."},
        {"type": "image_url", "image_url": data_uri}
    ])

    llm_structured = llm.with_structured_output(Diagram)

    with st.spinner("Analyzing image with LLM..."):
        try:
            start = time.time()
            response = llm_structured.invoke([message])
            end = time.time()
            st.success(f"Analysis complete! Time Taken : { end - start }")
            # Display groups and services nicely
            for group in response.Groups:
                with st.expander(f"📦 Group: {group.name}"):
                    for service in group.services:
                        st.markdown(f"**Service:** {service.name}")
                        st.markdown(f"- Type: `{service.type}`")
                        st.markdown(f"- Description: {service.description}")
                        st.markdown(f"- Account Context: `{service.account_context}`")
                        st.markdown(f"- Count: {service.count}")
                        if service.relations:
                            st.markdown("**Relations:**")
                            for rel in service.relations:
                                st.markdown(f"  ->`{rel.target}`: {rel.description}")
                        st.markdown("---")

        except Exception as e:
            st.error(f"Error occurred: {e}")
