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

from langgraph.graph import StateGraph,START, END
from langchain_core.runnables import RunnableLambda

# -------------------------
# 1. LLM + Models
# -------------------------
gemini = ChatGoogleGenerativeAI(
    api_key="AIzaSyDPaPrhk1d9JDjWH-rnE4CdEWUgkWUgO7E",
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
    configuration: Optional[Dict[str, Any]] = None

class Diagram(TypedDict):
    services: List[Service]

class DiagramWithServices(BaseModel):
    services: List[Service]

    # def group_services(self) -> Dict[str, List[Service]]:
    #     """Group services by their type or other criteria."""
    #     grouped = {}
    #     for service in self.services:
    #         group_key = service.type 
    #         if group_key not in grouped:
    #             grouped[group_key] = []
    #         grouped[group_key].append(service)
    #     return grouped

class PricingService(Service):
    cost: float
    explanation: List[str]

class Cost(BaseModel):
    cost: float
    explanation: Optional[str] = None

class PricingState(TypedDict):
    image_path : str
    queue: List[Service]
    completed: List[PricingService]

image = r"C:\VScodeMaster\Inferencing\Extractor\application\Experiment\data\image.png"

def get_services_from_diagram(input: dict):
    # """Extract services from the diagram image."""
    image_path = input['image_path']
    # image_path = input["image_path"]
    with open(image_path, "rb") as f:
        image_data = f.read()
    b64 = base64.b64encode(image_data).decode('utf-8')
    sysmsg = SystemMessage(content="You are an expert in AWS architecture diagrams. Your task is to extract all AWS services from the provided image and group them based on there diagram context. Each service should include its name, type, description, account context, and any relations to other services.")
    msg = HumanMessage(content=[
        {"type": "text", "text": "Extract all AWS services from this image."},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
    ])
    res = gemini.with_structured_output(DiagramWithServices)
    response = res.invoke([sysmsg,msg])
    print(response)
    return {'queue': response.services}

def get_configration():
    pass

def get_pricing(input: DiagramWithServices):
    pass


graph = StateGraph(PricingState)
graph.add_node("extract",get_services_from_diagram)




graph.add_edge(START, "extract")
graph.add_edge("extract", END)  


# Compile and run
extract = graph.compile(debug=True)

extract.invoke({
    "image_path": image})