import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM client setup (Gemini)
gemini = ChatGoogleGenerativeAI(
    api_key="AIzaSyC3Fgt_Cx1PpWe8DEUA5sjpaIiWlTHOJNQ",
    model="gemini-2.0-flash"
)

# -------------------------
# Models (moved from app.py)
# -------------------------
class Relation(BaseModel):
    target: str
    description: str

class Service(BaseModel):
    name: str
    type: str  # Literal["AWS service", "other"]
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

from typing_extensions import TypedDict
class PricingState(TypedDict):
    queue: List[Service]
    completed: List[PricingService]

# -------------------------
# 2. Pricing Logic
# -------------------------

def compute_cost_lambda(config: Dict[str, Any]) -> Cost:
    """Compute the cost for AWS Lambda based on the provided configuration."""
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "data/lambda.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    user_desc = config.get("description", "10 million requests per month")
    msg_lambda = HumanMessage(content=[
        {"type": "text", "text": (
            "Given the AWS Lambda service, compute the cost based on the following Configuration: "
            f"{user_desc}. "
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result.")},
        {"type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations. use the json given : {contents}"},
    ])

    response = gemini.with_structured_output(Cost).invoke([msg_lambda])
    return response


def compute_cost_s3(config: Dict[str, Any]) -> Cost:
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "data/s3.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    msg_s3 = HumanMessage(content=[
        {"type": "text", "text": (
            "Given the AWS S3 service, compute the cost based 10 GB storage and 1000 Get requests per month. "
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result.")},
        {"type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations using the json given: {contents}"},
    ])
    response = gemini.with_structured_output(Cost).invoke([msg_s3])
    return response


def compute_cost_api_gateway(config: Dict[str, Any]) -> Cost:
    dirname = os.path.dirname(__file__)
    json_path = os.path.join(dirname, "data/apigateway.json")
    with open(json_path, "r", encoding="utf-8") as f:
        contents = f.read()
    msg_api = HumanMessage(content=[
        {"type": "text", "text": (
            "You are an expert in AWS pricing. Given the AWS API Gateway service and a usage of 10 Million Messages per month, "
            "compute the total monthly cost while explicitly ignoring any Free Tier pricing. "
            "Provide a detailed breakdown of the cost calculations. Additionally, include your internal chain-of-thought "
            "explanation as part of the result.")},
        {"type": "text", "text": f"Calculate the total monthly cost and provide a breakdown of the calculations using the json given: {contents}"},
    ])
    response = gemini.with_structured_output(Cost).invoke([msg_api])
    return response


def compute_cost(service: Service) -> Cost:
    name = service.name.lower()
    config = {"description": service.description}
    if "lambda" in name:
        return compute_cost_lambda(config)
    elif "s3" in name:
        return compute_cost_s3(config)
    elif "api gateway" in name:
        return compute_cost_api_gateway(config)
    else:
        return Cost(cost=0.0, explanation="Service not supported for pricing yet.")


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