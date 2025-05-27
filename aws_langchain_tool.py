from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel

class AWSPriceInput(BaseModel):
    service: str
    region: str

class AWSPriceTool(BaseTool):
    name = "aws_pricing_tool"
    description = "Get AWS pricing for a given service and region"
    args_schema: Type[BaseModel] = AWSPriceInput

    def __init__(self, pricing_data):
        super().__init__()
        self.pricing_data = pricing_data

    def _run(self, service: str, region: str):
        results = [
            entry for entry in self.pricing_data
            if service.lower() in entry['service_name'].lower()
            and region.lower() in entry['region'].lower()
        ]
        return results if results else f"No pricing data found for {service} in {region}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented")
