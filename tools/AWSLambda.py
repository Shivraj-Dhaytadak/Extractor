import json
import requests
from typing import Union, Dict
from langchain.tools import BaseTool

class LambdaPricingTool(BaseTool):
    name = "lambda_pricing"
    description = (
        "Given an AWS Lambda SKU (or rateCode) and a usage amount, "
        "returns the cost in USD by parsing the pricing JSON's 'terms.OnDemand' section. "
        "The JSON should follow AWS offer format v1.0 with 'products' and 'terms' keys."
    )

    def __init__(
        self,
        pricing_source: Union[str, Dict],
        region: str = "us-east-1",
    ):
        """
        pricing_source: path or URL to the AWS Lambda pricing JSON, or already-loaded dict.
        region: AWS region identifier (not used directly but kept for compatibility).
        The JSON must include top-level keys: 'products' and 'terms'.
        """
        self.region = region

        # Load JSON from file, URL, or dict
        if isinstance(pricing_source, str):
            if pricing_source.startswith("http"):
                resp = requests.get(pricing_source, timeout=10)
                resp.raise_for_status()
                self.pricing = resp.json()
            else:
                with open(pricing_source, "r") as f:
                    self.pricing = json.load(f)
        else:
            self.pricing = pricing_source

        # Ensure expected keys exist
        if "terms" not in self.pricing or "OnDemand" not in self.pricing["terms"]:
            raise ValueError("Invalid pricing JSON: missing 'terms.OnDemand' section.")

        # Build lookup index for price dimensions
        self._build_price_index()

    def _build_price_index(self):
        self.price_index = {}
        ondemand = self.pricing.get("terms", {}).get("OnDemand", {})
        for sku, term_objs in ondemand.items():
            for term_id, term_detail in term_objs.items():
                for dim_id, dim in term_detail.get("priceDimensions", {}).items():
                    rate = float(dim.get("pricePerUnit", {}).get("USD", 0))
                    entry = {
                        "sku": sku,
                        "rateCode": dim.get("rateCode"),
                        "description": dim.get("description"),
                        "unit": dim.get("unit"),
                        "rate": rate,
                    }
                    # index by dimension ID and rateCode
                    self.price_index[dim_id] = entry
                    if dim.get("rateCode"):
                        self.price_index[dim.get("rateCode")] = entry

    def _run(self, query: str) -> str:
        """
        Query formats:
          - "<rateCode_or_dim_id> <usage_quantity>"
          - "<sku> <usage_quantity>" to sum all dimensions under a SKU
        """
        parts = query.strip().split()
        if len(parts) != 2:
            return "Error: please provide '<rateCode|dim_id|sku> <quantity>'."

        key, qty_str = parts
        try:
            qty = float(qty_str)
        except ValueError:
            return f"Error: quantity must be numeric, got '{qty_str}'."

        # Direct lookup by rateCode or dim_id
        if key in self.price_index:
            info = self.price_index[key]
            cost = info["rate"] * qty
            return (
                f"{qty} {info['unit']} at ${info['rate']:.10f}/{info['unit']} "
                f"→ ${cost:.6f} USD\n"  
                f"({info['description']})"
            )

        # Sum all dims under a given SKU
        matches = [info for info in self.price_index.values() if info["sku"] == key]
        if matches:
            total = 0.0
            lines = []
            for info in matches:
                c = info["rate"] * qty
                total += c
                lines.append(f"- {info['rateCode']}: ${info['rate']:.10f} × {qty} = ${c:.6f}")
            return "Breakdown:\n" + "\n".join(lines) + f"\n\nTotal: ${total:.6f} USD"

        return f"Error: no matching rateCode, dim_id, or SKU found for '{key}'."

    async def _arun(self, query: str) -> str:
        return self._run(query)
