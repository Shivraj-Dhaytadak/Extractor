from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from cost_logic import PricingState, cost_node

# -------------------------
# LangGraph Setup
# -------------------------
graph = StateGraph(PricingState)
graph.add_node("cost", RunnableLambda(cost_node))
graph.set_entry_point("cost")
graph.set_finish_point("cost")
graph.add_conditional_edges("cost", lambda s: END if not s["queue"] else "cost")

# Compile runner
to_run = graph.compile()

# Expose cost_runner for app
cost_runner = to_run