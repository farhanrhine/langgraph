# from langchain_core.prompts import PromptTemplate
# from langchain.chains.router import MultiPromptChain
# from langchain.chains.router.llm_router import LLMRouterChain, Route
# from langchain.chains import ConversationChain
# from langchain_openai import ChatOpenAI
# from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


# # A simple chain for a specific topic (e.g., "Math")
# math_chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o-mini"), verbose=False)

# # 1. Define Routes/Destinations
# route_infos = [
#     Route(
#         name="Math_Solver",
#         description="Good for answering any question about mathematics or calculations.",
#         destination=math_chain,
#     ),
#     # ... more routes (e.g., "History_Expert", "Poetry_Generator")
# ]

# # 2. Define the Routing Logic (LLM determines the route)
# router_chain = LLMRouterChain.from_routes(routes=route_infos, llm=ChatOpenAI())

# # 3. Combine with a fallback (for when no route is matched)
# default_chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o-mini"))

# # 4. The MultiPromptChain puts it all together
# full_router_chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains={"Math_Solver": math_chain},
#     default_chain=default_chain,
#     verbose=True
# )

# # full_router_chain.invoke("What is the square root of 81?") # This would hit Math_Solver

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:0.6b")

route_system = "Route the user's query to either the animal or vegetable expert."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{input}"),
    ]
)


# Define schema for output:
class RouteQuery(TypedDict):
    # """Route query to destination expert."""

    destination: Literal["animal", "vegetable"]


# Instead of writing formatting instructions into the prompt, we
# leverage .with_structured_output to coerce the output into a simple
# schema.
chain = route_prompt | llm.with_structured_output(RouteQuery)


result = chain.invoke({"input": "What color are carrots?"})

print(result["destination"]) # pyright: ignore[reportIndexIssue]

