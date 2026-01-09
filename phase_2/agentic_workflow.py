# agentic_workflow.py

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open('Product-Spec-Email-Router.txt','r') as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = f"""
    Stories are defined from a product spec by identifying a persona, an action, and a desired outcome for each story.
    Each story represents a specific functionality of the product described in the specification.
    Features are defined by grouping related user stories.
    Tasks are defined for each story and represent the engineering work required to develop the product.
    A development Plan for a product contains all these components.
"""
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product."
    f"{product_spec}"
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, 
                                                                persona_product_manager, 
                                                                knowledge_product_manager
                                                                )

### Product Manager - Evaluation Agent
persona_product_manager_evaluator = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
max_interactions = 10

product_manager_evaluation_agent = EvaluationAgent(openai_api_key, 
                                                   persona_product_manager_evaluator, 
                                                   evaluation_criteria_product_manager, 
                                                   product_manager_knowledge_agent, 
                                                   max_interactions)

### Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'


program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, 
                                                                persona_program_manager, 
                                                                knowledge_program_manager
                                                                )

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = (
                      "The answer should be product features that follow the following structure: \n"
                      "Feature Name: A clear, concise title that identifies the capability\n"
                      "Description: A brief explanation of what the feature does and its purpose\n"
                      "Key Functionality: The specific capabilities or actions the feature provides\n"
                      "User Benefit: How this feature creates value for the user"
)

# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
program_manager_evaluation_agent = EvaluationAgent(openai_api_key, 
                                                   persona_program_manager_eval, 
                                                   evaluation_criteria_program_manager, 
                                                   program_manager_knowledge_agent, 
                                                   max_interactions)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, 
                                                                persona_dev_engineer, 
                                                                knowledge_dev_engineer
                                                                )

### Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_dev_engineer = (
                      "The answer should be tasks following this exact structure: \n" 
                      "Task ID: A unique identifier for tracking purposes\n"
                      "Task Title: Brief description of the specific development work\n"
                      "Related User Story: Reference to the parent user story\n"
                      "Description: Detailed explanation of the technical work required\n"
                      "Acceptance Criteria: Specific requirements that must be met for completion\n"
                      "Estimated Effort: Time or complexity estimation\n"
                      "Dependencies: Any tasks that must be completed first"
)
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key, 
                                                   persona_dev_engineer_eval, 
                                                   evaluation_criteria_dev_engineer, 
                                                   development_engineer_knowledge_agent, 
                                                   max_interactions)

### Routing Agent

# You will need to define a list of agent dictionaries (routes) for Product Manager, 
# Program Manager, and Development Engineer. 
# Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). 
# # Assign this list to the routing_agent's 'agents' attribute.


routing_agent = RoutingAgent(openai_api_key, {})

agents = [
    {
        "name":"product manager agent",
        "description": "Generates user stories according to product specs",
        "func": lambda x: product_manager_knowledge_agent.respond(x)
    },
    {
        "name":"program manager agent",
        "description": "Defines features for a product",
        "func": lambda x: program_manager_knowledge_agent.respond(x)
    },
    {
        "name":"development engineer agent",
        "description": "Defines development tasks for a product",
        "func": lambda x: development_engineer_knowledge_agent.respond(x)
    }
]

routing_agent.agents = agents

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)    
    evaluation = product_manager_evaluation_agent.evaluate(response)
    return evaluation['final_response']

def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)    
    evaluation = program_manager_evaluation_agent.evaluate(response)
    return evaluation['final_response']

def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)    
    evaluation = development_engineer_evaluation_agent.evaluate(response)
    return evaluation['final_response']

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
#workflow_prompt = "Create a comprehensive project plan for the Email Router product including user stories, key features, and development tasks."
# ****
print(f"==> Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\n==> Defining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).

steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []
for i, step in enumerate(steps):
    step = step.strip()
    if not step:
        continue
    print(i, step)

print("-" * 30)

for step in steps:
    step = step.strip()
    print(f"Running step {step}")
    response = routing_agent.routing(step)
    # print(response)
    completed_steps.append(response)
    print(f"Step {step} completed with result - \n{response}")

print("-" * 30)

print(" === FINAL OUTPUT === ")
for i, output in enumerate(completed_steps, start=1):
        print(f"Step {i} Output:\n{output}")
