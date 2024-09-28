from ai21.models.chat import ChatMessage

# system_prompt = """
# You are a professional biologist and therapeutics scientist. Use the supplied tools to answer the query provided by the user.
# Think step by step. You run in a loop of THOUGHT, ACTION, OBSERVATION. 
# At the end of the loop you output an ANSWER.
# Use THOUGHT to describe your thoughts about the question you have been asked.
# Use ACTION to run one of the actions available to you.
# OBSERVATION will be the result of running those actions.

# The ANSWER should directly and succintly answer the user's original query. You should not write code in the ANSWER, only provide a direct answer to the user's query.
# """

system_prompt = "You are a therapeutics data analyst and scientist. Use the supplied tools to answer the query provided."

messages = [
    ChatMessage(
        role="system",
        content=system_prompt),
    
    ChatMessage(role="user", content="Is STK38 a target for IBD?"),
    ChatMessage(role="user", content="What cell types have the highest number of targets for RA? Give the top 3."),
    ChatMessage(role="user", content="What tissues have the highest number of targets for IBD? Give the top 2.")
    ChatMessage(role="user", content="list the top 10 proteins name with the highest expression in connective tissue cell"),
    ChatMessage(role="user", content="Which cell type is most affected by ulcerative colitis?"),
    ChatMessage(role="user", content="Provide some drug repositioning opportunities for aspirin"),
]