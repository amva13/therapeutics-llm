from ai21.models.chat import ChatMessage, ToolMessage

system_prompt = "You are a professional biologist and therapeutics scientist. Answer the questions to the best of your ability."  # TODO

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