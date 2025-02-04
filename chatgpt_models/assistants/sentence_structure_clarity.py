from openai import AsyncOpenAI
import asyncio
from chatgpt_models.assistants.base import Assistant
from dataclasses import dataclass
from loguru import logger

@dataclass
class Rule:
    description: str

class ClarityAssistant(Assistant):
    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)

    def parse_examples(self,example):
        e = self.get_response(example)
        examples = e['examples']
        formatted_examples = ', '.join(f'"{example}"' for example in examples)
        return f"(e.g., {formatted_examples})"

    async def set_up_rules(self, verbose=False):
        rules               = [r.description for r in self.rules]
        tasks               = []

        if verbose: logger.debug(f"Total Jobs: {len(rules)}")
        if verbose: logger.debug(f"BEGINNING {self.agent_ident} AGENT: {self.agent_ident}")

        for r in rules:
            task = asyncio.create_task(self.assistant_apply(r))
            tasks.append(task)

        # Gather the results
        examples = await asyncio.gather(*tasks)
        rules_with_examples = [f"{r} {self.parse_examples(e)}" for r, e in zip(rules, examples)]
        return rules_with_examples

class Structure(ClarityAssistant):
    """
    ### 1. **Sentence Structure and Clarity**
    - **Subject-Verb Agreement**: Ensure that the subject and verb agree in number (singular/plural) consistently throughout the manuscript.
    - **Parallelism**: Maintain parallel structure in sentences and lists to ensure clarity and consistency.
    - **Active vs. Passive Voice**: Active voice is generally preferred for clarity and directness, though passive voice may be used when the focus is on the action rather than the actor.
    - **Avoiding Run-On Sentences**: Use proper punctuation to separate independent clauses; avoid stringing multiple clauses together without proper conjunctions or punctuation..
    """
    RULES = [
        Rule(description="Check Subject-Verb Agreement: Ensure that the subject and verb agree in number (singular/plural)."),
        Rule(description="Check parallelism: Maintain parallel structure in sentences and lists to ensure clarity and consistency."),
        Rule(description="Check Active vs Passive voice: Make sure the appropriate voice is used. Active voice is generally preferred for clarity and directness, though passive voice may be used when the focus is on the action rather than the actor."),
        Rule(description="Avoid Run-On Sentences: Use proper punctuation to separate independent clauses; avoid stringing multiple clauses together without proper conjunctions or punctuation"),
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "sentence_structure_and_clarity_agent"
        self.rules = self.RULES