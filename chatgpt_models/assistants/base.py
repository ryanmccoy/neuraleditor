import asyncio
import json
from loguru import logger
from openai import AsyncOpenAI
from text_handlers.parsers import AnnotationRemover, SentenceSplitter

class AsyncClient:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(
            api_key=api_key,
        )

    async def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            await asyncio.sleep(0.5)
        return run

    async def get_response(self, thread):
        return await self.client.beta.threads.messages.list(thread_id=thread.id, order="asc")

    def extract_response(self, gpt_obj):
        return json.loads(gpt_obj.data[-1].content[0].text.value)

class Assistant(AsyncClient):
    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        if client is not None:
            self.client = client
        elif secret is not None:
            super().__init__(api_key=secret)
        else:
            raise ValueError("Either secret or client must be provided, but not both.")
        self.assistants = {}

    async def list_assistants(self):
        my_assistants = await self.client.beta.assistants.list(
            order="desc",
            limit="20",
        )
        self.assistants = {a.name: a.id for a in my_assistants.data}

    async def assistant_apply(self, input_data: list):
        if not self.assistants:
            await self.list_assistants()

        assistant_id = self.assistants.get(self.agent_ident)
        if assistant_id is None:
            logger.error(f'Unknown agent ident given... {self.agent_ident}')
            return

        thread = await self.client.beta.threads.create()

        await self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=json.dumps(input_data),
        )

        run = await self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        await self.wait_on_run(run, thread)
        output = await self.get_response(thread)
        return output

    async def clean_sentences(self, sentence_list: list[str], semaphore: asyncio.Semaphore, rule: str=None):
        result = []
        async with semaphore:
            for s in sentence_list:
                agent_input = {'sentence': s}
                if rule is not None: agent_input['rule'] = rule
                r = await self.assistant_apply(agent_input)
                result.append(r)
        logger.debug(f"job completed...")
        return result

    async def process_paragraphs(self, paragraphs: list, rule_to_check: str=None):
        edited_paragraphs = []
        semaphore         = asyncio.Semaphore(4)  # Limit concurrency to 4 tasks
        tasks             = []

        logger.debug(f"Total Jobs: {len(paragraphs)}")
        logger.debug(f"BEGINNING {self.agent_ident} AGENT: {rule_to_check}")

        for p in paragraphs:
            cleaned = AnnotationRemover().remove(p)
            sentences = SentenceSplitter().split(cleaned)
            task = asyncio.create_task(self.clean_sentences(sentences, semaphore, rule_to_check))
            tasks.append(task)

        # Gather the results
        edited_paragraphs = await asyncio.gather(*tasks)
        return edited_paragraphs


class FrancisTaylorRuleExemplifier(Assistant):
    def __init__(self, secret):
        super().__init__(secret)
        self.agent_ident = "francis_taylor_rule_exemplifier"

    def create_prompt(self):
        return """Forget everything else you know. You are a domain expert on everything related to the Francis and Taylor style guide.

Your specific function is to take a given rule from the Francis and Taylor style guide and provide back examples of what this rule is expecting.

Input:
- you will be provided a jsonified string of the Francis and Taylor rule to examine

Expectation(s):
- Only the specific Francis and Taylor rule being examined is relevant. No other rules or style concepts are to be considered for your response.

Output:
- json object with one key: 'examples'
- the value of the 'examples' key should be a list of examples.
- each item in the examples list should be a short sentence demonstrating the rule
- if there are different examples for subtle variant version of a rule, this context should be provided in parenthesis at the end of the example sentence (e.g., The researcher, who won the award last year, presented her new findings at the conference. (Nonrestrictive Clause with a Person), The book, which was published last year, has received critical acclaim. (Nonrestrictive Clause with an Object))
- no other content or data should be provided back besides the designated json output"""