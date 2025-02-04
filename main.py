import asyncio
from datetime import datetime
from loguru import logger
from openai import AsyncOpenAI
from chatgpt_models.assistants.punctuation import (
    Comma, Semicolon, Colon, DashHyphen, QuotationMark,
    Apostrophe, Parentheses, Ellipsis, ExclamationMark, QuestionMark
)
from text_handlers.parsers import (
    ChapterLoader, ParagraphClassifier, ParagraphReconstructor, ParagraphCombiner
)

class TextProcessor:
    def __init__(self, client):
        self.client = client
        self.agent_network = self._create_agent_network()

    def _create_agent_network(self):
        return [
            Comma(client=self.client),
            Semicolon(client=self.client),
            Colon(client=self.client),
            DashHyphen(client=self.client),
            QuotationMark(client=self.client),
            Apostrophe(client=self.client),
            Parentheses(client=self.client),
            Ellipsis(client=self.client),
            ExclamationMark(client=self.client),
            QuestionMark(client=self.client)
        ]

    async def process_text(self, paragraphs):
        for agent in self.agent_network:
            logger.debug(f'Processing with {agent.agent_ident}')
            try:
                edited_paragraphs = await agent.process_paragraphs(paragraphs)
            except Exception as e:
                logger.error(f'Error processing with {agent.agent_ident}: {e}... ending agent network operation and stashing results so far')
                break
            else:
                full = []
                for p in edited_paragraphs:
                    sentences = [agent.extract_response(s) for s in p]
                    para = ParagraphReconstructor().reconstruct(sentences)
                    full.append(para)
                paragraphs = ParagraphCombiner().combine(full)

            self._save_paragraphs(paragraphs)
        return paragraphs

    def _save_paragraphs(self, paragraphs):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"paragraphs_{current_time}.txt"
        with open(filename, "w") as file:
            file.write(paragraphs)
        print(f"Paragraphs have been saved to {filename}")

def load_chapter():
    chapter12_content = ChapterLoader('raja').load('Ch12_SocialMarket_plaintext_cleaned.txt')
    uncertain_blocks, paragraphs = ParagraphClassifier().classify(chapter12_content)
    return paragraphs[1:]

def create_client(secret):
    return AsyncOpenAI(api_key=secret)

async def main():
    secret = ''
    client = create_client(secret)
    paragraphs = load_chapter()
    processor = TextProcessor(client)
    await processor.process_text(paragraphs)

if __name__ == '__main__':
    asyncio.run(main())
