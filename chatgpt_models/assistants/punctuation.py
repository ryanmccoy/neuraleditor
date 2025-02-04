from openai import AsyncOpenAI
import asyncio
from chatgpt_models.assistants.base import Assistant
from dataclasses import dataclass
from loguru import logger

@dataclass
class Rule:
    description: str

class PunctuationAssistant(Assistant):
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

    def create_punctuation_agent_prompt(self, formatted_rules):
        prompt = f"""Forget everything else you know or believe about the world. You are part of a content editing team. This team is a fleet of specially trained GPT assistants. Each member on the team is designed to perform a specific function that in aggregate, would provide a comprehensive editorial service to an author of a full manuscript using the Taylor & Francis style guide.

Your specific focus is on the enforcement of punctuation rules for the usage of {self.punctuation_type}.

Here are the specifics of your role:
Title: Punctuation Enforcement: {self.punctuation_type} Usage

Rules to Enforce:{formatted_rules}

Input:
- you will be provided a json object with one key: 'sentence'
- the value of the 'sentence' key will be the sentence you are to check

Expectation(s):
- check that the sentence obeys all punctuation rules defined
- if the rule is not obeyed by the sentence, rewrite the sentence so that it does, otherwise leave it unchanged
- you only make edits specific to the given rules. if you notice another grammatical or syntactical error, if it does not violate the rules you are supposed to be evaluating you leave the error in place.

Output:
- json object with two keys: 'sentence' and 'edited'
- the value of the 'sentence' key should be the given sentence. if it was not edited by you because it obeyed the rules, provide the same sentence back unaltered. if required edits because it did not obey the rules, provide the edited version so that it now obeys the given rule. this sentence should be passed back in the 'sentence' key of the output json
- the 'edited' key should be True or False, depending on if there were edits required
- no other content or data should be provided back besides the designated json output"""

        return prompt

    async def create_prompt(self):
        rules_with_examples = await self.set_up_rules()
        formatted_rules = "\n- " + "\n- ".join(rules_with_examples)
        return self.create_punctuation_agent_prompt(formatted_rules)

class Comma(PunctuationAssistant):
    """
    - **Commas**:
     - **Serial Comma**: Taylor & Francis style generally uses the serial (Oxford) comma before the final item in a list (e.g., apples, oranges, and bananas).
     - **Introductory Elements**: Use a comma after introductory phrases, clauses, or words that come before the main clause.
     - **Nonrestrictive Clauses**: Use commas to set off nonrestrictive clauses that add non-essential information to a sentence.
     - **Restrictive Clauses**: Do not use commas for restrictive clauses that are essential to the sentence’s meaning.
     - **Conjunctions**: Use a comma before coordinating conjunctions (and, but, or, nor, for, so, yet) when they join two independent clauses.
     - **Parenthetical Elements**: Set off non-essential elements in a sentence with commas.
    """
    RULES = [
        Rule(description="Serial Comma: Taylor & Francis style generally uses the serial (Oxford) comma before the final item in a list"),
        Rule(description="Introductory Elements: Use a comma after introductory phrases that come before the main clause."),
        Rule(description="Introductory Elements: Use a comma after introductory clauses that come before the main clause."),
        Rule(description="Introductory Elements: Use a comma after introductory words that come before the main clause."),
        Rule(description="Nonrestrictive Clauses: Use commas to set off nonrestrictive clauses that add non-essential information to a sentence."),
        Rule(description="Restrictive Clauses: Do not use commas for restrictive clauses that are essential to the sentence's meaning."),
        Rule(description="Conjunctions: Use a comma before coordinating conjunctions (and, but, or, nor, for, so, yet) when they join two independent clauses."),
        Rule(description="Parenthetical Elements: Set off non-essential elements in a sentence with commas.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_comma_agent"
        self.rules = self.RULES
        self.punctuation_type = "commas"

class Semicolon(PunctuationAssistant):
    """
    - **Semicolons**:
     - **Independent Clauses**: Use semicolons to link closely related independent clauses that are not joined by a conjunction.
     - **Complex Lists**: Use semicolons to separate items in a list when the items themselves contain commas for clarity.
    """
    RULES = [
        Rule(description="Independent Clauses: Use semicolons to link closely related independent clauses that are not joined by a conjunction."),
        Rule(description="Complex Lists: Use semicolons to separate items in a list when the items themselves contain commas for clarity.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_semicolon_agent"
        self.rules = self.RULES
        self.punctuation_type = "semicolons"

class Colon(PunctuationAssistant):
    """
    - **Colons**:
     - **Introducing Lists or Explanations**: Use a colon after a complete sentence to introduce a list, quote, or explanation.
     - **Time**: Use colons to separate hours and minutes in time expressions (e.g., 10:30 a.m.).
     - **Subtitles**: Use colons to separate a title from its subtitle (e.g., *Title: Subtitle*).
    """
    RULES = [
        Rule(description="Introducing Lists or Explanations: Use a colon after a complete sentence to introduce a list, quote, or explanation."),
        Rule(description="Time: Use colons to separate hours and minutes in time expressions."),
        Rule(description="Subtitles: Use colons to separate a title from its subtitle.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_colon_agent"
        self.rules = self.RULES
        self.punctuation_type = "colons"

class DashHyphen(PunctuationAssistant):
    """
    - **Dashes and Hyphens**:
     - **Hyphen**: Use hyphens to form compound adjectives and to avoid ambiguity (e.g., well-known, re-cover vs. recover).
     - **En Dash**: Use en dashes for number ranges (e.g., 2010–2020) and in compound adjectives involving open compounds (e.g., New York–London flight).
     - **Em Dash**: Use em dashes sparingly to set off parenthetical statements or for emphasis, without spaces on either side.
    """
    RULES = [
        Rule(description="Hyphen: Use hyphens to form compound adjectives and to avoid ambiguity."),
        Rule(description="En Dash: Use en dashes for number ranges and in compound adjectives involving open compounds."),
        Rule(description="Em Dash: Use em dashes sparingly to set off parenthetical statements or for emphasis, without spaces on either side.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_dash_hyphen_agent"
        self.rules = self.RULES
        self.punctuation_type = "dashes and hyphens"

class QuotationMark(PunctuationAssistant):
    """
    - **Quotation Marks**:
     - **Double Quotation Marks**: Use double quotation marks for direct quotes and titles of shorter works (e.g., articles, poems).
     - **Single Quotation Marks**: Use single quotation marks for quotes within quotes.
     - **Punctuation Inside Quotation Marks**: Periods and commas are placed inside quotation marks. Colons and semicolons are placed outside. Question marks and exclamation points are placed inside if part of the quoted material and outside if they apply to the entire sentence.
    """
    RULES = [
        Rule(description="Double Quotation Marks: Use double quotation marks for direct quotes and titles of shorter works."),
        Rule(description="Single Quotation Marks: Use single quotation marks for quotes within quotes."),
        Rule(description="Punctuation Inside Quotation Marks: Periods and commas are placed inside quotation marks. Colons and semicolons are placed outside. Question marks and exclamation points are placed inside if part of the quoted material and outside if they apply to the entire sentence.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_quotation_agent"
        self.rules = self.RULES
        self.punctuation_type = "quotation marks"

class Apostrophe(PunctuationAssistant):
    """
    - **Apostrophes**:
     - **Possessives**: Form the possessive of singular nouns with 's (e.g., the author’s study), and for plural nouns ending in s with just an apostrophe (e.g., the authors’ findings).
     - **Contractions**: Use apostrophes in contractions to indicate omitted letters (e.g., don’t, it’s).
     - **Plural Forms**: Avoid using apostrophes for plural forms of abbreviations or numbers (e.g., 1990s, MPs).
    """
    RULES = [
        Rule(description="Possessives: Form the possessive of singular nouns with 's, and for plural nouns ending in s with just an apostrophe."),
        Rule(description="Contractions: Use apostrophes in contractions to indicate omitted letters."),
        Rule(description="Plural Forms: Avoid using apostrophes for plural forms of abbreviations or numbers.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_apostrophe_agent"
        self.rules = self.RULES
        self.punctuation_type = "apostrophes"

class Parentheses(PunctuationAssistant):
    """
    - **Parentheses**:
     - **Non-Essential Information**: Use parentheses to enclose supplementary or non-essential information.
     - **Punctuation Placement**: Punctuation marks are placed outside the parentheses unless the entire sentence is within the parentheses.
    """
    RULES = [
        Rule(description="Non-Essential Information: Use parentheses to enclose supplementary or non-essential information."),
        Rule(description="Punctuation Placement: Punctuation marks are placed outside the parentheses unless the entire sentence is within the parentheses.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_parentheses_agent"
        self.rules = self.RULES
        self.punctuation_type = "parentheses"

class Ellipsis(PunctuationAssistant):
    """
    - **Ellipses**:
     - **Omission of Words**: Use ellipses to indicate omissions within a quotation. Three dots should be used for omissions within a sentence, and four dots for omissions between sentences.
     - **End of Sentence**: If an ellipsis appears at the end of a sentence, add a period before the ellipsis (making four dots in total).
    """
    RULES = [
        Rule(description="Omission of Words: Use ellipses to indicate omissions within a quotation. Three dots should be used for omissions within a sentence, and four dots for omissions between sentences."),
        Rule(description="End of Sentence: If an ellipsis appears at the end of a sentence, add a period before the ellipsis making four dots in total.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_ellipses_agent"
        self.rules = self.RULES
        self.punctuation_type = "ellipses"

class ExclamationMark(PunctuationAssistant):
    """
    - **Exclamation Points**:
     - **Use Sparingly**: Exclamation points should be used sparingly and only for strong emphasis.
    """
    RULES = [
        Rule(description="Exclamation points should be used sparingly and only for strong emphasis.")
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_exclamation_agent"
        self.rules = self.RULES
        self.punctuation_type = "exclamation marks"

class QuestionMark(PunctuationAssistant):
    """
    - **Question Marks**:
     - **Direct Questions**: Place a question mark at the end of a direct question.
     - **Indirect Questions**: Do not use a question mark for indirect questions.
    """
    RULES = [
        Rule(description="Direct Questions: Place a question mark at the end of a direct question."),
        Rule(description="Indirect Questions: Do not use a question mark for indirect questions."),
    ]

    def __init__(self, secret: str|None=None, client: AsyncOpenAI|None=None):
        super().__init__(secret, client)
        self.agent_ident = "punctuation_rules_question_marks_agent"
        self.rules = self.RULES
        self.punctuation_type = "question marks"