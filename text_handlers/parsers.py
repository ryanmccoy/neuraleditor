import re
import os
import nltk
from abc import ABC, abstractmethod
# nltk.download('punkt')
# nltk.download('punkt_tab')

class TextCleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        pass

class WhitespaceNormalizer(TextCleaner):
    def clean(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text)

class NonUTF8Remover(TextCleaner):
    def clean(self, text: str) -> str:
        return text.encode('utf-8', 'ignore').decode('utf-8')

class CompositeTextCleaner(TextCleaner):
    def __init__(self, cleaners: list[TextCleaner]):
        self.cleaners = cleaners

    def clean(self, text: str) -> str:
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text

class ChapterLoader:
    def __init__(self, input_directory: str):
        self.input_directory = input_directory

    def load(self, chapter_file: str) -> str:
        if chapter_file.endswith('.txt'):
            chapter_path = os.path.join(self.input_directory, chapter_file)
            with open(chapter_path, 'r', encoding='utf-8') as file:
                return file.read()
        return ""

class ParagraphClassifier:
    def classify(self, text: str) -> tuple[list[str], list[str]]:
        raw_paragraphs = re.split(r'\n\s*\n', text.strip())
        uncertain_blocks = []
        definite_paragraphs = []

        for para in raw_paragraphs:
            sub_paragraphs = para.split('\n')
            combined_para = " ".join([sub.strip() for sub in sub_paragraphs if len(sub.strip()) > 40 or sub.strip().endswith(('.', '!', '?'))])

            if combined_para:
                sentence_count = combined_para.count('.')
                if sentence_count == 0 or sentence_count == 1:
                    uncertain_blocks.append(combined_para)
                else:
                    definite_paragraphs.append(combined_para)

        return uncertain_blocks, definite_paragraphs

class AnnotationRemover:
    def remove(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\s*,\s*', ' ', text)
        text = re.sub(r'\s*\[\s*', ' ', text)
        text = re.sub(r'\s*\]\s*', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

class SentenceSplitter:
    def split(self, paragraph: str) -> list[str]:
        return nltk.sent_tokenize(paragraph)

class ParagraphReconstructor:
    def reconstruct(self, sentences: list) -> str:
        reconstructed_sentences = []
        for s in sentences:
            sentence = s['sentence'].strip()
            if not sentence.endswith('.'):
                sentence += '.'
            reconstructed_sentences.append(sentence)
        return ' '.join(reconstructed_sentences)


class ParagraphCombiner:
    def combine(self, paragraphs: list) -> str:
        return '\n\n'.join(paragraph.strip() for paragraph in paragraphs)