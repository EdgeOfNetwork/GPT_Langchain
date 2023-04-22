from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
import os
import config


def pdf_reader(raw_text):
    reader = PdfReader("Demian.pdf")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def summarizer():
    llm = OpenAI(temperature=0)
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    summarize_document_chain.run(raw_text)
    print(summarize_document_chain.run(raw_text))

def question():
    model = ChatOpenAI(model="gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4
    qa_chain = load_qa_chain(model, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    return qa_document_chain

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = config.api_key
    raw_text = pdf_reader("")

    #summarizer()

    qa_dc_chain = question()
    print(qa_dc_chain.run(input_document = raw_text, question = "싱클레어를 괴롭힌 사람은?"))
    print(qa_dc_chain.run(input_document=raw_text, question="싱클레어는 데미안을 어디서 만났지?"))
    print(qa_dc_chain.run(input_document=raw_text, question="데미안의 외모를 묘사해줘"))
