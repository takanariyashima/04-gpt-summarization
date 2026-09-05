"""
Text Summarization API(抽出型要約)。

入力された長文から重要な文を選んで抜き出す「抽出型(extractive)」要約。
sumy(LexRankアルゴリズム)を使用し、外部AI API・外部サイトへの
通信は一切発生しない(完全にローカルで処理を完結する)。

生成型(abstractive、新しい文章を作る方式)ではなく、
入力文の中から重要な文をそのまま選んで返す方式。
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = FastAPI(
    title="Text Summarization API",
    description="長文から重要な文を抽出して要約します(抽出型要約)。",
    version="1.0.0",
)


def summarize(text: str, sentence_count: int = 3) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="要約したい英文")
    sentence_count: int = Field(3, ge=1, le=10, description="抽出する文の数")


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/")
def root():
    return {"status": "ok", "service": "text-summarization"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    return SummarizeResponse(summary=summarize(req.text, req.sentence_count))
