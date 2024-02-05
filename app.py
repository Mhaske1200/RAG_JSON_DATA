import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route('/get_answer', methods=['GET'])
def get_answer():
    try:
        # Get the text input from the request
        text_input = request.args.get('text')
        print("Question :: ", text_input)

        # Use TextLoader to load text from a local file
        text_loader = TextLoader("json_data.txt")
        docs = text_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the text
        retriever = vectorstore.as_retriever()
        # Replace "rlm/rag-prompt" with the actual path to your RAG model prompt
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response_generate = rag_chain.invoke(text_input)

        print("RAG OUTPUT :: ", response_generate)

        # Extract the generated response from OpenAI
        generated_text = response_generate.content if hasattr(response_generate, 'content') else str(response_generate)

        vectorstore.delete_collection()

        # Return the generated response as JSON
        return jsonify({'answer': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
