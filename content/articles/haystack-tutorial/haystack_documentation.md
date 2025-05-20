| Title                        | Date       | Author          |
|------------------------------|------------|-----------------|
| Building RAGs using Haystack | 2025-05-20 | Antonio Del Rio |
---

# 1. Introduction and Terminology
This tutorial provides a comprehensive, step-by-step guide for building a RAG pipeline using the open-source Haystack 
framework. In this walkthrough, you will learn how to install and configure Haystack, prepare your knowledge base 
using various data formats (such as text files, PDFs, and website content), embed documents, and construct a fully functional RAG pipeline that answers questions grounded in external data.

To enhance your learning experience, an interactive Jupyter Notebook accompanies this tutorial. The notebook mirrors the content presented here and allows you to execute code, test components, and build your pipeline step by step in a live environment.

Note - This tutorial was inspired by and builds upon the concepts and implementations presented in the following 
resources:
- [Haystack Tutorial: Creating Your First QA Pipeline with Retrieval-Augmentation](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline)
- [DeepLearning.AI Short Course: Building AI Applications with Haystack](https://www.deeplearning.ai/short-courses/building-ai-applications-with-haystack/)


--- 

**Haystack** is an open-source framework for building search and question-answering systems powered by Natural 
Language Processing (NLP). It supports various components such as document retrieval, question answering, and 
generative models, making it suitable for implementing complex pipelines like Retrieval-Augmented Generation (RAG). 
The full documentation for Haystack can be found [here](https://docs.haystack.deepset.ai/docs/intro).

**Retrieval-Augmented Generation (RAG)** is a pipeline technique that retrieves relevant information from a 
knowledge base to produce more accurate and contextually relevant answers to a provided query. Instead of relying on 
the generative model's training data to generate a response, the RAG pipeline retrieves relevant documents from a 
defined set of documents (the knowledge base) and passes them onto the generative model to generate a response. This 
approach allows the 
model to access external up-to-date information to more accurately respond to queries. 

Example of queries to generative models with and without RAG pipeline:
- *Without* RAG pipeline: Answer the question (based on what you have been trained on)
- *With* RAG pipeline: Answer the question AND base your answer on the provided document(s)

Key terminology:

| Term                 | What It Means                                                                                               |
|----------------------|-------------------------------------------------------------------------------------------------------------|
| **Document**         | A unit of text (e.g. paragraph, webpage, etc.)                                                              |
| **Chunking**         | Splitting longer documents into smaller, more manageable parts (chunks) before passing through the embedder |
| **Embedder**         | Turns text into numerical vectors (embeddings)                                                              |
| **Document Store**   | A storage component that holds chunked embedded documents                                                   |
| **Retriever**        | Finds relevant documents based on a query (using vector similarity)                                         |
| **Reader/Generator** | Extracts/Generates answers from retrieved documents                                                         |

# 2. Haystack Installation and Environment Setup

## Haystack Installation

Haystack and its required packages can be installed from *haystack-ai* using pip, uv, or any other preferred package 
manager:

```console
pip install haystack-ai
```

To use Haystack components for Google's Gemini model, *google-ai-haystack* and its required packages need to 
 be installed in addition:
```console
pip install google-ai-haystack
```

## API Key Setup
When building a RAG pipeline, certain components may use AI models that require access to an API key.
In this documentation, the following components require an API key:

- OpenAIDocumentEmbedder → requires OPENAI_API_KEY
- OpenAITextEmbedder → requires OPENAI_API_KEY
- OpenAIChatGenerator → requires OPENAI_API_KEY
- GoogleAIGeminiChatGenerator → requires GOOGLE_API_KEY

To store API keys for use in the RAG pipeline, a .env file can be created in the project space and the API 
keys can be listed in the file:
```text
OPENAI_API_KEY = ...OpenAI API key goes here...
GOOGLE_API_KEY = ...Google Gemini API key goes here...
```

To access the .env file containing the API keys, the library *dotenv* is needed:
```console
pip install dotenv
```
When working with a standard Python script, the API keys in the .env file can be loaded by running the function 
*load_dotenv*:
```python
from dotenv import load_dotenv
load_dotenv()
```
When working in a jupyter notebook Python environment, the following code can be used instead of the *load_dotenv* 
function from above:
```python
%load_ext dotenv
%dotenv
```

# 3. Haystack Components for Building a RAG Pipeline

## Haystack Document

The idea of using a RAG pipeline is to retrieve relevant information from a given knowledge base. To set up 
this knowledge base, documents must first be provided. To do this, the Haystack *Document* object is used. In 
this object, the content and metadata of the individual document can be stored. For a set of documents, the 
individual *Document* objects can be added to a documents list:

```python
from haystack import Document
documents = [Document(content="Content of document 1", meta=["Metadata of document 1"]),
             Document(content="Content of document 2", meta=["Metadata of document 2"]),
             ...]
```

Instead of listing the contents of each document to be added to the knowledge base, one can also work with local 
datasets. To do this, the *datasets* library needs to first be installed:

```console
pip install datasets
```
For a local dataset stored as a .txt or as a .csv file, the function *load_dataset* from the library *datasets* can 
be used. With a new dataset object, its contents and metadata can be added iteratively to the Haystack document object.
```python
from datasets import load_dataset
dataset = load_dataset('txt', data_files='path/to/your/file.txt')
# dataset = load_dataset('csv', data_files='path/to/your/file.csv')

docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
```

## Document Store
To store the documents from the knowledge base, we can initialize a Haystack document store object. This object is 
essentially a database for the knowledge base which the RAG's retriever can then access when a query is provided. 
The simplest document store that can be used is *InMemoryDocumentStore()*:
```python
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
```
There are different document store types that can be used depending on the scope of the RAG pipeline being built. 
The different available document stores as well as descriptions of their advantages and disadvantages can be found 
[here](https://docs.haystack.deepset.ai/docs/choosing-a-document-store). 

Note: Chunked document embeddings rather than the original documents are stored inside the document store.

## Document Embedding - Manual Method

The document embedding process involves first "chunking", or dividing, documents in the knowledge base into a list 
of shorter text documents. This can be done using the *DocumentSplitter* component, which by default chunks the 
contents of the document list into chunks of 200 words:

```python
from haystack.components.preprocessors import DocumentSplitter
splitter = DocumentSplitter() # default: splits documents by 200 words
chunked_docs = splitter.run(documents=docs)
```

The chunked documents can then be passed through an embedder component, which converts text into embeddings using a 
certain pre-trained model. Haystack offers various components for doing this, a list of which can be found 
[here](https://docs.haystack.deepset.ai/docs/embedders). Certain embedder components are local embedders that run 
locally and do not need an API key, while others access a non-local embedder model, requiring an API key for use. 
Once the documents embeddings have been calculated, they are then stored in the defined document store object 
through the *write_documents* method.

Below is an example of a local document embedder using the *SentenceTransformersDocumentEmbedder* component and a 
non-local document embedder using the *OpenAIDocumentEmbedder* component (which uses an OpenAI embedding model):

### Example of local document embedder

```python
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up() # This embedder runs locally, so it needs to be preloaded
docs_with_embeddings = embedder.run(documents=chunked_docs)
document_store.write_documents(docs_with_embeddings["documents"])
```
### Example of non-local document embedder
```python
from haystack.components.embedders import OpenAIDocumentEmbedder

embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
# External OpenAI embedder used --> no local model to warm up (but requires API)
docs_with_embeddings = embedder.run(documents=chunked_docs)
document_store.write_documents(docs_with_embeddings["documents"])
```

## Document Embedding - Using a Pipeline

The above shows the "manual" method for embedding documents in the knowledge base, i.e. by calling individual 
methods of the used components (*splitter.run()*, *embedder.run()*, etc.). The document embedding step can be 
streamlined by creating a Haystack Pipeline object, adding and connecting the components to it, and feeding the 
documents through the object. 

Firstly, the necessary components need to be imported. In the below example, since the data source (knowledge base) 
is in the form of a .txt file, the component *TextFileToDocument* can be used to automatically convert the text file 
into a valid Haystack Document object. Haystack offers a wide variety of converter components to convert .csv, .html,
.doc, and more data sources into valid Haystack Document objects, all of which can be found 
[here](https://docs.haystack.deepset.ai/docs/converters).

```python
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter

data_source = "data/document_text_file.txt"

converter = TextFileToDocument()
splitter = DocumentSplitter() #default: splits documents by 200 words
embedder = OpenAIDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)
```

Next a Haystack Pipeline can be initialized, and the required components can be added to this object using the *.
add_component()" method:

```python
from haystack import Pipeline
indexing_pipeline = Pipeline()

indexing_pipeline.add_component(name="converter", instance=converter)
indexing_pipeline.add_component(name="splitter", instance=splitter)
indexing_pipeline.add_component(name="embedder", instance=embedder)
indexing_pipeline.add_component(name="writer", instance=writer)
```
The added components must then be connected correctly to build the pipeline. The steps for properly processing the 
documents for the knowledge base are:

1. Convert data source into a proper Haystack document object using the converter component
2. Chunk the documents into smaller documents using the splitter component
3. Embed the chunked documents using the embedder component
4. Write the embeddings to the document store using the writer component

The components can be connected using the *.connect()* method:

```python
indexing_pipeline.connect(sender="converter", receiver="splitter")
indexing_pipeline.connect(sender="splitter", receiver="embedder")
indexing_pipeline.connect(sender="embedder", receiver="writer")
```
Now that the pipeline for embedding the documents has been properly created, the pipeline can be run be calling the 
*.run()* method and defining the data source to be passed into the converter component (the first component of the 
pipeline):

```python
indexing_pipeline.run({"converter": {"sources": [data_source]}})
```
To visualize the created Haystack Pipeline and how the pipeline's components are connected to each other, the *.show()* 
method can be used:
```python
indexing_pipeline.show()
```
![indexing_pipeline_show.png](indexing_pipeline_show.png)

## Prompt Layout

In a RAG pipeline, the idea is for the AI model to answer the question and base its answer on the provided 
document(s). The prompt that goes to the AI model should therefore be:

- Given the following information, answer the question.
- Context: Loop through all documents in knowledge base
- Question: User's query / question
- Answer: To be given by AI model

Using the Haystack dataclass *ChatMessage* and Jinja2 looping syntax, the following RAG prompt template can be 
defined, which loops through the documents in the knowledge base and takes in the user's 
query:

```python
from haystack.dataclasses import ChatMessage

prompt = [ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")
          ]
```

## Query Embedding, Prompt Building, and Putting RAG Together

With the prompt template defined above, a new Haystack Pipeline can be built to take the user's query and embedded 
chunked documents, pass them through an AI model, and return an answer to the user's query. 

Firstly, the following necessary components need to be imported and added to the RAG Haystack Pipeline object:
- *query_embedder* → Embeds text (which will be the user's query)
- *retriever* → Retrieves chunked document embeddings stored in the defined document store
- *prompt_builder* → Builds the prompt based on the prompt template to be given to the AI model
- *chat_generator* → Takes in the user's full prompt, passes it through the Gemini AI model, and 
  returns an answer to the user's query
  - *GoogleAIGeminiChatGenerator* → Uses Google's Gemini chat model for generating an answer to the prompt
  - *OpenAIChatGenerator* → Uses OpenAI's chat model for generating an answer to the prompt

```python
from haystack import Pipeline

from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator
# from haystack.components.generators.chat import OpenAIChatGenerator --> OpenAI Chat Generator

query_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt)
chat_generator = GoogleAIGeminiChatGenerator()
# chat_generator = OpenAIChatGenerator(model="gpt-4o-mini") --> use for OpenAI Chat Generator

rag = Pipeline()
rag.add_component("query_embedder", query_embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt_builder", prompt_builder)
rag.add_component("llm", chat_generator)
```

To complete the RAG pipeline, the above-added components need to be connected correctly. The 
steps for properly 
processing the user's query in a RAG pipeline are:

1. Embed the user's query using the query embedder component
2. Retrieve the chunked document embeddings stored in the defined document store using the retriever component
3. Build a prompt using the defined prompt template and the prompt builder component
4. Pass the full prompt to the AI model to generate an answer to the user's query using the chat generator component

```python
rag.connect("query_embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt_builder.documents")
rag.connect("prompt_builder.prompt", "llm.messages")
```
To visualize the created Haystack Pipeline and how the pipeline's components are connected to each other, the *.show()* 
method can be used:
```python
rag.show()
```
![rag_view.png](rag_view.png)


## Asking Question

With the RAG pipeline fully built, it can be run by calling the *.run()* method. To run the RAG pipeline, the user must:
1. Define their query; and
2. Specify the integer value for the *top_k* parameter, which controls the number of top document chunks are 
   retrieved based on the similarity to the user's query

After running the RAG pipeline, the response returned by the RAG pipeline can then be called and displayed:
```python
question = "...ask question here...?" 

response = rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 5},
        "prompt_builder": {"question": question},
    }
)

rag_response = response["llm"]["replies"][0].text
print(rag_response)
```

# 4. Building Custom Haystack Components

The RAG pipeline example from above has been built entirely with existing Haystack components. Haystack also has 
support for building custom components. Each Haystack component must fulfill the following requirements:
1. It must be a class with the ```@component``` decorator
2. It must include a ```run()``` method with a ```@component.output_types```decorator
3. It must return a dictionary

An example of a simple component is the following:

```python
from haystack import component

@component # class with @component decorator
class favorite_animal:

    @component.output_types(output=str) # run() method with @component.output_types decorator
    def run(self, animal_name: str):
        return {"output": f"My favorite animal is the {animal_name}!"} # dictionary returned
```
With the custom component ```favorite_animal``` built, it can be called and run:

```python
my_favorite_animal = favorite_animal()
my_favorite_animal.run(animal_name="dog")
```

As with the preset Haystack components, custom Haystack components can be added to Haystack pipelines. In the below example, the pipeline will:

1. Construct the sentence "My favorite animal is the ```{{ favorite_animal }}```!" given the inputted ```favorite_animal``` (custom component ```favorite_animal()```)
2. Prompts the Google Gemini AI agent to write a short paragraph about the given favorite animal (preset component ```ChatPromptBuilder()```)
3. Generate the short paragraph about the given favorite animal given the existing knowledge of the AI agent (preset component ```GoogleAIGeminiChatGenerator()```)

First, initialize the prompt to pass to the AI agent:

```python
from haystack.dataclasses import ChatMessage

prompt = [ChatMessage.from_user("""
You are given the name of the favorite animal of the user. Write a short paragraph about the animal.
Start of paragraph: {{ favorite_animal }}
Paragraph:
""")]
```
Next, initialize the two preset components:

```python
# custom component
my_favorite_animal = favorite_animal()

# preset components
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

prompt_builder = ChatPromptBuilder(template=prompt)
chat_generator = GoogleAIGeminiChatGenerator()
```
Next, initialize the Haystack pipeline, add the three components, and connect them correctly:

```python
from haystack import Pipeline

favorite_animal_pipeline = Pipeline()
favorite_animal_pipeline.add_component("my_favorite_animal", my_favorite_animal)
favorite_animal_pipeline.add_component("prompt_builder", prompt_builder)
favorite_animal_pipeline.add_component("llm", chat_generator)

favorite_animal_pipeline.connect("my_favorite_animal.output", "prompt_builder.favorite_animal")
favorite_animal_pipeline.connect("prompt_builder.prompt", "llm.messages")
```
Finally, generate the desired answer for your favorite animal (in this example, the author's favorite animal is the dog):

```python
# run the pipeline to generate a complete answer
answer = favorite_animal_pipeline.run({"my_favorite_animal": {"animal_name": "dog"}})

# display the text of the AI agent's reply
print(answer["llm"]["replies"][0].text)
```

# 5. Converting PDFs into Documents

The above examples have considered information for the RAG's knowledge base as pure text stored in a .txt file. Haystack has various converter components that can convert data in various forms into the embedding form for the RAG to be able to use. This section covers the simplest method to convert documents in PDF form into Haystack documents.

To convert a PDF into content that can be stored in the knowledge base of the RAG, the Haystack component *PyPDFToDocument* is used. The *pypdf* package needs to be installed to use the Haystack component, which needs to be installed:

```console
pip install pypdf
```

To demonstrate the functionality of the converter, the below code can be run. The file path to a document. The converter is first initialized and the document's file path is passed into the run method of the converter. The *sources* parameter of the run method can take in a list of file paths to convert multiple PDFs into a list of document text.

```python
from haystack.components.converters import PyPDFToDocument

file_path = 'path_to_file.pdf'
converter = PyPDFToDocument()
docs = converter.run(sources=[file_path])
```

The contents of the above *docs* object can be inspected by printing out the contents of each document:

```python
for i, doc in enumerate(docs['documents']):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.content)
```

The process of converting PDFs into document embeddings for a RAG pipeline to use can be streamlined using a Haystack Pipeline. The object *pdf_embedder_pipeline* has the following components:
- Converter → Converts PDF into a text document
- Cleaner → Cleans the text document by removing things like empty lines, unsupported characters, extra whitespaces, etc.
- Splitter → Chunks the document into a list of smaller documents
- Embedder → Turns the chunked documents into numerical vectors
- Writer → Stores chunked document embeddings in the document store object

After adding and connecting the pipeline's components, the pipeline can be run for the file paths of the PDFs.

```python
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter

document_store = InMemoryDocumentStore()
file_paths = ["document_uzh.pdf"]

pdf_embedder_pipeline = Pipeline()

converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter()
embedder = OpenAIDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)

pdf_embedder_pipeline.add_component("converter", converter)
pdf_embedder_pipeline.add_component("cleaner", cleaner)
pdf_embedder_pipeline.add_component("splitter", splitter)
pdf_embedder_pipeline.add_component("embedder", embedder)
pdf_embedder_pipeline.add_component("writer", writer)

pdf_embedder_pipeline.connect("converter", "cleaner")
pdf_embedder_pipeline.connect("cleaner", "splitter")
pdf_embedder_pipeline.connect("splitter", "embedder")
pdf_embedder_pipeline.connect("embedder", "writer")

pdf_embedder_pipeline.run({"converter": {"sources": file_paths}})
```

As was done for previous RAG pipelines, the following prompt template can be defined:

```python
from haystack.dataclasses import ChatMessage

prompt = [ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")
          ]
```

The RAG pipeline can then be built in the same was as was shown previously:

```python
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

query_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt)
chat_generator = GoogleAIGeminiChatGenerator()

pdf_document_rag = Pipeline()

pdf_document_rag.add_component("query_embedder", query_embedder)
pdf_document_rag.add_component("retriever", retriever)
pdf_document_rag.add_component("prompt_builder", prompt_builder)
pdf_document_rag.add_component("llm", chat_generator)

pdf_document_rag.connect("query_embedder.embedding", "retriever.query_embedding")
pdf_document_rag.connect("retriever.documents", "prompt_builder.documents")
pdf_document_rag.connect("prompt_builder.prompt", "llm.messages")
```

With the PDF to embeddings pipeline and RAG pipeline ready, questions relating to the contents of the PDF document can now be asked in the same way as was shown previously:

```python
question = "...ask question here...?"

response = pdf_document_rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 5},
        "prompt_builder": {"question": question},
    }
)

rag_response = response["llm"]["replies"][0].text
print(rag_response)
```

It is important to note that the *pypdf* package used by the *PyPDFToDocument* Haystack component is limited in how well it can convert PDFs into text. Data such as tables, images, and hyperlinks cannot be properly converted into text using this component alone. There are solutions such as [Docling](https://github.com/docling-project/docling) that can be implemented to address these issues.


# 6. Converting Website Content into Documents

This section covers the simplest method to convert data stored on websites into Haystack documents.

To convert the content of a website into a form that a RAG can use, the Haystack components *LinkContentFetcher* and *HTMLToDocument* are used. The former fetches data from a website link as an HTML file, while the latter converts the fetched HTML file into a document object.

To demonstrate the functionality of the converter, the below code can be run. The website links can be specified in a list object. The fetcher and converter are first initialized, the website links are then passed into the run method of the fetcher, and then the HTML objects are passed into the run method of the converter. Both the *urls* and *sources* parameters of the fetcher and converter respectively are lists.

```python
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument

urls = ["https://...website url goes here..."]

fetcher = LinkContentFetcher()
converter = HTMLToDocument()

website_fetcher = fetcher.run(urls=urls)

website_docs = converter.run(sources=website_fetcher['streams'])
```

The contents of the above *docs* object can be inspected by printing out the contents of each document:

```python
for i, doc in enumerate(website_docs['documents']):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.content)
```

The process of converting website content into document embeddings for a RAG pipeline to use can be streamlined using a Haystack Pipeline. The object *website_embedder_pipeline* is almost exactly the same as the *pdf_embedder_pipeline* from the previous section, except there is an additional *LinkContentFetcher*  component and the *PyPDFToDocument* component is replaced with the *HTMLToDocument* component.

After adding and connecting the pipeline's components, the pipeline can be run for website links.

```python
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

document_store = InMemoryDocumentStore()

websites = ["https://www.df.uzh.ch/en/studies/bachelor-master/master/ma-banking-finance.html"]

website_embedder_pipeline = Pipeline()

fetcher = LinkContentFetcher()
converter = HTMLToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter()
embedder = OpenAIDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)

website_embedder_pipeline.add_component("fetcher", fetcher)
website_embedder_pipeline.add_component("converter", converter)
website_embedder_pipeline.add_component("cleaner", cleaner)
website_embedder_pipeline.add_component("splitter", splitter)
website_embedder_pipeline.add_component("embedder", embedder)
website_embedder_pipeline.add_component("writer", writer)

website_embedder_pipeline.connect("fetcher", "converter")
website_embedder_pipeline.connect("converter", "cleaner")
website_embedder_pipeline.connect("cleaner", "splitter")
website_embedder_pipeline.connect("splitter", "embedder")
website_embedder_pipeline.connect("embedder", "writer")

website_embedder_pipeline.run({"fetcher": {"urls": websites}})
```

The same prompt template and RAG pipeline from the previous section can be reused:

```python
from haystack.dataclasses import ChatMessage

prompt = [ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")
          ]
```

```python
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

query_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt)
chat_generator = GoogleAIGeminiChatGenerator()

website_rag = Pipeline()

website_rag.add_component("query_embedder", query_embedder)
website_rag.add_component("retriever", retriever)
website_rag.add_component("prompt_builder", prompt_builder)
website_rag.add_component("llm", chat_generator)

website_rag.connect("query_embedder.embedding", "retriever.query_embedding")
website_rag.connect("retriever.documents", "prompt_builder.documents")
website_rag.connect("prompt_builder.prompt", "llm.messages")
```

With the website to embeddings pipeline and RAG pipeline ready, questions relating to the contents of the website can now be asked in the same way as was shown previously:

```python
question = "...question goes here...?"

response = website_rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 5},
        "prompt_builder": {"question": question},
    }
)

rag_response = response["llm"]["replies"][0].text
print(rag_response)
```
As with the PDF converter, the fetcher and converter components used for retrieving website data cannot convert more complicated forms of data, such as tables, photos, and hyperlinks, into text format. There are solutions such as [Firecrawl](https://www.firecrawl.dev) that can be implemented to address these issues.
