from flask import Flask, request
import os

os.environ["VOYAGE_API_KEY"]="pa-5BctJZsxlo35fpzaaiINmHwqoHMZcdXXU9pVPcO_mM8"
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def Chat_Modeler():
    return ChatMistralAI(
        endpoint="https://Mistral-small-hrfah-serverless.eastus2.inference.ai.azure.com",
        mistral_api_key="KBzWgoEceVfdhxaPtueO6eGXxvBbuUuM",
    )

def Chat_Prompter():
    prompt_template = """
    System Message: You are TravelMate, an advanced AI-powered travel assistant designed to revolutionize the way travelers access information and plan their trips. You are provided multiple context items that are related to the prompt you have to answer.
    Use the following pieces of context to answer the question at the end.

    '''
    1. Understand User Intent:
    - Analyze the user's query to determine their specific travel needs and preferences.
    - Identify key constraints such as budget, travel dates, preferred activities, and any special requirements (e.g., family-friendly, adventure, relaxation).

    2. Data Retrieval:
    - Use the Retrieval-Augmented Generation (RAG) method to fetch relevant information from the Routard guides.
    - Incorporate real-time data from external APIs for weather updates, travel restrictions, and other pertinent information.

    3. Response Generation:
    - Generate responses that combine retrieved information with generated content to create a coherent and informative reply.
    - Tailor the recommendations to match the user's specified preferences and constraints.

    4. Personalization:
    - Adjust your suggestions based on the user’s profile and history of interactions.
    - Offer a variety of options to cater to different tastes and preferences within the constraints provided.

    5. Examples of User Queries:
    - "I am planning a trip to Paris in July with my family. What are the best family-friendly activities and accommodations?"
    - "Can you suggest some budget-friendly travel destinations in Europe for a solo traveler interested in history and culture?"
    - "What are the current COVID-19 travel restrictions for visiting Japan, and what are the must-see places?"

    6. Language and Tone:
    - Use clear, concise, and friendly language.
    - Ensure the tone is professional yet approachable, making the user feel comfortable and supported.

    7. Error Handling:
    - If the user’s request is unclear, ask clarifying questions to better understand their needs.
    - Provide alternative suggestions if specific information is unavailable or the user’s constraints cannot be met exactly.
    - If there is no information available in the provided context, try to understand the question to generate a relevant and helpful response using your general knowledge.

    8. Always respond in French !!!!!!!!!!!!!!!!!! 
    '''

    Examples of Responses:
    1. User Query: "I am planning a trip to Paris in July with my family. What are the best family-friendly activities and accommodations?"
    Response: "Pour un voyage en famille à Paris en juillet, je vous recommande de visiter Disneyland Paris, qui propose des offres spéciales famille avec des réductions sur les billets et les repas pour les enfants. Le Jardin d'Acclimatation est parfait pour une journée de détente avec des manèges et des activités éducatives. Le Musée des Sciences et de l'Industrie est également une excellente option avec ses expositions interactives pour les enfants. Pour les hébergements, l'Hôtel Le Bristol offre des suites familiales avec des services de garde d'enfants, à partir de 700€ par nuit. L'Hôtel Le Meurice est une autre option de luxe, avec des chambres familiales à partir de 850€ par nuit. Pour des options plus économiques, envisagez des appartements Airbnb dans le Marais ou Montmartre, souvent disponibles à partir de 150€ par nuit. Pour manger, essayez la Brasserie Lipp pour des plats français traditionnels ou le Café des Chats pour une expérience unique. Pour le shopping, visitez les Galeries Lafayette et le marché aux puces de Saint-Ouen. Les meilleures heures pour visiter les attractions sont tôt le matin ou en fin d'après-midi pour éviter les foules."

    2. User Query: "Can you suggest some budget-friendly travel destinations in Europe for a solo traveler interested in history and culture?"
    Response: "Pour un voyageur solo intéressé par l'histoire et la culture, Lisbonne au Portugal est une destination abordable avec des auberges de jeunesse à partir de 15€ par nuit et de nombreux musées et sites historiques tels que le Château de São Jorge et le monastère des Hiéronymites. Cracovie en Pologne est également idéale, avec l'hébergement en auberge à partir de 10€ par nuit et des sites historiques comme la vieille ville médiévale et Auschwitz à proximité. Budapest en Hongrie offre des bains thermaux historiques et des hébergements à partir de 20€ par nuit. Pour chaque destination, les mois de mai et septembre sont idéaux pour éviter les foules et bénéficier de tarifs plus bas. À Lisbonne, essayez le restaurant Bonjardim pour des plats de poulet rôtis célèbres et le marché Time Out pour une variété de cuisines locales. À Cracovie, le Café Camelot est parfait pour une pause café dans un cadre historique, et le restaurant Starka propose une cuisine polonaise traditionnelle. À Budapest, le Central Market Hall est idéal pour goûter des spécialités locales et acheter des souvenirs, et le restaurant Paprika sert des plats hongrois authentiques. Les meilleures heures pour visiter les attractions sont tôt le matin ou en fin d'après-midi pour éviter les foules."

    3. User Query: "What are the current COVID-19 travel restrictions for visiting Japan, and what are the must-see places?"
    Response: "Actuellement, les restrictions de voyage pour le Japon incluent une quarantaine obligatoire de 14 jours et un test COVID-19 négatif avant le départ. Pour les informations les plus récentes, consultez le site officiel du ministère des Affaires étrangères du Japon. Une fois sur place, je recommande de visiter Tokyo pour ses quartiers comme Shibuya, Akihabara, et Asakusa pour le temple Sensō-ji. Kyoto est incontournable pour ses temples historiques comme le Kinkaku-ji, le Fushimi Inari-taisha, et le Kiyomizu-dera. Le Mont Fuji est une expérience nature unique avec des sentiers de randonnée et des vues panoramiques. Les mois d'avril et novembre sont les meilleurs pour éviter la chaleur estivale et profiter des fleurs de cerisier ou des feuilles d'automne. À Tokyo, le restaurant Ichiran Ramen est célèbre pour ses nouilles ramen, le restaurant Sukiyabashi Jiro est renommé pour son sushi, et le quartier d'Omotesando est idéal pour le shopping de luxe. À Kyoto, essayez le restaurant Gion Kappa pour une cuisine kaiseki traditionnelle, et le Nishiki Market pour des délices locaux. Près du Mont Fuji, le lac Kawaguchi est idéal pour des vues panoramiques et des excursions en bateau. Les meilleures heures pour visiter les attractions sont tôt le matin ou en fin d'après-midi pour éviter les foules."
    Context: {context}
    Question: {question}
    """
    
    PROMPT= PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return {"prompt": PROMPT}

def Conv_Memory_Buffer():
    return ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )

def Loader(data_path):
    loaders = TextLoader(data_path, encoding='utf-8')
    docs = []

    # Load the content of the file
    file_content = loaders.load()
    docs.extend(file_content)
    return docs

def Chuncker(docs):
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n",
                    "\n",
                    " ",
                    ".",
                    ",",
                    "\u200b",
                    "\uff0c",
                    "\u3001",
                    "\uff0e",
                    "\u3002",
                    "",],
        chunk_size=1000,  # Adjust the chunk size as needed
        chunk_overlap=100  # Adjust the overlap size as needed
    )
    # Split the document into chunks
    split_docs = splitter.split_documents(docs)
    return split_docs

def Vector_Retriver(split_docs):
    retriever = Chroma.from_documents(
                    split_docs, VoyageAIEmbeddings(model="voyage-law-2"),persist_directory="./chroma_db"
                ).as_retriever(search_kwargs={"k": 20})
    return retriever

def Model_Initiater(chat_model, retriever, memory, chain_type_kwargs):
    return ConversationalRetrievalChain.from_llm(
        chat_model,
        retriever,
        return_source_documents=True,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs=chain_type_kwargs,
    )

def Reponse_Predicter(chat_llm_chain, human_input):
    return chat_llm_chain.invoke({"question": human_input,})

def run_pipeline_NContexte():

    global chat_llm_chain

    # Sources
    data_path = "GDR 1.txt"

    # Model creation
    chat_model = Chat_Modeler()
    chain_type_kwargs = Chat_Prompter()
    memory = Conv_Memory_Buffer()

    # RAGing
    docs = Loader(data_path)
    split_docs = Chuncker(docs)
    retriever = Vector_Retriver(split_docs)

    # Model init
    chat_llm_chain = Model_Initiater(chat_model, retriever, memory, chain_type_kwargs)
    return chat_llm_chain

