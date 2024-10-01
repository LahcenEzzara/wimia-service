from django.contrib.auth.decorators import user_passes_test
from django.contrib import auth
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings

import os
import torch
import pandas as pd
import re
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .models import Chat

# Load environment variables
load_dotenv()

# Load the fine-tuned model and tokenizer
model_name = os.path.join(settings.BASE_DIR, 'models', 'fine-tuned-model')
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load and clean the dataset
file_path = os.path.join(settings.BASE_DIR, 'models', 'chats.csv')
df = pd.read_csv(file_path, delimiter=',')


def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()


df['cleaned_message'] = df['Message'].apply(clean_text)

# Convert WimTech responses into Document objects
documents = [Document(page_content=text)
             for text in df['cleaned_message'].tolist()]

# Initialize embeddings and FAISS index
embedding_model = OpenAIEmbeddings()
faiss_index = FAISS.from_documents(documents, embedding_model)

# Set up the retriever
retriever = faiss_index.as_retriever()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define a custom prompt template
template = """
You are an AI assistant at WimTech. Below is the context of previous conversations.
Use this context to provide the best response to the client.

Context:
{context}

Client's message: {question}

Your response:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Function to generate response using the fine-tuned model and QA chain


def ask_openai(message):
    # Generate the response using the RetrievalQA chain
    response = qa_chain({"query": message})
    return response['result']

# View to check if the user is authenticated and not a superuser


def is_non_superuser(user):
    return user.is_authenticated and not user.is_superuser


@user_passes_test(is_non_superuser, login_url='login')
def chatbot(request):
    chats = Chat.objects.filter(user=request.user)
    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)
        chat = Chat(user=request.user, message=message,
                    response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            if user.is_superuser:
                error_message = "Vous êtes un superutilisateur. Veuillez vous connecter en tant qu'utilisateur régulier"
                return render(request, 'login.html', {'error_message': error_message})
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = "Nom d'utilisateur ou mot de passe invalide"
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Passwords don\'t match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')
