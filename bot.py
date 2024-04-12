import streamlit as st
from streamlit_chat import message
import json
from pathlib import Path
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
os.environ["OPENAI_API_KEY"] = <openai-key>
from openai import OpenAI

genmed = 'GenMedGPT.json'
meddia = 'MedDialogue-5k.json'
embeddings = HuggingFaceEmbeddings(model_name="bge-small-en-v1.5")
db = FAISS.load_local("faiss_index", embeddings)

# RAG
def RAG(input,k=5):
    ret_docs = db.max_marginal_relevance_search(input,k)
    rag = []
    for doc in ret_docs:
        if doc.metadata["source"] == '/kaggle/input/raw-medicalqa/GenMedGPT.json':
            context = json.loads(Path(genmed).read_text())[doc.metadata['seq_num']]
            input = context["input"]
            output = context["output"]
            rag.append(f"patient: {input}\ndoctor: {output}")
        if doc.metadata["source"] == '/kaggle/input/raw-medicalqa/MedDialog.json':
            context = json.loads(Path(meddia).read_text())[doc.metadata['seq_num']]
            input = context["input"]
            output = context["output"]
            rag.append(f"Patient: {input}\nDoctor: {output}")
    rag = "\n\n".join(rag)
    return rag


# Prompt 
def prompt_template(prompt):
    rag = RAG(prompt)
    prompt_ = f'''system message: you are a doctor who provide solution to patient's symptom.
Here is some relevant talk between doctors and patients.
####
{rag}
####
user message: Patient: {prompt}.
Assistant role: Doctor: '''
    return prompt_
    
# query GPT
def generate_response(prompt):
  '''
  This function input a prompt, call the api of openai and response.
  '''
  client = OpenAI()
  completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {"role": "system", "content": "you are an expert doctor, Waston."},
          {"role": "user", "content": prompt}
        ],
          temperature=1e-7,
          presence_penalty =1.1,
          top_p =1
      )
  return completion.choices[0].message.content

# Extraction Part
# format constraint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
response_schemas = [
    ResponseSchema(
        name="name",
        description="the name of patient",
        type="list"
    ),
    ResponseSchema(
        name="age", description="the age of patient",
        type="list"
    ),
    ResponseSchema(
        name="symptoms", description="the symptoms experienced by patient",
        type="list"
    ),    
    ResponseSchema(
        name="precautions", description="the precaution provided by doctor",
        type="list"
    ),
    ResponseSchema(
        name="drugs", description="drugs or medications prescribed by the doctor",
        type="list"
    )
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions=output_parser.get_format_instructions()

# extraction function
def auto_gpt_extract(context):
    client = OpenAI()
    prompt = f'''
do the tasks step by step
correct the grammar if necessary, search for the potention norms, judge whether they are the patient's name, patient's age, patient'scondition, patient's symptoms, precautions and drug advised by doctor from the note .
take a breath
extract them from the text, do not include the adjective, do not include adverb, do not include targets or results
input = {context}
{format_instructions}
output = 
'''    
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {"role": "system", "content": "you are a doctor trying to extract information from your clinical notes"},
        {"role": "user", "content": prompt}
      ],
        temperature=0,
        presence_penalty =1.1
    )
    response = completion.choices[0].message.content
    return response
    

st.title("MedGPT")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input=st.text_input("You:",key='input')
command = "wrap it up"
history = ""
for i in range(len(st.session_state['generated'])):
    if str(st.session_state['past'][i]) == command:
        pass
    history += "Patient: "+str(st.session_state['past'][i])+"\n"+"Doctor: "+str(st.session_state['generated'][i])+"\n"
if user_input:
    if str(user_input).lower() == command:
        dialogue = ""
        for i in range(len(st.session_state['generated'])):
            if str(st.session_state['past'][i]) == command:
                pass
            dialogue += "Patient: "+str(st.session_state['past'][i])+"\n"+"Doctor: "+str(st.session_state['generated'][i])+"\n"
        output = auto_gpt_extract(dialogue)
    else:
        prompt = prompt_template(user_input)
        final_prompt = f"Chat History:{history}\n\n"+prompt
        output=generate_response(final_prompt)
        print("__________")
        print(final_prompt)
        print("___________")
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
  
