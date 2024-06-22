import json
import requests
import streamlit as st
from templates.custom_css import CSSGenerator


class InferenceClient:
    def __init__(self, model_name):
        with open('./config/env.json', 'r') as file:
            self.env_vars = json.load(file)
        self.uua_url = self.env_vars["AICORE_AUTH_URL"]
        self.client_id = self.env_vars["AICORE_CLIENT_ID"]
        self.client_secret = self.env_vars["AICORE_CLIENT_SECRET"]
        self.tst_url = self.env_vars["TST_URL"]
        self.slm_url = self.env_vars["SLM_URL"]
        self.resource_group = self.env_vars["RESOURCE_GROUP"]
        self.model_name = model_name
        

    def get_token(self):
        params = {"grant_type": "client_credentials" }
        resp = requests.post(f"{self.uua_url}/oauth/token",
                            auth=(self.client_id, self.client_secret),
                            params=params)
        return resp.json()["access_token"]

    def get_headers(self):
        return {
            'Content-Type': 'application/json',
            'AI-Resource-Group': self.resource_group,
            'Authorization': f'Bearer {self.get_token()}'
        }

    def get_inference_url(self):
        suffix = '/v2/generate'
        if self.model_name == 'shakespeare-text-generator':
            return self.slm_url + suffix
        elif self.model_name == 'shakespeare-style-transfer':
            return self.tst_url + suffix
        else:
            raise ValueError("Invalid model name")

    def get_payload(self, max_tokens, temperature, top_k, top_p, prompt=None):
        if self.model_name == 'shakespeare-text-generator':
            return {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        elif self.model_name == 'shakespeare-style-transfer':
            return {
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        else:
            raise ValueError("Invalid model name")

def Run():
    st.title("Shakespearean Language Model")
    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox("Select the Shakespeare model you want:", ['shakespeare-text-generator', 'shakespeare-style-transfer'])

    st.sidebar.header("Parameters")
    max_tokens = st.sidebar.slider("Max Tokens", min_value=0, max_value=4096, value=250, step=10)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.5) 
    top_k = st.sidebar.slider("Top-K", min_value=0, max_value=50, value=0)
    top_p = st.sidebar.slider("Top-P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    
    infc = InferenceClient(model_name)

    custom_css = CSSGenerator.generate_custom_css(max_tokens)
    st.markdown(custom_css, unsafe_allow_html=True)

    if model_name == 'shakespeare-style-transfer':
        prompt = st.text_input("Prompt", key="prompt_input")
        if st.session_state.prompt_input:
            headers = infc.get_headers()
            inference_url = infc.get_inference_url()
            payload = infc.get_payload(max_tokens, temperature, top_k, top_p, prompt)

            response = requests.post(inference_url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()

                st.subheader("Shakespeare Style Text:")
                #st.write(f"Prompt: {data['prompt']}")
                lines = data['completion']
                formatted_text = "<br>".join(lines)
                styled_text = f'<div class="section"><ul class="list">{formatted_text}</ul></div>'
                st.markdown(styled_text, unsafe_allow_html=True)

                st.subheader("Metadata:")
                metadata_html = (
                    f"Model Name: {data['model_details']['model_name']}<br>"
                    f"Temperature: {data['model_details']['temperature']}<br>"
                    f"Length: {data['model_details']['length']}<br>"
                    f"Top-K: {data['model_details']['top_k']}<br>"
                    f"Top-P: {data['model_details']['top_p']}"
                )
                st.markdown(metadata_html, unsafe_allow_html=True)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
    else:
        prompt = None
        if st.sidebar.button("Generate") or (st.session_state.prompt_input if model_name == 'shakespeare-style-transfer' else False):
            headers = infc.get_headers()
            inference_url = infc.get_inference_url()
            payload = infc.get_payload(max_tokens, temperature, top_k, top_p, prompt)

            response = requests.post(inference_url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()

                st.subheader("Generated Text:")
                lines = data['generated_text']
                formatted_text = "<br>".join(lines)
                styled_text = f'<div class="section"><ul class="list">{formatted_text}</ul></div>'
                st.markdown(styled_text, unsafe_allow_html=True)

                st.subheader("Metadata:")
                metadata_html = (
                    f"Model Name: {data['model_details']['model_name']}<br>"
                    f"Temperature: {data['model_details']['temperature']}<br>"
                    f"Length: {data['model_details']['length']}<br>"
                    f"Top-K: {data['model_details']['top_k']}<br>"
                    f"Top-P: {data['model_details']['top_p']}"
                )
                st.markdown(metadata_html, unsafe_allow_html=True)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    Run()