import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline#, AutoConfig
from huggingface_hub import login
import psycopg2
# import time
import yolopandas
from yolopandas import pd
# import torch
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from accelerate import Accelerator
from pandasai import Agent
from pandasai.llm.local_llm import LocalLLM

# import pandas as pd

import pandasai as pai


load_dotenv()




class chat_llm():
    def __init__(self):
        self.llm = None
        self.hf_token = os.getenv("HF_LOGIN_TOKEN")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")
        self.db_table = os.getenv("DB_TABLE")
        

    def load_llm(self,model_id,provider):
        """
        Args: 
            model_id: huggingface model id e.g pandasai/bamboo-llm for hf and path to gguf file for llamacpp
        return:
            llm: LargeLanguageModel
        """
        # model_id = "meta-llama/CodeLlama-7b-Python-hf" # not loading cuase of memory issue 32GB GPU not working
        # model_id = "meta-llama/CodeLlama-7b-Instruct-hf" # Not running out of memory 32GBGPU
        # model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Out of memory
        # model_id = "mistralai/Mistral-7B-Instruct-v0.3" # Out of memory
        # model_id = "openai-community/gpt2" # Context window
        # model_id = "bigscience/bloom-7b1" # Cuda error
        # model_id = "meta-llama/Meta-Llama-3-8B" #Out of memory
        # model_id = "pandasai/bamboo-llm" #No package metadata was found for bitsandbytes
        # model_id = ""
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if provider == "hf":
            try:
                login(self.hf_token)
            except:
                pass
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"./{model_id}")
                model = AutoModelForCausalLM.from_pretrained(f"./{model_id}")
                # model.save_pretrained()
                # tokenizer.pad_token_id = tokenizer.eos_token_id
            #     model.to(device) # Load model on GPU
            except Exception as e:
                # os.environ["TRANSFORMERS_OFFLINE"] = "1"
                # model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                print("tokernizer")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_id)#,device_map="auto")
                except RuntimeError:
                    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=False)
                base_path = os.getcwd()
                full_path = os.path.join(base_path, model_id).replace("\\","/")
                CHECK_FOLDER = os.path.exists(full_path)
                if not CHECK_FOLDER:
                    os.makedirs(full_path)
                    print("folder")
                tokenizer.save_pretrained(f"./{model_id}")
                model.save_pretrained(f"./{model_id}")
                # print(e)
                # exit()
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("model")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
            print("pipeline")
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("llm loaded")
        elif provider == "llmcpp":
            n_gpu_layers = 1   # Metal set to 1 is enough.
            n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            # Make sure the model path is correct for your system!
            llm = LlamaCpp(
                # model_path="D:/Ammar Data/Metis/QA_BOT/gpt4all/wizardlm-13b-v1.2.Q4_0.gguf",
                # model_path = "d:/Ammar Data/Metis/QA_BOT/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf",
                # model_path = "D:/Ammar Data/Metis/QA_BOT/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf",
                model_path = model_id,
                # model_path="d:/Ammar Data/Metis/QA_BOT/llama3/Llama-3-16B-Instruct-v0.1.Q4_K_M.gguf",
                # model_path=r"d:/Ammar Data/Metis/QA_BOT/llama3/llama-3-8b-Instruct.Q4_K_M.gguf",
                # model_path="D:/Ammar Data/Metis/QA_BOT/gpt4all/gpt4all-falcon-newbpe-q4_0.gguf",
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                n_ctx=31000,
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                callback_manager=callback_manager,
            #     verbose=False,
                temperature=0.1,
                device=0,
                streaming=False
            )


            accelerator = Accelerator()
            self.llm = accelerator.prepare(llm)
            print("llm loaded")
        else:
            pai.clear_cache()

            self.llm = LocalLLM(api_base="http://localhost:1234/v1")
            # self.llm = LocalLLM(api_base="http://20.173.112.182:5000/v1")
            


    def create_llm_chain(self,model_id,provider):
        if self.llm is None:
            self.load_llm(model_id,provider)
        
        conn = psycopg2.connect(database=self.db_name, user=self.db_user, password=self.db_password, host=self.db_host)

        cur = conn.cursor()

        sql_query = f"SELECT * FROM {self.db_table};"
        df = pd.read_sql_query(sql_query, conn)

        conn.close()
        if provider == ["hf", "llmcpp"]:
            yolopandas.set_llm(self.llm)

            return df
        else:
            # df = pd.read_sql_query()
            agent = Agent(df, config={"llm": self.llm})
            return agent



