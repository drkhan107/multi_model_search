
def get_llms():
    import os
    from dotenv import load_dotenv
    

    from langchain_google_genai import ChatGoogleGenerativeAI
   

    load_dotenv()
    model_name=["gemini-2.0-flash","gemini-2.0-flash-lite","gemini-2.5-pro-exp-03-25", "gemini-2.5-flash-preview-04-17"]
    api_key=os.getenv("GOOGLE_API_KEY")
    llms=[]
    for model in model_name:
        try:
            llm= ChatGoogleGenerativeAI(model=model,
                                                temperature=0.3,
                                                max_tokens=2106,
                                                timeout=None,
                                                max_retries=2,
                                                google_api_key=api_key,
                                            )
            llms.append(llm)
        except:
            print("Exception")
    
    
    return llms

import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')