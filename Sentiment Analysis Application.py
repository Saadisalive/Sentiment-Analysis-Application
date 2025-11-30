import requests
from config import API_key
def calssify_text(text):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

    headers = {
        "Authorization": f"Bearer {API_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            result = response.json()[0]
            return [
                {"label": item["label"],"Score": round(item["score"], 3)}
                for item in result
            ]
        except Exception as e:
            return {"error": "UNexpected reponse format","details": response.json()}
    else:
        return {"error": {response.status_code} - {response.text}}
    
if __name__ == "__main__":
    sample_text = input("Enter text to analyze sentiment: ")
    result = calssify_text(sample_text)
    print(result)
    