import requests
import json
import os


def format_prompt(data):
    """Format the data into the model's expected prompt structure"""
    messages_text = "\n".join(
        [f"{msg['timestamp']} | {msg['content']}" for msg in data["messages"]]
    )

    prompt = f"""[INST] You are a chat summarization assistant. Given a conversation and its date range, create a concise yet comprehensive summary that captures the key points, emotional undertones, and progression of the relationship between participants.

Please summarize the following chat conversation that occurred between {data['start_date']} and {data['end_date']}.

[START DATE]
{data['start_date']}
[END DATE]
{data['end_date']}
[CHAT MESSAGES]
{messages_text} [/INST]
[SUMMARY]
"""

    return prompt


def test_inference_endpoint(jsonl_path):
    API_URL = "https://jzyutjh6xvrcylwx.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer hf_XXXXXXX",
        "Content-Type": "application/json",
    }

    # Read test data
    with open(jsonl_path, "r") as f:
        test_cases = [json.loads(line) for line in f]

    for test_case in test_cases:
        print(f"\nProcessing test case ID: {test_case.get('id', 'unknown')}")
        print(f"Date range: {test_case['start_date']} to {test_case['end_date']}")
        print("-" * 50)

        # Format the prompt
        formatted_prompt = format_prompt(test_case)

        payload = {
            "model": "tgi",
            "messages": [
                {"role": "user", "content": formatted_prompt},
            ],
            "max_tokens": 330,
            "temperature": 0.8,
            "stream": True,
        }

        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, stream=True
            )
            response.raise_for_status()

            print("Generated Summary:")
            summary = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        try:
                            json_response = json.loads(line)
                            if "choices" in json_response:
                                content = (
                                    json_response["choices"][0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    print(content, end="", flush=True)
                                    summary += content
                        except json.JSONDecodeError:
                            continue

            print("\n\n Expected Summary (Top of the Line Model):")
            print(test_case["summary"])
            print("\n" + "=" * 80 + "\n")

        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            if hasattr(e.response, "text"):
                print(f"Error details: {e.response.text}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_jsonl_path = "data/test/test.jsonl"
    test_inference_endpoint(test_jsonl_path)
