Chat Memory 




# Chat Memory Fine Tuned llama3.1 8B

Tyler Gilman, [Add your names]

## Training
We will finetune the base model, NOT instruct version (instruct is a censored chatbot)
This will enable us to overfit a encoding that the server can parse into game moves.
Non-determanistic but should work.
The best way I know how to train is on collab (.ipynb), but if we can use any machine w > 30GB vram
https://www.youtube.com/watch?v=Us5ZFp16PaU&t=406s
We will need to train it to not only give us text responses, but an encoding that will influence the servers game state


## Encoding
We should use supervised lora, but can also do unsupervised.
Supervised will give it more recall, while unsupervised affects the personality as a whole (bad explaination)
I also saw something about how we can overfit for better results.
Here is a example prompt we would feed it. Need to manufacture data, but quality is much more important than quantity. Only need around 10-20 detailed examples.
*Main thing is good training data is necessarily for good model.*
""" \
[START DATE] \
{Encoded start date for the summary period} \
[END DATE] \
{Encoded end date for the summary period} \ 
[CHAT MESSAGES] \
{date_encoded_1} | {user_1}: {message_1} \ 
{date_encoded_2} | {user_2}: {message_2} \
{date_encoded_3} | {user_1}: {message_3} \
...
{date_encoded_n} | {user_n}: {message_n} \
[SUMMARY] \
{Concise summary of the chat messages within the specified date range} \
"""
