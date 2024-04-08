#- from transformers import AutoModelForCausalLM
from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
  "mistralai/Mistral-7B-Instruct-v0.2",
    use_fp8=True, 
    
)

# model_inputs = tokenizer(["How is autonomous vehicle technology transforming the future of transportation and urban planning?"], return_tensors="pt").to("cuda")

# generated_ids = model.generate(
#     **model_inputs, 
#     top_k=40, 
#     top_p=0.7, 
#     repetition_penalty=10,
# )

# tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))