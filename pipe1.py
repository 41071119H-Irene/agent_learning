from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("好爽窩")

print(res)