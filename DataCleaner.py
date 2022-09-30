import json
with open('D:\\test\\train-v2.0.json') as json_file:
    data = json.load(json_file)

print(type(data['data']))

dataset = {}
intents = []

beta = ""

alpha = 0
for intent in data['data']:
    tagname = intent['title']


    #intents[-1]['tag'] = tagname


    for i in intent['paragraphs']:
        #print(i.keys())  # qas and context
        intents.append(dict())
        intents[-1]['tag'] = alpha
        alpha += 1

        for qas in i['qas']:
            #print(qas.keys())
            if 'question' in qas:
                intents[-1]['patterns'] = list()
                intents[-1]['patterns'].append(qas['question'])
                #if "When did Beyonce start becoming popular?" in qas['question']:
                #    print(len(intents))
                #    print(intents[0])
            if 'plausible_answers' in qas:
                intents[-1]['responses'] = list()
                intents[-1]['responses'].append(qas['plausible_answers'][0]['text'])
            elif 'answers' in qas:
                intents[-1]['responses'] = list()
                intents[-1]['responses'].append(qas['answers'][0]['text'])
        beta += str(intents[-1])
        if alpha == 1:
            print(intents[0])

#if 'When did Beyonce start becoming popular?' in beta:
#    print("hola")

#for i in intents:
 #   for j in i['patterns']:
 #       if j[0] == "When did Beyonce start becoming popular?":
 #           print("Yes")
#print(intents[0])



dataset['intents'] = intents[:100]
print(type(dataset))
#json_object = json.dumps(dataset, indent = 4) 
#print(json_object)
with open("sample2.json", "w") as outfile:
    json.dump(dataset, outfile)

