import json

in_file = open('data/setting2_test.json')

for line in in_file:
    json_instance = json.loads(line)

    for utr in json_instance['context']:
        print(utr)
    print('')
    for inf_id in json_instance['inferences'].keys():
        print(json_instance['inferences'][inf_id])
        for resp in json_instance['GPT_human_prompts'][inf_id]['responses']:
            print('\t' + resp['text'].replace('\n', ''))
    print('----------------------------------------------')

