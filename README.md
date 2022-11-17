# Reflect Not Reflex: Inference-Based Common Ground Improves Dialogue Response Quality
Data and Code for Paper "Reflect Not Reflex: Inference-Based Common Ground Improves Dialogue Response Quality" (EMNLP 2022)

[Project Website] (https://inklab.usc.edu/Reflect/)
[Paper] (https://inklab.usc.edu/Reflect/)

**Reflect** is a dataset that annotates dialogues with explicit CG (materialized as inferences approximating shared knowledge and beliefs) and solicits 9k diverse human-generated responses each following one common ground.

<img src="http://inklab.usc.edu/Reflect/reflect_data" width="700">

## Content
- `data` contains our main dataset (`data/organized_Reflect_9k_responses.json`) in json file. Each dictionary in the file contains the following keywards: 
    - `dialogue`: the dialogue history where each utterance is separated by `<br>`; 
    - `speaker`: the speaker name (note that our collected responses and reactions are from the perspective of `Friend`); 
    - `reaction_1`: the inference answer we collect in stage 1 following the questions `"How would you describe Speaker?"`
    - `reaction_2`: the inference answer we collect in stage 1 following the questions `"What might have happened before?"`
    - `reaction_1`: the inference answer we collect in stage 1 following the questions `"What might happen after?"`
    - `reaction_1`: the inference answer we collect in stage 1 following the questions `"What is Speaker feeling now?"`
    - `reaction_1`: the inference answer we collect in stage 1 following the questions `"What are you feeling now?"`
    - `responses_1` to `responses_5`: responses (3 for each inference dimension) we collect in stage 2 following each of the corresponding inference answer/reaction.
    - `utterances`: the dialogue history as a list of utterances (the first speaker is always the person in `speaker` ); 


- `exps` contains code we used to fine-tune BlenderBot on Reflect and some generated results from BlenderBot and GPT-3.



## Contact

Feel free to directly email peiz[at]usc[dot]edu if you have any feedback. 

## Citation
Please cite our EMNLP 2022 paper if you find this data helpful.
```
@inproceedings{zhou2022reflect,
		title={Reflect Not Reflex: Inference-Based Common Ground Improves Dialogue Response Quality},
		author={Zhou, Pei and Cho, Hyundong J. and Jandaghi, Pegah and Lee, Dong-Ho and Lin, Bill Yuchen and Pujara, Jay and Ren, Xiang},
		booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
		year={2022}
	  }
```


