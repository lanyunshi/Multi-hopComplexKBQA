# **Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases**
This is the code for the paper:
**[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases]
()**\
Yunshi Lan, Jing Jiang\
[ACL 2020](https://acl2020.org/) .

If you find this code useful in your research, please cite
>@inproceedings{lan:acl2020,\
>title={Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases},\
>author={Lan, Yunshi and Jiang, Jing},\
>booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},\
>year={2020}\
>}

## **Setups**
All code was developed and tested in the following environment.
- Ubuntu 16.04
- Python 3.7.1
- Pytorch 1.1.0

Download the code and data:
```
git clone https://github.com/lanyunshi/Multi-hopComplexKBQA.git
pip install requirements.txt
```

## **Download Pre-processed Data**
We evaluate our methods on [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763), [ComplexWebQuestions](https://www.tau-nlp.org/compwebq) and [ComplexQuestions](https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions).

The processed data can be downloaded from [link](https://drive.google.com/drive/folders/1sAOUiFbk2ujfXUityIq51p14j9E4HCZ8?usp=sharing). Please put the folders under the path *data/*.
There are folders for the splits of each dataset, which denote as *splitname_datasetname*. Each folder contains:

- **q.txt**: The file of questions.
- **te.txt**: The file of topic entities.
- **con.txt**: The file of detected constraints, which usually cannot be detected as topic entities.
- **a.txt**: The file of answers.
- **g.txt**: The file of ground truth query graphs (No such file is provided for ComplexQuestions).

## **Save Freebase dump in Your Machine**
As we are querying Freebase dynamically, you need install a database engine in your machine. We are using [Virtuoso Open-Source](https://github.com/openlink/virtuoso-opensource). You can follow the instruction to install.
Once you installed the database engine, you can download freebase dump from [link](https://developers.google.com/freebase) and setup your database.
After the database is installed, you need to replace the *SPARQLPATH* in *code/SPARQL_test.py* file with *your/path/to/database*.
To do a quick test, you can run:
```
python code/SPARQL_test.py
```
The output is as follows:
```
{'head': {'link': [], 'vars': ['r']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'r': {'type': 'uri', 'value': 'http://rdf.freebase.com/ns/meteorology.forecast_zone.weather_service'}}]}}
```

Please note if you fail to save Freebase dump in your machine, you **cannot train** a new model using our code.
But you can **still test** our pre-trained models by using our searched caches (Maybe with little performance loss).
The caches of tested datasets are stored in *data/datasetname*. Each folder contains:

- **kb_cache.json**: A dictionary of the cache of searched query graphs *{query: answer}*.
- **m2n_cache.json**: A dictionary of the cache of searched mid *{mid: surface name}*.
- **query_cache.json**: A set of the cache of search query *set(query)*.

*CWQ/kb_cache.json* is a cache on ComplexWebQuestions collected during tesing, which is to validate our pre-trained model. The complete cache collected during training and testing is too much to update.

## **Download Pre-trained Model**
You can download our pre-trained models from the [link](https://drive.google.com/drive/folders/1Kw1kNXR6IaFTmjLDoPcb_YcwWt8NTc8t?usp=sharing) and put the folders under the path *trained_model/*.

## **Test the Pre-trained Model**
To test our pre-trained model, simply run shell files:
```
./[CWQ|WBQ|CQ]_Runner.sh
```
The predicted query graphs are saved in *trained_model/Best_predcp.txt*. Following the official evaluation matrix of WebQuestionsSP, we measure [hit@1|accuracy|precision|recall|f1] based on predicted query graph.
You can simply run:
```
python code/Evaluation.py \
    --data_path trained_model/[CWQ|WBQ|CQ] \
    --data_file Best \
    --mode eval \
```
If there is no any performance loss due to the absence of the database engine, we can obtain results as follows:

|Dataset|CWQ|WBQ|CQ|
|---|---|---|---|
|Hit@1|39.3|73.3|41.3|
|Accuracy|30.0|66.1|34.0|
|Precision|37.8|73.6|42.0|
|Recall|55.2|80.5|56.6|
|F1|40.4|73.8|43.3|

Please note the hit@1 is calculated based on the accumulated scores of the predicted answers. If there are multiple answers with same scores, we randomly sample one from them.

For official hit@1 of ComplexWebQuestions reported in the paper, instead of considering the top1 ranked query graph, we save top2 ranked query graphs and extract answers based on the accumualted scores. Then we transform the mid answer to surface name answers and send to the author for evaluation following the instruction in their [Webpage](https://www.tau-nlp.org/compwebq).

- Add command ``*--top_k 2*'' to *./CWQ_Runner.sh* and re-run *./CWQ_Runner.sh*
- Run code to re-format the answers:
```
python code/Evaluation.py \
    --data_path trained_model/CWQ \
    --data_file Best \
    --mode trans
```

## **Train a New Model**
If you want to train your model, for example WebQuestionsSP, you can input
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_WBQ \
        --dev_folder data/dev_WBQ \
        --test_folder data/test_WBQ \
        --vocab_file data/WBQ/vocab.txt \
        --KB_file data/WBQ/kb_cache.json \
        --M2N_file data/WBQ/m2n_cache.json \
        --QUERY_file data/WBQ/query_cache.json \
        --output_dir trained_model/WBQ \
        --config config/bert_config.json \
        --gpu_id 1\
        --save_model my_model \
        --max_hop_num 2 \
        --num_train_epochs 50 \
        --do_train 1 \
        --do_eval 1\
        --do_policy_gradient 1
```

It takes long time to train the model at the first begining as random exploration is conducting, especially on ComplexWebQuestions. With the convergence of the model, the query time will decrease and the training processing becomes faster and faster.
